"""
train_scratch.py — ResNet-18 multi-label classifier trained FROM SCRATCH.

Stage:
  1. Loads SimCLR backbone weights (from pretrain_simclr.py output)
  2. Attaches a 4-class head
  3. Trains with BCEWithLogitsLoss + pos_weight, MixUp, AMP, cosine LR

Usage:
    py -3.11 train_scratch.py [--epochs 80] [--batch 64]

Output:
    checkpoints/scratch_best.pth   ← best val mAP checkpoint
    checkpoints/scratch_latest.pth ← most recent checkpoint
"""

import argparse
import sys
import time
from pathlib import Path

SITE_PKGS = Path(__file__).parent.parent / "site_pkgs"
if SITE_PKGS.exists() and str(SITE_PKGS) not in sys.path:
    sys.path.insert(0, str(SITE_PKGS))

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torchvision.models import resnet18
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from part1_foundations.dataset import (
    get_dataloader, get_pos_weights, mixup_batch, NUM_CLASSES, CLASSES
)

CKPT_DIR = Path(__file__).parent.parent / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)


# ─── Model ────────────────────────────────────────────────────────────────────

def build_model(simclr_backbone_path: Path = None) -> nn.Module:
    model = resnet18(weights=None)
    feat_dim = model.fc.in_features  # 512
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(feat_dim, NUM_CLASSES),
    )

    if simclr_backbone_path and simclr_backbone_path.exists():
        state = torch.load(simclr_backbone_path, map_location="cpu")
        # Load backbone weights (fc will mismatch — ignore it)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded SimCLR backbone from {simclr_backbone_path}")
        print(f"  Missing keys (expected — new head): {len(missing)}")
        print(f"  Unexpected keys: {len(unexpected)}")
    else:
        print("No SimCLR backbone found — training fully from scratch (cold init)")
        # Xavier init for better convergence than default kaiming
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    return model


# ─── LR Schedule (cosine with linear warmup) ──────────────────────────────────

def get_scheduler(optimizer, epochs: int, warmup: int = 5):
    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

class EarlyStopper:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None

    def __call__(self, score):
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# ─── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def calibrate_thresholds(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        with autocast(enabled=(device.type == "cuda")):
            logits = model(imgs)
        probs = torch.sigmoid(logits).cpu()
        all_probs.append(probs)
        all_labels.append(labels)
        
    probs_np = torch.cat(all_probs).numpy()
    labels_np = torch.cat(all_labels).numpy()
    
    best_thresholds = [0.5] * NUM_CLASSES
    for i, cls in enumerate(CLASSES):
        best_f1 = 0
        best_t = 0.5
        for t in np.arange(0.1, 0.95, 0.05):
            preds = (probs_np[:, i] >= t).astype(float)
            tp = ((preds == 1) & (labels_np[:, i] == 1)).sum()
            fp = ((preds == 1) & (labels_np[:, i] == 0)).sum()
            fn = ((preds == 0) & (labels_np[:, i] == 1)).sum()
            p = tp / max(tp + fp, 1)
            r = tp / max(tp + fn, 1)
            f1 = 2 * p * r / max(p + r, 1e-8)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        best_thresholds[i] = float(best_t)
    return best_thresholds

@torch.no_grad()
def evaluate(model, loader, device, thresholds=None):
    model.eval()
    if thresholds is None:
        thresholds = [0.5] * NUM_CLASSES
    threshold_tensor = torch.tensor(thresholds).float()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        with autocast(enabled=(device.type == "cuda")):
            logits = model(imgs)
        preds = (torch.sigmoid(logits).cpu() >= threshold_tensor).float()
        all_preds.append(preds)
        all_labels.append(labels)

    preds_np  = torch.cat(all_preds).numpy()
    labels_np = torch.cat(all_labels).numpy()

    # Per-class precision, recall, F1
    results = {}
    for i, cls in enumerate(CLASSES):
        tp = ((preds_np[:, i] == 1) & (labels_np[:, i] == 1)).sum()
        fp = ((preds_np[:, i] == 1) & (labels_np[:, i] == 0)).sum()
        fn = ((preds_np[:, i] == 0) & (labels_np[:, i] == 1)).sum()
        p  = tp / max(tp + fp, 1)
        r  = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-8)
        results[cls] = {"precision": p, "recall": r, "f1": f1}

    mAP = np.mean([v["f1"] for v in results.values()])
    return results, mAP


# ─── Training Loop ────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Data
    train_loader = get_dataloader("train", batch_size=args.batch,
                                  num_workers=args.workers, img_size=224)
    val_loader   = get_dataloader("valid", batch_size=args.batch * 2,
                                  num_workers=args.workers, img_size=224)

    # Model
    backbone_path = CKPT_DIR / "simclr_backbone.pth"
    model = build_model(backbone_path).to(device)

    # Loss with class balancing
    pos_weights = get_pos_weights("train", device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = get_scheduler(optimizer, args.epochs, warmup=5)
    scaler    = GradScaler(enabled=(device.type == "cuda"))

    # Resume
    resume_epoch = 0
    best_mAP = 0.0
    latest_path = CKPT_DIR / "scratch_latest.pth"
    if latest_path.exists() and not args.fresh:
        ckpt = torch.load(latest_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        resume_epoch = ckpt["epoch"] + 1
        best_mAP = ckpt.get("best_mAP", 0.0)
        print(f"Resumed from epoch {resume_epoch}, best_mAP={best_mAP:.4f}")

    early_stopper = EarlyStopper(patience=args.patience)

    GRAD_ACCUM = 2  # effective batch = args.batch * 2 = 128

    print(f"\nTraining from-scratch ResNet-18: {args.epochs} epochs, "
          f"batch={args.batch} (accum={GRAD_ACCUM}→eff {args.batch*GRAD_ACCUM})")
    print("-" * 70)

    for epoch in range(resume_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        t0 = time.time()

        for step, (imgs, labels) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        ):
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # MixUp
            imgs, labels = mixup_batch(imgs, labels, alpha=0.2)

            with autocast(enabled=(device.type == "cuda")):
                logits = model(imgs)
                loss   = criterion(logits, labels) / GRAD_ACCUM

            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * GRAD_ACCUM

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # Print GPU memory after first epoch
        if epoch == 0 and device.type == "cuda":
            peak_mb = torch.cuda.max_memory_allocated() / 1e6
            print(f"\n  *** Peak VRAM after epoch 1: {peak_mb:.0f} MB "
                  f"({100*peak_mb/6144:.1f}% of 6144 MB) ***\n")

        # Validate every 2 epochs
        if (epoch + 1) % 2 == 0 or (epoch + 1) == args.epochs:
            thresholds = calibrate_thresholds(model, val_loader, device)
            val_results, val_mAP = evaluate(model, val_loader, device, thresholds=thresholds)
            elapsed = time.time() - t0
            lr_now  = scheduler.get_last_lr()[0]
            mem_mb  = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else 0

            print(f"Epoch {epoch+1:>3}/{args.epochs}  "
                  f"loss={avg_loss:.4f}  "
                  f"mAP={val_mAP:.4f}  "
                  f"lr={lr_now:.2e}  "
                  f"time={elapsed:.1f}s  "
                  f"VRAM={mem_mb:.0f}MB")

            for i, cls in enumerate(CLASSES):
                m = val_results[cls]
                print(f"  {cls:<12} (thr={thresholds[i]:.2f}) P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")

            # Save best
            if val_mAP > best_mAP:
                best_mAP = val_mAP
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "val_results": val_results,
                    "val_mAP": val_mAP,
                    "best_mAP": best_mAP,
                    "thresholds": thresholds,
                }, CKPT_DIR / "scratch_best.pth")
                print(f"  [OK] New best! scratch_best.pth (mAP={best_mAP:.4f})")
                
            if early_stopper(val_mAP):
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break
        else:
            elapsed = time.time() - t0
            print(f"Epoch {epoch+1:>3}/{args.epochs}  loss={avg_loss:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  time={elapsed:.1f}s")

        # Save latest every 5 epochs
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_mAP": best_mAP,
                "thresholds": thresholds if 'thresholds' in locals() else [0.5]*NUM_CLASSES,
            }, latest_path)

    print(f"\nTraining complete! Best val mAP: {best_mAP:.4f}")
    print(f"Best checkpoint: {CKPT_DIR / 'scratch_best.pth'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet-18 from scratch")
    parser.add_argument("--epochs",  type=int, default=80)
    parser.add_argument("--batch",   type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fresh",   action="store_true", help="Ignore existing checkpoint")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    args = parser.parse_args()
    train(args)
