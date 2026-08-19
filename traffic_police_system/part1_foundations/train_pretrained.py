"""
train_pretrained.py — ResNet-18 with ImageNet weights (COMPARISON BASELINE ONLY).

NOT for deployment. Used side-by-side with scratch model to show accuracy gap.

Usage:
    py -3.11 train_pretrained.py [--epochs 20] [--batch 64]

Output:
    checkpoints/pretrained_best.pth
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from part1_foundations.dataset import (
    get_dataloader, get_pos_weights, NUM_CLASSES, CLASSES
)
from part1_foundations.train_scratch import evaluate, get_scheduler

CKPT_DIR = Path(__file__).parent.parent / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[BASELINE] Device: {device}")

    train_loader = get_dataloader("train", batch_size=args.batch,
                                  num_workers=args.workers, img_size=224)
    val_loader   = get_dataloader("valid", batch_size=args.batch * 2,
                                  num_workers=args.workers, img_size=224)

    # ImageNet-pretrained backbone
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    feat_dim = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(feat_dim, NUM_CLASSES),
    )
    model = model.to(device)

    pos_weights = get_pos_weights("train", device=device)
    criterion   = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # Fine-tune: lower LR for backbone, higher for head
    head_params     = list(model.fc.parameters())
    backbone_params = [p for p in model.parameters() if not any(p is h for h in head_params)]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": 1e-4},
        {"params": head_params,     "lr": 1e-3},
    ], weight_decay=1e-4)

    scheduler = get_scheduler(optimizer, args.epochs, warmup=2)
    scaler    = GradScaler(enabled=(device.type == "cuda"))

    best_mAP = 0.0
    print(f"\n[BASELINE] ImageNet fine-tune: {args.epochs} epochs, batch={args.batch}")
    print("-" * 70)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for imgs, labels in tqdm(train_loader,
                                  desc=f"[BASELINE] Epoch {epoch+1}/{args.epochs}",
                                  leave=False):
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda")):
                logits = model(imgs)
                loss   = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        scheduler.step()

        if (epoch + 1) % 2 == 0 or (epoch + 1) == args.epochs:
            val_results, val_mAP = evaluate(model, val_loader, device)
            elapsed = time.time() - t0
            print(f"[BASELINE] Epoch {epoch+1:>3}/{args.epochs}  "
                  f"loss={total_loss/len(train_loader):.4f}  "
                  f"mAP={val_mAP:.4f}  time={elapsed:.1f}s")
            for cls, m in val_results.items():
                print(f"  {cls:<12} P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")

            if val_mAP > best_mAP:
                best_mAP = val_mAP
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "val_results": val_results,
                    "val_mAP": val_mAP,
                }, CKPT_DIR / "pretrained_best.pth")
                print(f"  ✓ New best! pretrained_best.pth (mAP={best_mAP:.4f})")

    print(f"\n[BASELINE] Done. Best val mAP: {best_mAP:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ImageNet baseline (comparison only)")
    parser.add_argument("--epochs",  type=int, default=20)
    parser.add_argument("--batch",   type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    train(args)
