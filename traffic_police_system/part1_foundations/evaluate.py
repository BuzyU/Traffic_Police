"""
evaluate.py — Print side-by-side per-class metrics for both models.

Usage:
    py -3.11 evaluate.py          # uses test split by default
    py -3.11 evaluate.py --split valid
    py -3.11 evaluate.py --tta    # use test-time augmentation (5 views)
"""

import argparse
import sys
from pathlib import Path

SITE_PKGS = Path(__file__).parent.parent / "site_pkgs"
if SITE_PKGS.exists() and str(SITE_PKGS) not in sys.path:
    sys.path.insert(0, str(SITE_PKGS))

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from part1_foundations.dataset import (
    VehicleDataset, get_val_transform, get_train_transform,
    NUM_CLASSES, CLASSES
)
from part1_foundations.train_scratch import build_model

CKPT_DIR = Path(__file__).parent.parent / "checkpoints"


# ─── Test-Time Augmentation ───────────────────────────────────────────────────

@torch.no_grad()
def predict_tta(model, loader, device, n_views: int = 5, threshold: float = 0.5):
    """Average predictions over n_views augmented versions of each image."""
    model.eval()
    all_preds, all_labels = [], []

    for imgs, labels in tqdm(loader, desc="TTA inference", leave=False):
        # imgs is the base-transform version; re-augment n_views times
        B = imgs.size(0)
        probs_accum = torch.zeros(B, NUM_CLASSES)

        with autocast(enabled=(device.type == "cuda")):
            for _ in range(n_views):
                logits = model(imgs.to(device, non_blocking=True))
                probs  = torch.sigmoid(logits).cpu()
                probs_accum += probs

        avg_probs = probs_accum / n_views
        preds = (avg_probs >= threshold).float()
        all_preds.append(preds)
        all_labels.append(labels)

    preds_np  = torch.cat(all_preds).numpy()
    labels_np = torch.cat(all_labels).numpy()
    return preds_np, labels_np


@torch.no_grad()
def predict_standard(model, loader, device, threshold: float = 0.5):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in tqdm(loader, desc="Inference", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        with autocast(enabled=(device.type == "cuda")):
            logits = model(imgs)
        preds = (torch.sigmoid(logits).cpu() >= threshold).float()
        all_preds.append(preds)
        all_labels.append(labels)
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


def compute_metrics(preds_np, labels_np):
    results = {}
    for i, cls in enumerate(CLASSES):
        tp = ((preds_np[:, i] == 1) & (labels_np[:, i] == 1)).sum()
        fp = ((preds_np[:, i] == 1) & (labels_np[:, i] == 0)).sum()
        fn = ((preds_np[:, i] == 0) & (labels_np[:, i] == 1)).sum()
        p  = float(tp) / max(tp + fp, 1)
        r  = float(tp) / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-8)
        results[cls] = {"precision": p, "recall": r, "f1": f1, "tp": int(tp), "fp": int(fp), "fn": int(fn)}
    mAP = float(np.mean([v["f1"] for v in results.values()]))
    return results, mAP


def load_scratch_model(device):
    path = CKPT_DIR / "scratch_best.pth"
    if not path.exists():
        return None, None
    model = build_model(simclr_backbone_path=None).to(device)
    ckpt  = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt


def load_pretrained_model(device):
    path = CKPT_DIR / "pretrained_best.pth"
    if not path.exists():
        return None, None
    model = resnet18(weights=None)
    import torch.nn as nn
    model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.fc.in_features, NUM_CLASSES))
    model = model.to(device)
    ckpt  = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt


def print_table(scratch_results, scratch_mAP, pretrained_results, pretrained_mAP):
    W = 78
    print("=" * W)
    print("  PER-CLASS METRICS COMPARISON (threshold=0.5)")
    print("=" * W)
    print(f"  {'Class':<13}  {'FROM-SCRATCH':^28}  {'PRETRAINED (baseline)':^28}")
    print(f"  {'':<13}  {'P':>6} {'R':>6} {'F1':>6} {'':>6}  {'P':>6} {'R':>6} {'F1':>6}")
    print(f"  {'-'*70}")

    for cls in CLASSES:
        s = scratch_results.get(cls, {}) if scratch_results else {}
        p = pretrained_results.get(cls, {}) if pretrained_results else {}
        s_str = (f"{s['precision']:6.3f} {s['recall']:6.3f} {s['f1']:6.3f}"
                 if s else "   N/A    N/A    N/A")
        p_str = (f"{p['precision']:6.3f} {p['recall']:6.3f} {p['f1']:6.3f}"
                 if p else "   N/A    N/A    N/A")
        print(f"  {cls:<13}  {s_str}  {'':>6}  {p_str}")

    print(f"  {'-'*70}")
    s_map = f"{scratch_mAP:.4f}" if scratch_mAP is not None else "   N/A"
    p_map = f"{pretrained_mAP:.4f}" if pretrained_mAP is not None else "   N/A"
    print(f"  {'mean-F1 (mAP)':<13}  {s_map:^28}  {p_map:^28}")
    print("=" * W)


def run_evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on '{args.split}' split  |  TTA={args.tta}")
    print(f"Device: {device}\n")

    # TTA loader uses augmentation transforms; standard uses val transforms
    if args.tta:
        ds = VehicleDataset(args.split, transform=get_train_transform(224))
    else:
        ds = VehicleDataset(args.split, transform=get_val_transform(224))
    loader = DataLoader(ds, batch_size=64, shuffle=False,
                        num_workers=args.workers, pin_memory=(device.type=="cuda"))

    scratch_results, scratch_mAP       = None, None
    pretrained_results, pretrained_mAP = None, None

    # Evaluate scratch model
    scratch_model, scratch_ckpt = load_scratch_model(device)
    if scratch_model:
        print("Evaluating FROM-SCRATCH model...")
        if args.tta:
            preds, labels = predict_tta(scratch_model, loader, device, n_views=5)
        else:
            preds, labels = predict_standard(scratch_model, loader, device)
        scratch_results, scratch_mAP = compute_metrics(preds, labels)
        print(f"  Scratch model  mAP (mean-F1) = {scratch_mAP:.4f}")
    else:
        print("[WARNING] scratch_best.pth not found - run train_scratch.py first")

    # Evaluate pretrained baseline
    pretrained_model, pretrained_ckpt = load_pretrained_model(device)
    if pretrained_model:
        print("Evaluating PRETRAINED BASELINE model...")
        if args.tta:
            preds, labels = predict_tta(pretrained_model, loader, device, n_views=5)
        else:
            preds, labels = predict_standard(pretrained_model, loader, device)
        pretrained_results, pretrained_mAP = compute_metrics(preds, labels)
        print(f"  Pretrained model mAP (mean-F1) = {pretrained_mAP:.4f}")
    else:
        print("[WARNING] pretrained_best.pth not found - run train_pretrained.py first")

    print()
    print_table(scratch_results, scratch_mAP, pretrained_results, pretrained_mAP)

    if scratch_mAP and pretrained_mAP:
        gap = pretrained_mAP - scratch_mAP
        sign = "+" if gap >= 0 else "-"
        print(f"\n  Accuracy gap (pretrained − scratch): {sign}{abs(gap):.4f} mAP\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and compare models")
    parser.add_argument("--split",   default="test", choices=["train", "valid", "test"])
    parser.add_argument("--tta",     action="store_true", help="Use test-time augmentation")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run_evaluation(args)
