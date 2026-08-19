"""
pretrain_simclr.py — SimCLR self-supervised pretraining on all 18,998 images.

No labels used. Uses the NT-Xent (InfoNCE) loss.
Saves: checkpoints/simclr_backbone.pth  (backbone weights, no projection head)

Usage:
    py -3.11 pretrain_simclr.py [--epochs 50] [--batch 128] [--img-size 128]
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from part1_foundations.dataset import SimCLRDataset

CKPT_DIR = Path(__file__).parent.parent / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)


# ─── SimCLR Projection Head ───────────────────────────────────────────────────

class SimCLRModel(nn.Module):
    def __init__(self, proj_dim: int = 128):
        super().__init__()
        self.backbone = resnet18(weights=None)
        feat_dim = self.backbone.fc.in_features  # 512 for ResNet-18
        self.backbone.fc = nn.Identity()          # strip classifier

        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, proj_dim),
        )

    def forward(self, x):
        h = self.backbone(x)       # (B, 512)
        z = self.projector(h)      # (B, proj_dim)
        return F.normalize(z, dim=1)


# ─── NT-Xent Loss ─────────────────────────────────────────────────────────────

class NTXentLoss(nn.Module):
    """Normalized temperature-scaled cross-entropy (SimCLR loss)."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temp = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)          # (2B, D)
        sim = torch.mm(z, z.T) / self.temp       # (2B, 2B)

        # Mask out self-similarity
        mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
        sim.masked_fill_(mask, -1e9)

        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.arange(B, device=z.device)
        labels = torch.cat([labels + B, labels])  # (2B,)

        loss = F.cross_entropy(sim, labels)
        return loss


# ─── Training ─────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    dataset = SimCLRDataset(img_size=args.img_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    model = SimCLRModel(proj_dim=128).to(device)
    criterion = NTXentLoss(temperature=0.07)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # Resume if checkpoint exists
    resume_epoch = 0
    ckpt_path = CKPT_DIR / "simclr_latest.pth"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        resume_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {resume_epoch}")

    print(f"\nStarting SimCLR pretraining: {args.epochs} epochs, "
          f"batch={args.batch}, img={args.img_size}px")
    print("-" * 60)

    for epoch in range(resume_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for (v1, v2) in tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            v1, v2 = v1.to(device, non_blocking=True), v2.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda")):
                z1 = model(v1)
                z2 = model(v2)
                loss = criterion(z1, z2)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)
        elapsed = time.time() - t0

        if device.type == "cuda":
            mem_mb = torch.cuda.max_memory_allocated() / 1e6
            print(f"Epoch {epoch+1:>3}/{args.epochs}  loss={avg_loss:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  "
                  f"time={elapsed:.1f}s  peak_vram={mem_mb:.0f}MB")
        else:
            print(f"Epoch {epoch+1:>3}/{args.epochs}  loss={avg_loss:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  time={elapsed:.1f}s")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
            ckpt_data = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "loss": avg_loss,
            }
            torch.save(ckpt_data, ckpt_path)
            # Also save backbone-only weights at final epoch
            if (epoch + 1) == args.epochs:
                backbone_path = CKPT_DIR / "simclr_backbone.pth"
                torch.save(model.backbone.state_dict(), backbone_path)
                print(f"\nBackbone saved → {backbone_path}")

    print("\nSimCLR pretraining complete!")
    print(f"Backbone weights: {CKPT_DIR / 'simclr_backbone.pth'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR pretraining")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch",  type=int, default=128)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    train(args)
