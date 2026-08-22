import os
import csv
import sys
from pathlib import Path
from typing import Optional, Tuple

SITE_PKGS = Path(__file__).parent.parent / "site_pkgs"
if SITE_PKGS.exists() and str(SITE_PKGS) not in sys.path:
    sys.path.insert(0, str(SITE_PKGS))

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

DATASET_ROOT = Path(os.environ.get("TRAFFIC_DATASET_ROOT", r"C:\Users\Umer Zingu\Desktop\Learning\Traffic_Police\Vehicles-coco.v2i.multiclass"))
CLASSES = ["bus", "car", "motorcycle", "truck"]
NUM_CLASSES = len(CLASSES)

CLASS_COLORS_RGB = {
    "bus":        (0,   0,   0),
    "car":        (255, 220, 0),
    "motorcycle": (30,  100, 255),
    "truck":      (220, 30,  30),
}
CLASS_COLORS_BGR = {k: (v[2], v[1], v[0]) for k, v in CLASS_COLORS_RGB.items()}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
    ])


def get_val_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_simclr_transform(img_size: int = 128) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.3, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=int(0.1 * img_size) | 1)
        ], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class VehicleDataset(Dataset):
    def __init__(
        self,
        split: str,
        transform=None,
        root: Path = DATASET_ROOT,
    ):
        assert split in ("train", "valid", "test"), f"Unknown split: {split}"
        self.split = split
        self.root = Path(root)
        self.img_dir = self.root / split
        self.transform = transform or get_val_transform()
        self.samples: list[Tuple[str, np.ndarray]] = []
        self._load_csv()

    def _load_csv(self):
        csv_path = self.img_dir / "_classes.csv"
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            col_map = {}
            for cls in CLASSES:
                for key in reader.fieldnames:
                    if key.strip() == cls:
                        col_map[cls] = key
                        break
            for row in reader:
                fname = row["filename"].strip()
                img_path = self.img_dir / fname
                if not img_path.exists():
                    continue
                label = np.array(
                    [float(row[col_map[cls]].strip()) for cls in CLASSES],
                    dtype=np.float32,
                )
                self.samples.append((str(img_path), label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.from_numpy(label)


class SimCLRDataset(Dataset):
    def __init__(self, img_size: int = 128, root: Path = DATASET_ROOT):
        self.transform = get_simclr_transform(img_size)
        self.img_paths: list[str] = []
        for split in ("train", "valid", "test"):
            img_dir = Path(root) / split
            self.img_paths.extend([str(p) for p in img_dir.glob("*.jpg")])
        print(f"SimCLRDataset: {len(self.img_paths)} images (all splits, unlabeled)")

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        view1 = self.transform(img)
        view2 = self.transform(img)
        return view1, view2


def print_distribution():
    print("=" * 60)
    print("  Dataset Class Distribution")
    print("=" * 60)
    for split in ("train", "valid", "test"):
        ds = VehicleDataset(split, transform=get_val_transform())
        labels = np.stack([s[1] for s in ds.samples])
        total = len(ds)
        multi = int((labels.sum(axis=1) > 1).sum())
        zero  = int((labels.sum(axis=1) == 0).sum())
        print(f"\n{split.upper()} split — {total} images, {multi} multi-label, {zero} no-label")
        print(f"  {'Class':<12} {'Count':>6}  {'%':>6}  {'Pos-weight':>10}")
        print(f"  {'-'*40}")
        for i, cls in enumerate(CLASSES):
            count = int(labels[:, i].sum())
            pct   = 100.0 * count / total
            neg   = total - count
            pos_w = neg / max(count, 1)
            print(f"  {cls:<12} {count:>6}  {pct:>5.1f}%  {pos_w:>10.2f}")
    print()


def get_pos_weights(split: str = "train", device=None) -> torch.Tensor:
    ds = VehicleDataset(split, transform=get_val_transform())
    labels = np.stack([s[1] for s in ds.samples])
    total = len(ds)
    pos_weights = []
    for i in range(NUM_CLASSES):
        count = labels[:, i].sum()
        neg   = total - count
        pos_weights.append(neg / max(count, 1))
    t = torch.tensor(pos_weights, dtype=torch.float32)
    if device:
        t = t.to(device)
    return t


def get_dataloader(
    split: str,
    batch_size: int = 64,
    num_workers: int = 4,
    img_size: int = 224,
    pin_memory: bool = True,
) -> DataLoader:
    is_train = split == "train"
    transform = get_train_transform(img_size) if is_train else get_val_transform(img_size)
    ds = VehicleDataset(split, transform=transform)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=is_train,
    )


def mixup_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if alpha <= 0:
        return images, labels
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    idx = torch.randperm(batch_size, device=images.device)
    mixed_images = lam * images + (1 - lam) * images[idx]
    mixed_labels = lam * labels + (1 - lam) * labels[idx]
    return mixed_images, mixed_labels


if __name__ == "__main__":
    print_distribution()
    for split in ("train", "valid", "test"):
        loader = get_dataloader(split, batch_size=8, num_workers=0)
        imgs, lbls = next(iter(loader))
        print(f"{split}: batch shape={imgs.shape}, labels shape={lbls.shape}, dtype={lbls.dtype}")
