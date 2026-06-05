"""Fine-tune UNet (EfficientNet-B3 encoder, ImageNet weights) on OralSkop dental data.

Labels must be in YOLO polygon format (.txt alongside images); they are rasterized to
per-pixel class masks on the fly.  Pixel convention: 0 = background, 1–9 = foreground.

    python models/unet/train.py
    python models/unet/train.py --epochs 50 --batch 8 --device cpu  # smoke test
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[3]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# AlphaDent classes: 0=bg, then 9 foreground classes
CLASS_NAMES = [
    "background",
    "abrasion", "filling", "crown",
    "caries_1", "caries_2", "caries_3",
    "caries_4", "caries_5", "caries_6",
]
NC_FG = 9           # foreground classes
NC = NC_FG + 1      # total classes including background


# ── Augmentation ─────────────────────────────────────────────────────────────

def _build_transforms(imgsz: int, augment: bool):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    if augment:
        return A.Compose([
            A.RandomResizedCrop(height=imgsz, width=imgsz, scale=(0.7, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(height=imgsz, width=imgsz),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ── YOLO polygon → pixel mask ─────────────────────────────────────────────────

def rasterize_polygons(label_path: Path, size: int) -> np.ndarray:
    """YOLO-seg polygon file → (H, W) uint8 mask [0=bg, cls+1=foreground]."""
    mask = np.zeros((size, size), dtype=np.uint8)
    if not label_path.exists():
        return mask
    polys: list[tuple[int, np.ndarray]] = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 7 or len(parts) % 2 == 0:
            continue
        try:
            cls = int(parts[0])
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        pts = np.array(
            [(coords[i] * size, coords[i + 1] * size) for i in range(0, len(coords) - 1, 2)],
            dtype=np.int32,
        )
        if len(pts) >= 3:
            polys.append((cls, pts))
    # Paint larger polygons first so small lesions stay visible on top
    for cls, pts in sorted(polys, key=lambda cp: cv2.contourArea(cp[1]), reverse=True):
        cv2.fillPoly(mask, [pts], cls + 1)
    return mask


# ── Dataset ───────────────────────────────────────────────────────────────────

class DentalDataset(Dataset):
    def __init__(self, split: str, imgsz: int = 512, augment: bool = False,
                 data_root: Path = ROOT / "ai" / "dataset" / "AlphaDent"):
        self.imgsz = imgsz
        self.transform = _build_transforms(imgsz, augment)

        images_dir = data_root / split / "images"
        labels_dir = data_root / split / "labels"

        self.samples: list[tuple[Path, Path]] = []
        if images_dir.is_dir():
            for img in sorted(images_dir.iterdir()):
                if img.suffix.lower() in IMAGE_EXTS:
                    self.samples.append((img, labels_dir / f"{img.stem}.txt"))

        if not self.samples:
            raise RuntimeError(
                f"No images found in {images_dir}.\n"
                f"Expected: dataset/AlphaDent/{split}/images/"
            )
        print(f"[unet] {split}: {len(self.samples)} images")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, lbl_path = self.samples[idx]
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise RuntimeError(f"Cannot read: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        mask = rasterize_polygons(lbl_path, self.imgsz)

        out = self.transform(image=rgb, mask=mask)
        image = out["image"]                          # float32 [3,H,W]
        target = out["mask"].long()                   # int64 [H,W]
        return image, target


# ── Metrics ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_metrics(model, loader, device) -> dict:
    model.eval()
    conf = torch.zeros(NC, NC, dtype=torch.int64, device=device)
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        preds = model(images).argmax(1)
        valid = (targets >= 0) & (targets < NC)
        idx = NC * targets[valid] + preds[valid]
        conf += torch.bincount(idx, minlength=NC * NC).reshape(NC, NC)
    inter = conf.diag().float()
    union = conf.sum(0).float() + conf.sum(1).float() - inter
    iou = inter / union.clamp(min=1)
    present = conf.sum(1) > 0
    return {
        "miou":      iou[present].mean().item() if present.any() else 0.0,
        "pixel_acc": (inter.sum() / conf.sum().clamp(min=1)).item(),
    }


# ── Training ──────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Fine-tune UNet (EfficientNet-B3) for dental seg.")
    p.add_argument("--epochs",  type=int,   default=100)
    p.add_argument("--batch",   type=int,   default=8)
    p.add_argument("--imgsz",   type=int,   default=512)
    p.add_argument("--lr",      type=float, default=2e-4)
    p.add_argument("--device",  default="0")
    p.add_argument("--workers", type=int,   default=4)
    p.add_argument("--name",    default="unet_dental_v1")
    p.add_argument("--out",     default=str(ROOT / "runs" / "unet"))
    p.add_argument("--seed",    type=int,   default=42)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import segmentation_models_pytorch as smp

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    d = args.device
    if d == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    elif d.isdigit():
        device = torch.device(f"cuda:{d}")
    else:
        device = torch.device(d)

    train_ds = DentalDataset("train", imgsz=args.imgsz, augment=True)
    val_ds   = DentalDataset("val",   imgsz=args.imgsz, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
    )

    model = smp.Unet(
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NC,          # 10: bg + 9 dental classes
        activation=None,     # raw logits → CrossEntropyLoss
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_dir = (Path(args.out) / args.name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    best_miou = -1.0
    print(f"[unet] device={device} | {NC} classes (0=bg) | "
          f"train={len(train_ds)} val={len(val_ds)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total_loss, seen = time.time(), 0.0, 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss   = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * images.size(0)
            seen       += images.size(0)
        scheduler.step()

        m    = compute_metrics(model, val_loader, device)
        line = (f"epoch {epoch:3d}/{args.epochs}  "
                f"loss {total_loss / max(seen, 1):.4f}  "
                f"val_mIoU {m['miou']:.4f}  "
                f"pixel_acc {m['pixel_acc']:.4f}  "
                f"({time.time() - t0:.0f}s)")

        ckpt = {
            "model": model.state_dict(), "epoch": epoch,
            "arch": "unet_efficientnet-b3", "num_classes": NC,
            "class_names": CLASS_NAMES,
        }
        torch.save(ckpt, out_dir / "last.pt")
        if m["miou"] > best_miou:
            best_miou = m["miou"]
            torch.save({**ckpt, "miou": best_miou}, out_dir / "best.pt")
            line += "  *best*"
        print(line)

    print(f"\n[unet] Done. Best val mIoU {best_miou:.4f}. Weights → {out_dir}")


if __name__ == "__main__":
    main()
