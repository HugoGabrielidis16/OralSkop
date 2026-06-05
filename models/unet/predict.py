"""UNet segmentation inference on a folder or single image.

    python models/unet/predict.py --source dataset/AlphaDent/test/images/
    python models/unet/predict.py --source path/to/image.jpg --weights runs/unet/unet_dental_v1/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = ROOT / "runs" / "unet" / "unet_dental_v1" / "best.pt"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# BGR palette — one colour per class (index = class id)
_PALETTE = np.array([
    [  0,   0,   0],  # 0 background
    [255,  80,  80],  # 1 abrasion
    [ 80, 200,  80],  # 2 filling
    [ 80, 120, 255],  # 3 crown
    [255, 180,  50],  # 4 caries_1
    [200,  80, 200],  # 5 caries_2
    [ 50, 220, 220],  # 6 caries_3
    [180, 255,  80],  # 7 caries_4
    [255, 120, 200],  # 8 caries_5
    [120,  80, 255],  # 9 caries_6
], dtype=np.uint8)

CLASS_NAMES = [
    "background", "abrasion", "filling", "crown",
    "caries_1", "caries_2", "caries_3", "caries_4", "caries_5", "caries_6",
]


def _overlay(bgr: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    """Blend per-pixel class mask onto the original image."""
    color_mask = _PALETTE[mask.clip(0, len(_PALETTE) - 1)]
    fg = mask > 0
    vis = bgr.copy()
    vis[fg] = cv2.addWeighted(bgr, 1 - alpha, color_mask, alpha, 0)[fg]
    return vis


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="UNet segmentation inference.")
    p.add_argument("--source",  required=True, help="Image file or directory.")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p.add_argument("--imgsz",   type=int,   default=512)
    p.add_argument("--device",  default="0")
    p.add_argument("--alpha",   type=float, default=0.5, help="Mask overlay opacity (0–1).")
    p.add_argument("--save",    default=str(ROOT / "runs" / "unet" / "predict"),
                   help="Output directory for annotated images.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import segmentation_models_pytorch as smp
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights not found: {weights_path}\n"
            f"Run training first:  python run.py --model unet --task train"
        )

    ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    nc = ckpt.get("num_classes", 10)

    d = args.device
    if d == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    elif d.isdigit():
        device = torch.device(f"cuda:{d}")
    else:
        device = torch.device(d)

    model = smp.Unet(
        encoder_name="efficientnet-b3",
        encoder_weights=None,
        in_channels=3,
        classes=nc,
        activation=None,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    transform = A.Compose([
        A.Resize(height=args.imgsz, width=args.imgsz),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    source = Path(args.source)
    images = (
        [p for p in sorted(source.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
        if source.is_dir() else [source]
    )

    out_dir = Path(args.save)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[unet] Inferring on {len(images)} image(s) → {out_dir}")

    with torch.no_grad():
        for img_path in images:
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                print(f"[unet] Warning: cannot read {img_path}, skipping.")
                continue
            orig_h, orig_w = bgr.shape[:2]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            tensor = transform(image=rgb)["image"].unsqueeze(0).to(device)
            pred = model(tensor).argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
            pred = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            vis = _overlay(bgr, pred, alpha=args.alpha)
            cv2.imwrite(str(out_dir / img_path.name), vis)

    print(f"[unet] Done. Segmented images saved → {out_dir}")


if __name__ == "__main__":
    main()
