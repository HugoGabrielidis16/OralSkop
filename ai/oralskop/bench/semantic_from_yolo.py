"""Evaluate YOLO instance masks as semantic segmentation on a fixed split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from oralskop.torchseg.dataset import YoloSegDataset, rasterize_polygons
from oralskop.torchseg.train import format_per_class, seg_class_names


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rasterize YOLO predictions and compute semantic metrics.")
    p.add_argument("--weights", required=True, help="YOLO .pt checkpoint.")
    p.add_argument("--data", required=True, help="Built dataset data.yaml.")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--imgsz", type=int, default=512, help="Semantic metric raster size.")
    p.add_argument("--infer-imgsz", type=int, default=960, help="YOLO inference size.")
    p.add_argument("--device", default="0")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--limit", type=int, help="Limit images for smoke tests.")
    p.add_argument("--out", help="Optional JSON metrics output path.")
    return p.parse_args(argv)


def _pred_semantic(result, size: int, num_classes: int) -> np.ndarray:
    pred = np.zeros((size, size), dtype=np.uint8)
    if result.masks is None or result.boxes is None or result.boxes.cls is None:
        return pred

    masks = result.masks.data.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    items = []
    for cls, mask in zip(classes, masks):
        if cls < 0 or cls >= num_classes - 1:
            continue
        resized = cv2.resize(mask.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR) >= 0.5
        items.append((int(cls), resized, int(resized.sum())))

    for cls, mask, _ in sorted(items, key=lambda item: item[2], reverse=True):
        pred[mask] = cls + 1
    return pred


def _metrics(conf: torch.Tensor) -> dict[str, float]:
    true_pixels = conf.sum(1).float()
    pred_pixels = conf.sum(0).float()
    inter = conf.diag().float()
    union = pred_pixels + true_pixels - inter
    iou = inter / union.clamp(min=1)
    dice = (2 * inter) / (pred_pixels + true_pixels).clamp(min=1)
    class_acc = inter / true_pixels.clamp(min=1)
    present = true_pixels > 0
    fg_present = present.clone()
    fg_present[0] = False
    fg_true = true_pixels[1:].sum()
    return {
        "val_loss": float("nan"),
        "pixel_acc": (inter.sum() / conf.sum().clamp(min=1)).item(),
        "fg_pixel_acc": (inter[1:].sum() / fg_true.clamp(min=1)).item(),
        "mean_acc": class_acc[present].mean().item() if present.any() else 0.0,
        "miou": iou[present].mean().item() if present.any() else 0.0,
        "fg_miou": iou[fg_present].mean().item() if fg_present.any() else 0.0,
        "mean_dice": dice[present].mean().item() if present.any() else 0.0,
        "fg_dice": dice[fg_present].mean().item() if fg_present.any() else 0.0,
        "per_class_iou": iou.tolist(),
        "per_class_dice": dice.tolist(),
        "per_class_acc": class_acc.tolist(),
        "per_class_support": true_pixels.long().tolist(),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not Path(args.weights).exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    from ultralytics import YOLO

    dataset = YoloSegDataset(data_yaml=args.data, split=args.split, imgsz=args.imgsz, augment=False)
    num_classes = dataset.num_seg_classes
    conf = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    model = YOLO(args.weights)

    samples = dataset.samples[: args.limit] if args.limit else dataset.samples
    for img_path, label_path in samples:
        result = model.predict(
            source=str(img_path),
            imgsz=args.infer_imgsz,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
        )[0]
        pred = _pred_semantic(result, args.imgsz, num_classes)
        text = label_path.read_text() if label_path.exists() else ""
        target = rasterize_polygons(text, args.imgsz)
        valid = (target >= 0) & (target < num_classes)
        idx = num_classes * target[valid].reshape(-1) + pred[valid].reshape(-1)
        conf += torch.bincount(torch.from_numpy(idx.astype(np.int64)), minlength=num_classes**2).reshape(
            num_classes, num_classes
        )

    metrics = _metrics(conf)
    class_names = seg_class_names(dataset.class_names)
    print(json.dumps({k: v for k, v in metrics.items() if not k.startswith("per_class")}, indent=2))
    print("per_class " + format_per_class(metrics, class_names))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"class_names": class_names, **metrics}, indent=2))
        print(f"Metrics written to {out}")


if __name__ == "__main__":
    main()
