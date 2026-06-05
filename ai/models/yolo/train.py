"""Fine-tune YOLOv8m-seg on OralSkop dental data (4 classes).

    python models/yolo/train.py
    python models/yolo/train.py --epochs 1 --batch 2 --device cpu  # smoke test
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WEIGHTS_DIR = ROOT / "ai" / "models" / "weights"
DEFAULT_WEIGHTS = WEIGHTS_DIR / "yolov8m-seg.pt"
DEFAULT_DATA = ROOT / "ai" / "models" / "configs" / "dataset.yaml"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Fine-tune YOLOv8m-seg for dental segmentation.")
    p.add_argument("--data",     default=str(DEFAULT_DATA))
    p.add_argument("--weights",  default=str(DEFAULT_WEIGHTS))
    p.add_argument("--epochs",   type=int,   default=100)
    p.add_argument("--imgsz",    type=int,   default=640)
    p.add_argument("--batch",    type=int,   default=16)
    p.add_argument("--device",   default="0", help="0 / '0,1' / 'cpu'")
    p.add_argument("--name",     default="yolo_dental_v1")
    p.add_argument("--project",  default=str(ROOT / "runs" / "yolo"))
    p.add_argument("--patience", type=int,   default=30)
    return p.parse_args(argv)


def _get_model(weights_path: Path):
    from ultralytics import YOLO

    if weights_path.exists():
        return YOLO(str(weights_path))

    print(f"[yolo] {weights_path.name} not found — downloading via ultralytics ...")
    # ultralytics downloads to cwd; we move the file afterwards
    model = YOLO("yolov8m-seg.pt")
    downloaded = Path("yolov8m-seg.pt")
    if downloaded.exists():
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(downloaded), str(weights_path))
        print(f"[yolo] Weights saved → {weights_path}")
    return model


def main(argv=None):
    args = parse_args(argv)

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")

    model = _get_model(Path(args.weights))

    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        project=args.project,
        patience=args.patience,
        exist_ok=True,
        # Augmentation tuned for intraoral photos
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,   # mouths have up/down orientation — keep off
        mosaic=1.0,
    )
    print(f"[yolo] Training complete. Best weights: {results.save_dir}")


if __name__ == "__main__":
    main()
