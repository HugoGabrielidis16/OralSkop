"""YOLO segmentation inference on a folder or single image.

    python models/yolo/predict.py --source dataset/AlphaDent/test/images/
    python models/yolo/predict.py --source path/to/image.jpg --weights runs/yolo/yolo_dental_v1/weights/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = ROOT / "runs" / "yolo" / "yolo_dental_v1" / "weights" / "best.pt"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="YOLO segmentation inference.")
    p.add_argument("--source",  required=True, help="Image file or directory.")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p.add_argument("--imgsz",   type=int,   default=640)
    p.add_argument("--conf",    type=float, default=0.25)
    p.add_argument("--device",  default="0")
    p.add_argument("--save",    default=str(ROOT / "runs" / "yolo" / "predict"),
                   help="Output directory for annotated images.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from ultralytics import YOLO

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(
            f"Weights not found: {weights}\n"
            f"Run training first:  python run.py --model yolo --task train"
        )

    model = YOLO(str(weights))
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        project=str(Path(args.save).parent),
        name=Path(args.save).name,
        save=True,
        exist_ok=True,
    )
    print(f"[yolo] Predictions saved → {args.save}")


if __name__ == "__main__":
    main()
