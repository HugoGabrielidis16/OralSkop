"""
OralSkop — YOLO11m-seg fine-tuning script.

Develop locally, push to git, clone on SageMaker, run there on GPU.

Usage
-----
    # local smoke test (CPU, 1 epoch, tiny batch)
    python models/train.py --device cpu --epochs 1 --batch 2

    # SageMaker / full GPU run
    python models/train.py --data /path/to/data --epochs 100

Outputs land in  runs/segment/oralskop_v1/weights/best.pt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune YOLO11m-seg for OralSkop.")
    p.add_argument(
        "--data",
        default=Path(__file__).parent / "dataset.yaml",
        help="Path to dataset.yaml (default: models/dataset.yaml)",
    )
    p.add_argument("--model",    default="yolo11m-seg.pt")
    p.add_argument("--epochs",   type=int,   default=100)
    p.add_argument("--imgsz",    type=int,   default=640)
    p.add_argument("--batch",    type=int,   default=16,   help="-1 = auto-batch")
    p.add_argument("--device",   default="0",              help="0=GPU0, cpu, 0,1=multi-GPU")
    p.add_argument("--workers",  type=int,   default=8)
    p.add_argument("--name",     default="oralskop_v1")
    p.add_argument("--patience", type=int,   default=30)
    p.add_argument("--resume",   action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"dataset.yaml not found at: {data_path}\n"
            "Place your images under data/images/{{train,val}}/ and labels under\n"
            "data/labels/{{train,val}}/ at the project root, then re-run."
        )

    from ultralytics import YOLO

    model = YOLO(args.model)

    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=42,
        patience=args.patience,
        optimizer="auto",
        cos_lr=True,
        # augmentation — intraoral photos vary a lot in framing and lighting
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,    # teeth have up/down orientation — keep off
        mosaic=1.0,
        close_mosaic=10,
        project="runs/segment",
        name=args.name,
        exist_ok=False,
        resume=args.resume,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nBest weights: {best}")

    # If running inside SageMaker, copy best.pt to the output model dir.
    sm_model_dir = os.environ.get("SM_MODEL_DIR")
    if sm_model_dir:
        import shutil
        dest = Path(sm_model_dir) / "best.pt"
        shutil.copy2(best, dest)
        print(f"Copied to SM_MODEL_DIR: {dest}")


if __name__ == "__main__":
    main()
