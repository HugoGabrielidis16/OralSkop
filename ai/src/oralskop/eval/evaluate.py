"""Per-class segmentation evaluation on the held-out val split.

Runs Ultralytics validation and prints per-class mask mAP, surfacing rare-class
performance (AlphaDent is heavily imbalanced). Ultralytics writes PR curves and the
confusion matrix into the run's save_dir.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a YOLO11-seg checkpoint.")
    p.add_argument("--weights", required=True, help="Path to trained .pt weights.")
    p.add_argument("--data", required=True, help="Path to data.yaml.")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--split", default="val", choices=["val", "test", "train"])
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not Path(args.weights).exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    from ultralytics import YOLO

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split=args.split,
    )

    names = model.names  # {idx: name}
    seg = metrics.seg     # mask metrics

    print("\nPer-class mask metrics (held-out {}):".format(args.split))
    print(f"  {'class':<16}{'mAP50':>10}{'mAP50-95':>12}")
    # ap_class_index lists the class ids that were actually evaluated.
    for i, cls_id in enumerate(getattr(metrics.box, "ap_class_index", [])):
        name = names.get(int(cls_id), str(cls_id))
        ap50 = seg.ap50[i] if i < len(seg.ap50) else float("nan")
        ap = seg.ap[i] if i < len(seg.ap) else float("nan")
        print(f"  {name:<16}{ap50:>10.3f}{ap:>12.3f}")

    print("\nOverall mask mAP50   : {:.3f}".format(seg.map50))
    print("Overall mask mAP50-95: {:.3f}".format(seg.map))
    print(f"\nArtifacts (PR curves, confusion matrix): {metrics.save_dir}")


if __name__ == "__main__":
    main()
