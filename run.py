"""OralSkop — top-level training / inference runner.

Training:
    python run.py --model yolo [--epochs N] [--batch N] [--device 0|cpu]
    python run.py --model unet [--epochs N] [--batch N] [--device 0|cpu]

Inference:
    python run.py --model yolo --task predict --source path/to/images/
    python run.py --model unet --task predict --source path/to/images/
    python run.py --model sam2 --image PATH [--boxes "x1,y1,x2,y2" ...] [--save PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="OralSkop dental AI — train or predict with a chosen model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model",  required=True, choices=["yolo", "unet", "sam2"])
    p.add_argument("--task",   default="train", choices=["train", "predict"],
                   help="train: fine-tune and save weights | predict: run inference")
    p.add_argument("--device", default=None, help="0 / '0,1' / 'cpu'")
    p.add_argument("--name",   default=None, help="Run name for output directory.")

    # Training args (yolo / unet)
    g = p.add_argument_group("training (yolo / unet)")
    g.add_argument("--epochs", type=int,   default=None)
    g.add_argument("--batch",  type=int,   default=None)
    g.add_argument("--imgsz",  type=int,   default=None)

    # Inference args (yolo / unet)
    i = p.add_argument_group("inference (yolo / unet)")
    i.add_argument("--source",  default=None, help="Image file or folder to run inference on.")
    i.add_argument("--weights", default=None, help="Weights file (.pt). Defaults to latest training run.")
    i.add_argument("--conf",    type=float, default=None, help="YOLO confidence threshold (default 0.25).")
    i.add_argument("--alpha",   type=float, default=None, help="UNet mask overlay opacity (default 0.5).")
    i.add_argument("--out",     default=None, help="Output directory for predictions.")

    # SAM2 inference
    s = p.add_argument_group("sam2 inference")
    s.add_argument("--image", default=None, help="Input image path.")
    s.add_argument("--boxes", nargs="+", metavar="X1,Y1,X2,Y2",
                   help="Bounding boxes in xyxy pixel format.")
    s.add_argument("--save",  default=None, help="Path to save the overlay image.")
    s.add_argument("--show",  action="store_true", help="Display result in a window.")

    return p.parse_args(argv)


def _argv_train(args) -> list[str]:
    out = []
    if args.epochs is not None: out += ["--epochs", str(args.epochs)]
    if args.batch  is not None: out += ["--batch",  str(args.batch)]
    if args.imgsz  is not None: out += ["--imgsz",  str(args.imgsz)]
    if args.device is not None: out += ["--device", args.device]
    if args.name   is not None: out += ["--name",   args.name]
    return out


def _argv_predict(args, *, has_conf=False, has_alpha=False) -> list[str]:
    if not args.source:
        print("Error: --task predict requires --source PATH")
        sys.exit(1)
    out = ["--source", args.source]
    if args.weights is not None: out += ["--weights", args.weights]
    if args.device  is not None: out += ["--device",  args.device]
    if args.imgsz   is not None: out += ["--imgsz",   str(args.imgsz)]
    if args.out     is not None: out += ["--save",     args.out]
    if has_conf  and args.conf  is not None: out += ["--conf",  str(args.conf)]
    if has_alpha and args.alpha is not None: out += ["--alpha", str(args.alpha)]
    return out


def main(argv=None):
    args = parse_args(argv)

    if args.model == "yolo":
        if args.task == "train":
            from models.yolo.train import main as _main
            _main(_argv_train(args))
        else:
            from models.yolo.predict import main as _main
            _main(_argv_predict(args, has_conf=True))

    elif args.model == "unet":
        if args.task == "train":
            from models.unet.train import main as _main
            _main(_argv_train(args))
        else:
            from models.unet.predict import main as _main
            _main(_argv_predict(args, has_alpha=True))

    elif args.model == "sam2":
        if not args.image:
            print("Error: --model sam2 requires --image PATH")
            print("Example: python run.py --model sam2 --image data/images/test/sample.jpg")
            sys.exit(1)
        from models.sam2.pipeline import main as _main
        child = ["--image", args.image]
        if args.boxes:  child += ["--boxes"] + args.boxes
        if args.device: child += ["--device", args.device]
        if args.save:   child += ["--save",   args.save]
        if args.show:   child += ["--show"]
        _main(child)


if __name__ == "__main__":
    main()
