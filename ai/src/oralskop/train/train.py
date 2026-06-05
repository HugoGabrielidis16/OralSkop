"""Config-driven YOLO11-seg training entrypoint.

Thin wrapper over Ultralytics ``YOLO(...).train()``. Everything comes from the YAML
config (configs/train/yolo11_seg.yaml); CLI ``--override`` lets you tweak any key,
e.g. for a CPU smoke test:

    python -m oralskop.train.train --config configs/train/yolo11_seg.yaml \
        --override epochs=1 device=cpu batch=2
"""

from __future__ import annotations

import argparse
from pathlib import Path

from oralskop.config import apply_overrides, load_yaml


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO11-seg for OralSkop.")
    p.add_argument("--config", required=True, help="Path to train YAML config.")
    p.add_argument(
        "--override",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Override any config key, e.g. epochs=1 device=cpu.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = apply_overrides(load_yaml(args.config), args.override)

    # `model` selects the pretrained checkpoint; the rest are train() kwargs.
    model_name = cfg.pop("model")

    data_path = Path(cfg["data"])
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset descriptor {data_path} not found. "
            "Run scripts/prepare_alphadent.py first."
        )

    # Imported here so `--help` works without the heavy torch/ultralytics import.
    from ultralytics import YOLO

    model = YOLO(model_name)
    results = model.train(**cfg)
    print(f"Training complete. Best weights: {getattr(results, 'save_dir', cfg.get('project'))}")


if __name__ == "__main__":
    main()
