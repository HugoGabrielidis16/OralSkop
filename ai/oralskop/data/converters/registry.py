"""Converter registry: build the right converter from a dataset config's `converter` field.

To support a new dataset *format*, implement a converter and register it here. Datasets
whose format already has a converter (AlphaDent's YOLO-seg, Roboflow YOLO-seg) are then
added with **config only** — no code.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from oralskop.data.converters.alphadent import AlphaDentConverter
from oralskop.data.converters.mask_semantic import SemanticMaskConverter
from oralskop.data.converters.roboflow import RoboflowYoloSegConverter

# config `converter:` value -> converter class. Each takes (config_path, base_dir).
REGISTRY = {
    "alphadent": AlphaDentConverter,
    "roboflow_yoloseg": RoboflowYoloSegConverter,
    "mask_semantic": SemanticMaskConverter,
}


def build_converter(config_path: Path, base_dir: Path):
    cfg = yaml.safe_load(Path(config_path).read_text())
    ctype = cfg.get("converter")
    if ctype not in REGISTRY:
        raise ValueError(
            f"{config_path}: unknown converter {ctype!r}. "
            f"Known: {', '.join(sorted(REGISTRY))}."
        )
    return REGISTRY[ctype](Path(config_path), base_dir)
