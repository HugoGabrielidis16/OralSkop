"""AlphaDent -> canonical YOLO-seg converter."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import yaml

from oralskop.data.converters.base import Converter, SampleRecord, remap_label_text
from oralskop.data.taxonomy import Taxonomy
from oralskop.data.verify import IMAGE_EXTS, VerifyReport, verify_dataset


class AlphaDentConverter(Converter):
    name = "alphadent"

    def __init__(self, config_path: Path, base_dir: Path):
        cfg = yaml.safe_load(Path(config_path).read_text())
        self.base_dir = base_dir
        # `raw_root` in the config is resolved relative to base_dir (the ai/ directory).
        self.raw_root = (base_dir / cfg["raw_root"]).resolve()
        self.images_dir = self.raw_root / cfg["images_subdir"]
        self.labels_dir = self.raw_root / cfg["labels_subdir"]
        self.patient_re = re.compile(cfg["patient_id_regex"])
        # native id -> canonical NAME (resolved to index later against the taxonomy)
        self.class_map_names: dict[int, str] = {int(k): v for k, v in cfg["class_map"].items()}
        self.val_fraction: float = float(cfg.get("split", {}).get("val_fraction", 0.15))
        self.seed: int = int(cfg.get("split", {}).get("seed", 42))

    def group_of(self, stem: str) -> str:
        m = self.patient_re.match(stem)
        return m.group(1) if m else stem

    def verify(self) -> list[VerifyReport]:
        return [verify_dataset(self.images_dir, self.labels_dir)]

    def records(self, taxonomy: Taxonomy) -> Iterator[SampleRecord]:
        native_to_canonical = {
            native: taxonomy.index_of(name) for native, name in self.class_map_names.items()
        }

        report = verify_dataset(self.images_dir, self.labels_dir)
        image_by_stem = {
            p.stem: p
            for p in self.images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        }

        # Only matched (image AND label) stems are usable. Orphan labels / images
        # without labels are reconciled by simply being skipped here.
        for stem in report.matched:
            label_path = self.labels_dir / f"{stem}.txt"
            lines = remap_label_text(label_path.read_text(), native_to_canonical)
            if not lines:
                continue  # nothing of interest after remap -> skip
            yield SampleRecord(
                image_path=image_by_stem[stem],
                label_lines=lines,
                group=self.group_of(stem),
            )
