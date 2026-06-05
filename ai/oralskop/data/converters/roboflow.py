"""Generic Roboflow YOLO-seg export -> canonical YOLO-seg converter.

Roboflow "YOLOv8/YOLO11 segmentation" exports look like::

    <root>/
      data.yaml                 # names: [Caries, Calculus, ...]  (native class names)
      train/images/*.jpg        train/labels/*.txt
      valid/images/*.jpg        valid/labels/*.txt
      test/images/*.jpg         test/labels/*.txt

This converter is format-generic across Roboflow seg exports. The dataset config maps
native class *names* -> canonical class names (robust to Roboflow re-export index
churn). All splits are pooled and re-split by *source image* (the filename portion
before Roboflow's ``.rf.<hash>`` suffix) so augmented variants never leak across the
train/val boundary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import yaml

from oralskop.data.converters.base import SampleRecord, remap_label_text
from oralskop.data.taxonomy import Taxonomy
from oralskop.data.verify import IMAGE_EXTS, VerifyReport, verify_dataset


def _load_native_names(data_yaml: Path) -> dict[int, str]:
    """Read the native ``names`` from a Roboflow data.yaml (list or dict form)."""
    names = yaml.safe_load(data_yaml.read_text()).get("names")
    if isinstance(names, list):
        return dict(enumerate(names))
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    raise ValueError(f"Could not read `names` from {data_yaml}.")


def _source_group(stem: str) -> str:
    """Group key = source image id (strip Roboflow's `.rf.<hash>` augmentation suffix)."""
    return stem.split(".rf.")[0]


class RoboflowYoloSegConverter:
    name = "roboflow_yoloseg"

    def __init__(self, config_path: Path, base_dir: Path):
        cfg = yaml.safe_load(Path(config_path).read_text())
        self.name = cfg.get("name", "roboflow")
        self.base_dir = base_dir
        self.raw_root = (base_dir / cfg["raw_root"]).resolve()

        self.splits: list[str] = cfg.get("splits", ["train", "valid", "test"])
        self.images_subdir_name: str = cfg.get("images_subdir_name", "images")
        self.labels_subdir_name: str = cfg.get("labels_subdir_name", "labels")
        self.roboflow_data_yaml: str = cfg.get("roboflow_data_yaml", "data.yaml")

        # native class NAME -> canonical class NAME (must exist in the taxonomy)
        self.class_map_names: dict[str, str] = dict(cfg["class_map"])

        self.val_fraction: float = float(cfg.get("split", {}).get("val_fraction", 0.15))
        self.seed: int = int(cfg.get("split", {}).get("seed", 42))

    # -- directory helpers ----------------------------------------------------
    def _split_dirs(self) -> list[tuple[Path, Path]]:
        """Existing (images_dir, labels_dir) pairs across the configured splits.

        Supports both Roboflow's `train/images` layout and the `images/train` layout.
        """
        pairs: list[tuple[Path, Path]] = []
        for split in self.splits:
            a_img = self.raw_root / split / self.images_subdir_name
            a_lbl = self.raw_root / split / self.labels_subdir_name
            b_img = self.raw_root / self.images_subdir_name / split
            b_lbl = self.raw_root / self.labels_subdir_name / split
            if a_img.is_dir() and a_lbl.is_dir():
                pairs.append((a_img, a_lbl))
            elif b_img.is_dir() and b_lbl.is_dir():
                pairs.append((b_img, b_lbl))
        return pairs

    def verify(self) -> list[VerifyReport]:
        return [verify_dataset(img, lbl) for img, lbl in self._split_dirs()]

    # -- conversion -----------------------------------------------------------
    def _native_index_to_canonical(self, taxonomy: Taxonomy) -> dict[int, int]:
        data_yaml = self.raw_root / self.roboflow_data_yaml
        if not data_yaml.exists():
            raise FileNotFoundError(
                f"Roboflow {data_yaml} not found; needed to resolve native class names."
            )
        native_names = _load_native_names(data_yaml)
        mapping: dict[int, int] = {}
        for idx, native_name in native_names.items():
            canonical_name = self.class_map_names.get(native_name)
            if canonical_name is None:
                continue  # native class not selected -> its polygons are dropped
            mapping[idx] = taxonomy.index_of(canonical_name)
        if not mapping:
            raise ValueError(
                f"No native class in {data_yaml} matched class_map keys "
                f"{list(self.class_map_names)}."
            )
        return mapping

    def records(self, taxonomy: Taxonomy) -> Iterator[SampleRecord]:
        native_to_canonical = self._native_index_to_canonical(taxonomy)

        for images_dir, labels_dir in self._split_dirs():
            report = verify_dataset(images_dir, labels_dir)
            image_by_stem = {
                p.stem: p
                for p in images_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            }
            for stem in report.matched:
                lines = remap_label_text(
                    (labels_dir / f"{stem}.txt").read_text(), native_to_canonical
                )
                if not lines:
                    continue
                yield SampleRecord(
                    image_path=image_by_stem[stem],
                    label_lines=lines,
                    group=_source_group(stem),
                )
