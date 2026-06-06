"""Semantic pixel-mask dataset -> canonical YOLO-seg converter.

For datasets annotated as **pixel masks** (one indexed PNG per image where each pixel's
value identifies its class) rather than YOLO polygons — e.g. BMC Oral Health 2024.
Each connected component of each class becomes one instance polygon, extracted with
OpenCV contour detection and emitted as a canonical YOLO-seg line.

Assumed layout (all paths configurable)::

    <raw_root>/
      <images_subdir>/<stem><img_ext>
      <masks_subdir>/<stem><mask_suffix><mask_ext>

The mask is read as a single-channel indexed image; `class_map` maps **pixel value ->
canonical class name**. IMPORTANT: confirm the actual pixel encoding of a real download
(see the tip in configs/data/bmc_oral_health.yaml) and adjust `class_map` — different
exports use different value/color conventions.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import yaml

from oralskop.data.converters.base import SampleRecord
from oralskop.data.taxonomy import Taxonomy
from oralskop.data.verify import IMAGE_EXTS, VerifyReport


class SemanticMaskConverter:
    name = "mask_semantic"

    def __init__(self, config_path: Path, base_dir: Path):
        cfg = yaml.safe_load(Path(config_path).read_text())
        self.name = cfg.get("name", "mask_semantic")
        self.base_dir = base_dir
        self.raw_root = (base_dir / cfg["raw_root"]).resolve()
        self.images_dir = self.raw_root / cfg.get("images_subdir", "images")
        self.masks_dir = self.raw_root / cfg.get("masks_subdir", "masks")
        self.mask_suffix: str = cfg.get("mask_suffix", "")
        self.mask_ext: str = cfg.get("mask_ext", ".png")

        # pixel value (int) -> canonical class NAME (must exist in the taxonomy)
        self.class_map: dict[int, str] = {int(k): v for k, v in cfg["class_map"].items()}

        self.min_area: int = int(cfg.get("min_area", 25))       # drop tiny noise blobs (px)
        self.poly_epsilon: float = float(cfg.get("poly_epsilon", 1.5))  # approxPolyDP (px)
        pid = cfg.get("patient_id_regex")
        self.patient_re = re.compile(pid) if pid else None

        self.val_fraction: float = float(cfg.get("split", {}).get("val_fraction", 0.2))
        self.seed: int = int(cfg.get("split", {}).get("seed", 42))

    # -- helpers --------------------------------------------------------------
    def _image_by_stem(self) -> dict[str, Path]:
        return {
            p.stem: p
            for p in self.images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        } if self.images_dir.is_dir() else {}

    def _mask_path(self, stem: str) -> Path:
        return self.masks_dir / f"{stem}{self.mask_suffix}{self.mask_ext}"

    def group_of(self, stem: str) -> str:
        if self.patient_re:
            m = self.patient_re.match(stem)
            if m:
                return m.group(1)
        return stem

    def verify(self) -> list[VerifyReport]:
        images = self._image_by_stem()
        report = VerifyReport(images_dir=self.images_dir, labels_dir=self.masks_dir)
        mask_stems = set()
        for stem in images:
            if self._mask_path(stem).exists():
                report.matched.append(stem)
                mask_stems.add(stem)
            else:
                report.images_without_label.append(stem)
        if self.masks_dir.is_dir():
            for p in self.masks_dir.iterdir():
                stem = p.stem[: -len(self.mask_suffix)] if self.mask_suffix and \
                    p.stem.endswith(self.mask_suffix) else p.stem
                if p.is_file() and stem not in images:
                    report.labels_without_image.append(stem)
        return [report]

    # -- conversion -----------------------------------------------------------
    def _mask_to_lines(self, mask: np.ndarray, value_to_canonical: dict[int, int]) -> list[str]:
        h, w = mask.shape[:2]
        lines: list[str] = []
        for value, canonical in value_to_canonical.items():
            binary = (mask == value).astype(np.uint8)
            if not binary.any():
                continue
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < self.min_area:
                    continue
                approx = cv2.approxPolyDP(contour, self.poly_epsilon, True).reshape(-1, 2)
                if len(approx) < 3:
                    continue
                coords = []
                for x, y in approx:
                    coords.append(f"{min(max(x / w, 0.0), 1.0):.6f}")
                    coords.append(f"{min(max(y / h, 0.0), 1.0):.6f}")
                lines.append(f"{canonical} " + " ".join(coords))
        return lines

    def records(self, taxonomy: Taxonomy) -> Iterator[SampleRecord]:
        value_to_canonical = {
            value: taxonomy.index_of(name) for value, name in self.class_map.items()
        }
        for stem, img_path in self._image_by_stem().items():
            mask_path = self._mask_path(stem)
            if not mask_path.exists():
                continue
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            lines = self._mask_to_lines(mask, value_to_canonical)
            if not lines:
                continue
            yield SampleRecord(image_path=img_path, label_lines=lines, group=self.group_of(stem))
