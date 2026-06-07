"""Dataset integrity checks, reusable across datasets.

Reports the problems we found in the shipped AlphaDent copy: label files with no
matching image, images with no label, and the per-class polygon histogram.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class VerifyReport:
    images_dir: Path
    labels_dir: Path
    matched: list[str] = field(default_factory=list)        # stems with image AND label
    images_without_label: list[str] = field(default_factory=list)
    labels_without_image: list[str] = field(default_factory=list)
    empty_labels: list[str] = field(default_factory=list)
    class_histogram: Counter = field(default_factory=Counter)
    malformed_lines: list[str] = field(default_factory=list)  # "stem:lineno"

    def summary(self) -> str:
        lines = [
            f"Dataset verify: {self.images_dir}",
            f"  matched image+label pairs : {len(self.matched)}",
            f"  images without label      : {len(self.images_without_label)}",
            f"  labels without image      : {len(self.labels_without_image)}",
            f"  empty label files         : {len(self.empty_labels)}",
            f"  malformed polygon lines   : {len(self.malformed_lines)}",
            "  class histogram (native id -> polygons):",
        ]
        for cls in sorted(self.class_histogram):
            lines.append(f"      {cls:>3} : {self.class_histogram[cls]}")
        return "\n".join(lines)


def _stem_index(directory: Path, exts: set[str] | None) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not directory.is_dir():
        return out
    for p in directory.iterdir():
        if not p.is_file():
            continue
        if exts is not None and p.suffix.lower() not in exts:
            continue
        out[p.stem] = p
    return out


def verify_dataset(images_dir: Path, labels_dir: Path) -> VerifyReport:
    images = _stem_index(images_dir, IMAGE_EXTS)
    labels = _stem_index(labels_dir, {".txt"})

    report = VerifyReport(images_dir=images_dir, labels_dir=labels_dir)

    for stem in sorted(set(images) | set(labels)):
        has_img, has_lbl = stem in images, stem in labels
        if has_img and has_lbl:
            report.matched.append(stem)
        elif has_img:
            report.images_without_label.append(stem)
        else:
            report.labels_without_image.append(stem)

    # Parse matched labels for the class histogram and malformed-line detection.
    for stem in report.matched:
        text = labels[stem].read_text().strip()
        if not text:
            report.empty_labels.append(stem)
            continue
        for lineno, line in enumerate(text.splitlines()):
            parts = line.split()
            # YOLO-seg polygon: class + >=3 (x,y) pairs => >= 7 tokens, odd count.
            if len(parts) < 7 or len(parts) % 2 == 0:
                report.malformed_lines.append(f"{stem}:{lineno}")
                continue
            try:
                report.class_histogram[int(parts[0])] += 1
            except ValueError:
                report.malformed_lines.append(f"{stem}:{lineno}")

    return report
