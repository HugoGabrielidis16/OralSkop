"""Converter interface: raw dataset -> canonical YOLO-seg records.

A converter does NOT write files. It yields :class:`SampleRecord`s (image path +
remapped label lines + grouping key); :mod:`oralskop.data.build` is responsible for
splitting and materializing them. This keeps splitting/orchestration logic in one
place and makes each converter small.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

from oralskop.data.taxonomy import Taxonomy


@dataclass
class SampleRecord:
    image_path: Path
    # YOLO-seg label lines already remapped to CANONICAL class indices.
    label_lines: list[str]
    group: str  # patient id (or other) used to prevent train/val leakage


class Converter(Protocol):
    name: str

    def records(self, taxonomy: Taxonomy) -> Iterator[SampleRecord]:
        """Yield canonical records for every usable sample in the raw dataset."""
        ...


def remap_label_text(
    text: str,
    native_to_canonical: dict[int, int],
) -> list[str]:
    """Rewrite the leading class id of each YOLO-seg line to canonical indices.

    Lines whose native class is not in the map are dropped. Malformed lines are
    skipped. Polygon coordinates are passed through untouched.
    """
    out: list[str] = []
    for line in text.strip().splitlines():
        parts = line.split()
        if len(parts) < 7 or len(parts) % 2 == 0:
            continue
        try:
            native = int(parts[0])
        except ValueError:
            continue
        canonical = native_to_canonical.get(native)
        if canonical is None:
            continue
        out.append(" ".join([str(canonical), *parts[1:]]))
    return out
