"""Canonical dental-condition taxonomy: the single source of truth for class ids.

Datasets declare their native classes by *name*; this module resolves those names
to the canonical integer indices that get written into YOLO labels and the
generated ``data.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Taxonomy:
    version: int
    name_to_index: dict[str, int]

    @property
    def num_classes(self) -> int:
        return len(self.name_to_index)

    def index_of(self, name: str) -> int:
        try:
            return self.name_to_index[name]
        except KeyError as exc:
            valid = ", ".join(sorted(self.name_to_index))
            raise KeyError(
                f"Unknown canonical class {name!r}. Known classes: {valid}"
            ) from exc

    def names_in_index_order(self) -> list[str]:
        return [name for name, _ in sorted(self.name_to_index.items(), key=lambda kv: kv[1])]

    def names_mapping(self) -> dict[int, str]:
        """Ultralytics-style ``{index: name}`` mapping for data.yaml."""
        return {idx: name for name, idx in self.name_to_index.items()}


def load_taxonomy(path: str | Path) -> Taxonomy:
    data = yaml.safe_load(Path(path).read_text())
    classes = data.get("classes", [])
    if not classes:
        raise ValueError(f"Taxonomy {path} has no classes.")

    name_to_index: dict[str, int] = {}
    indices: list[int] = []
    for entry in classes:
        idx, name = entry["index"], entry["name"]
        if name in name_to_index:
            raise ValueError(f"Duplicate class name {name!r} in taxonomy.")
        if idx in indices:
            raise ValueError(f"Duplicate class index {idx} in taxonomy.")
        name_to_index[name] = idx
        indices.append(idx)

    expected = list(range(len(indices)))
    if sorted(indices) != expected:
        raise ValueError(
            f"Taxonomy indices must be contiguous from 0; got {sorted(indices)}."
        )

    return Taxonomy(version=int(data.get("version", 0)), name_to_index=name_to_index)
