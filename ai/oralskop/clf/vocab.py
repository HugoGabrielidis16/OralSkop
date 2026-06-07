"""Label vocabulary for the multi-label manifest classifier.

The vocabulary fixes the class -> index mapping the model predicts. Prefer a
committed YAML (``configs/clf/labels_{coarse,fine}.yaml``) so indices are stable
regardless of which subset of the manifest is present; otherwise derive a sorted
vocabulary from the manifest column. A run always persists its resolved
vocabulary to ``vocab.json`` next to the checkpoints.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from oralskop.config import load_yaml

# Train-only micro-classes (doc §7.1): too rare to learn or score, never predicted.
TRAIN_ONLY_MICRO = {"restauration", "usure_dentaire"}
# Sentinel label carried by the unlabelled MetaDent pretraining corpus (doc §6.1).
PRETRAIN_LABEL = "image_contexte_non_labellisee"


def split_labels(cell: object) -> list[str]:
    """Split a ``canonical_*`` manifest cell (``"a|b"``) into clean label names.

    Tolerates NaN / empty cells (returns ``[]``) and strips whitespace. Drops the
    pretraining sentinel so it can never leak into a supervised target.
    """
    if cell is None:
        return []
    text = str(cell).strip()
    if not text or text.lower() in {"nan", "none"}:
        return []
    out = []
    for part in text.split("|"):
        name = part.strip()
        if name and name != PRETRAIN_LABEL:
            out.append(name)
    return out


@dataclass
class Vocab:
    """An ordered class vocabulary with multi-hot encoding."""

    names: list[str]
    level: str

    def __post_init__(self) -> None:
        self.index = {name: i for i, name in enumerate(self.names)}

    def __len__(self) -> int:
        return len(self.names)

    def encode(self, labels: list[str]) -> np.ndarray:
        """Multi-hot float32 vector for ``labels`` (unknown names are ignored)."""
        vec = np.zeros(len(self.names), dtype=np.float32)
        for name in labels:
            i = self.index.get(name)
            if i is not None:
                vec[i] = 1.0
        return vec

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"level": self.level, "classes": self.names}, indent=2))


def load_label_file(path: str | Path) -> list[str]:
    """Read a committed ``configs/clf/labels_*.yaml`` -> ordered class list."""
    cfg = load_yaml(path)
    classes = cfg.get("classes")
    if not classes:
        raise ValueError(f"{path}: expected a non-empty 'classes:' list.")
    return [str(c) for c in classes]


def derive_from_frame(labels_per_row: list[list[str]], exclude_micro: bool) -> list[str]:
    """Derive a sorted vocabulary from observed labels (descending frequency)."""
    counts: Counter[str] = Counter()
    for labels in labels_per_row:
        counts.update(labels)
    for micro in TRAIN_ONLY_MICRO if exclude_micro else set():
        counts.pop(micro, None)
    # Most-frequent first, ties broken alphabetically — deterministic.
    return [name for name, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def build_vocab(
    level: str,
    *,
    labels_file: str | Path | None,
    labels_per_row: list[list[str]] | None = None,
    exclude_micro: bool = True,
) -> Vocab:
    """Resolve the vocabulary from a committed file, else derive it from the data.

    ``labels_per_row`` is required only for the derive path. When a committed file
    is used, ``exclude_micro`` has no effect (the file is authoritative).
    """
    if labels_file:
        names = load_label_file(labels_file)
    else:
        if labels_per_row is None:
            raise ValueError("Deriving a vocabulary needs labels_per_row.")
        names = derive_from_frame(labels_per_row, exclude_micro)
    if not names:
        raise ValueError("Resolved an empty vocabulary — check level/labels_file/manifest.")
    return Vocab(names=names, level=level)
