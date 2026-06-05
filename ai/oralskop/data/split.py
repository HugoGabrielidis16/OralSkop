"""Patient-grouped train/val split (no patient on both sides)."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.model_selection import GroupShuffleSplit

from oralskop.data.converters.base import SampleRecord


@dataclass
class Split:
    train: list[SampleRecord]
    val: list[SampleRecord]


def grouped_split(records: list[SampleRecord], val_fraction: float, seed: int) -> Split:
    """Split records by their ``group`` (patient id) using GroupShuffleSplit.

    Falls back gracefully when there are too few groups to split.
    """
    if not records:
        raise ValueError("No records to split.")

    groups = [r.group for r in records]
    n_groups = len(set(groups))
    if n_groups < 2 or val_fraction <= 0:
        return Split(train=list(records), val=[])

    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(splitter.split(records, groups=groups))
    return Split(
        train=[records[i] for i in train_idx],
        val=[records[i] for i in val_idx],
    )
