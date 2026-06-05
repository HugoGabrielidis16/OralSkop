"""Orchestrate: converters -> grouped split -> materialized YOLO dataset + data.yaml.

Designed so that multiple datasets can be merged into one training set: pass several
converters and their records are pooled under a shared canonical taxonomy. Output
filenames are dataset-prefixed to avoid collisions.
"""

from __future__ import annotations

import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

from oralskop.data.converters.base import Converter, SampleRecord
from oralskop.data.split import Split, grouped_split
from oralskop.data.taxonomy import Taxonomy


@dataclass
class BuildResult:
    out_dir: Path
    data_yaml: Path
    n_train: int
    n_val: int
    train_class_hist: Counter
    val_class_hist: Counter


def _materialize(records: list[SampleRecord], dataset_prefix: str, img_dir: Path, lbl_dir: Path) -> Counter:
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    hist: Counter = Counter()
    for rec in records:
        # Prefix to keep filenames unique when several datasets are merged.
        stem = f"{dataset_prefix}__{rec.image_path.stem}"
        shutil.copy2(rec.image_path, img_dir / f"{stem}{rec.image_path.suffix.lower()}")
        (lbl_dir / f"{stem}.txt").write_text("\n".join(rec.label_lines) + "\n")
        for line in rec.label_lines:
            hist[int(line.split()[0])] += 1
    return hist


def build_dataset(
    converters: list[tuple[Converter, float, int]],
    taxonomy: Taxonomy,
    out_dir: Path,
    clean: bool = True,
) -> BuildResult:
    """Build a merged canonical YOLO-seg dataset.

    ``converters`` is a list of ``(converter, val_fraction, seed)``. Each dataset is
    split independently (patient-grouped) before pooling, so the val fraction holds
    per dataset.
    """
    out_dir = Path(out_dir)
    if clean and out_dir.exists():
        shutil.rmtree(out_dir)

    train_img, train_lbl = out_dir / "images/train", out_dir / "labels/train"
    val_img, val_lbl = out_dir / "images/val", out_dir / "labels/val"

    train_hist: Counter = Counter()
    val_hist: Counter = Counter()
    n_train = n_val = 0

    for converter, val_fraction, seed in converters:
        records = list(converter.records(taxonomy))
        split: Split = grouped_split(records, val_fraction=val_fraction, seed=seed)
        train_hist += _materialize(split.train, converter.name, train_img, train_lbl)
        val_hist += _materialize(split.val, converter.name, val_img, val_lbl)
        n_train += len(split.train)
        n_val += len(split.val)

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(out_dir.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": taxonomy.names_mapping(),
            },
            sort_keys=False,
        )
    )

    return BuildResult(
        out_dir=out_dir,
        data_yaml=data_yaml,
        n_train=n_train,
        n_val=n_val,
        train_class_hist=train_hist,
        val_class_hist=val_hist,
    )
