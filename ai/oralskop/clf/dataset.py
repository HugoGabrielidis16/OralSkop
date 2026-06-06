"""Manifest-driven multi-label classification dataset.

Reads the curated ``manifest_03_master_FINAL.csv`` (local or ``s3://``), keeps the
supervised rows, and yields ``(image_tensor, multi_hot_target)`` pairs. Images are
opened through fsspec so the same code path serves a local directory or an S3
bucket (the doc's intended AWS-notebook setup); an optional on-disk cache
amortises repeated S3 reads across epochs.

Filtering rules follow ``ai/PASSATION_DATA_OralSkop.md``:
* drop ``annotation_format == "unlabeled-pretraining"`` and ``split == "pretrain"``
  (the 48k unlabelled MetaDent corpus, doc §6.1);
* drop rows with no image-level label for the chosen level (e.g. the
  segmentation-mask rows whose labels are pixel-level, doc §6.2);
* drop rows whose label set is empty *after* mapping onto the vocabulary
  (e.g. train-only micro-classes, or fine labels outside the committed list).
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from oralskop.clf.vocab import Vocab, split_labels

AI_ROOT = Path(__file__).resolve().parents[2]

# ImageNet normalization (matches the rest of the repo).
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)

_PRETRAIN_FORMAT = "unlabeled-pretraining"
_PRETRAIN_SPLIT = "pretrain"


def read_manifest(path: str | Path):
    """Load the manifest CSV into a DataFrame (supports ``s3://`` via s3fs)."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(
            "The classifier path needs the `clf` extra: `uv sync --extra clf` "
            "(installs pandas + s3fs)."
        ) from exc
    # dtype=str keeps label cells verbatim; we parse them ourselves.
    return pd.read_csv(str(path), dtype=str, keep_default_na=False)


def load_supervised_frame(
    manifest: str | Path,
    level: str,
    *,
    image_path_prefixes: list[str] | None = None,
    limit: int | None = None,
):
    """Read + filter to the supervised set and attach a parsed ``_labels`` column.

    Returns ``(df, labels_per_row)`` where ``df`` has the original columns plus
    ``_labels`` (list of label names) and ``labels_per_row`` mirrors ``df._labels``
    (handy for deriving a vocabulary before any split is selected).
    """
    label_col = f"canonical_{level}"
    df = read_manifest(manifest)
    for col in ("image_path", "split", "annotation_format", label_col):
        if col not in df.columns:
            raise ValueError(f"Manifest missing required column {col!r}. Found: {list(df.columns)}")

    keep = (df["annotation_format"].str.strip() != _PRETRAIN_FORMAT) & (
        df["split"].str.strip() != _PRETRAIN_SPLIT
    )
    df = df[keep].copy()

    if image_path_prefixes:
        prefixes = tuple(p.lstrip("/") for p in image_path_prefixes)
        df = df[df["image_path"].str.lstrip("/").str.startswith(prefixes)].copy()

    df["_labels"] = df[label_col].map(split_labels)
    df = df[df["_labels"].map(len) > 0].copy()  # need an image-level label

    if limit:
        df = df.head(int(limit)).copy()

    return df, list(df["_labels"])


def build_transforms(imgsz: int, *, train: bool):
    """On-the-fly augmentation for train; deterministic resize for valid/test."""
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(imgsz, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])
    return transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


def _join_uri(image_root: str, rel: str) -> str:
    return f"{image_root.rstrip('/')}/{rel.lstrip('/')}"


def _load_image(uri: str, cache_dir: str | None) -> Image.Image:
    """Open an image from a local path or an fsspec URL (``s3://`` etc.)."""
    if "://" in uri:
        import fsspec

        if cache_dir:
            of = fsspec.open(f"simplecache::{uri}", simplecache={"cache_storage": cache_dir})
        else:
            of = fsspec.open(uri)
        with of as f:
            return Image.open(f).convert("RGB")
    return Image.open(uri).convert("RGB")


def pos_weight_from(targets: np.ndarray) -> torch.Tensor:
    """Per-class ``neg/pos`` for ``BCEWithLogitsLoss`` (doc §7.4 imbalance)."""
    pos = targets.sum(axis=0)
    neg = targets.shape[0] - pos
    weight = np.where(pos > 0, neg / np.clip(pos, 1, None), 1.0)
    return torch.tensor(weight, dtype=torch.float32)


class ManifestClfDataset(Dataset):
    """A single split of the manifest, encoded against ``vocab``.

    Rows whose multi-hot target is all-zero after encoding are dropped (their
    labels fall entirely outside the vocabulary, e.g. train-only micro-classes).
    """

    def __init__(
        self,
        df,
        vocab: Vocab,
        *,
        image_root: str,
        imgsz: int,
        train: bool,
        cache_dir: str | None = None,
    ):
        self.vocab = vocab
        self.image_root = image_root
        self.cache_dir = cache_dir
        self.transform = build_transforms(imgsz, train=train)

        self.paths: list[str] = []
        targets: list[np.ndarray] = []
        dropped_empty = 0
        for rel, labels in zip(df["image_path"], df["_labels"]):
            vec = vocab.encode(labels)
            if vec.sum() == 0:
                dropped_empty += 1
                continue
            self.paths.append(str(rel))
            targets.append(vec)
        self.targets = np.stack(targets) if targets else np.zeros((0, len(vocab)), np.float32)
        self.dropped_empty = dropped_empty
        self._missing: set[int] = set()
        self._missing_logged = 0

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        # Skip-with-substitute on a missing/corrupt image so batch shapes stay valid.
        tries = 0
        idx = i
        while tries < 8:
            if idx not in self._missing:
                uri = _join_uri(self.image_root, self.paths[idx])
                try:
                    img = self.transform(_load_image(uri, self.cache_dir))
                    target = torch.from_numpy(self.targets[idx])
                    return img, target
                except Exception as exc:  # missing key, decode error, transient S3 error
                    self._missing.add(idx)
                    if self._missing_logged < 10:
                        print(f">> clf: skipping unreadable image {uri} ({exc})")
                        self._missing_logged += 1
            idx = random.randrange(len(self.paths))
            tries += 1
        raise RuntimeError(
            "Too many unreadable images in a row — check image_root / S3 access / "
            "image_path_prefixes."
        )
