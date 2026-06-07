"""Manifest-driven object-detection dataset (YOLO bbox -> DETR targets).

Filters the manifest to the **yolo-bbox** rows that have a referenced ``.txt`` label
and exactly one in-vocabulary coarse class, then yields ``(pixel_values, target)``
where ``target`` is the HF-DETR contract:

    {"class_labels": LongTensor[n], "boxes": FloatTensor[n, 4]}   # boxes = cx,cy,w,h norm

Box class IDs in the ``.txt`` are native per-source and are **ignored** — every box in
an image is assigned that image's single ``canonical_coarse`` class (the user's chosen
weak-multi-class scheme). YOLO's normalized ``cx cy w h`` is exactly DETR's target box
format, so no conversion is needed. Images + label files are streamed via fsspec, so the
same code serves a local dir or an S3 bucket (reuses the `clf` loaders).
"""

from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Reuse the classifier's manifest reader, fsspec image loader, and label parsing.
from oralskop.clf.dataset import AI_ROOT, _join_uri, _load_image, read_manifest  # noqa: F401
from oralskop.clf.vocab import Vocab, split_labels

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)
_PRETRAIN_FORMAT = "unlabeled-pretraining"
_PRETRAIN_SPLIT = "pretrain"


def load_bbox_frame(
    manifest,
    level: str,
    *,
    image_path_prefixes: list[str] | None = None,
    limit: int | None = None,
):
    """Filter the manifest to single-class bbox rows; attach a parsed ``_label``.

    Returns ``(df, labels_per_row)``: rows carry ``image_path``, ``label_path`` (a
    ``.txt``) and ``_label`` (the single coarse class name). ``labels_per_row`` mirrors
    ``_label`` for vocabulary derivation (matches the clf convention).
    """
    label_col = f"canonical_{level}"
    df = read_manifest(manifest)
    for col in ("image_path", "label_path", "split", "annotation_format", label_col):
        if col not in df.columns:
            raise ValueError(f"Manifest missing required column {col!r}. Found: {list(df.columns)}")

    keep = (
        df["annotation_format"].str.contains("yolo-bbox")
        & df["label_path"].str.strip().str.endswith(".txt")
        & (df["split"].str.strip() != _PRETRAIN_SPLIT)
    )
    df = df[keep].copy()
    if image_path_prefixes:
        prefixes = tuple(p.lstrip("/") for p in image_path_prefixes)
        df = df[df["image_path"].str.lstrip("/").str.startswith(prefixes)].copy()

    labels = df[label_col].map(split_labels)
    df = df[labels.map(len) == 1].copy()  # exactly one coarse class -> weak box labels
    df["_label"] = df[label_col].map(lambda s: split_labels(s)[0])
    if limit:
        df = df.head(int(limit)).copy()
    return df, [[v] for v in df["_label"]]


def _load_text(uri: str, cache_dir: str | None) -> str:
    """Read a small text file from a local path or fsspec URL (``s3://`` …)."""
    if "://" in uri:
        import fsspec

        if cache_dir:
            of = fsspec.open(f"simplecache::{uri}", simplecache={"cache_storage": cache_dir}, mode="rt")
        else:
            of = fsspec.open(uri, mode="rt")
        with of as f:
            return f.read()
    with open(uri, encoding="utf-8") as f:
        return f.read()


def parse_yolo_boxes(text: str) -> list[tuple[float, float, float, float]]:
    """Parse YOLO ``class cx cy w h`` lines -> list of (cx, cy, w, h), normalized.

    Only 5-token (bbox) lines are kept; polygon/segmentation lines (odd >5 tokens) and
    degenerate boxes are dropped. The native class id is discarded.
    """
    boxes = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            cx, cy, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        except ValueError:
            continue
        if w <= 0 or h <= 0:
            continue
        boxes.append((min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0),
                      min(w, 1.0), min(h, 1.0)))
    return boxes


def build_transforms(imgsz: int, mean=_MEAN, std=_STD):
    """Resize-only + normalize (normalized boxes are invariant to resize)."""
    return transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def det_collate_fn(batch):
    """Stack pixel_values; keep targets as a list of dicts (HF DETR contract)."""
    pixel_values = torch.stack([b[0] for b in batch])
    labels = [b[1] for b in batch]
    return pixel_values, labels


class ManifestDetDataset(Dataset):
    """A split of the bbox manifest as (pixel_values, DETR-target) pairs."""

    def __init__(
        self,
        df,
        vocab: Vocab,
        *,
        image_root: str,
        imgsz: int,
        cache_dir: str | None = None,
        mean=_MEAN,
        std=_STD,
        unreadable_log_limit: int = 0,
    ):
        self.vocab = vocab
        self.image_root = image_root
        self.cache_dir = cache_dir
        self.transform = build_transforms(imgsz, mean=mean, std=std)
        self.unreadable_log_limit = max(int(unreadable_log_limit or 0), 0)

        self.samples: list[tuple[str, str, int]] = []  # (image_path, label_path, class_idx)
        dropped = 0
        for img, lbl, name in zip(df["image_path"], df["label_path"], df["_label"]):
            idx = vocab.index.get(name)
            if idx is None:
                dropped += 1
                continue
            self.samples.append((str(img), str(lbl), idx))
        self.dropped_off_vocab = dropped
        self._missing: set[int] = set()
        self._missing_logged = 0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        tries, idx = 0, i
        while tries < 8:
            if idx not in self._missing:
                rel_img, rel_lbl, cls = self.samples[idx]
                try:
                    img = self.transform(_load_image(_join_uri(self.image_root, rel_img), self.cache_dir))
                    boxes = parse_yolo_boxes(_load_text(_join_uri(self.image_root, rel_lbl), self.cache_dir))
                    if not boxes:
                        raise ValueError("no boxes")
                    target = {
                        "class_labels": torch.full((len(boxes),), cls, dtype=torch.long),
                        "boxes": torch.tensor(boxes, dtype=torch.float32),
                    }
                    return img, target
                except Exception as exc:  # missing/empty/corrupt image or label
                    self._missing.add(idx)
                    if self._missing_logged < self.unreadable_log_limit:
                        print(f">> det: skipping {rel_img} / {rel_lbl} ({exc})")
                        self._missing_logged += 1
            idx = random.randrange(len(self.samples))
            tries += 1
        raise RuntimeError("Too many unreadable images/labels — check image_root / S3 access.")
