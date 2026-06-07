"""Manifest-driven object-detection dataset (YOLO bbox -> DETR targets).

Filters the manifest to the **yolo-bbox** rows that have a referenced ``.txt`` label,
then yields ``(pixel_values, target)`` where ``target`` is the HF-DETR contract:

    {"class_labels": LongTensor[n], "boxes": FloatTensor[n, 4]}   # boxes = cx,cy,w,h norm

Native YOLO class IDs can be mapped per source to the detector vocabulary. The legacy
image-label fallback is still available for comparison, but the preferred path is
``box_label_source=native`` so boxes keep their own source class. YOLO's normalized
``cx cy w h`` is exactly DETR's target box format, so no box conversion is needed.
Images + label files are streamed via fsspec, so the same code serves a local dir or
an S3 bucket (reuses the `clf` loaders).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from oralskop.config import load_yaml
# Reuse the classifier's manifest reader, fsspec image loader, and label parsing.
from oralskop.clf.dataset import AI_ROOT, _join_uri, _load_image, read_manifest  # noqa: F401
from oralskop.clf.vocab import Vocab, split_labels

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)
_PRETRAIN_FORMAT = "unlabeled-pretraining"
_PRETRAIN_SPLIT = "pretrain"
_BOX_LABEL_IMAGE = "image"
_BOX_LABEL_NATIVE = "native"
_BOX_LABEL_CLASS_AGNOSTIC = "class_agnostic"
_UNKNOWN_DROP = "drop"
_UNKNOWN_ERROR = "error"
_UNKNOWN_IMAGE = "image"
_UNKNOWN_CLASS_AGNOSTIC = "class_agnostic"


@dataclass(frozen=True)
class BoxLabelConfig:
    """Detector box-labeling policy resolved from config."""

    source: str = _BOX_LABEL_IMAGE
    class_maps: dict[str, dict[int, str]] | None = None
    unknown_policy: str = _UNKNOWN_DROP
    class_agnostic_label: str = "object"

    def __post_init__(self) -> None:
        source = str(self.source).lower().replace("-", "_")
        unknown = str(self.unknown_policy).lower().replace("-", "_")
        if source not in {_BOX_LABEL_IMAGE, _BOX_LABEL_NATIVE, _BOX_LABEL_CLASS_AGNOSTIC}:
            raise ValueError(
                f"Unknown box_label_source {self.source!r}. "
                "Options: image, native, class_agnostic."
            )
        if unknown not in {_UNKNOWN_DROP, _UNKNOWN_ERROR, _UNKNOWN_IMAGE, _UNKNOWN_CLASS_AGNOSTIC}:
            raise ValueError(
                f"Unknown unknown_box_class_policy {self.unknown_policy!r}. "
                "Options: drop, error, image, class_agnostic."
            )
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "unknown_policy", unknown)
        object.__setattr__(self, "class_maps", self.class_maps or {})


def load_box_label_config(
    path: str | Path | None,
    *,
    source: str = _BOX_LABEL_IMAGE,
    unknown_policy: str = _UNKNOWN_DROP,
    class_agnostic_label: str = "object",
) -> BoxLabelConfig:
    """Load per-dataset native YOLO class-id maps for detector training.

    YAML schema:

        datasets:
          source_dataset_name:
            0: carie
            1: maladie_parodontale
    """
    maps: dict[str, dict[int, str]] = {}
    if path:
        cfg = load_yaml(path)
        for dataset, raw_map in (cfg.get("datasets") or {}).items():
            if not isinstance(raw_map, dict):
                raise ValueError(f"{path}: datasets.{dataset} must be a mapping.")
            parsed: dict[int, str] = {}
            for raw_key, raw_value in raw_map.items():
                try:
                    class_id = int(raw_key)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{path}: datasets.{dataset}.{raw_key!r} must be an integer class id."
                    ) from exc
                parsed[class_id] = str(raw_value)
            maps[str(dataset)] = parsed
    return BoxLabelConfig(
        source=source,
        class_maps=maps,
        unknown_policy=unknown_policy,
        class_agnostic_label=class_agnostic_label,
    )


def labels_from_box_config(
    cfg: BoxLabelConfig,
    *,
    fallback_labels_per_row: list[list[str]] | None = None,
) -> list[list[str]]:
    """Return labels useful for deriving a vocab when no labels_file is provided."""
    if cfg.source == _BOX_LABEL_CLASS_AGNOSTIC:
        return [[cfg.class_agnostic_label]]
    if cfg.source == _BOX_LABEL_NATIVE:
        label_set = {name for by_id in cfg.class_maps.values() for name in by_id.values()}
        if cfg.unknown_policy == _UNKNOWN_CLASS_AGNOSTIC:
            label_set.add(cfg.class_agnostic_label)
        labels = sorted(label_set)
        return [[name] for name in labels]
    return fallback_labels_per_row or []


def load_bbox_frame(
    manifest,
    level: str,
    *,
    image_path_prefixes: list[str] | None = None,
    limit: int | None = None,
    box_label_config: BoxLabelConfig | None = None,
):
    """Filter the manifest to bbox rows and attach image labels for optional fallback.

    Returns ``(df, labels_per_row)``: rows carry ``image_path``, ``label_path`` (a
    ``.txt``) and ``_labels`` (the image-level canonical labels). ``labels_per_row`` is
    used only for vocabulary derivation when no labels file is configured.
    """
    box_label_config = box_label_config or BoxLabelConfig()
    if (
        box_label_config.source == _BOX_LABEL_NATIVE
        and box_label_config.unknown_policy in {_UNKNOWN_DROP, _UNKNOWN_ERROR}
        and not box_label_config.class_maps
    ):
        raise ValueError(
            "box_label_source=native with unknown_box_class_policy=drop/error requires "
            "a non-empty box_label_map. Use class_agnostic mode or add source maps."
        )
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
    if box_label_config.source == _BOX_LABEL_NATIVE and box_label_config.unknown_policy in {
        _UNKNOWN_DROP, _UNKNOWN_ERROR,
    }:
        known_sources = set(box_label_config.class_maps)
        if known_sources:
            df = df[df["dataset"].astype(str).isin(known_sources)].copy()
            labels = labels.loc[df.index]

    needs_single_image_label = (
        box_label_config.source == _BOX_LABEL_IMAGE
        or (
            box_label_config.source == _BOX_LABEL_NATIVE
            and box_label_config.unknown_policy == _UNKNOWN_IMAGE
        )
    )
    if needs_single_image_label:
        df = df[labels.map(len) == 1].copy()
        labels = labels.loc[df.index]

    df["_labels"] = labels.map(tuple)
    df["_label"] = labels.map(lambda xs: xs[0] if len(xs) == 1 else "")
    if limit:
        df = df.head(int(limit)).copy()
    if box_label_config.source == _BOX_LABEL_NATIVE:
        labels_per_row = labels_from_box_config(box_label_config)
    elif box_label_config.source == _BOX_LABEL_CLASS_AGNOSTIC:
        labels_per_row = [[box_label_config.class_agnostic_label]]
    else:
        labels_per_row = [list(v) for v in df["_labels"]]
    return df, labels_per_row


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


def parse_yolo_box_records(text: str) -> list[tuple[int, tuple[float, float, float, float]]]:
    """Parse YOLO ``class cx cy w h`` lines -> ``(class_id, box)`` records.

    Only 5-token (bbox) lines are kept; polygon/segmentation lines (odd >5 tokens) and
    degenerate boxes are dropped.
    """
    records = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(float(parts[0]))
            cx, cy, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        except ValueError:
            continue
        if w <= 0 or h <= 0:
            continue
        box = (
            min(max(cx, 0.0), 1.0),
            min(max(cy, 0.0), 1.0),
            min(w, 1.0),
            min(h, 1.0),
        )
        records.append((class_id, box))
    return records


def parse_yolo_boxes(text: str) -> list[tuple[float, float, float, float]]:
    """Parse YOLO bbox text and return boxes only, preserving the old helper API."""
    boxes = [box for _, box in parse_yolo_box_records(text)]
    return boxes


def build_transforms(imgsz: int, mean=_MEAN, std=_STD):
    """Resize-only + normalize (normalized boxes are invariant to resize)."""
    return transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def _letterbox(img, boxes, imgsz: int):
    """Aspect-preserving resize to ``imgsz`` + center pad; remap normalized boxes.

    The square ``Resize`` squashes objects (a tall lesion becomes wide), so we instead
    scale the long side to ``imgsz`` and pad the short side. ``boxes`` are normalized
    ``cx,cy,w,h`` to the original WxH; this re-normalizes them to the padded square.
    Returns ``(resized_PIL_image, remapped_boxes_list)``.
    """
    from PIL import Image as _Image

    w, h = img.size
    scale = imgsz / max(w, h)
    nw, nh = max(round(w * scale), 1), max(round(h * scale), 1)
    pad_x, pad_y = (imgsz - nw) / 2.0, (imgsz - nh) / 2.0
    canvas = _Image.new("RGB", (imgsz, imgsz), (114, 114, 114))
    canvas.paste(img.resize((nw, nh), _Image.BILINEAR), (round(pad_x), round(pad_y)))
    out = [(
        (cx * nw + pad_x) / imgsz,
        (cy * nh + pad_y) / imgsz,
        bw * nw / imgsz,
        bh * nh / imgsz,
    ) for cx, cy, bw, bh in boxes]
    return canvas, out


def _hflip(img, boxes):
    """Horizontal flip of the image and its normalized boxes (``cx -> 1 - cx``)."""
    from PIL import Image as _Image

    flipped = img.transpose(_Image.FLIP_LEFT_RIGHT)
    return flipped, [(1.0 - cx, cy, bw, bh) for cx, cy, bw, bh in boxes]


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
        box_label_config: BoxLabelConfig | None = None,
        unreadable_log_limit: int = 0,
        train: bool = False,
        augment: bool = True,
        letterbox: bool = True,
        hflip: bool = True,
        color_jitter: float = 0.2,
    ):
        self.vocab = vocab
        self.image_root = image_root
        self.cache_dir = cache_dir
        self.imgsz = int(imgsz)
        self.letterbox = bool(letterbox)
        # Random aug only on the train split; val/test get letterbox-only geometry.
        self.do_aug = bool(train and augment)
        self.do_hflip = self.do_aug and bool(hflip)
        cj = float(color_jitter or 0.0)
        self.jitter = (transforms.ColorJitter(brightness=cj, contrast=cj, saturation=cj)
                       if self.do_aug and cj > 0 else None)
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(mean, std)])
        # Fallback square resize when letterbox is off (normalized boxes are resize-invariant).
        self.square_resize = transforms.Resize((self.imgsz, self.imgsz))
        self.box_label_config = box_label_config or BoxLabelConfig()
        self.unreadable_log_limit = max(int(unreadable_log_limit or 0), 0)

        self.samples: list[tuple[str, str, str, tuple[str, ...]]] = []
        dropped = 0
        for _, row in df.iterrows():
            labels = tuple(row.get("_labels") or ())
            if self.box_label_config.source == _BOX_LABEL_IMAGE:
                if len(labels) != 1 or labels[0] not in vocab.index:
                    dropped += 1
                    continue
            elif (
                self.box_label_config.source == _BOX_LABEL_CLASS_AGNOSTIC
                and self.box_label_config.class_agnostic_label not in vocab.index
            ):
                dropped += 1
                continue
            self.samples.append((
                str(row["image_path"]),
                str(row["label_path"]),
                str(row.get("dataset", "")),
                labels,
            ))
        self.dropped_off_vocab = dropped
        self._missing: set[int] = set()
        self._missing_logged = 0
        self.skipped_unmapped_boxes = 0

    def _box_label_name(self, dataset: str, native_class_id: int, image_labels: tuple[str, ...]) -> str | None:
        cfg = self.box_label_config
        if cfg.source == _BOX_LABEL_CLASS_AGNOSTIC:
            return cfg.class_agnostic_label
        if cfg.source == _BOX_LABEL_IMAGE:
            return image_labels[0] if len(image_labels) == 1 else None

        label = cfg.class_maps.get(dataset, {}).get(native_class_id)
        if label is not None:
            return label
        if cfg.unknown_policy == _UNKNOWN_ERROR:
            raise ValueError(f"unmapped native class id {native_class_id} for dataset {dataset!r}")
        if cfg.unknown_policy == _UNKNOWN_IMAGE:
            return image_labels[0] if len(image_labels) == 1 else None
        if cfg.unknown_policy == _UNKNOWN_CLASS_AGNOSTIC:
            return cfg.class_agnostic_label
        return None

    def _target_from_records(
        self,
        records: list[tuple[int, tuple[float, float, float, float]]],
        *,
        dataset: str,
        image_labels: tuple[str, ...],
    ) -> dict[str, torch.Tensor] | None:
        labels: list[int] = []
        boxes: list[tuple[float, float, float, float]] = []
        skipped = 0
        for native_class_id, box in records:
            label_name = self._box_label_name(dataset, native_class_id, image_labels)
            if label_name is None:
                skipped += 1
                continue
            class_idx = self.vocab.index.get(label_name)
            if class_idx is None:
                skipped += 1
                continue
            labels.append(class_idx)
            boxes.append(box)
        self.skipped_unmapped_boxes += skipped
        if not boxes:
            return None
        return {
            "class_labels": torch.tensor(labels, dtype=torch.long),
            "boxes": torch.tensor(boxes, dtype=torch.float32),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        tries, idx = 0, i
        while tries < 8:
            if idx not in self._missing:
                rel_img, rel_lbl, dataset, image_labels = self.samples[idx]
                try:
                    pil = _load_image(_join_uri(self.image_root, rel_img), self.cache_dir)
                    records = parse_yolo_box_records(_load_text(_join_uri(self.image_root, rel_lbl), self.cache_dir))
                    if not records:
                        raise ValueError("no boxes")
                    target = self._target_from_records(records, dataset=dataset, image_labels=image_labels)
                    if target is None:
                        raise ValueError("no mapped boxes")
                    # Apply geometry jointly to image + boxes (boxes stay aligned with class_labels).
                    boxes = [tuple(b) for b in target["boxes"].tolist()]
                    if self.letterbox:
                        pil, boxes = _letterbox(pil, boxes, self.imgsz)
                    else:
                        pil = self.square_resize(pil)  # boxes are normalized -> unchanged
                    if self.do_hflip and random.random() < 0.5:
                        pil, boxes = _hflip(pil, boxes)
                    if self.jitter is not None:
                        pil = self.jitter(pil)
                    target["boxes"] = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
                    img = self.to_tensor(pil)
                    return img, target
                except Exception as exc:  # missing/empty/corrupt image or label
                    self._missing.add(idx)
                    if self._missing_logged < self.unreadable_log_limit:
                        print(f">> det: skipping {rel_img} / {rel_lbl} ({exc})")
                        self._missing_logged += 1
            idx = random.randrange(len(self.samples))
            tries += 1
        raise RuntimeError("Too many unreadable images/labels — check image_root / S3 access.")
