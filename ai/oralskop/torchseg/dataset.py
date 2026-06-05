"""PyTorch datasets for custom (non-YOLO) semantic-segmentation training.

Consumes the SAME built data the YOLO path uses (``data/<name>/`` with images +
canonical YOLO-seg polygon labels), but rasterizes the polygons into a per-pixel
class mask so any torchvision/`nn.Module` segmentation model can train on it.

Pixel target convention: ``0 = background``; canonical taxonomy class ``c`` -> ``c + 1``.
So ``num_seg_classes = len(taxonomy names) + 1``. Because every built dataset writes the
full canonical taxonomy into its ``data.yaml``, all datasets share one label space and
merge cleanly via :class:`MergedSegDataset`.
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import ConcatDataset, Dataset

from oralskop.data.verify import IMAGE_EXTS

AI_ROOT = Path(__file__).resolve().parents[2]

# ImageNet normalization (torchvision pretrained backbones expect it).
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _split_dirs(data_yaml: Path, split: str) -> list[tuple[Path, Path]]:
    cfg = yaml.safe_load(data_yaml.read_text())
    root = Path(cfg.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    keys = ["train", "val", "test"] if split == "all" else [split]
    pairs = []
    for key in keys:
        if key not in cfg:
            continue
        images_dir = (root / cfg[key]).resolve()
        labels_dir = Path(str(images_dir).replace("/images/", "/labels/"))
        if images_dir.is_dir():
            pairs.append((images_dir, labels_dir))
    return pairs


def _class_names(data_yaml: Path) -> dict[int, str]:
    names = yaml.safe_load(data_yaml.read_text()).get("names", {})
    if isinstance(names, list):
        return dict(enumerate(names))
    return {int(k): v for k, v in names.items()}


def _resolve_data_yaml(name: str | None, data_yaml: str | Path | None, base_dir: Path) -> Path:
    if data_yaml:
        return Path(data_yaml)
    if name:
        p = base_dir / "data" / name / "data.yaml"
        if not p.exists():
            raise FileNotFoundError(
                f"Dataset '{name}' not built: {p} missing. Run "
                f"`python -m oralskop.data.prepare --datasets {name}` first."
            )
        return p
    raise ValueError("Provide either `name` or `data_yaml`.")


def rasterize_polygons(label_text: str, size: int) -> np.ndarray:
    """YOLO-seg polygon lines -> (size, size) uint8 mask (0=bg, class c -> c+1).

    Larger polygons are painted first so small lesions stay visible on top.
    """
    mask = np.zeros((size, size), dtype=np.uint8)
    polys: list[tuple[int, np.ndarray]] = []
    for line in label_text.strip().splitlines():
        parts = line.split()
        if len(parts) < 7 or len(parts) % 2 == 0:
            continue
        try:
            cls = int(parts[0])
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        pts = np.array(
            [(coords[i] * size, coords[i + 1] * size) for i in range(0, len(coords) - 1, 2)],
            dtype=np.int32,
        )
        if len(pts) >= 3:
            polys.append((cls, pts))
    for cls, pts in sorted(polys, key=lambda cp: cv2.contourArea(cp[1]), reverse=True):
        cv2.fillPoly(mask, [pts], cls + 1)
    return mask


class YoloSegDataset(Dataset):
    """One built dataset/split -> (image_tensor[3,S,S], target_mask[S,S] long)."""

    def __init__(
        self,
        name: str | None = None,
        *,
        data_yaml: str | Path | None = None,
        split: str = "train",
        imgsz: int = 512,
        augment: bool = False,
        base_dir: Path = AI_ROOT,
    ):
        self.data_yaml = _resolve_data_yaml(name, data_yaml, base_dir)
        self.imgsz = imgsz
        self.augment = augment
        self.class_names = _class_names(self.data_yaml)
        self.num_seg_classes = len(self.class_names) + 1  # + background

        self.samples: list[tuple[Path, Path]] = []
        for images_dir, labels_dir in _split_dirs(self.data_yaml, split):
            for img in sorted(images_dir.iterdir()):
                if img.is_file() and img.suffix.lower() in IMAGE_EXTS:
                    self.samples.append((img, labels_dir / f"{img.stem}.txt"))
        if not self.samples:
            raise RuntimeError(f"No samples for split={split} in {self.data_yaml}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, label_path = self.samples[idx]
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise RuntimeError(f"Unreadable image: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

        text = label_path.read_text() if label_path.exists() else ""
        mask = rasterize_polygons(text, self.imgsz)

        if self.augment and random.random() < 0.5:          # horizontal flip
            rgb = np.ascontiguousarray(rgb[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])

        image = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
        image = (image - _MEAN) / _STD
        target = torch.from_numpy(mask.astype(np.int64))
        return image, target


class MergedSegDataset(ConcatDataset):
    """Concatenate several built datasets into one (shared canonical label space)."""

    def __init__(self, names: list[str], **kwargs):
        datasets = [YoloSegDataset(name=n, **kwargs) for n in names]
        ncls = {d.num_seg_classes for d in datasets}
        if len(ncls) != 1:
            raise ValueError(f"Datasets disagree on class count: {ncls}. Rebuild them.")
        super().__init__(datasets)
        self.num_seg_classes = datasets[0].num_seg_classes
        self.class_names = datasets[0].class_names


def build_seg_dataset(names: list[str], split: str, **kwargs):
    """Single :class:`YoloSegDataset` for one name, else a :class:`MergedSegDataset`."""
    if len(names) == 1:
        return YoloSegDataset(name=names[0], split=split, **kwargs)
    return MergedSegDataset(names, split=split, **kwargs)
