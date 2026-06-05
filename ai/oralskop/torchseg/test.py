"""Qualitative testing for the custom torchseg (semantic-segmentation) path.

Samples N images from a built dataset, runs a trained model on them, and shows a
three-panel comparison per image — **raw image | model prediction | ground truth** —
with a shared color legend (canonical taxonomy classes, colors consistent with the
visualizer).

Designed for a Jupyter notebook: call :func:`predict_and_show` from a *Python cell*
so the matplotlib figure renders inline (a ``!python -m`` subprocess cannot draw into
the notebook — use the CLI's ``--save DIR`` for that, headless).

Notebook usage (Python cell)::

    from oralskop.torchseg.test import predict_and_show
    predict_and_show(
        weights="runs/seg/deeplabv3_alphadent/best.pt",
        datasets=["alphadent"],
        arch="deeplabv3_resnet50",
        imgsz=512,
        device="cuda",
        num_imgs=8,
    )

CLI usage (headless; writes a PNG instead of showing)::

    python -m oralskop.torchseg.test --config configs/train/seg_torch.yaml \
        --datasets alphadent --weights runs/seg/deeplabv3_alphadent/best.pt \
        --num_imgs 8 --save runs/seg/test_preds
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from oralskop.config import apply_overrides, load_yaml
from oralskop.torchseg.dataset import AI_ROOT, build_seg_dataset
from oralskop.torchseg.model import build_model
from oralskop.viz.visualize import color_for  # shared BGR palette

# ImageNet normalization (matches dataset.py) — used to un-normalize for display.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# --------------------------------------------------------------------------- model
def load_model(
    num_seg_classes: int,
    *,
    weights: str | Path | None,
    arch: str,
    pretrained: bool,
    device: torch.device,
) -> tuple[torch.nn.Module, str, dict[int, str] | None]:
    """Build the model and (if given) load a checkpoint.

    Returns ``(model, arch, class_names_or_None)``. A checkpoint's stored ``arch`` /
    ``class_names`` take precedence so you don't have to remember them.
    """
    ckpt_class_names = None
    if weights:
        ckpt = torch.load(str(weights), map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            state = ckpt["model"]
            arch = ckpt.get("arch", arch)
            ckpt_class_names = ckpt.get("class_names")
            ckpt_ncls = ckpt.get("num_classes")
            if ckpt_ncls is not None and ckpt_ncls != num_seg_classes:
                raise ValueError(
                    f"Checkpoint has {ckpt_ncls} classes but the dataset has "
                    f"{num_seg_classes}. Rebuild the dataset or use matching weights."
                )
        else:
            state = ckpt  # a bare state_dict
        model = build_model(num_seg_classes, arch=arch, pretrained=False)
        model.load_state_dict(state)
    else:
        print("WARNING: no --weights given; using an UNtrained head — predictions "
              "will be meaningless (only useful to smoke-test this pipeline).")
        model = build_model(num_seg_classes, arch=arch, pretrained=pretrained)
    return model.to(device).eval(), arch, ckpt_class_names


# ----------------------------------------------------------------------- rendering
def _denormalize(image: torch.Tensor) -> np.ndarray:
    """Normalized [3,S,S] tensor -> HxWx3 float RGB in [0, 1] for display."""
    rgb = image.cpu().numpy().transpose(1, 2, 0) * _STD + _MEAN
    return np.clip(rgb, 0.0, 1.0)


def _colorize(mask: np.ndarray) -> np.ndarray:
    """Seg mask (0=bg, taxonomy class c -> c+1) -> HxWx3 uint8 RGB (bg stays black)."""
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for value in np.unique(mask):
        if value == 0:
            continue
        b, g, r = color_for(int(value) - 1)  # palette is BGR, indexed by taxonomy id
        out[mask == value] = (r, g, b)
    return out


def _overlay(rgb01: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    """Blend the colorized mask onto the image only where there's a foreground class."""
    out = rgb01.copy()
    color = _colorize(mask).astype(np.float32) / 255.0
    fg = mask > 0
    out[fg] = (1.0 - alpha) * rgb01[fg] + alpha * color[fg]
    return out


def _fg_miou(pred: np.ndarray, gt: np.ndarray, num_seg_classes: int) -> float:
    """Mean IoU over foreground classes present in the ground truth (NaN if none)."""
    ious = []
    for c in range(1, num_seg_classes):
        g = gt == c
        if not g.any():
            continue
        p = pred == c
        union = (p | g).sum()
        ious.append((p & g).sum() / union if union else 0.0)
    return float(np.mean(ious)) if ious else float("nan")


# ------------------------------------------------------------------------ main API
def predict_and_show(
    *,
    weights: str | Path | None = None,
    config: str | Path | None = None,
    datasets: list[str] | None = None,
    arch: str = "deeplabv3_resnet50",
    imgsz: int = 512,
    device: str = "cuda",
    pretrained: bool = True,
    num_imgs: int = 8,
    split: str = "val",
    seed: int = 42,
    alpha: float = 0.5,
    save: str | Path | None = None,
):
    """Run a trained seg model on `num_imgs` sampled images and show raw/pred/GT panels.

    Call from a notebook Python cell to render inline. Pass ``save=DIR`` to write a PNG
    instead (for headless / subprocess use). Any explicit kwarg overrides ``config``.
    Returns the matplotlib Figure.
    """
    import os
    # A `!python -m ...` subprocess inherits MPLBACKEND=module://matplotlib_inline...
    # from the Jupyter kernel; that backend is only valid inside IPython and makes a
    # plain `import matplotlib` raise. When saving (headless), force Agg BEFORE import.
    if save is not None:
        os.environ["MPLBACKEND"] = "Agg"
    import matplotlib
    if save is not None:
        matplotlib.use("Agg")  # no display needed when saving
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    cfg = load_yaml(config) if config else {}
    datasets = datasets or cfg.get("datasets")
    if not datasets:
        raise ValueError("Provide `datasets` (or a config that sets `datasets`).")
    arch = arch if arch != "deeplabv3_resnet50" or "arch" not in cfg else cfg.get("arch", arch)
    imgsz = cfg.get("imgsz", imgsz) if config and imgsz == 512 else imgsz

    want_cuda = device != "cpu"
    dev = torch.device("cuda" if want_cuda and torch.cuda.is_available() else "cpu")
    if want_cuda and dev.type == "cpu":
        print("WARNING: CUDA requested but unavailable; running on CPU.")

    ds = build_seg_dataset(datasets, split=split, imgsz=imgsz, augment=False)
    num_seg_classes = ds.num_seg_classes
    print(f"Datasets {datasets} split={split}: {len(ds)} images | "
          f"{num_seg_classes} seg classes (0=bg) | device={dev.type}")

    model, arch, ckpt_names = load_model(
        num_seg_classes, weights=weights, arch=arch, pretrained=pretrained, device=dev)
    # taxonomy index -> name (prefer the checkpoint's, fall back to the dataset's)
    class_names = ckpt_names or ds.class_names

    n = min(num_imgs, len(ds))
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), n)

    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n), squeeze=False)
    present: set[int] = set()  # taxonomy ids appearing in any pred/gt, for the legend

    for row, idx in enumerate(indices):
        image, target = ds[idx]
        with torch.no_grad():
            logits = model(image.unsqueeze(0).to(dev))["out"]
        pred = logits.argmax(1)[0].cpu().numpy()
        gt = target.cpu().numpy()
        rgb = _denormalize(image)

        present.update(int(v) - 1 for v in np.unique(pred) if v != 0)
        present.update(int(v) - 1 for v in np.unique(gt) if v != 0)
        stem = Path(ds.samples[idx][0]).stem if hasattr(ds, "samples") else f"#{idx}"
        miou = _fg_miou(pred, gt, num_seg_classes)

        axes[row][0].imshow(rgb)
        axes[row][1].imshow(_overlay(rgb, pred, alpha))
        axes[row][2].imshow(_overlay(rgb, gt, alpha))
        axes[row][0].set_ylabel(stem, fontsize=9)
        if row == 0:
            axes[row][0].set_title("Raw image")
            axes[row][1].set_title("Prediction")
            axes[row][2].set_title("Ground truth")
        axes[row][1].set_xlabel(f"fg mIoU {miou:.2f}" if miou == miou else "no GT fg")
        for col in range(3):
            axes[row][col].set_xticks([])
            axes[row][col].set_yticks([])

    # Shared legend (only classes that actually appear).
    handles = []
    for cid in sorted(present):
        b, g, r = color_for(cid)
        name = class_names.get(cid, f"class {cid}") if isinstance(class_names, dict) else f"class {cid}"
        handles.append(mpatches.Patch(color=(r / 255, g / 255, b / 255), label=f"{cid} {name}"))
    if handles:
        fig.legend(handles=handles, loc="lower center",
                   ncol=min(len(handles), 6), bbox_to_anchor=(0.5, -0.02))

    title = f"{arch} on {'+'.join(datasets)} ({split})"
    if weights:
        title += f"  —  {Path(weights).name}"
    fig.suptitle(title, y=1.0)
    fig.tight_layout()

    if save is not None:
        save_dir = Path(save)
        if not save_dir.is_absolute():
            save_dir = AI_ROOT / save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / f"preds_{'_'.join(datasets)}_{split}_n{n}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print(f"Saved {out}")
    else:
        plt.show()
    return fig


# ----------------------------------------------------------------------------- CLI
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qualitative testing for torchseg models.")
    p.add_argument("--config", help="seg_torch.yaml for defaults (optional).")
    p.add_argument("--datasets", nargs="+", help="Built dataset name(s).")
    p.add_argument("--weights", help="Checkpoint (best.pt/last.pt) to load.")
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE",
                   help="arch=, imgsz=, device=, pretrained= (override the config).")
    p.add_argument("--num_imgs", "--num", type=int, default=8, dest="num_imgs")
    p.add_argument("--split", default="val", choices=["train", "val", "test", "all"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.5, help="Mask overlay opacity.")
    p.add_argument("--save", help="Write a PNG here instead of showing (headless).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = apply_overrides(load_yaml(args.config), args.override) if args.config else \
        apply_overrides({}, args.override)
    datasets = args.datasets or cfg.get("datasets")
    # A bare `!python -m ...` subprocess can't render inline, so default to saving.
    save = args.save or (None if cfg.get("_force_show") else "runs/seg/test_preds")
    predict_and_show(
        weights=args.weights,
        datasets=datasets,
        arch=cfg.get("arch", "deeplabv3_resnet50"),
        imgsz=cfg.get("imgsz", 512),
        device=cfg.get("device", "cuda"),
        pretrained=bool(cfg.get("pretrained", True)),
        num_imgs=args.num_imgs,
        split=args.split,
        seed=args.seed,
        alpha=args.alpha,
        save=save,
    )


if __name__ == "__main__":
    main()
