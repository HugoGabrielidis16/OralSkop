"""Custom PyTorch semantic-segmentation training loop (non-YOLO path).

Trains a torchvision segmentation model on any built dataset, or on a merged set, using
the canonical taxonomy as the per-pixel label space. Config-driven; `--override key=val`
and `--datasets a b` override the YAML.

    python -m oralskop.torchseg.train --config configs/train/seg_torch.yaml
    python -m oralskop.torchseg.train --config configs/train/seg_torch.yaml \
        --datasets alphadent bmc_oral_health --override epochs=1 device=cpu batch=2
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

# ImageNet normalization (matches dataset.py) — to un-normalize images for W&B previews.
_NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_NORM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

from oralskop.config import apply_overrides, load_yaml
from oralskop.torchseg.dataset import AI_ROOT, build_seg_dataset
from oralskop.torchseg.losses import build_criterion
from oralskop.torchseg.lora import apply_lora, parameter_counts
from oralskop.torchseg.model import build_model, has_aux
from oralskop.viz.visualize import color_for  # shared BGR palette


@torch.no_grad()
def evaluate(
    model,
    loader,
    num_classes,
    device,
    limit: int | None = None,
    desc: str = "val",
    progress: bool = True,
    criterion=None,
    aux_weight: float = 0.0,
) -> dict[str, float]:
    """Return semantic-segmentation metrics from a confusion matrix.

    If ``criterion`` is given, also computes the mean validation loss (matching the
    training objective, including the aux head when ``aux_weight`` > 0).
    """
    model.eval()
    conf = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)
    loss_sum, loss_n = 0.0, 0
    batches = tqdm(
        loader,
        total=loader_batches(loader, limit),
        desc=desc,
        leave=False,
        dynamic_ncols=True,
        disable=not progress,
    )
    for i, (images, targets) in enumerate(batches):
        if limit and i >= limit:
            break
        images, targets = images.to(device), targets.to(device)
        out = model(images)
        logits = out["out"]
        if criterion is not None:
            loss = criterion(logits, targets)
            if aux_weight and "aux" in out:
                loss = loss + aux_weight * criterion(out["aux"], targets)
            loss_sum += loss.item() * images.size(0)
            loss_n += images.size(0)
        preds = logits.argmax(1)
        k = (targets >= 0) & (targets < num_classes)
        idx = num_classes * targets[k].view(-1) + preds[k].view(-1)
        conf += torch.bincount(idx, minlength=num_classes**2).reshape(num_classes, num_classes)
    true_pixels = conf.sum(1).float()
    pred_pixels = conf.sum(0).float()
    inter = conf.diag().float()
    union = pred_pixels + true_pixels - inter
    iou = inter / union.clamp(min=1)
    dice = (2 * inter) / (pred_pixels + true_pixels).clamp(min=1)
    class_acc = inter / true_pixels.clamp(min=1)
    present = true_pixels > 0  # classes that actually appear in val
    fg_present = present.clone()
    fg_present[0] = False
    fg_true = true_pixels[1:].sum()
    return {
        "val_loss": (loss_sum / loss_n) if loss_n else float("nan"),
        "pixel_acc": (inter.sum() / conf.sum().clamp(min=1)).item(),
        "fg_pixel_acc": (inter[1:].sum() / fg_true.clamp(min=1)).item(),
        "mean_acc": class_acc[present].mean().item() if present.any() else 0.0,
        "miou": iou[present].mean().item() if present.any() else 0.0,
        "fg_miou": iou[fg_present].mean().item() if fg_present.any() else 0.0,
        "mean_dice": dice[present].mean().item() if present.any() else 0.0,
        "fg_dice": dice[fg_present].mean().item() if fg_present.any() else 0.0,
        "per_class_iou": iou.tolist(),
        "per_class_dice": dice.tolist(),
        "per_class_acc": class_acc.tolist(),
        "per_class_support": true_pixels.long().tolist(),
    }


def train_pixel_accuracy(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> tuple[int, int]:
    """Return correct/total train pixels for one batch."""
    preds = logits.argmax(1)
    valid = (targets >= 0) & (targets < num_classes)
    correct = ((preds == targets) & valid).sum().item()
    total = valid.sum().item()
    return correct, total


def loader_batches(loader, limit: int | None = None) -> int:
    """Number of batches shown in progress bars."""
    if limit:
        return min(len(loader), limit)
    return len(loader)


def seg_class_names(class_names: dict[int, str]) -> list[str]:
    """Return segmentation class names, including background at index 0."""
    return ["background"] + [class_names[i] for i in sorted(class_names)]


def format_per_class(metrics: dict[str, float], class_names: list[str]) -> str:
    """Compact per-class IoU/Dice/accuracy table for console logs."""
    parts = []
    for idx, name in enumerate(class_names):
        support = metrics["per_class_support"][idx]
        if support <= 0:
            continue
        parts.append(
            f"{name}:iou={metrics['per_class_iou'][idx]:.3f},"
            f"dice={metrics['per_class_dice'][idx]:.3f},"
            f"acc={metrics['per_class_acc'][idx]:.3f}"
        )
    return " | ".join(parts)


def class_weights_from(dataset, num_classes) -> torch.Tensor:
    """Inverse-frequency pixel weights (median-frequency balancing) for CE loss."""
    counts = torch.zeros(num_classes)
    for _, target in dataset:
        counts += torch.bincount(target.view(-1), minlength=num_classes).float()
    freq = counts / counts.sum().clamp(min=1)
    med = freq[freq > 0].median()
    w = torch.where(freq > 0, med / freq.clamp(min=1e-6), torch.zeros_like(freq))
    return w


def build_optimizer(cfg: dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    """Build the selected optimizer while preserving AdamW as the default."""
    name = str(cfg.get("optimizer", "adamw")).lower()
    lr = cfg.get("lr", 2e-4)
    weight_decay = cfg.get("weight_decay", 1e-4)
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=cfg.get("momentum", 0.9),
            weight_decay=weight_decay,
            nesterov=bool(cfg.get("nesterov", False)),
        )
    raise ValueError("Unknown optimizer {!r}. Options: adamw, sgd".format(name))


def build_scheduler(cfg: dict, optimizer: torch.optim.Optimizer):
    """Build an epoch-stepped LR scheduler with optional linear warmup."""
    name = str(cfg.get("scheduler", "none")).lower()
    epochs = int(cfg.get("epochs", 50))
    warmup_epochs = int(cfg.get("warmup_epochs", 0) or 0)
    min_lr_factor = float(cfg.get("min_lr_factor", 0.0))
    poly_power = float(cfg.get("poly_power", 0.9))

    if name not in {"none", "cosine", "poly"}:
        raise ValueError("Unknown scheduler {!r}. Options: none, cosine, poly".format(name))
    if name == "none" and warmup_epochs <= 0:
        return None

    def lr_lambda(epoch_idx: int) -> float:
        if warmup_epochs > 0 and epoch_idx < warmup_epochs:
            return max((epoch_idx + 1) / warmup_epochs, 1e-8)
        if name == "none":
            return 1.0
        progress_epochs = max(epochs - warmup_epochs, 1)
        progress = min(max((epoch_idx - warmup_epochs + 1) / progress_epochs, 0.0), 1.0)
        if name == "cosine":
            return min_lr_factor + (1.0 - min_lr_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))
        if name == "poly":
            return min_lr_factor + (1.0 - min_lr_factor) * ((1.0 - progress) ** poly_power)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def wandb_setup(cfg: dict, run_name: str, out_dir: Path):
    """Init a Weights & Biases run if ``cfg['wandb']`` is true; else return ``None``.

    Never aborts training: a missing package or failed login just disables logging
    (with a clear message). Returns the ``wandb`` module when logging is active.
    """
    if not cfg.get("wandb", False):
        return None
    try:
        import wandb
    except ImportError:
        print(">> wandb requested but not installed — run `uv sync --extra wandb`. "
              "Continuing WITHOUT W&B.")
        return None
    try:
        wandb.init(
            project=cfg.get("wandb_project", "oralskop-seg"),
            entity=cfg.get("wandb_entity"),
            name=run_name,
            config=cfg,
            dir=str(out_dir),
        )
    except Exception as exc:  # not logged in / no API key / network
        print(f">> wandb.init failed ({exc}); continuing WITHOUT W&B. Set WANDB_API_KEY "
              "or run `wandb login <token>` in the notebook first.")
        return None
    print(f">> W&B logging enabled: project='{cfg.get('wandb_project', 'oralskop-seg')}' "
          f"run='{run_name}' url={wandb.run.url}")
    return wandb


@torch.no_grad()
def _denorm_rgb(image: torch.Tensor) -> np.ndarray:
    """Normalized [3,H,W] tensor -> HxWx3 uint8 RGB."""
    rgb = image.cpu().numpy().transpose(1, 2, 0) * _NORM_STD + _NORM_MEAN
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


def _palette_rgb(class_id: int) -> tuple[int, int, int]:
    b, g, r = color_for(class_id)
    return (r, g, b)


def _seg_overlay_rgb(rgb: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    """Blend the palette-colored seg mask onto an RGB image (background, 0, untouched)."""
    out = rgb.copy()
    for value in np.unique(mask):
        if value == 0:
            continue
        color = np.array(_palette_rgb(int(value) - 1), dtype=np.float32)
        sel = mask == value
        out[sel] = ((1 - alpha) * rgb[sel].astype(np.float32) + alpha * color).astype(np.uint8)
    return out


def _fg_miou(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> float:
    """Mean IoU over foreground classes present in the ground truth (NaN if none)."""
    ious = []
    for c in range(1, num_classes):
        g = gt == c
        if not g.any():
            continue
        p = pred == c
        union = (p | g).sum()
        ious.append((p & g).sum() / union if union else 0.0)
    return float(np.mean(ious)) if ious else float("nan")


def _draw_legend_rgb(image: np.ndarray, present: list[int], names: dict[int, str]) -> np.ndarray:
    """Burn a compact class legend into an RGB panel."""
    if not present:
        return image
    out = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(out)
    pad, sw, row_h = 8, 16, 22
    width = min(out.width, 280)
    height = pad * 2 + row_h * len(present)
    draw.rectangle((0, 0, width, height), fill=(32, 32, 32))
    y = pad
    for cid in present:
        draw.rectangle((pad, y + 2, pad + sw, y + sw + 2), fill=_palette_rgb(cid), outline=(255, 255, 255))
        draw.text((pad + sw + 8, y + 1), names.get(cid, str(cid)), fill=(255, 255, 255))
        y += row_h
    return np.asarray(out)


def _title_bar_rgb(panel: np.ndarray, text: str, bar_h: int = 26) -> np.ndarray:
    """Stack a black title bar with white text on top of an RGB panel."""
    bar = Image.new("RGB", (panel.shape[1], bar_h), (0, 0, 0))
    ImageDraw.Draw(bar).text((8, 6), text, fill=(255, 255, 255))
    return np.vstack([np.asarray(bar), panel])


def _sample_stem(dataset, idx: int) -> str:
    """Best-effort source filename stem for a (possibly merged/concat) dataset index."""
    if hasattr(dataset, "samples"):
        return Path(dataset.samples[idx][0]).stem
    if hasattr(dataset, "datasets") and hasattr(dataset, "cumulative_sizes"):
        d = bisect.bisect_right(dataset.cumulative_sizes, idx)
        sub = dataset.datasets[d]
        local = idx - (dataset.cumulative_sizes[d - 1] if d > 0 else 0)
        if hasattr(sub, "samples"):
            return Path(sub.samples[local][0]).stem
    return f"#{idx}"


def wandb_prediction_images(wb, model, dataset, device, n: int,
                            seg_labels: dict[int, str], num_classes: int,
                            alpha: float = 0.5):
    """Build N annotated W&B Images: ``prediction | ground-truth`` side by side.

    Everything is burned into the pixels so it's readable with no hover/toggling:
    masks use the visualizer's palette, a class-color **legend** of the classes present
    is drawn on the prediction panel, and each panel carries a title (the prediction
    title also shows the per-image foreground mIoU). The image caption is the source
    filename. Uses the first N (deterministic) val samples so the same images are
    tracked across epochs. ``seg_labels`` maps pixel value -> name (0=bg, class c -> c+1).
    """
    model.eval()
    # Legend wants taxonomy-id -> name; seg_labels is pixel-value -> name (id = value-1).
    legend_names = {v - 1: name for v, name in seg_labels.items() if v != 0}
    images = []
    for i in range(min(n, len(dataset))):
        image, target = dataset[i]
        pred = model(image.unsqueeze(0).to(device))["out"].argmax(1)[0].cpu().numpy()
        gt = target.cpu().numpy()
        rgb = _denorm_rgb(image)

        pred_panel = _seg_overlay_rgb(rgb, pred, alpha)
        gt_panel = _seg_overlay_rgb(rgb, gt, alpha)

        # Burn a legend of the classes appearing in either mask onto the prediction panel.
        present = sorted({int(v) - 1 for v in np.unique(pred) if v != 0}
                         | {int(v) - 1 for v in np.unique(gt) if v != 0})
        pred_panel = _draw_legend_rgb(pred_panel, present, legend_names)

        miou = _fg_miou(pred, gt, num_classes)
        miou_txt = f"fg mIoU {miou:.2f}" if miou == miou else "no GT fg"
        pred_panel = _title_bar_rgb(pred_panel, f"prediction  -  {miou_txt}")
        gt_panel = _title_bar_rgb(gt_panel, "ground truth")

        panel = np.hstack([pred_panel, gt_panel])
        images.append(wb.Image(panel, caption=_sample_stem(dataset, i)))
    return images


def resolve_run_dir(base: Path, name: str, exist_ok: bool) -> Path:
    """Output dir for this run, auto-incrementing the name if it already exists.

    Mirrors Ultralytics: ``exist_ok=True`` reuses/overwrites ``base/name``; otherwise
    if ``base/name`` exists, bump to ``base/name2``, ``base/name3``, ... (first free),
    so a new run never clobbers a previous run's checkpoints.
    """
    candidate = base / name
    if exist_ok or not candidate.exists():
        return candidate
    i = 2
    while (base / f"{name}{i}").exists():
        i += 1
    return base / f"{name}{i}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Custom PyTorch seg training.")
    p.add_argument("--config", required=True)
    p.add_argument("--datasets", nargs="+", help="Override the config's dataset list.")
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = apply_overrides(load_yaml(args.config), args.override)
    if args.datasets:
        cfg["datasets"] = args.datasets

    device = torch.device(cfg.get("device", "cuda") if cfg.get("device") != "cpu"
                          and torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.get("seed", 42))

    train_ds = build_seg_dataset(cfg["datasets"], split=cfg.get("split_train", "train"),
                                 imgsz=cfg.get("imgsz", 512), augment=True,
                                 aug=cfg.get("aug", "flip"))
    val_ds = build_seg_dataset(cfg["datasets"], split=cfg.get("split_val", "val"),
                               imgsz=cfg.get("imgsz", 512), augment=False, aug="none")
    num_classes = train_ds.num_seg_classes
    print(f"Datasets {cfg['datasets']}: train={len(train_ds)} val={len(val_ds)} "
          f"| {num_classes} seg classes (0=bg) | device={device.type}")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.get("batch", 8), shuffle=True,
        num_workers=cfg.get("workers", 4), pin_memory=(device.type == "cuda"), drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg.get("batch", 8), shuffle=False, num_workers=cfg.get("workers", 4))

    model = build_model(num_classes, arch=cfg.get("arch", "deeplabv3_resnet50"),
                        pretrained=cfg.get("pretrained", True)).to(device)
    arch = cfg.get("arch", "deeplabv3_resnet50")
    if bool(cfg.get("grad_checkpointing", False)) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if bool(cfg.get("lora", False)):
        model = apply_lora(
            model,
            r=cfg.get("lora_r", 8),
            alpha=cfg.get("lora_alpha", 16),
            targets=cfg.get("lora_targets"),
        ).to(device)
    trainable_params, total_params = parameter_counts(model)
    print(f"Parameters: trainable={trainable_params:,} total={total_params:,} "
          f"({trainable_params / max(total_params, 1):.1%} trainable)")

    weights = None
    if cfg.get("class_weights") == "auto":
        print("Computing class weights (median-frequency balancing)...")
        weight_ds = build_seg_dataset(
            cfg["datasets"],
            split=cfg.get("split_train", "train"),
            imgsz=cfg.get("imgsz", 512),
            augment=False,
            aug="none",
        )
        weights = class_weights_from(weight_ds, num_classes).to(device)
    criterion = build_criterion(cfg.get("loss", "ce"), class_weights=weights)

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    aux_w = cfg.get("aux_loss_weight", 0.4) if has_aux(arch) else 0.0

    run_name = cfg.get("name", "seg")
    exist_ok = bool(cfg.get("exist_ok", False))
    out_dir = resolve_run_dir((AI_ROOT / cfg.get("out", "runs/seg")).resolve(), run_name, exist_ok)
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_dir.name != run_name:
        print(f">> Run dir '{run_name}' already exists; writing to '{out_dir.name}' "
              f"instead (pass exist_ok=true to reuse/overwrite).")
    metrics_path = out_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    class_names = seg_class_names(train_ds.class_names)
    best_miou = -1.0
    limit = cfg.get("limit_batches")  # cap batches/epoch for quick smoke tests
    progress = bool(cfg.get("progress", True))
    save_model = bool(cfg.get("save_model", True))

    # Weights & Biases (optional, config-gated). Disabled cleanly if unavailable.
    wb = wandb_setup(cfg, out_dir.name, out_dir)
    seg_labels = {0: "background", **{c + 1: n for c, n in train_ds.class_names.items()}}
    wandb_n_images = int(cfg.get("wandb_images", 8))

    for epoch in range(1, cfg.get("epochs", 50) + 1):
        model.train()
        t0, running, seen = time.time(), 0.0, 0
        train_correct, train_total = 0, 0
        train_batches = tqdm(
            train_loader,
            total=loader_batches(train_loader, limit),
            desc=f"epoch {epoch}/{cfg.get('epochs', 50)} train",
            leave=False,
            dynamic_ncols=True,
            disable=not progress,
        )
        for i, (images, targets) in enumerate(train_batches):
            if limit and i >= limit:
                break
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                out = model(images)
                loss = criterion(out["out"], targets)
                if aux_w and "aux" in out:
                    loss = loss + aux_w * criterion(out["aux"], targets)
            correct, total = train_pixel_accuracy(out["out"].detach(), targets, num_classes)
            train_correct += correct
            train_total += total
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * images.size(0)
            seen += images.size(0)
            running_loss = running / max(seen, 1)
            running_acc = train_correct / max(train_total, 1)
            train_batches.set_postfix(
                loss=f"{running_loss:.4f}",
                acc=f"{running_acc:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
        train_loss = running / max(seen, 1)
        train_acc = train_correct / max(train_total, 1)

        line = f"epoch {epoch:3d}/{cfg.get('epochs', 50)}  loss {train_loss:.4f}  train_acc {train_acc:.4f}"
        saved_checkpoints = []
        if epoch % cfg.get("val_interval", 1) == 0:
            m = evaluate(
                model,
                val_loader,
                num_classes,
                device,
                limit=limit,
                desc=f"epoch {epoch}/{cfg.get('epochs', 50)} val",
                progress=progress,
                criterion=criterion,
                aux_weight=aux_w,
            )
            line += (
                f"  val_loss {m['val_loss']:.4f}"
                f"  val_mIoU {m['miou']:.4f}  val_fg_mIoU {m['fg_miou']:.4f}"
                f"  val_dice {m['mean_dice']:.4f}  val_fg_dice {m['fg_dice']:.4f}"
                f"  val_acc {m['pixel_acc']:.4f}  val_fg_acc {m['fg_pixel_acc']:.4f}"
                f"  val_mean_acc {m['mean_acc']:.4f}"
            )
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_pixel_acc": train_acc,
                **m,
                "class_names": class_names,
                "lr": optimizer.param_groups[0]["lr"],
                "trainable_params": trainable_params,
                "total_params": total_params,
                "seconds": time.time() - t0,
            }
            with metrics_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
            if save_model:
                torch.save({"model": model.state_dict(), "arch": arch,
                            "num_classes": num_classes, "class_names": train_ds.class_names,
                            "epoch": epoch, "metrics": m}, out_dir / "last.pt")
                saved_checkpoints.append("last.pt")
            if m["fg_miou"] > best_miou:
                best_miou = m["fg_miou"]
                if save_model:
                    torch.save({"model": model.state_dict(), "arch": arch,
                                "num_classes": num_classes, "class_names": train_ds.class_names,
                                "epoch": epoch, "fg_miou": best_miou, "metrics": m}, out_dir / "best.pt")
                    saved_checkpoints.append("best.pt")
                line += "  *best*"
        if saved_checkpoints:
            line += f"  checkpoint_epoch={epoch} saved {','.join(saved_checkpoints)}"
        else:
            line += f"  checkpoint_epoch={epoch} saved none"
        if scheduler is not None:
            scheduler.step()
        tqdm.write(f"{line}  ({time.time() - t0:.0f}s)")
        if cfg.get("log_per_class", True) and epoch % cfg.get("val_interval", 1) == 0:
            tqdm.write("  per_class " + format_per_class(m, class_names))

        if wb is not None:
            log = {"train/loss": train_loss, "train/pixel_acc": train_acc,
                   "lr": optimizer.param_groups[0]["lr"], "epoch_seconds": time.time() - t0,
                   "params/trainable": trainable_params, "params/total": total_params}
            if epoch % cfg.get("val_interval", 1) == 0:
                log.update({
                    "val/loss": m["val_loss"],
                    "val/mIoU": m["miou"], "val/fg_mIoU": m["fg_miou"],
                    "val/dice": m["mean_dice"], "val/fg_dice": m["fg_dice"],
                    "val/pixel_acc": m["pixel_acc"], "val/fg_pixel_acc": m["fg_pixel_acc"],
                    "val/mean_acc": m["mean_acc"],
                })
                for idx, name in enumerate(class_names):
                    if m["per_class_support"][idx] > 0:
                        log[f"val_iou/{name}"] = m["per_class_iou"][idx]
                        log[f"val_dice/{name}"] = m["per_class_dice"][idx]
                        log[f"val_acc/{name}"] = m["per_class_acc"][idx]
                if wandb_n_images > 0:
                    log["val/predictions"] = wandb_prediction_images(
                        wb, model, val_ds, device, wandb_n_images, seg_labels, num_classes)
                wb.summary["best_val_fg_mIoU"] = best_miou
            wb.log(log, step=epoch)

    if wb is not None:
        wb.summary["best_val_fg_mIoU"] = best_miou
        wb.finish()

    if save_model:
        print(f"\nDone. Best val fg_mIoU {best_miou:.4f}. Weights in {out_dir}")
    else:
        print(f"\nDone. Best val fg_mIoU {best_miou:.4f}. Model saving disabled.")
    print(f"Metrics log: {metrics_path}")


if __name__ == "__main__":
    main()
