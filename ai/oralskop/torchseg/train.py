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
import json
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

from oralskop.config import apply_overrides, load_yaml
from oralskop.torchseg.dataset import AI_ROOT, build_seg_dataset
from oralskop.torchseg.model import build_model, has_aux


@torch.no_grad()
def evaluate(
    model,
    loader,
    num_classes,
    device,
    limit: int | None = None,
    desc: str = "val",
    progress: bool = True,
) -> dict[str, float]:
    """Return semantic-segmentation metrics from a confusion matrix."""
    model.eval()
    conf = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)
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
        preds = model(images)["out"].argmax(1)
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
                                 imgsz=cfg.get("imgsz", 512), augment=True)
    val_ds = build_seg_dataset(cfg["datasets"], split=cfg.get("split_val", "val"),
                               imgsz=cfg.get("imgsz", 512), augment=False)
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

    weights = None
    if cfg.get("class_weights") == "auto":
        print("Computing class weights (median-frequency balancing)...")
        weights = class_weights_from(train_ds, num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 2e-4),
                                  weight_decay=cfg.get("weight_decay", 1e-4))
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
        if epoch % cfg.get("val_interval", 1) == 0:
            m = evaluate(
                model,
                val_loader,
                num_classes,
                device,
                limit=limit,
                desc=f"epoch {epoch}/{cfg.get('epochs', 50)} val",
                progress=progress,
            )
            line += (
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
                "seconds": time.time() - t0,
            }
            with metrics_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
            if save_model:
                torch.save({"model": model.state_dict(), "arch": arch,
                            "num_classes": num_classes, "class_names": train_ds.class_names,
                            "epoch": epoch, "metrics": m}, out_dir / "last.pt")
            if m["miou"] > best_miou:
                best_miou = m["miou"]
                if save_model:
                    torch.save({"model": model.state_dict(), "arch": arch,
                                "num_classes": num_classes, "class_names": train_ds.class_names,
                                "epoch": epoch, "miou": best_miou, "metrics": m}, out_dir / "best.pt")
                line += "  *best*"
        tqdm.write(f"{line}  ({time.time() - t0:.0f}s)")
        if cfg.get("log_per_class", True) and epoch % cfg.get("val_interval", 1) == 0:
            tqdm.write("  per_class " + format_per_class(m, class_names))

    if save_model:
        print(f"\nDone. Best val mIoU {best_miou:.4f}. Weights in {out_dir}")
    else:
        print(f"\nDone. Best val mIoU {best_miou:.4f}. Model saving disabled.")
    print(f"Metrics log: {metrics_path}")


if __name__ == "__main__":
    main()
