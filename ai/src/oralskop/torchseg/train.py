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
import time
from pathlib import Path

import torch

from oralskop.config import apply_overrides, load_yaml
from oralskop.torchseg.dataset import AI_ROOT, build_seg_dataset
from oralskop.torchseg.model import build_model, has_aux


@torch.no_grad()
def evaluate(model, loader, num_classes, device, limit: int | None = None) -> dict[str, float]:
    """Return overall pixel accuracy and mean IoU via a confusion matrix."""
    model.eval()
    conf = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)
    for i, (images, targets) in enumerate(loader):
        if limit and i >= limit:
            break
        images, targets = images.to(device), targets.to(device)
        preds = model(images)["out"].argmax(1)
        k = (targets >= 0) & (targets < num_classes)
        idx = num_classes * targets[k].view(-1) + preds[k].view(-1)
        conf += torch.bincount(idx, minlength=num_classes**2).reshape(num_classes, num_classes)
    inter = conf.diag().float()
    union = conf.sum(0).float() + conf.sum(1).float() - inter
    iou = inter / union.clamp(min=1)
    present = conf.sum(1) > 0  # classes that actually appear in val
    return {
        "pixel_acc": (inter.sum() / conf.sum().clamp(min=1)).item(),
        "miou": iou[present].mean().item() if present.any() else 0.0,
        "per_class_iou": iou.tolist(),
    }


def class_weights_from(dataset, num_classes) -> torch.Tensor:
    """Inverse-frequency pixel weights (median-frequency balancing) for CE loss."""
    counts = torch.zeros(num_classes)
    for _, target in dataset:
        counts += torch.bincount(target.view(-1), minlength=num_classes).float()
    freq = counts / counts.sum().clamp(min=1)
    med = freq[freq > 0].median()
    w = torch.where(freq > 0, med / freq.clamp(min=1e-6), torch.zeros_like(freq))
    return w


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

    out_dir = (AI_ROOT / cfg.get("out", "runs/seg") / cfg.get("name", "seg")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    best_miou = -1.0
    limit = cfg.get("limit_batches")  # cap batches/epoch for quick smoke tests

    for epoch in range(1, cfg.get("epochs", 50) + 1):
        model.train()
        t0, running, seen = time.time(), 0.0, 0
        for i, (images, targets) in enumerate(train_loader):
            if limit and i >= limit:
                break
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                out = model(images)
                loss = criterion(out["out"], targets)
                if aux_w and "aux" in out:
                    loss = loss + aux_w * criterion(out["aux"], targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * images.size(0)
            seen += images.size(0)
        train_loss = running / max(seen, 1)

        line = f"epoch {epoch:3d}/{cfg.get('epochs', 50)}  loss {train_loss:.4f}"
        if epoch % cfg.get("val_interval", 1) == 0:
            m = evaluate(model, val_loader, num_classes, device, limit=limit)
            line += f"  val_mIoU {m['miou']:.4f}  pixel_acc {m['pixel_acc']:.4f}"
            torch.save({"model": model.state_dict(), "arch": arch,
                        "num_classes": num_classes, "class_names": train_ds.class_names,
                        "epoch": epoch}, out_dir / "last.pt")
            if m["miou"] > best_miou:
                best_miou = m["miou"]
                torch.save({"model": model.state_dict(), "arch": arch,
                            "num_classes": num_classes, "class_names": train_ds.class_names,
                            "epoch": epoch, "miou": best_miou}, out_dir / "best.pt")
                line += "  *best*"
        print(f"{line}  ({time.time() - t0:.0f}s)")

    print(f"\nDone. Best val mIoU {best_miou:.4f}. Weights in {out_dir}")


if __name__ == "__main__":
    main()
