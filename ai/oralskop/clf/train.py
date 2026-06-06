"""Multi-label image-classification training on the curated manifest.

Config-driven; ``--override key=value`` patches the YAML (same convention as the
other entrypoints).

    python -m oralskop.clf.train --config configs/clf/manifest_clf.yaml
    python -m oralskop.clf.train --config configs/clf/manifest_clf.yaml \
        --override level=coarse epochs=1 batch=4 limit=64 device=cpu amp=false
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from oralskop.config import apply_overrides, load_yaml
from oralskop.clf.dataset import (
    AI_ROOT,
    ManifestClfDataset,
    load_supervised_frame,
    pos_weight_from,
)
from oralskop.clf.metrics import format_per_class, multilabel_metrics
from oralskop.clf.model import build_classifier
from oralskop.clf.vocab import build_vocab

# Manifest split names (doc §2) — note "valid", not "val".
_SPLIT_TRAIN = "train"
_SPLIT_VAL = "valid"


def build_optimizer(cfg: dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    """AdamW (default) or SGD — mirrors oralskop.torchseg.train.build_optimizer."""
    name = str(cfg.get("optimizer", "adamw")).lower()
    lr = cfg.get("lr", 3e-4)
    weight_decay = cfg.get("weight_decay", 0.05)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr, momentum=cfg.get("momentum", 0.9),
            weight_decay=weight_decay, nesterov=bool(cfg.get("nesterov", False)),
        )
    raise ValueError(f"Unknown optimizer {name!r}. Options: adamw, sgd")


def build_scheduler(cfg: dict, optimizer: torch.optim.Optimizer):
    """Epoch-stepped LR schedule with optional linear warmup (cosine/poly/none)."""
    name = str(cfg.get("scheduler", "none")).lower()
    epochs = int(cfg.get("epochs", 30))
    warmup_epochs = int(cfg.get("warmup_epochs", 0) or 0)
    min_lr_factor = float(cfg.get("min_lr_factor", 0.0))
    poly_power = float(cfg.get("poly_power", 0.9))
    if name not in {"none", "cosine", "poly"}:
        raise ValueError(f"Unknown scheduler {name!r}. Options: none, cosine, poly")
    if name == "none" and warmup_epochs <= 0:
        return None

    def lr_lambda(epoch_idx: int) -> float:
        if warmup_epochs > 0 and epoch_idx < warmup_epochs:
            return max((epoch_idx + 1) / warmup_epochs, 1e-8)
        if name == "none":
            return 1.0
        progress = min(max((epoch_idx - warmup_epochs + 1) / max(epochs - warmup_epochs, 1), 0.0), 1.0)
        if name == "cosine":
            return min_lr_factor + (1.0 - min_lr_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_factor + (1.0 - min_lr_factor) * ((1.0 - progress) ** poly_power)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def wandb_setup(cfg: dict, run_name: str, out_dir: Path):
    """Init W&B if cfg['wandb']; never aborts training on failure."""
    if not cfg.get("wandb", False):
        return None
    try:
        import wandb
    except ImportError:
        print(">> wandb requested but not installed — `uv sync --extra wandb`. Continuing WITHOUT W&B.")
        return None
    try:
        wandb.init(project=cfg.get("wandb_project", "oralskop-clf"), entity=cfg.get("wandb_entity"),
                   name=run_name, config=cfg, dir=str(out_dir))
    except Exception as exc:
        print(f">> wandb.init failed ({exc}); continuing WITHOUT W&B.")
        return None
    print(f">> W&B logging enabled: run='{run_name}' url={wandb.run.url}")
    return wandb


def resolve_run_dir(base: Path, name: str, exist_ok: bool) -> Path:
    """`base/name`, or `base/name{N}` if it exists and exist_ok is False."""
    candidate = base / name
    if exist_ok or not candidate.exists():
        return candidate
    i = 2
    while (base / f"{name}{i}").exists():
        i += 1
    return base / f"{name}{i}"


@torch.no_grad()
def collect_scores(model, loader, device, *, limit: int | None = None, desc: str = "val"):
    """Run the model over a loader -> (y_true, y_score) numpy arrays [N, C]."""
    model.eval()
    trues, scores = [], []
    seen = 0
    for imgs, targets in tqdm(loader, desc=desc, leave=False):
        logits = model(imgs.to(device))
        scores.append(torch.sigmoid(logits).float().cpu().numpy())
        trues.append(targets.numpy())
        seen += imgs.shape[0]
        if limit and seen >= limit:
            break
    if not trues:
        return np.zeros((0, 0)), np.zeros((0, 0))
    return np.concatenate(trues), np.concatenate(scores)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-label manifest classifier training.")
    p.add_argument("--config", required=True)
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = apply_overrides(load_yaml(args.config), args.override)

    device = torch.device(cfg.get("device", "cuda") if cfg.get("device") != "cpu"
                          and torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))

    level = cfg.get("level", "coarse")
    df, labels_per_row = load_supervised_frame(
        cfg["manifest"], level,
        image_path_prefixes=cfg.get("image_path_prefixes"),
        limit=cfg.get("limit"),
    )
    vocab = build_vocab(
        level, labels_file=cfg.get("labels_file"), labels_per_row=labels_per_row,
        exclude_micro=bool(cfg.get("exclude_train_only_microclasses", True)),
    )
    num_classes = len(vocab)
    print(f"Level={level}: {num_classes} classes | supervised rows={len(df)} "
          f"| vocab from {'file' if cfg.get('labels_file') else 'manifest'}")

    image_root = cfg.get("image_root", "s3://datastoraged4gen/02_PROCESSED")
    imgsz = int(cfg.get("imgsz", 224))
    cache_dir = cfg.get("cache_dir")
    train_ds = ManifestClfDataset(df[df["split"].str.strip() == _SPLIT_TRAIN], vocab,
                                  image_root=image_root, imgsz=imgsz, train=True, cache_dir=cache_dir)
    val_ds = ManifestClfDataset(df[df["split"].str.strip() == _SPLIT_VAL], vocab,
                                image_root=image_root, imgsz=imgsz, train=False, cache_dir=cache_dir)
    print(f"train={len(train_ds)} (dropped {train_ds.dropped_empty} off-vocab) "
          f"valid={len(val_ds)} (dropped {val_ds.dropped_empty}) | device={device.type}")
    if len(train_ds) == 0:
        raise SystemExit("No trainable rows — check manifest/level/image_path_prefixes.")

    pin = device.type == "cuda"
    workers = int(cfg.get("num_workers", 8))
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.get("batch", 64), shuffle=True, num_workers=workers,
        pin_memory=pin, drop_last=len(train_ds) > cfg.get("batch", 64))
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg.get("batch", 64), shuffle=False, num_workers=workers, pin_memory=pin)

    model = build_classifier(num_classes, arch=cfg.get("arch", "convnext_tiny"),
                             pretrained=bool(cfg.get("pretrained", True))).to(device)

    pw_cfg = cfg.get("pos_weight", "auto")
    if pw_cfg == "auto":
        pos_weight = pos_weight_from(train_ds.targets).to(device)
    elif isinstance(pw_cfg, (list, tuple)):
        pos_weight = torch.tensor(pw_cfg, dtype=torch.float32, device=device)
    else:
        pos_weight = None
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    run_name = cfg.get("name", f"clf_{level}")
    out_dir = resolve_run_dir((AI_ROOT / cfg.get("out_dir", "runs/clf")).resolve(), run_name,
                              bool(cfg.get("exist_ok", False)))
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab.to_json(out_dir / "vocab.json")
    metrics_path = out_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    wb = wandb_setup(cfg, run_name, out_dir)

    epochs = int(cfg.get("epochs", 30))
    threshold = float(cfg.get("threshold", 0.5))
    best_map = -1.0
    print(f"Training {cfg.get('arch', 'convnext_tiny')} for {epochs} epochs -> {out_dir}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum, n = 0.0, 0
        for imgs, targets in tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False):
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(imgs), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss) * imgs.shape[0]
            n += imgs.shape[0]
        train_loss = loss_sum / max(n, 1)

        line = f"epoch {epoch:3d}  train_loss {train_loss:.4f}"
        m = None
        if len(val_ds) > 0:
            y_true, y_score = collect_scores(model, val_loader, device, desc=f"val {epoch}")
            m = multilabel_metrics(y_true, y_score, vocab.names, threshold=threshold)
            line += (f"  macro_mAP {m['macro_map']:.4f}  micro_AP {m['micro_ap']:.4f}"
                     f"  macro_F1 {m['macro_f1']:.4f}  micro_F1 {m['micro_f1']:.4f}")

        record = {"epoch": epoch, "train_loss": train_loss, "lr": optimizer.param_groups[0]["lr"]}
        if m is not None:
            record.update({k: m[k] for k in
                           ("macro_map", "micro_ap", "macro_f1", "micro_f1")})
        with metrics_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

        ckpt = {"model": model.state_dict(), "arch": cfg.get("arch", "convnext_tiny"),
                "num_classes": num_classes, "class_names": vocab.names, "level": level,
                "imgsz": imgsz, "epoch": epoch, "metrics": m}
        torch.save(ckpt, out_dir / "last.pt")
        if m is not None and m["macro_map"] == m["macro_map"] and m["macro_map"] > best_map:
            best_map = m["macro_map"]
            torch.save(ckpt, out_dir / "best.pt")
            line += "  *best*"

        if scheduler is not None:
            scheduler.step()
        tqdm.write(f"{line}  ({time.time() - t0:.0f}s)")
        if m is not None and cfg.get("log_per_class", True):
            tqdm.write(format_per_class(m))

        if wb is not None:
            log = {"train/loss": train_loss, "lr": optimizer.param_groups[0]["lr"]}
            if m is not None:
                log.update({"val/macro_mAP": m["macro_map"], "val/micro_AP": m["micro_ap"],
                            "val/macro_F1": m["macro_f1"], "val/micro_F1": m["micro_f1"]})
            wb.log(log, step=epoch)

    if wb is not None:
        wb.summary["best_val_macro_mAP"] = best_map
        wb.finish()
    print(f"\nDone. Best val macro-mAP {best_map:.4f}. Checkpoints + vocab.json in {out_dir}")
    print(f"Metrics log: {metrics_path}")


if __name__ == "__main__":
    main()
