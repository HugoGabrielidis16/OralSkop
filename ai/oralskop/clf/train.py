"""Multi-label image-classification training on the curated manifest.

Two model families share this loop (see ``oralskop.clf.model``):
* torchvision backbones (full fine-tune), and
* HuggingFace foundation models fine-tuned with **QLoRA** (4-bit base + LoRA adapters,
  8-bit optimizer, gradient checkpointing, gradient accumulation) for a small footprint.

Config-driven; ``--override key=value`` patches the YAML.

    python -m oralskop.clf.train --config configs/clf/manifest_clf.yaml
    uv run --extra clf --extra qlora python -m oralskop.clf.train \
        --config configs/clf/qlora_dinov2.yaml --override limit=128 epochs=1 batch=4
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
from oralskop.clf.model import build_model
from oralskop.clf.vocab import build_vocab
from oralskop.torchseg.lora import parameter_counts  # reused trainable/total counter

# Manifest split names (doc §2) — note "valid", not "val".
_SPLIT_TRAIN = "train"
_SPLIT_VAL = "valid"


def build_optimizer(cfg: dict, params) -> torch.optim.Optimizer:
    """AdamW / SGD / 8-bit (paged) AdamW over the given (trainable) params."""
    name = str(cfg.get("optimizer", "adamw")).lower().replace("-", "_")
    lr = cfg.get("lr", 3e-4)
    weight_decay = cfg.get("weight_decay", 0.05)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=cfg.get("momentum", 0.9),
                               weight_decay=weight_decay, nesterov=bool(cfg.get("nesterov", False)))
    if name in {"adamw8bit", "paged_adamw8bit", "adamw_8bit", "paged_adamw_8bit"}:
        try:
            import bitsandbytes as bnb
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError("8-bit optimizers need the `qlora` extra (bitsandbytes): "
                              "uv sync --extra qlora.") from exc
        Opt = bnb.optim.PagedAdamW8bit if "paged" in name else bnb.optim.AdamW8bit
        return Opt(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer {name!r}. Options: adamw, sgd, adamw8bit, paged_adamw8bit")


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


def forward_logits(model, imgs, is_hf: bool):
    """Uniform forward: HF image models return ``.logits``; torchvision returns logits."""
    if is_hf:
        return model(pixel_values=imgs).logits
    return model(imgs)


@torch.no_grad()
def collect_scores(model, loader, device, *, is_hf: bool = False, amp_dtype=torch.float16,
                   use_amp: bool = False, desc: str = "val", progress: bool = True):
    """Run the model over a loader -> (y_true, y_score) numpy arrays [N, C]."""
    model.eval()
    trues, scores = [], []
    for imgs, targets in tqdm(loader, desc=desc, leave=False, disable=not progress):
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = forward_logits(model, imgs.to(device), is_hf)
        scores.append(torch.sigmoid(logits.float()).cpu().numpy())
        trues.append(targets.numpy())
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

    # Precision: bf16 where supported (A10/A100), else fp16. GradScaler only for fp16.
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    compute_dtype = amp_dtype if device.type == "cuda" else torch.float32
    use_scaler = use_amp and amp_dtype == torch.float16

    level = cfg.get("level", "coarse")
    df, labels_per_row = load_supervised_frame(
        cfg["manifest"], level,
        image_path_prefixes=cfg.get("image_path_prefixes"), limit=cfg.get("limit"))
    vocab = build_vocab(level, labels_file=cfg.get("labels_file"), labels_per_row=labels_per_row,
                        exclude_micro=bool(cfg.get("exclude_train_only_microclasses", True)))
    num_classes = len(vocab)
    arch = str(cfg.get("arch", "convnext_tiny"))
    print(f"Level={level}: {num_classes} classes | supervised rows={len(df)} | arch={arch}")

    # Build the model first so preprocessing (normalization + input size) comes from it.
    model, preprocess, is_hf = build_model(num_classes, cfg, compute_dtype=compute_dtype)
    quantized = is_hf and str(cfg.get("quantize", "4bit")).lower() in {"4bit", "8bit"}
    if not quantized:  # quantized weights are placed on GPU by bitsandbytes — don't .to()
        model = model.to(device)
    trainable, total = parameter_counts(model)
    print(f"Parameters: trainable={trainable:,} total={total:,} "
          f"({trainable / max(total, 1):.2%} trainable) | precision={amp_dtype if use_amp else 'fp32'}")

    imgsz, mean, std = int(preprocess["imgsz"]), preprocess["mean"], preprocess["std"]
    cache_dir = cfg.get("cache_dir")
    image_root = cfg.get("image_root", "s3://datastoraged4gen/02_PROCESSED")
    unreadable_log_limit = int(cfg.get("unreadable_log_limit", 10))
    ds_kw = dict(image_root=image_root, imgsz=imgsz, cache_dir=cache_dir,
                 mean=mean, std=std, unreadable_log_limit=unreadable_log_limit)
    train_ds = ManifestClfDataset(df[df["split"].str.strip() == _SPLIT_TRAIN], vocab, train=True, **ds_kw)
    val_ds = ManifestClfDataset(df[df["split"].str.strip() == _SPLIT_VAL], vocab, train=False, **ds_kw)
    print(f"train={len(train_ds)} (dropped {train_ds.dropped_empty} off-vocab) "
          f"valid={len(val_ds)} (dropped {val_ds.dropped_empty}) | imgsz={imgsz} device={device.type}")
    if len(train_ds) == 0:
        raise SystemExit("No trainable rows — check manifest/level/image_path_prefixes.")

    pin = device.type == "cuda"
    workers = int(cfg.get("num_workers", 8))
    batch = int(cfg.get("batch", 64))
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch, shuffle=True, num_workers=workers,
        pin_memory=pin, drop_last=len(train_ds) > batch)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=pin)

    pw_cfg = cfg.get("pos_weight", "auto")
    if pw_cfg == "auto":
        pos_weight = pos_weight_from(train_ds.targets).to(device)
    elif isinstance(pw_cfg, (list, tuple)):
        pos_weight = torch.tensor(pw_cfg, dtype=torch.float32, device=device)
    else:
        pos_weight = None
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = build_optimizer(cfg, [p for p in model.parameters() if p.requires_grad])
    scheduler = build_scheduler(cfg, optimizer)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    grad_accum = max(int(cfg.get("grad_accum_steps", 1)), 1)

    run_name = cfg.get("name", f"clf_{level}")
    out_dir = resolve_run_dir((AI_ROOT / cfg.get("out_dir", "runs/clf")).resolve(), run_name,
                              bool(cfg.get("exist_ok", False)))
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab.to_json(out_dir / "vocab.json")
    meta = {"is_hf": is_hf, "arch": arch, "model_id": preprocess.get("model_id"),
            "num_classes": num_classes, "class_names": vocab.names, "level": level,
            "imgsz": imgsz, "mean": list(mean), "std": list(std),
            "quantize": cfg.get("quantize", "4bit") if is_hf else None,
            "lora": bool(cfg.get("lora", True)) if is_hf else False,
            "lora_r": cfg.get("lora_r", 16), "lora_alpha": cfg.get("lora_alpha", 32)}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    metrics_path = out_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    wb = wandb_setup(cfg, run_name, out_dir)

    def save_ckpt(tag: str, m: dict | None, epoch: int) -> None:
        if is_hf:
            dest = out_dir / f"adapter_{tag}"
            model.save_pretrained(str(dest))  # LoRA adapters + head
        else:
            dest = out_dir / f"{tag}.pt"
            torch.save({"model": model.state_dict(), "arch": arch, "num_classes": num_classes,
                        "class_names": vocab.names, "level": level, "imgsz": imgsz,
                        "mean": list(mean), "std": list(std), "epoch": epoch, "metrics": m},
                       dest)
        if m is not None:
            summary = (f"macro_mAP {m['macro_map']:.4f}, micro_AP {m['micro_ap']:.4f}, "
                       f"macro_F1 {m['macro_f1']:.4f}, micro_F1 {m['micro_f1']:.4f}")
        else:
            summary = "no validation metrics (empty val split)"
        tqdm.write(f">> Model saved [{tag}] epoch {epoch} with metrics: {summary} -> {dest}")

    epochs = int(cfg.get("epochs", 30))
    threshold = float(cfg.get("threshold", 0.5))
    progress = bool(cfg.get("progress", True))
    progress_leave = bool(cfg.get("progress_leave", False))
    log_interval = int(cfg.get("log_interval", 0) or 0)
    best_map = -1.0
    print(f"Training {arch} for {epochs} epochs (batch {batch} x accum {grad_accum}) -> {out_dir}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum, n = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=progress_leave, disable=not progress)
        step = 0
        for step, (imgs, targets) in enumerate(pbar, 1):
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = forward_logits(model, imgs, is_hf)
                loss = criterion(logits.float(), targets) / grad_accum
            (scaler.scale(loss) if use_scaler else loss).backward()
            if step % grad_accum == 0:
                if use_scaler:
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            bs = imgs.shape[0]
            loss_sum += float(loss) * grad_accum * bs
            n += bs
            if log_interval and step % log_interval == 0:
                running = loss_sum / max(n, 1)
                pbar.set_postfix(loss=f"{running:.4f}")
                if wb is not None:
                    wb.log({"train/running_loss": running, "lr": optimizer.param_groups[0]["lr"]})
        if step % grad_accum != 0:  # flush trailing partial accumulation
            if use_scaler:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        train_loss = loss_sum / max(n, 1)

        line = f"epoch {epoch:3d}  train_loss {train_loss:.4f}"
        m = None
        if len(val_ds) > 0:
            y_true, y_score = collect_scores(model, val_loader, device, is_hf=is_hf,
                                             amp_dtype=amp_dtype, use_amp=use_amp,
                                             desc=f"val {epoch}", progress=progress)
            m = multilabel_metrics(y_true, y_score, vocab.names, threshold=threshold)
            line += (f"  macro_mAP {m['macro_map']:.4f}  micro_AP {m['micro_ap']:.4f}"
                     f"  macro_F1 {m['macro_f1']:.4f}  micro_F1 {m['micro_f1']:.4f}")

        record = {"epoch": epoch, "train_loss": train_loss, "lr": optimizer.param_groups[0]["lr"]}
        if m is not None:
            record.update({k: m[k] for k in ("macro_map", "micro_ap", "macro_f1", "micro_f1")})
        with metrics_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

        save_ckpt("last", m, epoch)
        if m is not None and m["macro_map"] == m["macro_map"] and m["macro_map"] > best_map:
            best_map = m["macro_map"]
            save_ckpt("best", m, epoch)
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
    print(f"\nDone. Best val macro-mAP {best_map:.4f}. Checkpoints + vocab.json + meta.json in {out_dir}")
    print(f"Metrics log: {metrics_path}")


if __name__ == "__main__":
    main()
