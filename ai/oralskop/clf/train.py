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
import re
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

# Manifest split names (doc §2) — note "valid", not "val".
_SPLIT_TRAIN = "train"
_SPLIT_VAL = "valid"


def build_optimizer(cfg: dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    """AdamW / SGD / 8-bit (paged) AdamW over the trainable params.

    Optimizing only ``requires_grad`` params means a QLoRA run updates just the LoRA
    adapters + head (the 4-bit base is frozen).
    """
    name = str(cfg.get("optimizer", "adamw")).lower().replace("-", "_")
    lr = cfg.get("lr", 3e-4)
    weight_decay = cfg.get("weight_decay", 0.05)
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            params, lr=lr, momentum=cfg.get("momentum", 0.9),
            weight_decay=weight_decay, nesterov=bool(cfg.get("nesterov", False)),
        )
    if name in {"adamw8bit", "paged_adamw8bit", "adamw_8bit", "paged_adamw_8bit"}:
        try:
            import bitsandbytes as bnb
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError("8-bit optimizers need the `qlora` extra (bitsandbytes): "
                              "uv sync --extra qlora.") from exc
        Opt = bnb.optim.PagedAdamW8bit if "paged" in name else bnb.optim.AdamW8bit
        return Opt(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer {name!r}. Options: adamw, sgd, adamw8bit, paged_adamw8bit")


def forward_logits(model, imgs, is_hf: bool):
    """Uniform forward: HF image models return ``.logits``; torchvision returns logits."""
    if is_hf:
        return model(pixel_values=imgs).logits
    return model(imgs)


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


def _compact_count(n: int) -> str:
    for unit, scale in (("B", 1_000_000_000), ("M", 1_000_000), ("K", 1_000)):
        if n >= scale:
            value = f"{n / scale:.1f}".rstrip("0").rstrip(".")
            return f"{value}{unit}"
    return str(n)


def _slug(value: object, *, max_len: int = 40) -> str:
    text = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(value).strip()).strip("-_.").lower()
    return (text or "na")[:max_len]


def _prefix_scope_label(prefixes) -> str:
    if not prefixes:
        return "all"
    if isinstance(prefixes, str):
        raw = [p for p in prefixes.replace(",", " ").split() if p]
    else:
        raw = list(prefixes)
    labels = []
    for prefix in raw:
        first = str(prefix).strip("/").split("/", 1)[0]
        labels.append(_slug(first, max_len=18))
    if len(labels) > 3:
        return "+".join(labels[:3]) + f"+{len(labels) - 3}more"
    return "+".join(labels) if labels else "all"


def _model_param_counts(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def auto_wandb_name(
    cfg: dict,
    *,
    level: str,
    num_rows: int,
    num_classes: int,
    model_params: int,
) -> str:
    """Descriptive W&B run name independent of the local checkpoint folder."""
    parts = [
        "clf",
        _slug(level),
        _slug(cfg.get("arch", "model")),
        _compact_count(model_params),
        f"img{cfg.get('imgsz', 224)}",
        f"b{cfg.get('batch', 64)}",
        f"e{cfg.get('epochs', 30)}",
        f"c{num_classes}",
        f"n{num_rows}",
        _prefix_scope_label(cfg.get("image_path_prefixes")),
    ]
    return "-".join(parts)


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


def _empty_running_stats() -> dict[str, float]:
    return {
        "samples": 0,
        "label_total": 0,
        "label_correct": 0,
        "exact_correct": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "loss_sum": 0.0,
        "loss_n": 0,
    }


def _update_running_stats(
    stats: dict[str, float],
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    threshold: float,
    loss_value: float | None = None,
) -> None:
    """Accumulate cheap thresholded metrics for live tqdm postfixes."""
    with torch.no_grad():
        pred = torch.sigmoid(logits.detach()) >= threshold
        true = targets.detach().bool()
        correct = pred == true
        stats["samples"] += int(true.shape[0])
        stats["label_total"] += int(true.numel())
        stats["label_correct"] += int(correct.sum().item())
        stats["exact_correct"] += int(correct.all(dim=1).sum().item())
        stats["tp"] += int((pred & true).sum().item())
        stats["fp"] += int((pred & ~true).sum().item())
        stats["fn"] += int((~pred & true).sum().item())
        if loss_value is not None:
            batch_n = int(true.shape[0])
            stats["loss_sum"] += float(loss_value) * batch_n
            stats["loss_n"] += batch_n


def _running_summary(stats: dict[str, float]) -> dict[str, float]:
    label_total = max(int(stats["label_total"]), 1)
    samples = max(int(stats["samples"]), 1)
    tp, fp, fn = int(stats["tp"]), int(stats["fp"]), int(stats["fn"])
    return {
        "loss": stats["loss_sum"] / max(int(stats["loss_n"]), 1),
        "micro_accuracy": stats["label_correct"] / label_total,
        "exact_match_accuracy": stats["exact_correct"] / samples,
        "micro_f1": (2 * tp / max(2 * tp + fp + fn, 1)),
    }


@torch.no_grad()
def collect_scores(
    model,
    loader,
    device,
    *,
    limit: int | None = None,
    desc: str = "val",
    progress: bool = True,
    leave: bool = True,
    criterion=None,
    threshold: float = 0.5,
    use_amp: bool = False,
    amp_dtype=torch.float16,
    is_hf: bool = False,
    return_stats: bool = False,
):
    """Run the model over a loader -> (y_true, y_score) numpy arrays [N, C]."""
    model.eval()
    trues, scores = [], []
    seen = 0
    stats = _empty_running_stats()
    batches = tqdm(
        loader,
        desc=desc,
        leave=leave,
        dynamic_ncols=True,
        disable=not progress,
    )
    for imgs, targets in batches:
        imgs = imgs.to(device)
        targets_device = targets.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = forward_logits(model, imgs, is_hf).float()
            loss = criterion(logits, targets_device) if criterion is not None else None
        loss_value = float(loss.detach()) if loss is not None else None
        _update_running_stats(
            stats,
            logits,
            targets_device,
            threshold=threshold,
            loss_value=loss_value,
        )
        if progress:
            s = _running_summary(stats)
            postfix = {
                "acc": f"{s['micro_accuracy']:.3f}",
                "f1": f"{s['micro_f1']:.3f}",
                "exact": f"{s['exact_match_accuracy']:.3f}",
            }
            if loss is not None:
                postfix["loss"] = f"{s['loss']:.4f}"
            batches.set_postfix(**postfix)
        scores.append(torch.sigmoid(logits).float().cpu().numpy())
        trues.append(targets.numpy())
        seen += imgs.shape[0]
        if limit and seen >= limit:
            break
    if not trues:
        empty = (np.zeros((0, 0)), np.zeros((0, 0)))
        return (*empty, _running_summary(stats)) if return_stats else empty
    result = (np.concatenate(trues), np.concatenate(scores))
    return (*result, _running_summary(stats)) if return_stats else result


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

    # Build the model first so preprocessing (normalization + input size) comes from it
    # (torchvision -> ImageNet/imgsz; foundation -> its own image processor).
    model, preprocess, is_hf = build_model(num_classes, cfg, compute_dtype=compute_dtype)
    quantized = is_hf and str(cfg.get("quantize", "4bit")).lower() in {"4bit", "8bit"}
    if not quantized:  # 4-bit/8-bit weights are placed on GPU by bitsandbytes — don't .to()
        model = model.to(device)
    imgsz = int(preprocess["imgsz"])
    mean, std = preprocess["mean"], preprocess["std"]

    image_root = cfg.get("image_root", "s3://datastoraged4gen/02_PROCESSED")
    cache_dir = cfg.get("cache_dir")
    unreadable_log_limit = int(cfg.get("unreadable_log_limit", 0) or 0)
    ds_kw = dict(image_root=image_root, imgsz=imgsz, cache_dir=cache_dir, mean=mean, std=std,
                 unreadable_log_limit=unreadable_log_limit)
    train_ds = ManifestClfDataset(df[df["split"].str.strip() == _SPLIT_TRAIN], vocab,
                                  train=True, **ds_kw)
    val_ds = ManifestClfDataset(df[df["split"].str.strip() == _SPLIT_VAL], vocab,
                                train=False, **ds_kw)
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

    model_params, trainable_params = _model_param_counts(model)
    cfg["model_params"] = model_params
    cfg["trainable_params"] = trainable_params

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
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    grad_accum = max(int(cfg.get("grad_accum_steps", 1)), 1)

    run_name = cfg.get("name", f"clf_{level}")
    out_dir = resolve_run_dir((AI_ROOT / cfg.get("out_dir", "runs/clf")).resolve(), run_name,
                              bool(cfg.get("exist_ok", False)))
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab.to_json(out_dir / "vocab.json")
    meta = {"is_hf": is_hf, "arch": cfg.get("arch", "convnext_tiny"),
            "model_id": preprocess.get("model_id"), "num_classes": num_classes,
            "class_names": vocab.names, "level": level, "imgsz": imgsz,
            "mean": list(mean), "std": list(std),
            "quantize": cfg.get("quantize", "4bit") if is_hf else None,
            "lora": bool(cfg.get("lora", True)) if is_hf else False,
            "lora_r": cfg.get("lora_r", 16), "lora_alpha": cfg.get("lora_alpha", 32)}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    metrics_path = out_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    wandb_run_name = cfg.get("wandb_name") or auto_wandb_name(
        cfg,
        level=level,
        num_rows=len(df),
        num_classes=num_classes,
        model_params=model_params,
    )
    cfg["wandb_run_name"] = wandb_run_name
    wb = wandb_setup(cfg, wandb_run_name, out_dir)

    epochs = int(cfg.get("epochs", 30))
    threshold = float(cfg.get("threshold", 0.5))
    progress = bool(cfg.get("progress", True))
    progress_leave = bool(cfg.get("progress_leave", True))
    log_interval = int(cfg.get("log_interval", 50) or 0)
    best_map = -1.0

    def save_ckpt(tag: str, m: dict | None, epoch: int) -> None:
        if is_hf:  # save LoRA adapters + head (not the frozen 4-bit base)
            dest = out_dir / f"adapter_{tag}"
            model.save_pretrained(str(dest))
        else:
            dest = out_dir / f"{tag}.pt"
            torch.save({"model": model.state_dict(), "arch": cfg.get("arch", "convnext_tiny"),
                        "num_classes": num_classes, "class_names": vocab.names, "level": level,
                        "imgsz": imgsz, "mean": list(mean), "std": list(std),
                        "epoch": epoch, "metrics": m}, dest)
        if m is not None:
            summary = (f"macro_mAP {m['macro_map']:.4f}, micro_AP {m['micro_ap']:.4f}, "
                       f"macro_F1 {m['macro_f1']:.4f}, micro_F1 {m['micro_f1']:.4f}")
        else:
            summary = "no validation metrics (empty val split)"
        tqdm.write(f">> Model saved [{tag}] epoch {epoch} with metrics: {summary} -> {dest}")
    print(f"Model params: total={model_params:,} trainable={trainable_params:,}")
    if cfg.get("wandb", False):
        print(f"W&B run name: {wandb_run_name}")
    print(f"Training {cfg.get('arch', 'convnext_tiny')} for {epochs} epochs -> {out_dir}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_stats = _empty_running_stats()
        train_batches = tqdm(
            train_loader,
            desc=f"epoch {epoch}/{epochs}",
            leave=progress_leave,
            dynamic_ncols=True,
            disable=not progress,
        )
        optimizer.zero_grad(set_to_none=True)
        step = 0
        for step, (imgs, targets) in enumerate(train_batches, 1):
            imgs, targets = imgs.to(device), targets.to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = forward_logits(model, imgs, is_hf).float()
                loss = criterion(logits, targets)
            scaled = loss / grad_accum
            (scaler.scale(scaled) if use_scaler else scaled).backward()
            if step % grad_accum == 0:
                if use_scaler:
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            batch_loss = float(loss.detach())
            _update_running_stats(
                train_stats,
                logits,
                targets,
                threshold=threshold,
                loss_value=batch_loss,
            )
            train_summary = _running_summary(train_stats)
            lr = optimizer.param_groups[0]["lr"]
            if progress:
                train_batches.set_postfix(
                    loss=f"{train_summary['loss']:.4f}",
                    acc=f"{train_summary['micro_accuracy']:.3f}",
                    f1=f"{train_summary['micro_f1']:.3f}",
                    exact=f"{train_summary['exact_match_accuracy']:.3f}",
                    lr=f"{lr:.2e}",
                )
            if log_interval and (step == 1 or step % log_interval == 0 or step == len(train_loader)):
                tqdm.write(
                    f"epoch {epoch}/{epochs} batch {step}/{len(train_loader)} "
                    f"loss {train_summary['loss']:.4f} batch_loss {batch_loss:.4f} "
                    f"acc {train_summary['micro_accuracy']:.4f} "
                    f"f1 {train_summary['micro_f1']:.4f} "
                    f"exact {train_summary['exact_match_accuracy']:.4f} "
                    f"lr {lr:.2e} elapsed {time.time() - t0:.0f}s"
                )
        if step % grad_accum != 0:  # flush trailing partial accumulation
            if use_scaler:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        train_summary = _running_summary(train_stats)
        train_loss = train_summary["loss"]

        line = f"epoch {epoch:3d}  train_loss {train_loss:.4f}"
        line += (f"  train_acc {train_summary['micro_accuracy']:.4f}"
                 f"  train_F1 {train_summary['micro_f1']:.4f}"
                 f"  train_exact {train_summary['exact_match_accuracy']:.4f}")
        m = None
        val_summary = None
        if len(val_ds) > 0:
            y_true, y_score, val_summary = collect_scores(
                model,
                val_loader,
                device,
                desc=f"val {epoch}",
                progress=progress,
                leave=progress_leave,
                criterion=criterion,
                threshold=threshold,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                is_hf=is_hf,
                return_stats=True,
            )
            m = multilabel_metrics(y_true, y_score, vocab.names, threshold=threshold)
            line += f"  val_loss {val_summary['loss']:.4f}"
            line += (f"  macro_mAP {m['macro_map']:.4f}  micro_AP {m['micro_ap']:.4f}"
                     f"  macro_F1 {m['macro_f1']:.4f}  micro_F1 {m['micro_f1']:.4f}")
            line += (f"  macro_acc {m['macro_accuracy']:.4f}"
                     f"  micro_acc {m['micro_accuracy']:.4f}"
                     f"  exact_acc {m['exact_match_accuracy']:.4f}")

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_micro_accuracy": train_summary["micro_accuracy"],
            "train_micro_f1": train_summary["micro_f1"],
            "train_exact_match_accuracy": train_summary["exact_match_accuracy"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        if val_summary is not None:
            record["val_loss"] = val_summary["loss"]
        if m is not None:
            record.update({k: m[k] for k in
                           ("macro_map", "micro_ap", "macro_f1", "micro_f1",
                            "macro_accuracy", "micro_accuracy", "exact_match_accuracy")})
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
            log = {
                "train/loss": train_loss,
                "train/micro_accuracy": train_summary["micro_accuracy"],
                "train/micro_F1": train_summary["micro_f1"],
                "train/exact_match_accuracy": train_summary["exact_match_accuracy"],
                "lr": optimizer.param_groups[0]["lr"],
            }
            if val_summary is not None:
                log["val/loss"] = val_summary["loss"]
            if m is not None:
                log.update({"val/macro_mAP": m["macro_map"], "val/micro_AP": m["micro_ap"],
                            "val/macro_F1": m["macro_f1"], "val/micro_F1": m["micro_f1"],
                            "val/macro_accuracy": m["macro_accuracy"],
                            "val/micro_accuracy": m["micro_accuracy"],
                            "val/exact_match_accuracy": m["exact_match_accuracy"]})
            wb.log(log, step=epoch)

    if wb is not None:
        wb.summary["best_val_macro_mAP"] = best_map
        wb.finish()
    print(f"\nDone. Best val macro-mAP {best_map:.4f}. Checkpoints + vocab.json in {out_dir}")
    print(f"Metrics log: {metrics_path}")


if __name__ == "__main__":
    main()
