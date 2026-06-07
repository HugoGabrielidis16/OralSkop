"""Train a DINOv2-backbone DETR detector on the manifest's bbox subset.

The HF model computes the DETR set-prediction loss, so the loop just feeds
``model(pixel_values=..., labels=...)`` and backprops ``outputs.loss``. Reuses the
classifier's optimizer / scheduler / W&B / run-dir scaffolding.

    uv run --extra clf --extra qlora --extra det python -m oralskop.det.train \
        --config configs/det/qlora_dinov2_detr.yaml \
        --override image_root=datasets/02_PROCESSED limit=64 epochs=1 batch=2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from oralskop.config import apply_overrides, load_yaml
from oralskop.clf.train import build_optimizer, build_scheduler, resolve_run_dir, wandb_setup
from oralskop.clf.vocab import build_vocab
from oralskop.det.dataset import AI_ROOT, ManifestDetDataset, det_collate_fn, load_bbox_frame
from oralskop.det.metrics import format_per_class, new_map_metric, summarize_map
from oralskop.det.model import build_detector, build_detr_processor

_SPLIT_TRAIN = "train"
_SPLIT_VAL = "valid"


def _to_device_targets(targets, device):
    return [{k: v.to(device) for k, v in t.items()} for t in targets]


def _cxcywh_norm_to_xyxy_abs(boxes: torch.Tensor, size: int) -> torch.Tensor:
    """[n,4] normalized cx,cy,w,h -> [n,4] xyxy in pixels at a square `size`."""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = (cx - w / 2) * size
    y1 = (cy - h / 2) * size
    x2 = (cx + w / 2) * size
    y2 = (cy + h / 2) * size
    return torch.stack([x1, y1, x2, y2], dim=-1)


@torch.no_grad()
def evaluate_map(model, loader, device, processor, imgsz, *, is_quant, amp_dtype, use_amp,
                 class_names, score_threshold=0.0, desc="val", progress=True):
    """Run the detector over a loader and return a summarized mAP dict."""
    model.eval()
    metric = new_map_metric(class_metrics=True)
    for imgs, targets in tqdm(loader, desc=desc, leave=False, disable=not progress):
        imgs = imgs.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            outputs = model(pixel_values=imgs)
        sizes = torch.tensor([[imgsz, imgsz]] * imgs.shape[0])
        preds = processor.post_process_object_detection(outputs, threshold=score_threshold,
                                                        target_sizes=sizes)
        preds = [{k: v.float().cpu() if k != "labels" else v.cpu() for k, v in p.items()} for p in preds]
        tgts = [{"boxes": _cxcywh_norm_to_xyxy_abs(t["boxes"].cpu().float(), imgsz),
                 "labels": t["class_labels"].cpu()} for t in targets]
        metric.update(preds, tgts)
    return summarize_map(metric.compute(), class_names)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="DINOv2-DETR object detection training.")
    p.add_argument("--config", required=True)
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = apply_overrides(load_yaml(args.config), args.override)

    device = torch.device(cfg.get("device", "cuda") if cfg.get("device") != "cpu"
                          and torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))

    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    compute_dtype = amp_dtype if device.type == "cuda" else torch.float32
    use_scaler = use_amp and amp_dtype == torch.float16

    level = cfg.get("level", "coarse")
    df, labels_per_row = load_bbox_frame(cfg["manifest"], level,
                                         image_path_prefixes=cfg.get("image_path_prefixes"),
                                         limit=cfg.get("limit"))
    vocab = build_vocab(level, labels_file=cfg.get("labels_file"), labels_per_row=labels_per_row,
                        exclude_micro=bool(cfg.get("exclude_train_only_microclasses", True)))
    num_classes = len(vocab)
    arch = str(cfg.get("arch", "dinov2_base"))
    print(f"Level={level}: {num_classes} classes | bbox rows={len(df)} | arch={arch}")

    model, preprocess = build_detector(
        num_classes, arch, quantize=cfg.get("quantize", "none"), lora=bool(cfg.get("lora", True)),
        lora_r=cfg.get("lora_r", 16), lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        grad_checkpointing=bool(cfg.get("grad_checkpointing", True)),
        compute_dtype=compute_dtype, num_queries=int(cfg.get("num_queries", 100)),
        imgsz=cfg.get("imgsz"))
    is_quant = str(cfg.get("quantize", "none")).lower() == "4bit"
    if not is_quant:
        model = model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: trainable={trainable:,} total={total:,} "
          f"({trainable/max(total,1):.2%}) | precision={amp_dtype if use_amp else 'fp32'}")

    imgsz, mean, std = int(preprocess["imgsz"]), preprocess["mean"], preprocess["std"]
    ds_kw = dict(image_root=cfg.get("image_root", "s3://datastoraged4gen/02_PROCESSED"),
                 imgsz=imgsz, cache_dir=cfg.get("cache_dir"), mean=mean, std=std,
                 unreadable_log_limit=int(cfg.get("unreadable_log_limit", 0) or 0))
    train_ds = ManifestDetDataset(df[df["split"].str.strip() == _SPLIT_TRAIN], vocab, **ds_kw)
    val_ds = ManifestDetDataset(df[df["split"].str.strip() == _SPLIT_VAL], vocab, **ds_kw)
    print(f"train={len(train_ds)} (dropped {train_ds.dropped_off_vocab}) "
          f"valid={len(val_ds)} | imgsz={imgsz} device={device.type}")
    if len(train_ds) == 0:
        raise SystemExit("No trainable rows — check manifest/level/image_path_prefixes.")

    pin = device.type == "cuda"
    workers = int(cfg.get("num_workers", 4))
    batch = int(cfg.get("batch", 4))
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch, shuffle=True, num_workers=workers, pin_memory=pin,
        drop_last=len(train_ds) > batch, collate_fn=det_collate_fn)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=pin,
        collate_fn=det_collate_fn)

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    grad_accum = max(int(cfg.get("grad_accum_steps", 1)), 1)
    processor = build_detr_processor()

    run_name = cfg.get("name", "det_coarse")
    out_dir = resolve_run_dir((AI_ROOT / cfg.get("out_dir", "runs/det")).resolve(), run_name,
                              bool(cfg.get("exist_ok", False)))
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab.to_json(out_dir / "vocab.json")
    meta = {"arch": arch, "model_id": preprocess.get("model_id"), "num_classes": num_classes,
            "class_names": vocab.names, "level": level, "imgsz": imgsz,
            "mean": list(mean), "std": list(std), "quantize": cfg.get("quantize", "none"),
            "lora": bool(cfg.get("lora", True)), "num_queries": int(cfg.get("num_queries", 100))}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    metrics_path = out_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    wb = wandb_setup(cfg, cfg.get("wandb_name") or run_name, out_dir)

    def save_ckpt(tag, m, epoch):
        dest = out_dir / f"{tag}.pt"
        torch.save({"model": model.state_dict(), "meta": meta, "epoch": epoch, "metrics": m}, dest)
        s = (f"map {m['map']:.4f}, map50 {m['map_50']:.4f}" if m else "no val metrics")
        tqdm.write(f">> Model saved [{tag}] epoch {epoch} with metrics: {s} -> {dest}")

    epochs = int(cfg.get("epochs", 50))
    progress = bool(cfg.get("progress", True))
    progress_leave = bool(cfg.get("progress_leave", True))
    log_interval = int(cfg.get("log_interval", 50) or 0)
    best_map = -1.0
    print(f"Training {arch}-DETR for {epochs} epochs (batch {batch} x accum {grad_accum}) -> {out_dir}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum, n = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        bar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=progress_leave, disable=not progress)
        step = 0
        for step, (imgs, targets) in enumerate(bar, 1):
            imgs = imgs.to(device)
            targets = _to_device_targets(targets, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss = model(pixel_values=imgs, labels=targets).loss
            scaled = loss / grad_accum
            (scaler.scale(scaled) if use_scaler else scaled).backward()
            if step % grad_accum == 0:
                if use_scaler:
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            loss_sum += float(loss.detach()) * imgs.shape[0]
            n += imgs.shape[0]
            if progress:
                bar.set_postfix(loss=f"{loss_sum/max(n,1):.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
            if log_interval and (step == 1 or step % log_interval == 0):
                tqdm.write(f"epoch {epoch} batch {step}/{len(train_loader)} "
                           f"loss {loss_sum/max(n,1):.4f} ({time.time()-t0:.0f}s)")
        if step % grad_accum != 0:
            if use_scaler:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        train_loss = loss_sum / max(n, 1)

        line = f"epoch {epoch:3d}  train_loss {train_loss:.4f}"
        m = None
        if len(val_ds) > 0:
            m = evaluate_map(model, val_loader, device, processor, imgsz, is_quant=is_quant,
                             amp_dtype=amp_dtype, use_amp=use_amp, class_names=vocab.names,
                             score_threshold=float(cfg.get("eval_score_threshold", 0.0)),
                             desc=f"val {epoch}", progress=progress)
            line += f"  mAP {m['map']:.4f}  mAP50 {m['map_50']:.4f}  mAR100 {m['mar_100']:.4f}"

        record = {"epoch": epoch, "train_loss": train_loss, "lr": optimizer.param_groups[0]["lr"]}
        if m is not None:
            record.update({k: m[k] for k in ("map", "map_50", "map_75", "mar_100")})
        with metrics_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

        save_ckpt("last", m, epoch)
        if m is not None and m["map"] == m["map"] and m["map"] > best_map:
            best_map = m["map"]
            save_ckpt("best", m, epoch)
            line += "  *best*"
        if scheduler is not None:
            scheduler.step()
        tqdm.write(f"{line}  ({time.time()-t0:.0f}s)")
        if m is not None and cfg.get("log_per_class", True):
            tqdm.write(format_per_class(m))
        if wb is not None:
            log = {"train/loss": train_loss, "lr": optimizer.param_groups[0]["lr"]}
            if m is not None:
                log.update({"val/mAP": m["map"], "val/mAP50": m["map_50"], "val/mAR100": m["mar_100"]})
            wb.log(log, step=epoch)

    if wb is not None:
        wb.summary["best_val_mAP"] = best_map
        wb.finish()
    print(f"\nDone. Best val mAP {best_map:.4f}. Checkpoints + meta.json in {out_dir}")


if __name__ == "__main__":
    main()
