"""Evaluate a trained DINOv2-DETR detector on the test split (mAP + per-class AP).

    uv run --extra clf --extra qlora --extra det python -m oralskop.det.eval \
        --config configs/det/qlora_dinov2_detr.yaml \
        --weights runs/det/det_coarse_dinov2_detr_qlora/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from oralskop.config import apply_overrides, load_yaml
from oralskop.clf.vocab import Vocab
from oralskop.det.dataset import AI_ROOT, ManifestDetDataset, det_collate_fn, load_bbox_frame
from oralskop.det.metrics import format_per_class
from oralskop.det.model import build_detector, build_detr_processor
from oralskop.det.train import evaluate_map

_SPLIT_TEST = "test"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate the DINOv2-DETR detector on test.")
    p.add_argument("--config", required=True)
    p.add_argument("--weights", help="Checkpoint .pt (else cfg['weights'] or out_dir/<name>/best.pt).")
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE")
    return p.parse_args(argv)


def _resolve_weights(cfg, cli):
    if cli:
        return Path(cli)
    if cfg.get("weights"):
        return Path(cfg["weights"])
    default = AI_ROOT / cfg.get("out_dir", "runs/det") / cfg.get("name", "det_coarse") / "best.pt"
    if not default.exists():
        raise SystemExit(f"No --weights given and nothing at {default}. Pass --weights <path>.")
    return default


def main(argv=None):
    args = parse_args(argv)
    cfg = apply_overrides(load_yaml(args.config), args.override)

    device = torch.device(cfg.get("device", "cuda") if cfg.get("device") != "cpu"
                          and torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    compute_dtype = amp_dtype if device.type == "cuda" else torch.float32

    weights = _resolve_weights(cfg, args.weights)
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    meta = ckpt["meta"]
    vocab = Vocab(names=list(meta["class_names"]), level=meta["level"])
    model, _ = build_detector(len(vocab), meta["arch"], quantize=meta.get("quantize", "none"),
                              lora=bool(meta.get("lora", True)), compute_dtype=compute_dtype,
                              num_queries=int(meta.get("num_queries", 100)), imgsz=meta["imgsz"])
    model.load_state_dict(ckpt["model"], strict=False)
    is_quant = str(meta.get("quantize", "none")).lower() == "4bit"
    if not is_quant:
        model = model.to(device)
    print(f"Loaded {weights} | arch={meta['arch']} classes={len(vocab)} epoch={ckpt.get('epoch')}")

    df, _ = load_bbox_frame(cfg["manifest"], meta["level"],
                            image_path_prefixes=cfg.get("image_path_prefixes"), limit=cfg.get("limit"))
    test_ds = ManifestDetDataset(
        df[df["split"].str.strip() == _SPLIT_TEST], vocab,
        image_root=cfg.get("image_root", "s3://datastoraged4gen/02_PROCESSED"),
        imgsz=int(meta["imgsz"]), cache_dir=cfg.get("cache_dir"),
        mean=tuple(meta["mean"]), std=tuple(meta["std"]))
    print(f"test={len(test_ds)} rows (dropped {test_ds.dropped_off_vocab})")
    if len(test_ds) == 0:
        raise SystemExit("No test rows — check manifest split/filters.")

    loader = torch.utils.data.DataLoader(
        test_ds, batch_size=cfg.get("batch", 4), shuffle=False,
        num_workers=int(cfg.get("num_workers", 4)), pin_memory=device.type == "cuda",
        collate_fn=det_collate_fn)
    processor = build_detr_processor()
    m = evaluate_map(model, loader, device, processor, int(meta["imgsz"]), is_quant=is_quant,
                     amp_dtype=amp_dtype, use_amp=use_amp, class_names=vocab.names,
                     score_threshold=float(cfg.get("eval_score_threshold", 0.0)),
                     match_score_threshold=float(cfg.get("eval_match_score_threshold", 0.5)),
                     desc="test")
    print(f"\n== test ==  mAP {m['map']:.4f}  mAP50 {m['map_50']:.4f}  "
          f"mAP75 {m['map_75']:.4f}  mAR100 {m['mar_100']:.4f}")
    print(f"threshold {m['match_score_threshold']:.2f}  precision50 {m['precision_50']:.4f}  "
          f"recall50 {m['recall_50']:.4f}  F1_50 {m['f1_50']:.4f}  "
          f"mean_iou50 {m['mean_iou_50']:.4f}  "
          f"matched_class_accuracy50 {m['matched_class_accuracy_50']:.4f}")
    print(format_per_class(m))


if __name__ == "__main__":
    main()
