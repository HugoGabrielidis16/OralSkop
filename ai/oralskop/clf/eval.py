"""Evaluate a trained manifest classifier on the test split.

Handles both checkpoint kinds:
* torchvision ``.pt`` (full fine-tune), and
* a foundation-model run directory (``meta.json`` + ``adapter_best/``) from QLoRA.

The checkpoint's own class list / preprocessing is authoritative (it matches the head).

    python -m oralskop.clf.eval --config configs/clf/manifest_clf.yaml \
        --weights runs/clf/clf_coarse/best.pt
    python -m oralskop.clf.eval --config configs/clf/qlora_dinov2.yaml \
        --weights runs/clf/clf_coarse_dinov2_large_qlora        # run dir -> adapter_best
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from oralskop.config import apply_overrides, load_yaml
from oralskop.clf.dataset import AI_ROOT, ManifestClfDataset, load_supervised_frame
from oralskop.clf.metrics import format_per_class, multilabel_metrics
from oralskop.clf.model import build_classifier, build_foundation_model
from oralskop.clf.train import collect_scores
from oralskop.clf.vocab import Vocab

_SPLIT_TEST = "test"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the manifest classifier on test.")
    p.add_argument("--config", required=True)
    p.add_argument("--weights", help="A .pt file, a run dir, or an adapter_* dir "
                                     "(else cfg['weights'] or out_dir/<name>).")
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE")
    return p.parse_args(argv)


def _resolve_weights(cfg: dict, cli: str | None) -> Path:
    if cli:
        return Path(cli)
    if cfg.get("weights"):
        return Path(cfg["weights"])
    default = (AI_ROOT / cfg.get("out_dir", "runs/clf") / cfg.get("name", "clf_coarse"))
    if not (default.exists()):
        raise SystemExit(f"No --weights given and nothing at {default}. Pass --weights <path>.")
    return default


def _load_model(weights: Path, device, compute_dtype):
    """Return (model, vocab, preprocess, is_hf) for either checkpoint kind."""
    if weights.is_file() and weights.suffix == ".pt":  # torchvision
        ckpt = torch.load(weights, map_location=device, weights_only=False)
        vocab = Vocab(names=list(ckpt["class_names"]), level=ckpt["level"])
        model = build_classifier(len(vocab), arch=ckpt["arch"], pretrained=False).to(device)
        model.load_state_dict(ckpt["model"])
        preprocess = {"imgsz": int(ckpt.get("imgsz", 224)),
                      "mean": tuple(ckpt.get("mean", (0.485, 0.456, 0.406))),
                      "std": tuple(ckpt.get("std", (0.229, 0.224, 0.225)))}
        print(f"Loaded torchvision {weights} | arch={ckpt['arch']} classes={len(vocab)}")
        return model, vocab, preprocess, False

    # Foundation/QLoRA: weights is a run dir (has meta.json) or an adapter_* dir.
    run_dir = weights if (weights / "meta.json").exists() else weights.parent
    meta = json.loads((run_dir / "meta.json").read_text())
    adapter_dir = (weights if (weights / "adapter_config.json").exists()
                   else next((run_dir / d for d in ("adapter_best", "adapter_last")
                              if (run_dir / d).exists()), None))
    if adapter_dir is None:
        raise SystemExit(f"No adapter_best/adapter_last under {run_dir}.")
    from peft import PeftModel

    vocab = Vocab(names=list(meta["class_names"]), level=meta["level"])
    base, _, _ = build_foundation_model(len(vocab), meta["model_id"],
                                        quantize=meta.get("quantize", "4bit"), lora=False,
                                        grad_checkpointing=False, compute_dtype=compute_dtype,
                                        imgsz=meta["imgsz"])
    quantized = str(meta.get("quantize") or "none").lower() in {"4bit", "8bit"}
    if not quantized:
        base = base.to(device)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    if not quantized:
        model = model.to(device)
    preprocess = {"imgsz": int(meta["imgsz"]), "mean": tuple(meta["mean"]), "std": tuple(meta["std"])}
    print(f"Loaded QLoRA {adapter_dir} | model_id={meta['model_id']} classes={len(vocab)}")
    return model, vocab, preprocess, True


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = apply_overrides(load_yaml(args.config), args.override)

    device = torch.device(cfg.get("device", "cuda") if cfg.get("device") != "cpu"
                          and torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    compute_dtype = amp_dtype if device.type == "cuda" else torch.float32

    weights = _resolve_weights(cfg, args.weights)
    model, vocab, preprocess, is_hf = _load_model(weights, device, compute_dtype)

    df, _ = load_supervised_frame(cfg["manifest"], vocab.level,
                                  image_path_prefixes=cfg.get("image_path_prefixes"),
                                  limit=cfg.get("limit"))
    test_ds = ManifestClfDataset(
        df[df["split"].str.strip() == _SPLIT_TEST], vocab,
        image_root=cfg.get("image_root", "s3://datastoraged4gen/02_PROCESSED"),
        imgsz=preprocess["imgsz"], train=False, cache_dir=cfg.get("cache_dir"),
        mean=preprocess["mean"], std=preprocess["std"])
    print(f"test={len(test_ds)} rows (dropped {test_ds.dropped_empty} off-vocab)")
    if len(test_ds) == 0:
        raise SystemExit("No test rows — check the manifest split column / filters.")

    loader = torch.utils.data.DataLoader(
        test_ds, batch_size=cfg.get("batch", 64), shuffle=False,
        num_workers=int(cfg.get("num_workers", 8)), pin_memory=device.type == "cuda")
    y_true, y_score = collect_scores(model, loader, device, is_hf=is_hf, amp_dtype=amp_dtype,
                                     use_amp=use_amp, desc="test")
    m = multilabel_metrics(y_true, y_score, vocab.names, threshold=float(cfg.get("threshold", 0.5)))

    print(f"\n== test ({m['num_samples']} samples) ==")
    print(f"macro-mAP {m['macro_map']:.4f}  micro-AP {m['micro_ap']:.4f}  "
          f"macro-F1 {m['macro_f1']:.4f}  micro-F1 {m['micro_f1']:.4f}  (thr={m['threshold']})")
    print(format_per_class(m))


if __name__ == "__main__":
    main()
