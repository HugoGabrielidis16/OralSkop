"""Evaluate a trained manifest classifier on the test split.

Loads a checkpoint (its embedded class list is authoritative — it matches the
trained head), scores the manifest's ``test`` rows, and prints macro/micro
summaries plus a per-class table.

    python -m oralskop.clf.eval --config configs/clf/manifest_clf.yaml \
        --weights runs/clf/clf_coarse/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from oralskop.config import apply_overrides, load_yaml
from oralskop.clf.dataset import AI_ROOT, ManifestClfDataset, load_supervised_frame
from oralskop.clf.metrics import format_per_class, multilabel_metrics
from oralskop.clf.model import build_classifier
from oralskop.clf.train import collect_scores
from oralskop.clf.vocab import Vocab

_SPLIT_TEST = "test"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the manifest classifier on test.")
    p.add_argument("--config", required=True)
    p.add_argument("--weights", help="Checkpoint path (else cfg['weights'] or out_dir/best.pt).")
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE")
    return p.parse_args(argv)


def _resolve_weights(cfg: dict, cli: str | None) -> Path:
    if cli:
        return Path(cli)
    if cfg.get("weights"):
        return Path(cfg["weights"])
    default = (AI_ROOT / cfg.get("out_dir", "runs/clf") / cfg.get("name", "clf_coarse") / "best.pt")
    if not default.exists():
        raise SystemExit("No --weights given and no checkpoint at "
                         f"{default}. Pass --weights <path>.")
    return default


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = apply_overrides(load_yaml(args.config), args.override)

    device = torch.device(cfg.get("device", "cuda") if cfg.get("device") != "cpu"
                          and torch.cuda.is_available() else "cpu")
    weights = _resolve_weights(cfg, args.weights)
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    vocab = Vocab(names=list(ckpt["class_names"]), level=ckpt["level"])
    print(f"Loaded {weights} | arch={ckpt['arch']} level={ckpt['level']} "
          f"classes={len(vocab)} epoch={ckpt.get('epoch')}")

    df, _ = load_supervised_frame(cfg["manifest"], ckpt["level"],
                                  image_path_prefixes=cfg.get("image_path_prefixes"),
                                  limit=cfg.get("limit"))
    test_ds = ManifestClfDataset(df[df["split"].str.strip() == _SPLIT_TEST], vocab,
                                 image_root=cfg.get("image_root", "s3://datastoraged4gen/02_PROCESSED"),
                                 imgsz=int(ckpt.get("imgsz", cfg.get("imgsz", 224))),
                                 train=False, cache_dir=cfg.get("cache_dir"))
    print(f"test={len(test_ds)} rows (dropped {test_ds.dropped_empty} off-vocab)")
    if len(test_ds) == 0:
        raise SystemExit("No test rows — check the manifest split column / filters.")

    model = build_classifier(len(vocab), arch=ckpt["arch"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])

    loader = torch.utils.data.DataLoader(
        test_ds, batch_size=cfg.get("batch", 64), shuffle=False,
        num_workers=int(cfg.get("num_workers", 8)), pin_memory=device.type == "cuda")
    y_true, y_score = collect_scores(model, loader, device, desc="test")
    m = multilabel_metrics(y_true, y_score, vocab.names, threshold=float(cfg.get("threshold", 0.5)))

    print(f"\n== test ({m['num_samples']} samples) ==")
    print(f"macro-mAP {m['macro_map']:.4f}  micro-AP {m['micro_ap']:.4f}  "
          f"macro-F1 {m['macro_f1']:.4f}  micro-F1 {m['micro_f1']:.4f}  "
          f"macro-acc {m['macro_accuracy']:.4f}  micro-acc {m['micro_accuracy']:.4f}  "
          f"exact-acc {m['exact_match_accuracy']:.4f}  "
          f"(thr={m['threshold']})")
    print(format_per_class(m))


if __name__ == "__main__":
    main()
