"""Build a canonical YOLO-seg training set from one or more dataset configs.

Registry-driven and config-only for known formats:

    # single dataset
    python -m oralskop.data.prepare --datasets alphadent

    # merged multi-dataset (pooled under the shared taxonomy)
    python -m oralskop.data.prepare --datasets alphadent caries_roboflow --out-name merged

Each named dataset resolves to configs/data/<name>.yaml, whose `converter:` field
selects the converter (see converters/registry.py).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from oralskop.data.build import build_dataset
from oralskop.data.converters.registry import build_converter
from oralskop.data.taxonomy import load_taxonomy

# ai/ directory (this file is ai/oralskop/data/prepare.py -> parents[2] == ai/).
AI_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the canonical training dataset.")
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["alphadent"],
        help="Dataset config names under configs/data/ (without .yaml).",
    )
    p.add_argument("--taxonomy", default=str(AI_ROOT / "configs/taxonomy.yaml"))
    p.add_argument("--data-config-dir", default=str(AI_ROOT / "configs/data"))
    p.add_argument("--out-root", default=str(AI_ROOT / "data"))
    p.add_argument(
        "--out-name",
        default=None,
        help="Output folder under data/ (default: the dataset name, or 'merged').",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    taxonomy = load_taxonomy(args.taxonomy)
    config_dir = Path(args.data_config_dir)

    converters = []
    for name in args.datasets:
        cfg_path = config_dir / f"{name}.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Dataset config not found: {cfg_path}")
        converter = build_converter(cfg_path, base_dir=AI_ROOT)
        print("=" * 64)
        print(f"[{converter.name}] verify:")
        for report in converter.verify():
            print(report.summary())
        converters.append((converter, converter.val_fraction, converter.seed))
    print("=" * 64)

    out_name = args.out_name or (args.datasets[0] if len(args.datasets) == 1 else "merged")
    out_dir = Path(args.out_root) / out_name

    result = build_dataset(converters=converters, taxonomy=taxonomy, out_dir=out_dir)

    names = taxonomy.names_mapping()
    print(f"\nCanonical taxonomy v{taxonomy.version}: {taxonomy.names_in_index_order()}")
    print(f"Built dataset at: {result.out_dir}")
    print(f"  data.yaml : {result.data_yaml}")
    print(f"  train     : {result.n_train} images")
    print(f"  val       : {result.n_val} images")
    print("  train class histogram (canonical id -> polygons):")
    for cls in sorted(result.train_class_hist):
        print(f"      {cls} {names.get(cls, cls):<12}: {result.train_class_hist[cls]}")
    print("  val class histogram:")
    for cls in sorted(result.val_class_hist):
        print(f"      {cls} {names.get(cls, cls):<12}: {result.val_class_hist[cls]}")


if __name__ == "__main__":
    main()
