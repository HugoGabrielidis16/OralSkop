"""Merge-map / taxonomy coverage report across all dataset configs.

Shows how each dataset's native classes fold into the canonical taxonomy, making the
merge policy explicit: which canonical classes are SHARED across datasets (similar
classes merged) and which are specific to a single dataset. Also flags problems:
  * a class_map target that is not in the taxonomy (would crash prepare),
  * canonical classes no dataset contributes to (unused / reserved).

    uv run python -m oralskop.data.coverage
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import yaml

from oralskop.data.taxonomy import load_taxonomy

AI_ROOT = Path(__file__).resolve().parents[2]


def collect(config_dir: Path) -> dict[str, dict[str, list]]:
    """canonical_name -> {dataset_name -> [native keys mapped to it]}."""
    coverage: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for cfg_path in sorted(config_dir.glob("*.yaml")):
        if cfg_path.stem.endswith("example") or cfg_path.stem.endswith("template"):
            continue  # skip copy-me templates
        cfg = yaml.safe_load(cfg_path.read_text())
        if "class_map" not in cfg or "converter" not in cfg or cfg.get("template"):
            continue  # skip non-dataset yamls
        ds = cfg.get("name", cfg_path.stem)
        for native_key, canonical in cfg["class_map"].items():
            coverage[canonical][ds].append(native_key)
    return coverage


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Taxonomy merge-map / coverage report.")
    p.add_argument("--taxonomy", default=str(AI_ROOT / "configs/taxonomy.yaml"))
    p.add_argument("--data-config-dir", default=str(AI_ROOT / "configs/data"))
    args = p.parse_args(argv)

    taxonomy = load_taxonomy(args.taxonomy)
    coverage = collect(Path(args.data_config_dir))

    print(f"Canonical taxonomy v{taxonomy.version} — merge map\n" + "=" * 70)
    print(f"{'idx':>3}  {'class':<14}{'contributing datasets (native -> canonical)'}")
    for name in taxonomy.names_in_index_order():
        idx = taxonomy.index_of(name)
        contributors = coverage.get(name, {})
        if not contributors:
            tag, detail = "(unused)", ""
        else:
            tag = "(SHARED)" if len(contributors) > 1 else "(specific)"
            detail = ", ".join(
                f"{ds}{sorted(keys)}" for ds, keys in sorted(contributors.items())
            )
        print(f"{idx:>3}  {name:<14}{detail}  {tag}")

    # Problems: class_map targets not in the taxonomy.
    unknown = [c for c in coverage if c not in taxonomy.name_to_index]
    if unknown:
        print("\n!! class_map targets NOT in taxonomy (fix these):")
        for c in unknown:
            print(f"   {c!r} <- {dict(coverage[c])}")

    shared = [n for n, c in coverage.items() if len(c) > 1 and n in taxonomy.name_to_index]
    print(f"\nShared (merged) classes: {sorted(shared) or '(none)'}")
    print("Datasets seen:",
          sorted({ds for c in coverage.values() for ds in c}) or "(none)")


if __name__ == "__main__":
    main()
