#!/usr/bin/env python3
"""Download the manifest-referenced images from the `datastoraged4gen` S3 bucket
into ``ai/datasets/02_PROCESSED/`` and place the manifest where training expects it,
so ``oralskop.clf`` can train from local disk.

By default this downloads only the **supervised** images (the train/valid/test rows)
and skips the ~48k unlabelled MetaDent ``pretrain`` images — keeping the footprint to
~55k files. Existing files are skipped, so it is **resumable**: re-run it after an
interruption and it only fetches what's missing.

Run it from the AWS Jupyter notebook (the instance's IAM role provides S3 creds — no
`aws configure` needed). Examples:

    # from the repo root
    !python ai/scripts/download_data.py
    # only some category folders (saves space)
    !python ai/scripts/download_data.py --prefixes CARIES/ GINGIVITE/ OPMD_OSCC/
    # include the pretrain corpus too
    !python ai/scripts/download_data.py --all
    # quick check without downloading
    !python ai/scripts/download_data.py --dry-run
    # tiny smoke test (first 200 images)
    !python ai/scripts/download_data.py --limit 200

Then launch training (run from ai/):

    uv run python -m oralskop.clf.train --config configs/clf/manifest_clf.yaml \
        --override image_root=datasets/02_PROCESSED device=cuda

Requires pandas (the `clf` extra) and boto3 (preinstalled on SageMaker). The manifest
is read from ``ai/manifest_03_master_FINAL.csv`` if present, otherwise pulled from
``s3://<bucket>/manifest_03_master_FINAL.csv`` (override with ``--manifest``).
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

# ai/ is the parent of this script's scripts/ directory.
AI_DIR = Path(__file__).resolve().parents[1]

_PRETRAIN_FORMAT = "unlabeled-pretraining"
_PRETRAIN_SPLIT = "pretrain"
_MANIFEST_NAME = "manifest_03_master_FINAL.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bucket", default="datastoraged4gen", help="S3 bucket name (default: %(default)s).")
    p.add_argument("--src-prefix", default="02_PROCESSED",
                   help="Key prefix in the bucket that image_path is relative to (default: %(default)s).")
    p.add_argument("--dest", default=str(AI_DIR / "datasets" / "02_PROCESSED"),
                   help="Local destination for the images (default: ai/datasets/02_PROCESSED).")
    p.add_argument("--manifest", default=str(AI_DIR / _MANIFEST_NAME),
                   help="Manifest path: a local CSV, or an s3:// URI to fetch. "
                        "If the local default is missing, falls back to s3://<bucket>/<name>.")
    p.add_argument("--all", action="store_true",
                   help="Also download the unlabelled MetaDent pretrain images (default: supervised only).")
    p.add_argument("--prefixes", nargs="+", default=None, metavar="PREFIX",
                   help="Restrict to image_paths starting with these prefixes, e.g. CARIES/ GINGIVITE/.")
    p.add_argument("--limit", type=int, default=None, help="Only the first N images (smoke test).")
    p.add_argument("--workers", type=int, default=32, help="Parallel download threads (default: %(default)s).")
    p.add_argument("--dry-run", action="store_true", help="Report what would be downloaded; fetch nothing.")
    return p.parse_args(argv)


def _s3_client(bucket: str):
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - env-dependent
        sys.exit("boto3 is required to download. On SageMaker it's preinstalled; "
                 "otherwise `pip install boto3`.\n" + str(exc))
    return boto3.client("s3")


def resolve_manifest(manifest: str, bucket: str) -> Path:
    """Return a local path to the manifest, fetching it from S3 if needed."""
    if manifest.startswith("s3://"):
        dest = AI_DIR / _MANIFEST_NAME
        _, _, rest = manifest.partition("s3://")
        man_bucket, _, key = rest.partition("/")
        print(f">> Fetching manifest {manifest} -> {dest}")
        _s3_client(man_bucket).download_file(man_bucket, key, str(dest))
        return dest
    local = Path(manifest)
    if local.exists():
        return local
    # Default local path missing — try the bucket root.
    dest = AI_DIR / _MANIFEST_NAME
    key = _MANIFEST_NAME
    print(f">> {local} not found; fetching s3://{bucket}/{key} -> {dest}")
    try:
        _s3_client(bucket).download_file(bucket, key, str(dest))
    except Exception as exc:
        sys.exit(f"Could not locate the manifest. Pass --manifest <local.csv|s3://...>.\n{exc}")
    return dest


def select_image_paths(manifest_path: Path, *, include_pretrain: bool,
                       prefixes: list[str] | None, limit: int | None) -> list[str]:
    """Read the manifest and return the unique image_paths to download."""
    df = pd.read_csv(manifest_path, usecols=["image_path", "split", "annotation_format"],
                     dtype=str, keep_default_na=False)
    if not include_pretrain:
        df = df[(df["annotation_format"].str.strip() != _PRETRAIN_FORMAT)
                & (df["split"].str.strip() != _PRETRAIN_SPLIT)]
    if prefixes:
        pref = tuple(p.lstrip("/") for p in prefixes)
        df = df[df["image_path"].str.lstrip("/").str.startswith(pref)]
    paths = sorted(dict.fromkeys(df["image_path"]))  # unique, stable order
    if limit:
        paths = paths[:limit]
    return paths


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    dest_root = Path(args.dest)
    src_prefix = args.src_prefix.strip("/")

    manifest_path = resolve_manifest(args.manifest, args.bucket)
    paths = select_image_paths(manifest_path, include_pretrain=args.all,
                               prefixes=args.prefixes, limit=args.limit)
    print(f">> Manifest: {manifest_path}")
    print(f">> {len(paths)} images selected "
          f"({'all incl. pretrain' if args.all else 'supervised only'}"
          f"{'' if not args.prefixes else f', prefixes={args.prefixes}'}"
          f"{'' if not args.limit else f', limit={args.limit}'})")
    print(f">> Source : s3://{args.bucket}/{src_prefix}/<image_path>")
    print(f">> Dest   : {dest_root}/<image_path>")

    if args.dry_run:
        for ex in paths[:5]:
            print(f"   e.g. s3://{args.bucket}/{src_prefix}/{ex}  ->  {dest_root / ex}")
        print(">> Dry run — nothing downloaded.")
        return

    client = _s3_client(args.bucket)

    def fetch(rel: str) -> tuple[str, str]:
        local = dest_root / rel
        if local.exists() and local.stat().st_size > 0:
            return ("skip", rel)
        local.parent.mkdir(parents=True, exist_ok=True)
        try:
            client.download_file(args.bucket, f"{src_prefix}/{rel.lstrip('/')}", str(local))
            return ("ok", rel)
        except Exception as exc:  # noqa: BLE001 - report and continue
            return ("fail", f"{rel}\t{exc}")

    ok = skipped = failed = 0
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, rel) for rel in paths]
        for i, fut in enumerate(as_completed(futures), 1):
            status, info = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skip":
                skipped += 1
            else:
                failed += 1
                if len(failures) < 10:
                    failures.append(info)
            if i % 1000 == 0 or i == len(futures):
                print(f"   {i}/{len(futures)}  (downloaded {ok}, skipped {skipped}, failed {failed})")

    print(f"\n>> Done. downloaded={ok} skipped={skipped} failed={failed} -> {dest_root}")
    if failures:
        print(">> First failures (key\\terror):")
        for f in failures:
            print(f"   {f}")
        print("   A few missing keys are expected if you only have some folders; many "
              "failures usually mean an S3 permission/KMS issue — see the IAM steps.")
    print("\nNext — train from local disk (run from ai/):")
    print("   uv run python -m oralskop.clf.train --config configs/clf/manifest_clf.yaml \\")
    print(f"       --override image_root={Path(args.dest).name if Path(args.dest).is_absolute() else args.dest} "
          "device=cuda")
    print("   (image_root above assumes you run from ai/; use the full path if not: "
          f"image_root={dest_root})")


if __name__ == "__main__":
    main()
