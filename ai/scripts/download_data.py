#!/usr/bin/env python3
"""Download manifest images from S3 into the local training layout.

This stages images from ``s3://datastoraged4gen/02_PROCESSED`` into
``ai/datasets/02_PROCESSED`` and keeps ``manifest_03_master_FINAL.csv`` in
``ai/``. That lets the AWS notebook train from local EBS instead of repeatedly
streaming images from S3.

By default this downloads only supervised train/valid/test rows and skips the
unlabelled MetaDent pretrain corpus. Existing non-empty files are skipped, so
the script is resumable.

Notebook examples:

    !python ai/scripts/download_data.py --dry-run
    !python ai/scripts/download_data.py --limit 200
    !python ai/scripts/download_data.py
    !python ai/scripts/download_data.py --all

Then train from ``ai/``:

    uv run python -m oralskop.clf.train --config configs/clf/manifest_clf.yaml \\
        --override image_root=datasets/02_PROCESSED device=cuda
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

AI_DIR = Path(__file__).resolve().parents[1]

_MANIFEST_NAME = "manifest_03_master_FINAL.csv"
_PRETRAIN_FORMAT = "unlabeled-pretraining"
_PRETRAIN_SPLIT = "pretrain"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--bucket",
        default="datastoraged4gen",
        help="S3 bucket name (default: %(default)s).",
    )
    p.add_argument(
        "--src-prefix",
        default="02_PROCESSED",
        help="Bucket key prefix that image_path is relative to (default: %(default)s).",
    )
    p.add_argument(
        "--dest",
        default=str(AI_DIR / "datasets" / "02_PROCESSED"),
        help="Local image destination (default: ai/datasets/02_PROCESSED).",
    )
    p.add_argument(
        "--manifest",
        default=str(AI_DIR / _MANIFEST_NAME),
        help=(
            "Manifest path: local CSV or s3:// URI. If the default local path is "
            "missing, fetches s3://<bucket>/manifest_03_master_FINAL.csv."
        ),
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Also download unlabelled pretrain images (default: supervised only).",
    )
    p.add_argument(
        "--with-labels",
        action="store_true",
        help="Also download the yolo-bbox .txt label files (for oralskop.det detection).",
    )
    p.add_argument(
        "--prefixes",
        nargs="+",
        metavar="PREFIX",
        help="Restrict to image_path prefixes, e.g. CARIES/ GINGIVITE/ OPMD_OSCC/.",
    )
    p.add_argument("--limit", type=int, help="Only download the first N selected images.")
    p.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Parallel download threads (default: %(default)s).",
    )
    p.add_argument("--dry-run", action="store_true", help="Print what would happen.")
    return p.parse_args(argv)


def s3_client():
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - environment-dependent
        sys.exit(
            "boto3 is required. On SageMaker it is usually preinstalled; "
            "otherwise install it with `pip install boto3`.\n"
            f"{exc}"
        )
    return boto3.client("s3")


def resolve_manifest(manifest: str, bucket: str) -> Path:
    """Return a local manifest path, downloading it from S3 when needed."""
    client = None
    if manifest.startswith("s3://"):
        _, _, rest = manifest.partition("s3://")
        man_bucket, _, key = rest.partition("/")
        dest = AI_DIR / Path(key).name
        print(f">> Fetching manifest {manifest} -> {dest}")
        s3_client().download_file(man_bucket, key, str(dest))
        return dest

    local = Path(manifest)
    if local.exists():
        return local

    dest = AI_DIR / _MANIFEST_NAME
    key = _MANIFEST_NAME
    print(f">> {local} not found; fetching s3://{bucket}/{key} -> {dest}")
    try:
        client = client or s3_client()
        client.download_file(bucket, key, str(dest))
    except Exception as exc:  # noqa: BLE001 - turn into a clear CLI failure
        sys.exit(f"Could not locate the manifest. Pass --manifest <local.csv|s3://...>.\n{exc}")
    return dest


def select_image_paths(
    manifest_path: Path,
    *,
    include_pretrain: bool,
    prefixes: list[str] | None,
    limit: int | None,
) -> list[str]:
    """Read the manifest and return unique relative image paths."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment-dependent
        sys.exit(
            "pandas is required to read the manifest. In this repo, run with "
            "`uv run --extra clf python scripts/download_data.py`, or install "
            "pandas in the active notebook environment.\n"
            f"{exc}"
        )

    df = pd.read_csv(
        manifest_path,
        usecols=["image_path", "split", "annotation_format"],
        dtype=str,
        keep_default_na=False,
    )

    if not include_pretrain:
        df = df[
            (df["annotation_format"].str.strip() != _PRETRAIN_FORMAT)
            & (df["split"].str.strip() != _PRETRAIN_SPLIT)
        ]

    if prefixes:
        normalized = tuple(p.lstrip("/") for p in prefixes)
        df = df[df["image_path"].str.lstrip("/").str.startswith(normalized)]

    paths = sorted(dict.fromkeys(p.lstrip("/") for p in df["image_path"] if p.strip()))
    if limit:
        paths = paths[: int(limit)]
    return paths


def select_label_paths(manifest_path: Path, *, prefixes: list[str] | None, limit: int | None) -> list[str]:
    """Unique yolo-bbox ``.txt`` label paths (for the detection subset)."""
    import pandas as pd

    df = pd.read_csv(manifest_path, usecols=["image_path", "label_path", "split", "annotation_format"],
                     dtype=str, keep_default_na=False)
    df = df[df["annotation_format"].str.contains("yolo-bbox")
            & df["label_path"].str.strip().str.endswith(".txt")
            & (df["split"].str.strip() != _PRETRAIN_SPLIT)]
    if prefixes:
        normalized = tuple(p.lstrip("/") for p in prefixes)
        df = df[df["image_path"].str.lstrip("/").str.startswith(normalized)]
    paths = sorted(dict.fromkeys(p.lstrip("/") for p in df["label_path"] if p.strip()))
    if limit:
        paths = paths[: int(limit)]
    return paths


def make_key(src_prefix: str, rel: str) -> str:
    prefix = src_prefix.strip("/")
    rel = rel.lstrip("/")
    return f"{prefix}/{rel}" if prefix else rel


def image_root_override(dest_root: Path) -> str:
    """Return the image_root value to use when training from ai/."""
    try:
        return str(dest_root.resolve().relative_to(AI_DIR.resolve()))
    except ValueError:
        return str(dest_root.resolve())


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    dest_root = Path(args.dest)
    src_prefix = args.src_prefix.strip("/")

    manifest_path = resolve_manifest(args.manifest, args.bucket)
    paths = select_image_paths(
        manifest_path,
        include_pretrain=args.all,
        prefixes=args.prefixes,
        limit=args.limit,
    )

    label_paths = []
    if args.with_labels:
        label_paths = select_label_paths(manifest_path, prefixes=args.prefixes, limit=args.limit)
        paths = paths + label_paths  # same root; fetch() handles both image + label rels

    selection = "all incl. pretrain" if args.all else "supervised only"
    if args.prefixes:
        selection += f", prefixes={args.prefixes}"
    if args.limit:
        selection += f", limit={args.limit}"
    if args.with_labels:
        selection += f", +{len(label_paths)} bbox label files"

    print(f">> Manifest: {manifest_path}")
    print(f">> Selected: {len(paths)} files ({selection})")
    print(f">> Source  : s3://{args.bucket}/{src_prefix}/<image_path>")
    print(f">> Dest    : {dest_root}/<image_path>")

    if args.dry_run:
        for rel in paths[:5]:
            print(f"   e.g. s3://{args.bucket}/{make_key(src_prefix, rel)} -> {dest_root / rel}")
        print(">> Dry run: nothing downloaded.")
        return

    client = s3_client()

    def fetch(rel: str) -> tuple[str, str]:
        local = dest_root / rel
        if local.exists() and local.stat().st_size > 0:
            return ("skip", rel)
        local.parent.mkdir(parents=True, exist_ok=True)
        try:
            client.download_file(args.bucket, make_key(src_prefix, rel), str(local))
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
                print(f"   {i}/{len(futures)} (downloaded {ok}, skipped {skipped}, failed {failed})")

    print(f"\n>> Done. downloaded={ok} skipped={skipped} failed={failed} -> {dest_root}")
    if failures:
        print(">> First failures (image_path\\terror):")
        for failure in failures:
            print(f"   {failure}")
        print(">> Many failures usually mean an S3 permission, region, KMS, or prefix issue.")

    root_override = image_root_override(dest_root)
    print("\nNext, train from ai/:")
    print("   uv run python -m oralskop.clf.train --config configs/clf/manifest_clf.yaml \\")
    print(f"       --override image_root={root_override} device=cuda")


if __name__ == "__main__":
    main()
