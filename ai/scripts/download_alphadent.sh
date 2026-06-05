#!/usr/bin/env bash
# Download the AlphaDent zip parts from S3 and unzip them into ai/datasets/AlphaDent/,
# reproducing the exact layout we have locally:
#     AlphaDent/images/{train,valid,test}  +  AlphaDent/labels/{train,valid}
#
# Each zip carries the same top-level `AlphaDent/` prefix, so extracting all of them
# into datasets/ merges into one AlphaDent/ folder (overwriting collapses the ~175
# filenames that overlap between the older and newer downloads -> 1237 unique train).
#
# Usage (from a notebook cell, after cloning the repo):
#     !bash ai/scripts/download_alphadent.sh
#     !bash ai/scripts/download_alphadent.sh s3://my-other-bucket    # override bucket
#
# On a SageMaker notebook the instance's IAM role provides S3 credentials, so no
# `aws configure` is needed. Requires the AWS CLI and `unzip` (both preinstalled on
# SageMaker notebook instances).

set -euo pipefail

# S3 bucket holding the 4 AlphaDent-*.zip parts (override as arg 1).
BUCKET="${1:-s3://alphadent}"

# Resolve the ai/ project dir from this script's location (ai/scripts/<this>.sh).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_DIR="$(dirname "$SCRIPT_DIR")"
DATASETS_DIR="$AI_DIR/datasets"
ZIP_DIR="$DATASETS_DIR/zip"

mkdir -p "$ZIP_DIR"

# 1. Pull every .zip from the bucket into datasets/zip/.
echo ">> Downloading *.zip from $BUCKET -> $ZIP_DIR ..."
aws s3 cp "$BUCKET" "$ZIP_DIR" --recursive --exclude "*" --include "*.zip"

if ! ls "$ZIP_DIR"/*.zip >/dev/null 2>&1; then
    echo "!! No .zip files found in $BUCKET — check the bucket name/contents." >&2
    exit 1
fi

# 2. Extract all parts into datasets/ (they share the AlphaDent/ root, so they merge).
echo ">> Extracting into $DATASETS_DIR (merges into AlphaDent/) ..."
for z in "$ZIP_DIR"/*.zip; do
    echo "   - $(basename "$z")"
    unzip -o -q "$z" -d "$DATASETS_DIR"
done

# 3. Reclaim space — the zips are ~5.4 GB and no longer needed.
echo ">> Removing downloaded archives ..."
rm -rf "$ZIP_DIR"

# 4. Report the resulting layout.
echo ">> Done. AlphaDent layout under $DATASETS_DIR/AlphaDent:"
for d in images/train images/valid images/test labels/train labels/valid; do
    n=$(find "$DATASETS_DIR/AlphaDent/$d" -type f 2>/dev/null | wc -l | tr -d ' ')
    printf "   %-16s %s files\n" "$d" "$n"
done

echo ""
echo ">> Next: build the canonical dataset with"
echo "   cd $AI_DIR && uv run python -m oralskop.data.prepare --datasets alphadent"
