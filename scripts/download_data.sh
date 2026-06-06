#!/usr/bin/env bash
# Download the AlphaDent dataset from S3 into ai/dataset/AlphaDent/.
# Run once after git clone, from the project root:
#
#   bash scripts/download_data.sh
#   bash scripts/download_data.sh s3://my-other-bucket/OralSkop/dataset/AlphaDent
#
# On SageMaker the IAM role attached to the instance gives S3 access automatically.

set -euo pipefail

S3_URI="${1:-s3://YOUR-BUCKET/OralSkop/dataset/AlphaDent}"
DEST="ai/dataset/AlphaDent"

echo "Downloading AlphaDent dataset from ${S3_URI} → ${DEST}/"
mkdir -p "${DEST}"
aws s3 sync "${S3_URI}" "${DEST}/"
echo "Done. $(find "${DEST}" -type f | wc -l) files downloaded."
