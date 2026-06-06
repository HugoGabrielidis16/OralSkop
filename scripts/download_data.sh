#!/usr/bin/env bash
# Download and unpack the AlphaDent dataset from S3.
# Run from the project root on any machine (local or SageMaker):
#
#   bash scripts/download_data.sh s3://your-bucket/path/to/zips/
#
# The script downloads every AlphaDent*.zip from that S3 prefix,
# unpacks them all, and assembles the dataset under ai/dataset/AlphaDent/.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: bash scripts/download_data.sh s3://your-bucket/prefix/"
    echo "Example: bash scripts/download_data.sh s3://oralskop-data/AlphaDent/"
    exit 1
fi

S3_PREFIX="${1%/}"   # strip trailing slash if any
TMP_DIR="$(mktemp -d)"
DEST="ai/dataset/AlphaDent"

echo "==> Downloading zips from ${S3_PREFIX}/ ..."
aws s3 sync "${S3_PREFIX}/" "${TMP_DIR}/" --exclude "*" --include "AlphaDent*.zip"

ZIP_COUNT=$(find "${TMP_DIR}" -name "AlphaDent*.zip" | wc -l)
if [[ ${ZIP_COUNT} -eq 0 ]]; then
    echo "ERROR: No AlphaDent*.zip files found at ${S3_PREFIX}/"
    exit 1
fi
echo "    Found ${ZIP_COUNT} zip file(s)."

echo "==> Unpacking into ${DEST}/ ..."
mkdir -p "${DEST}"
for zip in "${TMP_DIR}"/AlphaDent*.zip; do
    echo "    unzipping $(basename "${zip}") ..."
    unzip -q -o "${zip}" -d "${TMP_DIR}/extracted"
done

# The zips may unpack into a nested folder — move contents up if needed.
EXTRACTED="${TMP_DIR}/extracted"
INNER=$(find "${EXTRACTED}" -mindepth 1 -maxdepth 1 -type d | head -1)
if [[ -n "${INNER}" && $(find "${EXTRACTED}" -mindepth 1 -maxdepth 1 | wc -l) -eq 1 ]]; then
    # single top-level folder — merge its contents directly into DEST
    cp -r "${INNER}/." "${DEST}/"
else
    cp -r "${EXTRACTED}/." "${DEST}/"
fi

rm -rf "${TMP_DIR}"

echo "==> Done. Dataset at ${DEST}/"
find "${DEST}" -type f | wc -l | xargs -I{} echo "    {} files total."
