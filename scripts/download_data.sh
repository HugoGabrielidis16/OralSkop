#!/usr/bin/env bash
# Push the AlphaDent dataset from this machine to a remote SageMaker instance.
# Run from the PROJECT ROOT on your local machine (not on SageMaker):
#
#   bash scripts/push_data.sh ubuntu@<sagemaker-ip>
#   bash scripts/push_data.sh ec2-user@<sagemaker-ip> ~/.ssh/my-key.pem
#
# Arguments:
#   $1  user@host  — SSH target of your SageMaker instance (required)
#   $2  key_path   — path to your .pem key (optional; omit if using ssh-agent)

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: bash scripts/push_data.sh user@host [/path/to/key.pem]"
    exit 1
fi

REMOTE="$1"
KEY_OPT=""
[[ $# -ge 2 ]] && KEY_OPT="-e \"ssh -i $2\""

SRC="ai/dataset/AlphaDent/"
DEST="${REMOTE}:~/OralSkop/ai/dataset/AlphaDent/"

echo "Syncing ${SRC} → ${DEST}"
eval rsync -avz --progress ${KEY_OPT} "${SRC}" "${DEST}"
echo "Done."
