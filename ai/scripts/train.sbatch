#!/bin/bash
#SBATCH --job-name=oralskop-yolo11seg
#SBATCH --partition=gpu              # <-- adjust to your cluster's GPU partition
#SBATCH --gres=gpu:1                 # request 1 GPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm-%j.out
#SBATCH --error=runs/slurm-%j.err

# SLURM launcher for OralSkop YOLO11-seg training.
# Assumes SLURM + Apptainer/Singularity (most HPC). If your cluster uses bare
# modules/conda instead, replace the apptainer block with `module load` + `uv run`.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs

CONFIG="${CONFIG:-configs/train/yolo11_seg.yaml}"
CONTAINER="${CONTAINER:-oralskop.sif}"

if command -v apptainer >/dev/null 2>&1; then
    RUNNER="apptainer"
elif command -v singularity >/dev/null 2>&1; then
    RUNNER="singularity"
else
    RUNNER=""
fi

if [[ -n "$RUNNER" && -f "$CONTAINER" ]]; then
    echo "Running inside $RUNNER container: $CONTAINER"
    "$RUNNER" exec --nv "$CONTAINER" \
        uv run python -m oralskop.train.train --config "$CONFIG" "$@"
else
    echo "No container found ($CONTAINER); falling back to local uv environment."
    uv run python -m oralskop.train.train --config "$CONFIG" "$@"
fi
