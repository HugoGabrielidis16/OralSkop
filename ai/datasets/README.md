# `datasets/` — raw datasets (not version-controlled)

This folder holds the **raw, downloaded** datasets in their original on-disk
layout. The bulk contents are large and **gitignored** — only this README and the
per-dataset placeholder READMEs are tracked, so the folder structure is visible in
GitHub without pushing gigabytes of images/labels.

`raw_root` in each `configs/data/<name>.yaml` is resolved relative to `ai/` and
points into this folder.

## Obtaining the data

- **AlphaDent** — `scripts/download_alphadent.sh` (fetch + unzip from S3).
- Other datasets — see `ai/DATASETS.md` for sources and `ai/COMMANDS.md` for the runbook.

Anything you drop in here (besides `README.md` files) stays local and is never committed.
