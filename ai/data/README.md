# `data/` — processed / built datasets (not version-controlled)

This folder holds the **canonical YOLO-seg datasets** materialized by the prepare
pipeline (`python -m oralskop.data.prepare ...`), each with its own `data.yaml`.
The contents are **regenerated from `datasets/`** and are therefore large and
**gitignored** — only this README and the per-dataset placeholder READMEs are
tracked so the structure is visible in GitHub.

## Regenerating

```bash
cd ai
uv run python -m oralskop.data.prepare --datasets alphadent
uv run python -m oralskop.data.prepare --datasets alphadent bmc_oral_health --out-name merged
```

Anything materialized here (besides `README.md` files) stays local and is never committed.
