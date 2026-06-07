# OralSkop

AI-assisted **dental screening from phone / intraoral RGB photos**. A user snaps a
photo, the models detect and classify dental conditions (caries, mucosal lesions,
periodontal disease, …), and the result is surfaced through a web app with an
optional LLM chat explainer.

The repo is three layers: **`ai/`** trains the models, **`backend/`** serves
predictions over an API, and **`frontend-next/`** is the user-facing app.

## Repository layout

| Folder | What it does |
|---|---|
| `ai/` | ML research & training monorepo — data pipeline, model training (classification / detection / segmentation), evaluation, and a serving layer. The bulk of the project. |
| `backend/` | FastAPI service exposing the app API: auth, screening upload + inference, history, and LLM chat. |
| `frontend-next/` | Next.js + Tailwind web app — the production frontend (patient & dentist views). |
| `frontend/` | Early static-HTML prototype (`index.html`, `dentist.html`) kept for reference. |
| `models/` | Standalone top-level model training entry (`train.py` + `dataset.yaml`). |

## Inside `ai/`

| Path | What it does |
|---|---|
| `configs/` | YAML configs: `taxonomy.yaml` (canonical classes) plus `data/`, `clf/`, `det/`, `train/` model + dataset configs. |
| `oralskop/` | The Python package (installed via `pyproject.toml` / `uv`). |
| `oralskop/data/` | Dataset pipeline: verify → split → build, taxonomy mapping, coverage & describe tools, plus format `converters/`. |
| `oralskop/clf/` | Multi-label classifier (DINOv2 + QLoRA): dataset, model, train, eval, metrics, vocab. |
| `oralskop/det/` | Object detector (DINOv2-backbone DETR): dataset, model, train, eval, metrics. |
| `oralskop/torchseg/` | Semantic segmentation (SegFormer / DINOv2) with LoRA, losses, and test tooling. |
| `oralskop/train/`, `eval/`, `bench/`, `viz/` | Generic training entry, evaluation metrics, benchmarking, and visualization. |
| `oralskop/serve/` | FastAPI/Bedrock serving + `/chat` LLM layer used by `backend/`. |
| `scripts/` | Data download (`download_data.py`) and environment/bootstrap helpers. |
| `notebooks/` | Exploration & training notebooks. |
| `datasets/`, `data/` | Raw inputs and processed/built datasets (bulk content gitignored). |

## Documentation

- [`ai/README.md`](ai/README.md) — AI package overview and layout.
- [`ai/DATASETS.md`](ai/DATASETS.md) — the master manifest: 26 source datasets, 103,878 images, canonical taxonomy.
- [`ai/COMMANDS.md`](ai/COMMANDS.md) — how to prepare data, train, and add a new dataset.
- [`ai/training_plan.md`](ai/training_plan.md) — modeling roadmap.
