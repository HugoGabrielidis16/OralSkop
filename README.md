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

## Documentation

- [`ai/README.md`](ai/README.md) — AI package overview and layout.
- [`ai/DATASETS.md`](ai/DATASETS.md) — the master manifest: 26 source datasets, 103,878 images, canonical taxonomy.
- [`ai/COMMANDS.md`](ai/COMMANDS.md) — how to prepare data, train, and add a new dataset.
- [`ai/training_plan.md`](ai/training_plan.md) — modeling roadmap.
