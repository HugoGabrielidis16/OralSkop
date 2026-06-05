# OralSkop — AI / Training

Dental-pathology **instance segmentation** from RGB photos (phone / intraoral camera).

This package fine-tunes **Ultralytics YOLO11-seg**. It is built to ingest **many
datasets** over time under a single **canonical taxonomy** — AlphaDent is dataset #1.

## Layout

```
ai/
  datasets/                # raw inputs (AlphaDent + future datasets; gitignored)
  configs/
    taxonomy.yaml          # canonical dental-condition classes (versioned)
    data/alphadent.yaml    # AlphaDent raw paths + native->canonical class map
    train/yolo11_seg.yaml  # model + hyperparameters
  src/oralskop/
    data/                  # verify, split, build (-> data.yaml), prepare CLI
      converters/          # base + registry + alphadent + roboflow_yoloseg
    train/train.py         # config-driven Ultralytics training entrypoint
    eval/evaluate.py       # per-class segmentation metrics
  scripts/
    prepare_alphadent.py   # build canonical dataset from datasets/AlphaDent
    train.sbatch           # SLURM job
    apptainer.def          # HPC container definition
  data/                    # processed canonical datasets (gitignored)
```

## Quickstart

```bash
cd ai
uv sync                                   # CPU resolution; see CUDA note below
uv run python scripts/prepare_alphadent.py    # verify + convert + split -> data/alphadent/data.yaml
uv run python -m oralskop.train.train --config configs/train/yolo11_seg.yaml \
    --override epochs=1 device=cpu batch=2    # smoke test
uv run python -m oralskop.eval.evaluate --weights runs/<run>/weights/best.pt \
    --data data/alphadent/data.yaml
```

### GPU / cluster

`uv sync` resolves a CPU-capable torch. On the GPU cluster install the matching CUDA
build, e.g.:

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Full training is launched via `sbatch scripts/train.sbatch` (SLURM + Apptainer assumed —
adjust to your cluster's scheduler/runtime).

## Dataset notes (AlphaDent)

- YOLO instance-seg format; **9 native classes**: `0=Abrasion, 1=Filling, 2=Crown,
  3..8 = Caries` (6 subtypes). Source: the AlphaDent repo's `convert_9_classes_to_4_classes`
  script ([ZFTurbo/AlphaDent](https://github.com/ZFTurbo/AlphaDent)). Exact caries-subtype
  names are unconfirmed and treated as `caries` variants — adjust in `configs/data/alphadent.yaml`.
- The shipped copy has **integrity issues** (orphan labels, images without labels) and **no
  usable val set** (test split has 4 images, no labels). `prepare_alphadent.py` reconciles
  orphans and builds a **patient-grouped** train/val split (grouped by `pNNN` patient id).

## Roadmap — Stage B (deferred)

For sharper mask boundaries, a **YOLO box → SAM2 (optionally QLoRA) mask-refinement** stage
can be added later. SAM2 is preferred over MCP-MedSAM (whose pretraining is radiology
modalities, not intraoral RGB). Not implemented yet — YOLO11-seg is the automatic,
multi-class foundation.
