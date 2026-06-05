# OralSkop — Commands Runbook

All commands run from the **`ai/`** directory.

```bash
cd ai
```

---

## 0. One-time setup

```bash
# Install the Python environment from the lockfile (reproducible).
uv sync
```

**On the GPU cluster**, after `uv sync`, install the CUDA-matched PyTorch build
(replace `cu124` with your cluster's CUDA version):

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Verify the install:

```bash
uv run python -c "import torch, ultralytics; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

---

## 1. Prepare the dataset(s)

Reconciles data-integrity issues, remaps native classes to the canonical taxonomy,
and writes a grouped train/val split + `data/<name>/data.yaml`.

```bash
# Single dataset
uv run python -m oralskop.data.prepare --datasets alphadent

# Merge several datasets into one training set (pooled under the shared taxonomy)
uv run python -m oralskop.data.prepare --datasets alphadent caries_roboflow --out-name merged
```

`scripts/prepare_alphadent.py` is a convenience wrapper for the AlphaDent-only case.

Output: `data/<name>/data.yaml` (+ `images/{train,val}`, `labels/{train,val}`).
Re-run this any time you change the taxonomy, a dataset config, or add a dataset.

---

## 2. Train

```bash
# Full training (defaults: yolo11m-seg, 960px, 100 epochs — see configs/train/yolo11_seg.yaml)
uv run python -m oralskop.train.train --config configs/train/yolo11_seg.yaml
```

Override any config key on the CLI with `--override key=value`:

```bash
# Quick CPU smoke test (no GPU)
uv run python -m oralskop.train.train --config configs/train/yolo11_seg.yaml \
    --override model=yolo11n-seg.pt epochs=1 imgsz=320 batch=2 device=cpu

# Pick GPU(s) / model size / image size
uv run python -m oralskop.train.train --config configs/train/yolo11_seg.yaml \
    --override device=0 model=yolo11x-seg.pt imgsz=960 batch=16

# Resume an interrupted run
uv run python -m oralskop.train.train --config configs/train/yolo11_seg.yaml \
    --override resume=true
```

Outputs (weights, curves, confusion matrix) land in `runs/segment/<name>/`.

---

## 3. Evaluate

```bash
uv run python -m oralskop.eval.evaluate \
    --weights runs/segment/yolo11m_seg_alphadent/weights/best.pt \
    --data data/alphadent/data.yaml
```

Prints per-class mask mAP50 / mAP50-95 and saves PR curves + confusion matrix.

---

## 3b. Visualize images + segmentation masks

**Read-only.** Opens a window and shows the sampled images **one by one** with their
masks overlaid and a **color legend** (class name + instance count), colored by
canonical class so colors are consistent across datasets. The dataset must be **built
first** (section 1) — this command only reads `data/<dataset>/`.

```bash
# Random N images from a built dataset (the simple interface)
uv run python -m oralskop.viz.visualize --dataset alphadent --num_imgs 12

# Restrict to a split and/or specific classes (4=calculus, 5=gingivitis)
uv run python -m oralskop.viz.visualize --dataset merged --num_imgs 20 --split val --classes 4,5

# Inspect a RAW (not-yet-built) dataset directly
uv run python -m oralskop.viz.visualize \
    --images-dir datasets/CariesRoboflow/train/images \
    --labels-dir datasets/CariesRoboflow/train/labels \
    --names-yaml datasets/CariesRoboflow/data.yaml --num_imgs 8
```

**Window keys:** `→`/`space` next · `←`/`a` previous · `s` save current frame · `q`/`esc` quit.

Flags: `--num_imgs`, `--split {train,val,test,all}` (default `all`), `--seed`,
`--classes`, `--alpha` (opacity), `--max-dim` (on-screen size).
On a headless machine add **`--save DIR`** to write overlays to a folder instead of
opening a window.

---

## 3c. Custom PyTorch training (non-YOLO, semantic segmentation)

A separate `torchseg` stack trains a **torchvision** segmentation model on any built
dataset (or a merged set) using a real `torch.utils.data.Dataset`. It rasterizes the
polygon labels into per-pixel masks (`0=background`, canonical class `c` → `c+1`), so it
predicts **semantic** segmentation (per-pixel class), not YOLO's instance masks. This is
independent of the Ultralytics path in section 2.

```bash
# Train on one dataset
uv run python -m oralskop.torchseg.train --config configs/train/seg_torch.yaml --datasets alphadent

# Train on a MERGED set (MergedSegDataset — pooled under the shared taxonomy)
uv run python -m oralskop.torchseg.train --config configs/train/seg_torch.yaml \
    --datasets alphadent bmc_oral_health

# CPU smoke test
uv run python -m oralskop.torchseg.train --config configs/train/seg_torch.yaml \
    --override arch=lraspp_mobilenet_v3_large pretrained=false imgsz=128 \
              batch=2 epochs=1 device=cpu limit_batches=3
```

Key config (`configs/train/seg_torch.yaml`): `arch`
(deeplabv3_resnet50 / deeplabv3_mobilenet_v3_large / fcn_resnet50 / lraspp_mobilenet_v3_large),
`imgsz`, `batch`, `lr`, `class_weights: auto` (median-frequency balancing for imbalance),
`limit_batches` (debug). Reports **val mIoU** + pixel accuracy; saves `best.pt`/`last.pt`
to `runs/seg/<name>/`. By default `exist_ok: false` **auto-increments** the run dir
(`<name>`, `<name>2`, `<name>3`, …) so a re-run never overwrites old checkpoints; set
`exist_ok=true` to reuse/overwrite `runs/seg/<name>/`.

The `MergedSegDataset` / `YoloSegDataset` classes (`oralskop/torchseg/dataset.py`) are
reusable on their own for custom loops, EDA, or the future SAM2 stage.

---

## 3d. Qualitatively test a torchseg model (raw | prediction | ground truth)

Samples N images, runs a trained checkpoint on them, and shows a three-panel comparison
per image — **raw image · model prediction · ground truth** — with a shared class legend
and per-image foreground mIoU.

**In a Jupyter notebook (Python cell — renders inline):**

```python
from oralskop.torchseg.test import predict_and_show
predict_and_show(
    weights="runs/seg/deeplabv3_alphadent/best.pt",
    datasets=["alphadent"],
    arch="deeplabv3_resnet50", imgsz=512, device="cuda",
    num_imgs=8, split="val",
)
```

The checkpoint stores its own `arch` / `class_names`, so those args are just fallbacks.
Inline display only works when called from a Python cell (the kernel process). A bare
`!python -m …` subprocess can't draw into the notebook — use `--save` for that:

**Headless / CLI (writes a PNG):**

```bash
uv run python -m oralskop.torchseg.test --datasets alphadent \
    --weights runs/seg/deeplabv3_alphadent/best.pt \
    --num_imgs 8 --split val --save runs/seg/test_preds
```

Args: `--datasets`, `--weights`, `--num_imgs` (alias `--num`), `--split`, `--seed`,
`--alpha` (overlay opacity), `--save DIR`, and `--override arch=… imgsz=… device=…`.
Without `--weights` it runs an untrained head (pipeline smoke test only).

---

## 4. Run on the cluster (SLURM + Apptainer)

```bash
# Build the container image once (on a node with fakeroot/sudo).
apptainer build oralskop.sif scripts/apptainer.def

# Submit the training job (edit partition/GPU directives in the file first).
sbatch scripts/train.sbatch

# Pass overrides through to train.py:
sbatch scripts/train.sbatch --override epochs=200 imgsz=1280
```

> Adjust `scripts/train.sbatch` (`--partition`, `--gres`, `--time`) and, if your
> cluster doesn't use Apptainer, the runner block, to match your environment.

---

## 5. Adding a NEW dataset

The pipeline is multi-dataset by design: every dataset maps its native classes onto
the shared canonical taxonomy. The model's output class count = the taxonomy size, so
it grows automatically as you add conditions — no manual head surgery.

### Merge policy (shared vs. specific classes)

`configs/taxonomy.yaml` is the shared class list; each dataset's `class_map` folds its
native classes onto canonical **names**. The rule when adding a dataset:

- **Similar class already in the taxonomy → map to it** (the classes merge). E.g. both
  AlphaDent and BMC map their decay classes to `caries`.
- **New/different condition → append a class** to `taxonomy.yaml` (keep existing indices,
  bump `version`) and map only that dataset to it (stays dataset-specific). E.g. `plaque`
  comes only from BMC; `abrasion/filling/crown` only from AlphaDent.
- "Similar but defined differently" is a judgement call — merge if the annotation
  guidelines match, otherwise append a distinct class (e.g. you *could* split AlphaDent's
  6 caries subtypes into their own classes instead of collapsing them to `caries`).

See the whole map (shared vs specific, plus mapping errors) at any time:

```bash
uv run python -m oralskop.data.coverage          # class merge map (shared vs specific)
uv run python -m oralskop.data.describe          # regenerate description.md (counts per dataset + merged)
```

Then build the merged set and audit it visually before trusting combined metrics:

```bash
uv run python -m oralskop.data.prepare --datasets alphadent bmc_oral_health --out-name merged
uv run python -m oralskop.viz.visualize --dataset merged --num_imgs 20
```

### Case A — Roboflow YOLO-seg export (config-only, no code)

The common case. Uses the built-in `roboflow_yoloseg` converter.

```bash
# 1. Export from Roboflow as "YOLOv8/YOLO11 (Instance Segmentation)" and unzip:
#    must contain data.yaml + train/valid/test with images/ + labels/
unzip CariesRoboflow.zip -d datasets/CariesRoboflow

# 2. (only if it introduces NEW conditions) append them to configs/taxonomy.yaml
#    at the end, keep existing indices stable, bump `version`.

# 3. Create configs/data/<name>.yaml from the template:
cp configs/data/roboflow_example.yaml configs/data/caries_roboflow.yaml
#    edit: raw_root + class_map (native Roboflow NAME -> canonical taxonomy name)

# 4. Build (single, or merged with AlphaDent):
uv run python -m oralskop.data.prepare --datasets caries_roboflow
uv run python -m oralskop.data.prepare --datasets alphadent caries_roboflow --out-name merged

# 5. Train on it:
uv run python -m oralskop.train.train --config configs/train/yolo11_seg.yaml \
    --override data=data/merged/data.yaml
```

### Downloading a dataset

Datasets go in `ai/datasets/<Name>/`. Some ship a download helper:

```bash
# BMC Oral Health 2024 (Google Drive, ~6.7k loose files -> images/ + lables/)
uv run --with gdown python scripts/download_bmc.py     # re-run to resume if throttled
```

Google Drive rate-limits loose-file folder pulls; if `gdown` keeps stalling, download
the folder from the browser (right-click → Download = server-side zip) and unzip into
`datasets/<Name>/`. The BMC `class_map` (mask value → class) still needs a one-time
visual confirmation — see `configs/data/bmc_oral_health.yaml`.

### Case B — A different format (COCO JSON, masks, VOC, …)

Needs a small one-time converter, then it's config-only forever after:

1. Add `oralskop/data/converters/<name>.py` implementing the `Converter` protocol
   (`records()` yielding `SampleRecord`s with canonical label lines + a grouping key;
   `verify()` returning `VerifyReport`s). Reuse `base.remap_label_text` for YOLO-seg.
2. Register it in `oralskop/data/converters/registry.py` (`REGISTRY["<name>"] = ...`).
3. Add `configs/data/<name>.yaml` with `converter: <name>` + paths + `class_map`.
4. Build & train exactly as in Case A steps 4–5.

---

## Quick reference

| Task                     | Command |
|--------------------------|---------|
| Install env              | `uv sync` |
| Prepare (single)         | `uv run python -m oralskop.data.prepare --datasets alphadent` |
| Prepare (merged)         | `uv run python -m oralskop.data.prepare --datasets alphadent caries_roboflow --out-name merged` |
| Train YOLO (full)        | `uv run python -m oralskop.train.train --config configs/train/yolo11_seg.yaml` |
| Train YOLO (smoke, CPU)  | `… --override model=yolo11n-seg.pt epochs=1 imgsz=320 device=cpu` |
| Train torch seg (merged) | `uv run python -m oralskop.torchseg.train --config configs/train/seg_torch.yaml --datasets alphadent bmc_oral_health` |
| Test torch seg (visual)  | `uv run python -m oralskop.torchseg.test --datasets alphadent --weights <best.pt> --num_imgs 8 --save runs/seg/test_preds` |
| Evaluate                 | `uv run python -m oralskop.eval.evaluate --weights <best.pt> --data data/alphadent/data.yaml` |
| Visualize masks          | `uv run python -m oralskop.viz.visualize --dataset alphadent --num_imgs 12` |
| Build container          | `apptainer build oralskop.sif scripts/apptainer.def` |
| Submit cluster job       | `sbatch scripts/train.sbatch` |
