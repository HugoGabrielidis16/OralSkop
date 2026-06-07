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

Optional exploration stack for Albumentations, SMP architectures, and LoRA:

```bash
uv sync --extra explore
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

A separate `torchseg` stack trains a **torchvision / SMP / DINOv2** segmentation model on
any built dataset (or a merged set) using a real `torch.utils.data.Dataset`. It rasterizes
the polygon labels into per-pixel masks (`0=background`, canonical class `c` -> `c+1`), so
it predicts **semantic** segmentation (per-pixel class), not YOLO's instance masks. This is
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

# DINOv2 semantic segmentation on AlphaDent
uv run --extra explore python -m oralskop.torchseg.train \
    --config configs/train/dinov2_seg.yaml
```

Key config (`configs/train/seg_torch.yaml`): `arch`
(deeplabv3_resnet50 / deeplabv3_mobilenet_v3_large / fcn_resnet50 /
lraspp_mobilenet_v3_large / unet / deeplabv3plus_resnet50 /
deeplabv3plus_efficientnet-b4 / segformer_mit_b2 / segformer_mit_b3 /
dinov2_small / dinov2_base / dinov2_large / `hf:<model_id>`),
`imgsz`, `batch`, `lr`, `class_weights: auto` (median-frequency balancing for imbalance),
`loss` (ce / ce_dice / focal / focal_dice / lovasz), `optimizer` (adamw / sgd),
`scheduler` (none / cosine / poly), `warmup_epochs`, `aug` (none / flip / light / strong),
`grad_checkpointing`, `lora`, and `limit_batches` (debug). Reports **val fg_mIoU** +
mIoU + dice + pixel accuracy; saves `best.pt`/`last.pt` to `runs/seg/<name>/`. By default
`exist_ok: false` **auto-increments** the run dir
(`<name>`, `<name>2`, `<name>3`, …) so a re-run never overwrites old checkpoints; set
`exist_ok=true` to reuse/overwrite `runs/seg/<name>/`.

Focused recipe smoke tests:

```bash
# Scheduler + loss + strong augmentation + SGD on the torchvision baseline
uv run python -m oralskop.torchseg.train --config configs/train/seg_torch.yaml \
    --override epochs=1 limit_batches=3 device=cpu imgsz=128 batch=2 pretrained=false \
              scheduler=cosine warmup_epochs=1 loss=ce_dice aug=strong optimizer=sgd save_model=false

# SMP DeepLabV3+ contender
uv run python -m oralskop.torchseg.train --config configs/train/deeplabv3plus.yaml \
    --override epochs=1 limit_batches=3 device=cpu imgsz=128 batch=2 pretrained=false save_model=false

# SegFormer full fine-tune vs. LoRA
uv run python -m oralskop.torchseg.train --config configs/train/segformer.yaml \
    --override epochs=1 limit_batches=3 device=cpu imgsz=128 batch=2 pretrained=false save_model=false
uv run python -m oralskop.torchseg.train --config configs/train/segformer.yaml \
    --override epochs=1 limit_batches=3 device=cpu imgsz=128 batch=2 pretrained=false lora=true save_model=false

# DINOv2 decoder smoke test (random DINOv2-small config; no HF checkpoint download)
uv run --extra explore python -m oralskop.torchseg.train --config configs/train/dinov2_seg.yaml \
    --override arch=dinov2_small pretrained=false epochs=1 limit_batches=3 \
              device=cpu imgsz=140 batch=2 workers=0 save_model=false
```

The DeepLabV3+, SegFormer, DINOv2, `aug=light|strong`, non-CE losses, and LoRA paths
require `uv sync --extra explore`.

### YOLO-to-semantic bridge

After training a YOLO11-seg checkpoint, rasterize its instance predictions onto the same
AlphaDent val split and compute the torchseg semantic metrics (`fg_mIoU`, per-class IoU,
dice, pixel accuracy):

```bash
uv run python -m oralskop.bench.semantic_from_yolo \
    --weights runs/segment/yolo11m_seg_alphadent/weights/best.pt \
    --data data/alphadent/data.yaml --split val --imgsz 512 --infer-imgsz 960 \
    --out runs/segment/yolo11m_seg_alphadent/semantic_metrics.json
```

Use this number for the cross-paradigm ranking, alongside YOLO's native mask mAP from
section 3.

The `MergedSegDataset` / `YoloSegDataset` classes (`oralskop/torchseg/dataset.py`) are
reusable on their own for custom loops, EDA, or the future SAM2 stage.

### Weights & Biases logging (optional)

The torchseg loop can stream metrics + live prediction overlays to W&B. One-time install:

```bash
uv sync --extra wandb
```

Authenticate (in the Jupyter notebook, before launching training) — either set the key
as an env var so the `!`-subprocess inherits it, or log in once (writes `~/.netrc`):

```python
import os; os.environ["WANDB_API_KEY"] = "<your-token>"   # option A (Python cell)
```
```bash
!wandb login <your-token>                                  # option B (persists)
```

Then enable it on the run:

```bash
uv run python -m oralskop.torchseg.train --config configs/train/seg_torch.yaml \
    --datasets alphadent --override device=0 wandb=true wandb_project=oralskop-seg
```

Config knobs (`seg_torch.yaml`): `wandb` (on/off), `wandb_project`, `wandb_entity`,
`wandb_images` (val prediction overlays logged each validation; `0` = none). Logged:
`train/loss`, `train/pixel_acc`, `lr`, `val/mIoU` + `val/fg_mIoU` + dice/acc, per-class
`val_iou/<class>`, and interactive prediction vs. ground-truth masks. The run name matches
the (auto-incremented) `runs/seg/<name>` dir. If wandb isn't installed or you're not
logged in, training prints a warning and continues **without** it (never crashes).

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

## 3e. Serve a torchseg model as an HTTP endpoint (FastAPI)

Wraps a trained **torchseg** checkpoint in a FastAPI server. The model needs nothing
but its `.pt` file — `arch` / `num_classes` / `class_names` are read from the checkpoint.
Install the serving deps once: `uv sync --extra serve`.

Endpoints: `GET /` (browser upload form), `GET /health`, `GET /info`,
`POST /predict` (multipart `file=@photo.jpg` → JSON: per-component class name,
confidence, bbox, area + a per-class coverage summary), `POST /predict/overlay`
(→ annotated PNG with colored masks).

**In a Jupyter notebook (share a public URL with a friend via ngrok):**

```python
from oralskop.serve.notebook import serve
server = serve(
    weights="runs/seg/deeplabv3_alphadent/best.pt",
    device="cuda",                          # or "cpu"
    ngrok_authtoken="<token>",              # https://dashboard.ngrok.com (free)
)
print(server.url)   # public https URL; open <url>/ in a browser or POST to <url>/predict
server.stop()       # shut down server + tunnel
```

The server runs on a background thread, so the kernel stays interactive. Without a
token it serves locally only (`http://localhost:8000` — reachable on your LAN, not the
public internet). See `notebooks/serve_api.ipynb` for the full walkthrough.

**Standalone (no notebook):**

```bash
uv run python -m oralskop.serve.app \
    --weights runs/seg/deeplabv3_alphadent/best.pt --device cuda --port 8000
# then: curl -F file=@photo.jpg http://localhost:8000/predict
```

Args: `--weights`, `--arch` (fallback if no metadata), `--imgsz`, `--device`, `--conf`
(min confidence per detection), `--min-area-frac`, `--host`, `--port`.

---

## 3d. Multi-label classification (manifest dataset, S3)

A third, independent path (`oralskop.clf`) trains a **multi-label image classifier** on the
curated manifest dataset (`manifest_03_master_FINAL.csv`, see `PASSATION_DATA_OralSkop.md`).
It does **not** use the converter / `data.yaml` pipeline — it reads the manifest CSV
directly (local or `s3://`), builds multi-hot targets from `canonical_coarse` /
`canonical_fine`, and trains a torchvision backbone with `BCEWithLogitsLoss`. Designed to
run in a **Jupyter notebook on AWS** that has bucket access (the notebook's IAM role
provides S3 credentials).

```bash
# Install the extra (pandas + s3fs; s3fs pulls fsspec + boto3 for s3:// reads)
uv sync --extra clf

# CPU smoke test — a few rows, one epoch (set manifest/image_root for your bucket first)
uv run python -m oralskop.clf.train --config configs/clf/manifest_clf.yaml \
    --override level=coarse epochs=1 batch=4 limit=64 device=cpu amp=false num_workers=0

# Fine-level smoke test
uv run python -m oralskop.clf.train --config configs/clf/manifest_clf.yaml \
    --override level=fine labels_file=configs/clf/labels_fine.yaml name=clf_fine \
              epochs=1 batch=4 limit=64 device=cpu amp=false num_workers=0

# Real run (AWS notebook, GPU)
uv run python -m oralskop.clf.train --config configs/clf/manifest_clf.yaml \
    --override device=cuda epochs=30 batch=64 cache_dir=/home/ec2-user/oralskop_cache

# Evaluate on the test split (checkpoint's class list is authoritative)
uv run python -m oralskop.clf.eval --config configs/clf/manifest_clf.yaml \
    --weights runs/clf/clf_coarse/best.pt
```

Key config (`configs/clf/manifest_clf.yaml`): `manifest` + `image_root` (S3 keys),
`level` (coarse/fine) with a committed `labels_file` for stable class indices,
`image_path_prefixes` (restrict to the category folders you have synced), `cache_dir`
(cache S3 images on EBS across epochs), `arch`
(convnext_tiny / resnet50 / efficientnet_v2_s), `pos_weight: auto` (neg/pos from train,
for the heavy class imbalance — doc §7.4). Uses the manifest's own `split` column
(train/valid/test); excludes the unlabelled MetaDent `pretrain` rows and the two train-only
micro-classes. Reports **macro-mAP / micro-AP / macro-F1**; saves `best.pt` / `last.pt` /
`vocab.json` / `metrics.jsonl` to `runs/clf/<name>/`.

### QLoRA fine-tuning of a foundation model (DINOv2-large)

The same path can fine-tune a HuggingFace foundation model with **QLoRA** for a small
memory footprint: a 4-bit NF4 base (frozen) + LoRA adapters + a trainable multi-label
head, an **8-bit paged optimizer**, **gradient checkpointing**, and **gradient
accumulation**. Tuned for an A10G (~24 GB); needs CUDA (bitsandbytes).

```bash
# Extra deps (transformers + peft + bitsandbytes + accelerate) — CUDA only
uv sync --extra clf --extra qlora

# Smoke test in the GPU notebook (a few rows, one epoch)
uv run --extra clf --extra qlora python -m oralskop.clf.train \
    --config configs/clf/qlora_dinov2.yaml \
    --override image_root=datasets/02_PROCESSED limit=128 epochs=1 batch=4 \
              grad_accum_steps=1 wandb=false

# Real run (DINOv2-large, imgsz 518, batch 16 x accum 2, W&B)
uv run --extra clf --extra qlora --extra wandb python -m oralskop.clf.train \
    --config configs/clf/qlora_dinov2.yaml --override image_root=datasets/02_PROCESSED

# Evaluate (point --weights at the run dir; it loads adapter_best + meta.json)
uv run --extra clf --extra qlora python -m oralskop.clf.eval \
    --config configs/clf/qlora_dinov2.yaml \
    --weights runs/clf/clf_coarse_dinov2_large_qlora
```

Key config (`configs/clf/qlora_dinov2.yaml`): `arch` (`dinov2_large` / `dinov2_base` /
`vit_large_384` / `hf:<model_id>`), `quantize` (`4bit`/`8bit`/`none`), `lora` + `lora_r` /
`lora_alpha` / `lora_dropout`, `grad_checkpointing`, `grad_accum_steps`,
`optimizer: paged_adamw8bit`, `imgsz` (518 native for DINOv2). Normalization/size come
from the model's image processor; bf16 auto-selected on A10/A100 (fp16 on T4). Saves LoRA
**`adapter_best/` / `adapter_last/` + `meta.json`** (not a full `.pt`); the
"Model saved [tag] … with metrics" line prints on every checkpoint.

---

## 3f. Object detection (DINOv2 + DETR, manifest bbox subset)

`oralskop.det` trains a **DINOv2-backbone DETR** detector on the manifest's `yolo-bbox`
rows. By default, each YOLO box keeps its native `class_id` and that id is mapped per
source through `configs/det/box_label_map_coarse.yaml`; the legacy image-level weak label
mode is still available with `box_label_source=image`. The DINOv2 backbone is adapted
with **LoRA** (bf16 by default; `quantize=4bit` for experimental QLoRA); the DETR head
trains full-precision, and HF computes the set-prediction loss. Reports detection
**mAP**. CUDA only.

```bash
# Deps (transformers/peft via qlora; torchmetrics/pycocotools/timm via det)
uv sync --extra clf --extra qlora --extra det

# Get the bbox images AND their .txt labels locally (or stream from s3://)
python scripts/download_data.py --with-labels --prefixes CARIES/ MULTI/ mouth_detection/

# Smoke test in the GPU notebook
uv run --extra clf --extra qlora --extra det python -m oralskop.det.train \
    --config configs/det/qlora_dinov2_detr.yaml \
    --override image_root=datasets/02_PROCESSED limit=64 epochs=1 batch=2 \
              grad_accum_steps=1 wandb=false name=det_smoke

# Real run (DINOv2-base backbone, imgsz 518, ~50 epochs — DETR converges slowly)
uv run --extra clf --extra qlora --extra det --extra wandb python -m oralskop.det.train \
    --config configs/det/qlora_dinov2_detr.yaml --override image_root=datasets/02_PROCESSED

# Evaluate (test mAP + per-class AP)
uv run --extra clf --extra qlora --extra det python -m oralskop.det.eval \
    --config configs/det/qlora_dinov2_detr.yaml \
    --weights runs/det/det_coarse_dinov2_detr_qlora/best.pt
```

Key config (`configs/det/qlora_dinov2_detr.yaml`): `arch` (`dinov2_small`/`base`/`large`
or `hf:<id>`), `quantize` (`none` bf16 — recommended — / `4bit` experimental), `lora_*`,
`box_label_source` (`native` / `image` / `class_agnostic`), `box_label_map`,
`unknown_box_class_policy`, `num_queries` (max boxes/image), `imgsz` (518),
`optimizer: paged_adamw8bit`, `grad_accum_steps`, `progress`, `log_interval`,
`wandb_log_interval`, and `eval_match_score_threshold` (confidence cutoff for
precision50/recall50/F1_50).
Saves a full `best.pt`/`last.pt` + `meta.json` to `runs/det/<name>/`.
Caveats: DETR is slow to converge (start at base, ~50 epochs, expect modest early mAP);
native id maps must be documented per source; unmapped boxes are dropped by default.

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
| Train classifier (manifest) | `uv run python -m oralskop.clf.train --config configs/clf/manifest_clf.yaml` |
| Eval classifier (test)   | `uv run python -m oralskop.clf.eval --config configs/clf/manifest_clf.yaml --weights runs/clf/clf_coarse/best.pt` |
| Evaluate                 | `uv run python -m oralskop.eval.evaluate --weights <best.pt> --data data/alphadent/data.yaml` |
| Visualize masks          | `uv run python -m oralskop.viz.visualize --dataset alphadent --num_imgs 12` |
| Serve as API             | `uv run python -m oralskop.serve.app --weights <best.pt> --port 8000` (or `serve()` in a notebook) |
| Build container          | `apptainer build oralskop.sif scripts/apptainer.def` |
| Submit cluster job       | `sbatch scripts/train.sbatch` |
