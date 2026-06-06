# Improve OralSkop segmentation performance (metrics-driven, single A10G)

## Context

We want better dental-lesion segmentation accuracy. Today there are two paths: the
**instance** path (Ultralytics **YOLO11-seg**, `train/train.py`, the project's stated
foundation) and the **semantic** path (`torchseg/`, currently **DeepLabV3-ResNet50**).
The semantic recipe has three large, cheap gaps: **no LR scheduler** (constant AdamW lr,
`train.py:336`), **plain CrossEntropy only** (`train.py:334`), and **horizontal-flip-only
augmentation** (`dataset.py:139`). On a small dataset these recipe gaps usually cost more
mIoU than the choice of architecture.

Decisions from clarification: **compare both paradigms** and recommend the winner by
metric; **explore LoRA but use whatever scores best** (full fine-tune of ResNet/MiT-size
models fits the 23 GB A10G and usually beats LoRA — LoRA is reserved as an experiment on
the transformer backbone and for the future SAM2 stage); **AlphaDent only**; **focused
breadth** = fix the recipe, then compare 1–2 top architectures. Hardware: one A10G, 23 GB.

Goal: a reproducible, ablatable benchmark on a fixed AlphaDent val split that ranks
recipe + architecture choices by a single primary metric, ending in a recommendation.

## Benchmark protocol (the backbone — do this first)

- **Data**: AlphaDent only, the existing patient-grouped split materialized at
  `data/alphadent/` (no leakage; reuse as-is). Keep the split fixed across all runs.
- **Primary metric**: `fg_mIoU` (foreground mean IoU over caries/abrasion/filling/crown),
  already computed in `evaluate()` (`torchseg/train.py:90`). Secondary: per-class IoU,
  `fg_dice`, and now `val_loss` (for overfitting watch). YOLO native metric: mask
  mAP50-95 via `eval/evaluate.py`.
- **Tracking**: W&B (already wired). One project, consistent run names; rely on
  auto-increment run dirs so nothing clobbers.
- **Ablation ladder** (each step config-gated so we can isolate its effect):
  baseline DeepLabV3 → +scheduler → +loss → +augmentation → architecture swaps → LoRA.

## Workstream A — torchseg recipe upgrades (config-gated, ablatable)

All in `oralskop/torchseg/train.py` + `configs/train/seg_torch.yaml`, each behind a knob
with current behavior as default so existing runs are unchanged until opted in.

1. **LR scheduler + warmup** — new `scheduler: none|cosine|poly` and `warmup_epochs`.
   Use `torch.optim.lr_scheduler` (CosineAnnealingLR; poly via `LambdaLR`), stepped
   per-epoch after the train loop; linear warmup for the first N epochs. Log effective lr
   (already logged at `train.py:451`).
2. **Loss selection** — new `loss: ce|ce_dice|focal|focal_dice|lovasz`. Reuse
   `segmentation_models_pytorch.losses` (DiceLoss/FocalLoss/LovaszLoss) combined with the
   existing weighted CE; put the builder in a new `oralskop/torchseg/losses.py`
   (`build_criterion(name, class_weights)`), used in both train and `evaluate()` so
   `val_loss` stays comparable. Keep median-frequency `class_weights=auto` support.
3. **Augmentation** — replace the flip-only branch in `dataset.py` with an
   **Albumentations** pipeline, gated by `aug: none|light|strong`. strong = scale/shift/
   rotate, brightness/contrast, hue/sat, h-flip, mild blur/CLAHE (joint image+mask). This
   is the highest-leverage change for ~1k images.
4. **Optimizer knob** — new `optimizer: adamw|sgd`, with SGD+momentum+poly as the classic
   DeepLab recipe option; AdamW stays default.

## Workstream B — new architectures (focused: 2 contenders via one library)

Add `segmentation_models_pytorch` (smp) and extend the factory
(`oralskop/torchseg/model/factory.py`) with builders that wrap smp models to return the
project's `{"out": logits}` contract (smp returns a bare tensor; wrap in a tiny
`nn.Module`, no aux head):

- **`deeplabv3plus_<encoder>`** — upgrade over plain DeepLabV3 (adds a decoder). Strong
  CNN contender, e.g. encoder `efficientnet-b4` or `resnet50` (ImageNet-pretrained).
- **`segformer_mit_b2`** (and optionally `b3`) — transformer (MiT) encoder, ADE/ImageNet
  pretrained; strong on small data with heavy aug, and the vehicle for the LoRA experiment.

Keep the existing torchvision archs and `unet` as baselines. `_BUILDERS` already maps
names → builders; just add entries + the wrapper.

## Workstream C — LoRA experiment (peft on the transformer encoder)

Add `peft` (optional) and a small `oralskop/torchseg/lora.py` with
`apply_lora(model, r, alpha, targets)` that wraps the **SegFormer/MiT encoder's attention
Linear layers** via `peft.LoraConfig`/`get_peft_model`, leaving the decode head fully
trainable. Config: `lora: false`, `lora_r`, `lora_alpha`, `lora_targets`. Log trainable-
vs-total parameter count. Run it head-to-head against full fine-tuning of the same
SegFormer; **keep whichever wins the primary metric** (expectation: full FT wins on
accuracy here; LoRA may win on speed/overfitting — report both).

## Workstream D — instance benchmark (YOLO11-seg) + cross-paradigm comparison

- Train YOLO11-seg on AlphaDent (`configs/train/yolo11_seg.yaml`, `data=data/alphadent/
  data.yaml`, `yolo11m-seg`/`yolo11l-seg`, imgsz 640–960, batch tuned to A10G). It brings
  its own strong augmentation/scheduler, so it's a serious contender with little tuning.
  Native metric via `eval/evaluate.py` (mask mAP50-95).
- **Apples-to-apples bridge**: new `oralskop/bench/semantic_from_yolo.py` that runs YOLO
  inference on the AlphaDent val images, **rasterizes predicted instance masks into a
  semantic map** (reuse `rasterize_polygons` logic / fill predicted polygons), and feeds
  them through the same confusion-matrix metric used in `torchseg.evaluate` to get
  `fg_mIoU` on the identical val set. This lets us rank instance vs semantic on one number.
- Document the caveat: instance seg can separate touching lesions and gives counts;
  semantic cannot. Final recommendation weighs the metric **and** the app need (the app
  likely wants per-lesion detection → instance), not the single number alone.

## Dependencies (new, isolated in an optional extra to keep base/cluster light)

`pyproject.toml` → `[project.optional-dependencies] explore = ["segmentation-models-pytorch>=0.3.4", "albumentations>=1.4", "peft>=0.11", "timm>=0.9"]`
(smp pulls timm encoders; peft only for Workstream C). Install with `uv sync --extra explore`.
Lazy-import smp/albumentations/peft so the base env still runs without the extra.

## A10G (23 GB) memory guidance

All fit with AMP at imgsz 512 (semantic) / 640–960 (YOLO): DeepLabV3+ resnet50 batch ~16;
DeepLabV3+ efficientnet-b4 batch ~12; SegFormer-b2 batch ~16, b3 ~8–12; YOLO11m-seg batch
~16, l-seg ~8. Start one notch lower and raise while watching `nvidia-smi`; drop on OOM.

## Files to create / modify

- Modify: `oralskop/torchseg/train.py` (scheduler, optimizer choice, loss via builder,
  optional LoRA wrap, per-epoch scheduler step), `oralskop/torchseg/dataset.py`
  (Albumentations), `oralskop/torchseg/model/factory.py` (smp builders + wrapper),
  `configs/train/seg_torch.yaml` (new knobs), `pyproject.toml` (extra), `COMMANDS.md`.
- Create: `oralskop/torchseg/losses.py`, `oralskop/torchseg/lora.py`,
  `oralskop/bench/semantic_from_yolo.py`, and optionally
  `configs/train/segformer.yaml` / `deeplabv3plus.yaml` preset configs.

## Verification

1. **CPU smoke (each new knob)**: 1 epoch, `limit_batches=3`, `device=cpu` — confirm
   `scheduler=cosine`, `loss=ce_dice`, `aug=strong`, `optimizer=sgd`, and
   `arch=deeplabv3plus_resnet50` / `segformer_mit_b2` all run and write metrics.
2. **LoRA**: build SegFormer with `lora=true`, assert trainable-param count << total and
   one training step runs.
3. **Bridge**: run `bench/semantic_from_yolo.py` on a tiny set; confirm it emits the same
   metric keys as `torchseg.evaluate`.
4. **A10G short runs (10–20 epochs)** for each contender with the fixed recipe; compare
   `fg_mIoU` in W&B; then a longer run for the leader. Sanity-check qualitatively with the
   existing `torchseg/test.py` (raw | pred | GT) and YOLO predictions.
5. Deliverable: a W&B comparison table + a short written recommendation (best paradigm +
   arch + recipe), with per-class IoU so rare classes (crown) are visible.

## Expected outcome / recommendation hypothesis

Biggest gains from Workstream A (scheduler + Dice/Focal loss + strong augmentation),
typically several mIoU points. Among architectures, SegFormer-b2/b3 or DeepLabV3+ should
beat plain DeepLabV3. For the app's likely need (locate/count discrete lesions), YOLO11-seg
is the probable practical winner; the semantic models remain useful for EDA and the future
SAM2 stage. LoRA reported as an efficiency option, not expected to top full fine-tuning at
this scale.
