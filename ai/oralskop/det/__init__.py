"""Object detection on the curated manifest's bounding-box subset.

An INDEPENDENT path, parallel to `oralskop.clf` (classification) and
`oralskop.torchseg` (segmentation). It reads the manifest's `yolo-bbox` rows
directly (local or `s3://`), assigns each box its image's single coarse label
(see `ai/PASSATION_DATA_OralSkop.md`), and fine-tunes a **DINOv2-backbone DETR**
detector with LoRA (optionally 4-bit/QLoRA) — the HF model computes the DETR
set-prediction loss; we wrap dataset / training loop / mAP eval around it.

Run config: `configs/det/qlora_dinov2_detr.yaml`. Reuses `oralskop.clf` helpers.
"""
