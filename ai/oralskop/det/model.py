"""DINOv2-backbone DETR detector (LoRA on the backbone; optional 4-bit/QLoRA).

We build a HF ``DetrForObjectDetection`` and replace its backbone with a **pretrained
DINOv2** (`AutoBackbone`). The DINOv2 backbone is adapted with **LoRA** (its base frozen);
the DETR encoder/decoder + class/box heads + object queries are trained full-precision.
Default precision is **bf16** (via autocast) with LoRA — reliable and memory-light on an
A10. ``quantize="4bit"`` enables true QLoRA on the backbone (experimental: 4-bit
checkpointing is not fully supported here — prefer the bf16 default).

The HF model computes the DETR set-prediction (Hungarian) loss internally when ``labels``
are passed, so the training loop stays thin.
"""

from __future__ import annotations

import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Friendly name -> HF model id (also accept `hf:<id>`).
_DINOV2 = {
    "dinov2_small": "facebook/dinov2-small",
    "dinov2_base": "facebook/dinov2-base",
    "dinov2_large": "facebook/dinov2-large",
}
_DEFAULT_LORA_TARGETS = ["query", "key", "value", "dense", "fc1", "fc2"]

# state_dict prefix of the injected backbone (DetrForObjectDetection layout).
BACKBONE_STATE_PREFIX = "model.backbone.conv_encoder.model."


def _detr_conv_encoder(model):
    """Return the DETR conv encoder across transformers layout variants."""
    backbone = model.model.backbone
    if hasattr(backbone, "conv_encoder"):
        backbone = backbone.conv_encoder
    if not hasattr(backbone, "model"):
        raise AttributeError(
            "Could not locate DETR backbone model slot. Expected "
            "`model.model.backbone.model` or `model.model.backbone.conv_encoder.model`."
        )
    return backbone


def _get_detr_backbone_model(model):
    return _detr_conv_encoder(model).model


def _set_detr_backbone_model(model, backbone):
    _detr_conv_encoder(model).model = backbone


def _warm_start_from_coco(model, pretrained_detr: str) -> int:
    """Load a COCO-pretrained DETR transformer into ``model`` (shape-matched).

    Only the backbone is pretrained when we build DETR from a DINOv2 ``backbone_config``;
    the encoder/decoder/object-queries/box head start random and need 300-500 epochs to
    converge. ``facebook/detr-resnet-50`` shares DETR's defaults (d_model 256, 6/6 layers,
    100 queries), so its transformer + ``bbox_predictor`` load directly. The DINOv2 backbone,
    the ``input_projection`` (channel count differs), and the COCO ``class_labels_classifier``
    (91+1 vs C+1) are shape-mismatched and skipped. Returns the number of tensors transferred.
    """
    from transformers import DetrForObjectDetection

    src = DetrForObjectDetection.from_pretrained(pretrained_detr).state_dict()
    own = model.state_dict()
    keep = {
        k: v for k, v in src.items()
        if k in own and own[k].shape == v.shape
        and not k.startswith(("model.backbone.", "model.input_projection"))
    }
    model.load_state_dict(keep, strict=False)
    return len(keep)


def resolve_model_id(arch: str) -> str:
    if arch.startswith("hf:"):
        return arch[len("hf:"):]
    return _DINOV2.get(arch, arch)


def _processor_preprocess(model_id: str, imgsz_override: int | None) -> dict:
    from transformers import AutoImageProcessor

    proc = AutoImageProcessor.from_pretrained(model_id)
    mean = tuple(getattr(proc, "image_mean", IMAGENET_MEAN))
    std = tuple(getattr(proc, "image_std", IMAGENET_STD))
    size = getattr(proc, "size", None) or {}
    proc_sz = size.get("shortest_edge") or size.get("height") or size.get("crop_size") or 518
    return {"mean": mean, "std": std, "imgsz": int(imgsz_override or proc_sz), "model_id": model_id}


def build_detr_processor():
    """A DetrImageProcessor used only for ``post_process_object_detection`` in eval."""
    from transformers import DetrImageProcessor

    return DetrImageProcessor()


def build_detector(
    num_classes: int,
    arch: str = "dinov2_base",
    *,
    quantize: str | None = "none",
    lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: list[str] | None = None,
    grad_checkpointing: bool = True,
    compute_dtype: torch.dtype = torch.bfloat16,
    num_queries: int = 100,
    imgsz: int | None = None,
    pretrained_detr: str | None = "facebook/detr-resnet-50",
):
    """Build a DINOv2-backbone DETR. Returns ``(model, preprocess)``."""
    try:
        from transformers import AutoBackbone, DetrConfig, DetrForObjectDetection
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(
            "The detector needs the `qlora` + `det` extras: "
            "`uv sync --extra qlora --extra det` (transformers, peft, torchmetrics)."
        ) from exc

    model_id = resolve_model_id(arch)
    quant = (quantize or "none").lower()
    bnb_config = None
    if quant == "4bit":
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype)
    elif quant not in {"none", "no", "false"}:
        raise ValueError(f"Unknown quantize {quantize!r}. Options: 4bit, none.")

    # Pretrained DINOv2 backbone, last stage as the DETR feature map.
    backbone = AutoBackbone.from_pretrained(model_id, out_indices=[-1], quantization_config=bnb_config)

    # DETR with a matching (random) backbone, then inject the pretrained one.
    config = DetrConfig(num_labels=num_classes, num_queries=int(num_queries),
                        use_timm_backbone=False, use_pretrained_backbone=False,
                        backbone_config=backbone.config)
    model = DetrForObjectDetection(config)
    _set_detr_backbone_model(model, backbone)

    # Warm-start the (otherwise random) DETR transformer + box head from COCO. Done before
    # the LoRA wrap so backbone key renaming is irrelevant (backbone keys are skipped anyway).
    if pretrained_detr:
        n = _warm_start_from_coco(model, pretrained_detr)
        print(f">> warm-started {n} tensors from {pretrained_detr} "
              f"(reinit: backbone, input_projection, class head)")

    if lora:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        bb = _get_detr_backbone_model(model)
        if bnb_config is not None:
            bb = prepare_model_for_kbit_training(
                bb, use_gradient_checkpointing=grad_checkpointing,
                gradient_checkpointing_kwargs={"use_reentrant": False})
        elif grad_checkpointing and hasattr(bb, "gradient_checkpointing_enable"):
            bb.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        lora_cfg = LoraConfig(
            r=int(lora_r), lora_alpha=int(lora_alpha), lora_dropout=float(lora_dropout),
            target_modules=list(lora_target_modules or _DEFAULT_LORA_TARGETS), bias="none")
        _set_detr_backbone_model(model, get_peft_model(bb, lora_cfg))

    preprocess = _processor_preprocess(model_id, imgsz)
    return model, preprocess
