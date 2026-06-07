"""Backbone factory for the multi-label classifier (torchvision models).

Returns an ImageNet-pretrained backbone with its final layer swapped for a
``num_classes`` linear head. Outputs are raw logits — the loss
(``BCEWithLogitsLoss``) applies the sigmoid, and inference thresholds
``sigmoid(logits)``.
"""

from __future__ import annotations

import torch
from torchvision import models

# arch name -> (constructor, default-weights enum, head-replacement fn)
_ARCHS = {
    "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.DEFAULT),
    "convnext_small": (models.convnext_small, models.ConvNeXt_Small_Weights.DEFAULT),
    "convnext_base": (models.convnext_base, models.ConvNeXt_Base_Weights.DEFAULT),
    "convnext_large": (models.convnext_large, models.ConvNeXt_Large_Weights.DEFAULT),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
    "efficientnet_v2_s": (models.efficientnet_v2_s, models.EfficientNet_V2_S_Weights.DEFAULT),
    "efficientnet_v2_m": (models.efficientnet_v2_m, models.EfficientNet_V2_M_Weights.DEFAULT),
    "efficientnet_v2_l": (models.efficientnet_v2_l, models.EfficientNet_V2_L_Weights.DEFAULT),
}


def _replace_head(model: torch.nn.Module, arch: str, num_classes: int) -> torch.nn.Module:
    if arch == "resnet50":
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif arch.startswith("convnext_"):
        in_features = model.classifier[2].in_features
        model.classifier[2] = torch.nn.Linear(in_features, num_classes)
    elif arch.startswith("efficientnet_v2_"):
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    else:  # pragma: no cover - guarded by build_classifier
        raise ValueError(arch)
    return model


def build_classifier(num_classes: int, *, arch: str = "convnext_tiny", pretrained: bool = True):
    """Build a multi-label classifier backbone with a fresh ``num_classes`` head."""
    if arch not in _ARCHS:
        raise ValueError(f"Unknown arch {arch!r}. Options: {', '.join(sorted(_ARCHS))}.")
    ctor, weights = _ARCHS[arch]
    model = ctor(weights=weights if pretrained else None)
    return _replace_head(model, arch, num_classes)


# --------------------------------------------------------------------------- #
# Foundation models (HuggingFace) fine-tuned with QLoRA: 4-bit NF4 base (frozen)
# + LoRA adapters + a trainable multi-label head, for a small memory footprint.
# --------------------------------------------------------------------------- #

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Friendly arch name -> HuggingFace model id (also accept `hf:<id>` passthrough).
_HF_MODELS = {
    "dinov2_large": "facebook/dinov2-large",
    "dinov2_base": "facebook/dinov2-base",
    "dinov2_small": "facebook/dinov2-small",
    "dinov2_giant": "facebook/dinov2-giant",
    "vit_large_384": "google/vit-large-patch16-384",
    "vit_base": "google/vit-base-patch16-224",
}

# LoRA targets for ViT/DINOv2-style attention + MLP linear layers.
_DEFAULT_LORA_TARGETS = ["query", "key", "value", "dense", "fc1", "fc2"]


def _resolve_hf_id(arch: str) -> str | None:
    if arch.startswith("hf:"):
        return arch[len("hf:"):]
    return _HF_MODELS.get(arch)


def _processor_preprocess(model_id: str, imgsz_override: int | None) -> dict:
    """Read normalization + input size from the model's image processor."""
    from transformers import AutoImageProcessor

    proc = AutoImageProcessor.from_pretrained(model_id)
    mean = tuple(getattr(proc, "image_mean", IMAGENET_MEAN))
    std = tuple(getattr(proc, "image_std", IMAGENET_STD))
    size = getattr(proc, "size", None) or {}
    proc_sz = size.get("shortest_edge") or size.get("height") or size.get("crop_size") or 224
    return {"mean": mean, "std": std, "imgsz": int(imgsz_override or proc_sz)}


def build_foundation_model(
    num_classes: int,
    model_id: str,
    *,
    quantize: str | None = "4bit",
    lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: list[str] | None = None,
    grad_checkpointing: bool = True,
    compute_dtype: torch.dtype = torch.bfloat16,
    imgsz: int | None = None,
):
    """Build a HF image classifier with optional 4-bit/8-bit quantization + LoRA (QLoRA)."""
    try:
        from transformers import AutoModelForImageClassification, BitsAndBytesConfig
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(
            "Foundation-model fine-tuning needs the `qlora` extra: "
            "`uv sync --extra qlora` (transformers + peft + bitsandbytes + accelerate)."
        ) from exc

    quant = (quantize or "none").lower()
    bnb_config = None
    # Keep the freshly-initialized classification head OUT of quantization — it is the
    # trainable multi-label head (also in LoRA's modules_to_save). Quantizing it makes
    # peft clone a 4-bit Linear, which fails bitsandbytes' quant-state check on forward.
    if quant == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype,
            llm_int8_skip_modules=["classifier"],
        )
    elif quant == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["classifier"])
    elif quant not in {"none", "no", "false"}:
        raise ValueError(f"Unknown quantize {quantize!r}. Options: 4bit, 8bit, none.")

    model = AutoModelForImageClassification.from_pretrained(
        model_id, num_labels=num_classes, problem_type="multi_label_classification",
        quantization_config=bnb_config, ignore_mismatched_sizes=True,
    )

    if lora:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if bnb_config is not None:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=grad_checkpointing,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        elif grad_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        peft_cfg = LoraConfig(
            r=int(lora_r), lora_alpha=int(lora_alpha), lora_dropout=float(lora_dropout),
            target_modules=list(lora_target_modules or _DEFAULT_LORA_TARGETS),
            bias="none", modules_to_save=["classifier"],
        )
        model = get_peft_model(model, peft_cfg)

    preprocess = _processor_preprocess(model_id, imgsz)
    return model, preprocess, True


def build_model(num_classes: int, cfg: dict, *, compute_dtype: torch.dtype = torch.bfloat16):
    """Dispatch on ``cfg['arch']`` -> (model, preprocess, is_hf).

    preprocess = {"mean", "std", "imgsz", optional "model_id"}. Torchvision archs use
    ImageNet stats and the config's ``imgsz`` (default 224); foundation models use their
    processor's values.
    """
    arch = str(cfg.get("arch", "convnext_tiny"))
    hf_id = _resolve_hf_id(arch)
    if hf_id is not None:
        targets = cfg.get("lora_target_modules")
        if isinstance(targets, str):
            targets = [t.strip() for t in targets.split(",") if t.strip()]
        model, preprocess, is_hf = build_foundation_model(
            num_classes, hf_id,
            quantize=cfg.get("quantize", "4bit"), lora=bool(cfg.get("lora", True)),
            lora_r=cfg.get("lora_r", 16), lora_alpha=cfg.get("lora_alpha", 32),
            lora_dropout=cfg.get("lora_dropout", 0.05), lora_target_modules=targets,
            grad_checkpointing=bool(cfg.get("grad_checkpointing", True)),
            compute_dtype=compute_dtype, imgsz=cfg.get("imgsz"),
        )
        preprocess["model_id"] = hf_id
        return model, preprocess, is_hf

    model = build_classifier(num_classes, arch=arch, pretrained=bool(cfg.get("pretrained", True)))
    preprocess = {"mean": IMAGENET_MEAN, "std": IMAGENET_STD, "imgsz": int(cfg.get("imgsz", 224))}
    return model, preprocess, False
