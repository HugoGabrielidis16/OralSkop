"""Optional LoRA utilities for transformer segmentation experiments."""

from __future__ import annotations

import torch.nn as nn


def apply_lora(
    model: nn.Module,
    r: int = 8,
    alpha: int = 16,
    targets: str | list[str] | tuple[str, ...] | None = None,
) -> nn.Module:
    """Apply PEFT LoRA adapters to matching linear modules.

    The default target names cover common MiT/SegFormer attention projections exposed by
    timm-backed encoders. Decode-head parameters are left trainable by PEFT.
    """
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "LoRA requires the optional explore dependencies. Install with "
            "`uv sync --extra explore`."
        ) from exc

    if targets is None:
        target_modules = ["q", "kv", "proj", "query", "key", "value"]
    elif isinstance(targets, str):
        target_modules = [part.strip() for part in targets.split(",") if part.strip()]
    else:
        target_modules = list(targets)

    cfg = LoraConfig(
        r=int(r),
        lora_alpha=int(alpha),
        target_modules=target_modules,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    model = get_peft_model(model, cfg)
    for name, param in model.named_parameters():
        if any(part in name for part in ("decoder", "segmentation_head", "classification_head")):
            param.requires_grad = True
    return model


def parameter_counts(model: nn.Module) -> tuple[int, int]:
    """Return trainable and total parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
