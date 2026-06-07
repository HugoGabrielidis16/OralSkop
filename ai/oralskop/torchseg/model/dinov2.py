"""DINOv2 backbone + lightweight semantic-segmentation decoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_DINOV2_MODEL_IDS = {
    "dinov2_small": "facebook/dinov2-small",
    "dinov2_base": "facebook/dinov2-base",
    "dinov2_large": "facebook/dinov2-large",
    "dinov2_giant": "facebook/dinov2-giant",
}

_DINOV2_CONFIGS = {
    "dinov2_small": dict(hidden_size=384, num_hidden_layers=12, num_attention_heads=6, intermediate_size=1536),
    "dinov2_base": dict(hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072),
    "dinov2_large": dict(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, intermediate_size=4096),
    "dinov2_giant": dict(hidden_size=1536, num_hidden_layers=40, num_attention_heads=24, intermediate_size=6144),
}


def resolve_model_id(arch: str) -> str:
    """Friendly name or `hf:<repo-id>` -> Hugging Face model id."""
    if arch.startswith("hf:"):
        return arch[len("hf:"):]
    return _DINOV2_MODEL_IDS.get(arch, arch)


def _local_dinov2_config(arch: str):
    """Build a DINOv2 config without downloading metadata for known friendly names."""
    try:
        from transformers import AutoConfig, Dinov2Config
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(
            "DINOv2 segmentation requires transformers. Install with "
            "`uv sync --extra explore` or `uv sync --extra qlora`."
        ) from exc

    if arch in _DINOV2_CONFIGS:
        return Dinov2Config(
            image_size=518,
            patch_size=14,
            out_indices=[-1],
            apply_layernorm=True,
            **_DINOV2_CONFIGS[arch],
        )
    cfg = AutoConfig.from_pretrained(resolve_model_id(arch))
    cfg.out_indices = [-1]
    cfg.apply_layernorm = True
    return cfg


def _build_backbone(arch: str, pretrained: bool) -> nn.Module:
    try:
        from transformers import AutoBackbone, Dinov2Backbone
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(
            "DINOv2 segmentation requires transformers. Install with "
            "`uv sync --extra explore` or `uv sync --extra qlora`."
        ) from exc

    if pretrained:
        return AutoBackbone.from_pretrained(resolve_model_id(arch), out_indices=[-1])
    if arch in _DINOV2_CONFIGS:
        return Dinov2Backbone(_local_dinov2_config(arch))
    return AutoBackbone.from_config(_local_dinov2_config(arch))


def _hidden_size(backbone: nn.Module) -> int:
    config = getattr(backbone, "config", None)
    hidden = getattr(config, "hidden_size", None)
    if hidden is None:
        hidden = getattr(config, "embed_dim", None)
    if hidden is None:
        raise AttributeError("Could not infer DINOv2 hidden size from backbone config.")
    return int(hidden)


class ConvNormAct(nn.Sequential):
    """Small decoder block used on low-resolution DINOv2 patch features."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )


class Dinov2SegmentationModel(nn.Module):
    """DINOv2 encoder with a compact dense-prediction head.

    The model follows the project torchseg contract: ``forward`` returns
    ``{"out": logits}``, where logits are upsampled to the input image size.
    """

    def __init__(
        self,
        num_classes: int,
        arch: str = "dinov2_base",
        *,
        pretrained: bool = True,
        decoder_channels: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.arch = arch
        self.backbone = _build_backbone(arch, pretrained=pretrained)
        hidden = _hidden_size(self.backbone)
        self.decoder = nn.Sequential(
            ConvNormAct(hidden, decoder_channels, kernel_size=1),
            ConvNormAct(decoder_channels, decoder_channels, kernel_size=3),
            nn.Dropout2d(float(dropout)),
            nn.Conv2d(decoder_channels, num_classes, kernel_size=1),
        )

    def gradient_checkpointing_enable(self) -> None:
        """Enable checkpointing on the HF backbone when supported."""
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.backbone(pixel_values=x)
        features = out.feature_maps[-1]
        logits = self.decoder(features)
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return {"out": logits}
