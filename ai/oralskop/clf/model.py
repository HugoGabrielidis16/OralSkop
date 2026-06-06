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
