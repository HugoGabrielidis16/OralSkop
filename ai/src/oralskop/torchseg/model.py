"""Segmentation model factory (torchvision) with the head swapped to our class count."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import segmentation as tvseg

_BUILDERS = {
    "deeplabv3_resnet50": (tvseg.deeplabv3_resnet50, True),
    "deeplabv3_mobilenet_v3_large": (tvseg.deeplabv3_mobilenet_v3_large, True),
    "fcn_resnet50": (tvseg.fcn_resnet50, True),
    "lraspp_mobilenet_v3_large": (tvseg.lraspp_mobilenet_v3_large, False),  # no aux head
}


def build_model(num_classes: int, arch: str = "deeplabv3_resnet50", pretrained: bool = True) -> nn.Module:
    """Build a torchvision segmentation model with `num_classes` outputs.

    Returns an `nn.Module` whose forward gives a dict with key ``"out"`` (and ``"aux"``
    for the deeplab/fcn variants).
    """
    if arch not in _BUILDERS:
        raise ValueError(f"Unknown arch {arch!r}. Options: {', '.join(_BUILDERS)}")
    builder, has_aux = _BUILDERS[arch]

    kwargs = {"weights": "DEFAULT" if pretrained else None}
    if has_aux:
        kwargs["aux_loss"] = True
    model = builder(**kwargs)

    # Replace the final 1x1 conv of the classifier head(s) with our class count.
    if arch.startswith("lraspp"):
        model.classifier.low_classifier = nn.Conv2d(
            model.classifier.low_classifier.in_channels, num_classes, 1)
        model.classifier.high_classifier = nn.Conv2d(
            model.classifier.high_classifier.in_channels, num_classes, 1)
    else:
        model.classifier[-1] = nn.Conv2d(model.classifier[-1].in_channels, num_classes, 1)
        if getattr(model, "aux_classifier", None) is not None:
            model.aux_classifier[-1] = nn.Conv2d(
                model.aux_classifier[-1].in_channels, num_classes, 1)
    return model


def has_aux(arch: str) -> bool:
    return _BUILDERS.get(arch, (None, False))[1]
