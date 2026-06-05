"""Segmentation model factory: torchvision builders plus our from-scratch U-Net.

All models share one interface: ``forward`` returns a dict with key ``"out"``
(and ``"aux"`` for the deeplab/fcn variants). ``num_classes`` includes the
background class at index 0.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import segmentation as tvseg

from oralskop.torchseg.model.unet import UNet

_BUILDERS = {
    "deeplabv3_resnet50": (tvseg.deeplabv3_resnet50, True),
    "deeplabv3_mobilenet_v3_large": (tvseg.deeplabv3_mobilenet_v3_large, True),
    "fcn_resnet50": (tvseg.fcn_resnet50, True),
    "lraspp_mobilenet_v3_large": (tvseg.lraspp_mobilenet_v3_large, False),  # no aux head
    "unet": (None, False),  # custom, from-scratch; no torchvision builder / pretrained weights
}


def build_model(num_classes: int, arch: str = "deeplabv3_resnet50", pretrained: bool = True) -> nn.Module:
    """Build a segmentation model with `num_classes` outputs.

    Returns an `nn.Module` whose forward gives a dict with key ``"out"`` (and ``"aux"``
    for the deeplab/fcn variants).
    """
    if arch not in _BUILDERS:
        raise ValueError(f"Unknown arch {arch!r}. Options: {', '.join(_BUILDERS)}")

    if arch == "unet":
        return UNet(num_classes)  # trained from scratch; `pretrained` does not apply

    builder, has_aux_head = _BUILDERS[arch]

    kwargs = {"weights": "DEFAULT" if pretrained else None}
    if not pretrained:
        # torchvision defaults `weights_backbone` to ImageNet even when weights=None,
        # so it would download the backbone anyway. None = truly from scratch / no
        # download (correct for `pretrained=False` and for loading our own checkpoint).
        kwargs["weights_backbone"] = None
    if has_aux_head:
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
