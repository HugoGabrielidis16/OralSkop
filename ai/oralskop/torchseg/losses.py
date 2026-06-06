"""Loss builders for semantic segmentation experiments."""

from __future__ import annotations

import torch
import torch.nn as nn


class CombinedLoss(nn.Module):
    """Sum multiple loss modules with fixed weights."""

    def __init__(self, losses: list[tuple[float, nn.Module]]):
        super().__init__()
        self.weights = [weight for weight, _ in losses]
        self.losses = nn.ModuleList([loss for _, loss in losses])

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total = logits.new_tensor(0.0)
        for weight, loss in zip(self.weights, self.losses):
            total = total + weight * loss(logits, targets)
        return total


def _smp_losses():
    try:
        from segmentation_models_pytorch import losses
    except ImportError as exc:
        raise ImportError(
            "Losses other than 'ce' require the optional explore dependencies. "
            "Install with `uv sync --extra explore`."
        ) from exc
    return losses


def build_criterion(name: str, class_weights: torch.Tensor | None = None) -> nn.Module:
    """Build a config-selected segmentation loss.

    ``ce`` preserves the current training objective. Composite losses retain the existing
    weighted CE term so ``class_weights=auto`` remains meaningful.
    """
    name = (name or "ce").lower()
    ce = nn.CrossEntropyLoss(weight=class_weights)
    if name == "ce":
        return ce

    losses = _smp_losses()
    mode = "multiclass"
    if name == "ce_dice":
        return CombinedLoss([(1.0, ce), (1.0, losses.DiceLoss(mode=mode, from_logits=True))])
    if name == "focal":
        return losses.FocalLoss(mode=mode)
    if name == "focal_dice":
        return CombinedLoss([
            (1.0, losses.FocalLoss(mode=mode)),
            (1.0, losses.DiceLoss(mode=mode, from_logits=True)),
        ])
    if name == "lovasz":
        return losses.LovaszLoss(mode=mode, from_logits=True)
    raise ValueError("Unknown loss {!r}. Options: ce, ce_dice, focal, focal_dice, lovasz".format(name))
