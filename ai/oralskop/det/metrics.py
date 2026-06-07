"""Detection metrics — a thin wrapper over torchmetrics mean average precision.

Predictions and targets are passed as lists of dicts with ``boxes`` in **xyxy absolute**
pixel coordinates (consistent between preds and targets), matching torchmetrics' default.
"""

from __future__ import annotations

import os

import torch


def _use_headless_matplotlib_backend():
    """Avoid notebook-only matplotlib backends in tmux/subprocess metric imports."""
    backend = os.environ.get("MPLBACKEND", "")
    if backend.startswith("module://"):
        os.environ["MPLBACKEND"] = "Agg"


def new_map_metric(class_metrics: bool = True):
    """Create a fresh ``MeanAveragePrecision`` (xyxy boxes, per-class AP)."""
    _use_headless_matplotlib_backend()
    try:
        from torchmetrics.detection import MeanAveragePrecision
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError("Detection mAP needs the `det` extra (torchmetrics + pycocotools): "
                          "uv sync --extra det.") from exc
    return MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=class_metrics)


def summarize_map(result: dict, class_names: list[str]) -> dict:
    """Flatten a torchmetrics MAP result into plain floats + per-class AP names."""
    def f(x):
        return float(x) if torch.is_tensor(x) else float(x)

    out = {
        "map": f(result.get("map", float("nan"))),
        "map_50": f(result.get("map_50", float("nan"))),
        "map_75": f(result.get("map_75", float("nan"))),
        "mar_100": f(result.get("mar_100", float("nan"))),
    }
    per_class = result.get("map_per_class")
    classes = result.get("classes")
    if per_class is not None and classes is not None:
        pc = per_class.tolist() if torch.is_tensor(per_class) else list(per_class)
        cls = classes.tolist() if torch.is_tensor(classes) else list(classes)
        # map_per_class can be a scalar (-1) when only one class is present
        if isinstance(pc, (int, float)):
            pc, cls = [pc], [cls if not isinstance(cls, list) else cls[0]]
        out["per_class_ap"] = {
            class_names[c] if 0 <= c < len(class_names) else f"class_{c}": ap
            for c, ap in zip(cls, pc)
        }
    return out


def format_per_class(summary: dict) -> str:
    pc = summary.get("per_class_ap") or {}
    lines = [f"{'class':28s} {'AP@0.5:0.95':>12s}"]
    for name, ap in sorted(pc.items(), key=lambda kv: -kv[1]):
        lines.append(f"{name:28s} {ap:12.4f}")
    return "\n".join(lines)
