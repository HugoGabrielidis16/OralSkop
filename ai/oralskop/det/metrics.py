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


def box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU for ``xyxy`` boxes."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)
    boxes1 = boxes1.float()
    boxes2 = boxes2.float()
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    return inter / (area1[:, None] + area2[None, :] - inter).clamp(min=1e-9)


def new_match_stats(score_threshold: float = 0.5, iou_threshold: float = 0.5) -> dict:
    """Running validation stats for thresholded detection metrics."""
    return {
        "score_threshold": float(score_threshold),
        "iou_threshold": float(iou_threshold),
        "preds": 0,
        "targets": 0,
        "class_tp": 0,
        "loc_matches": 0,
        "loc_iou_sum": 0.0,
        "loc_class_correct": 0,
    }


def _greedy_matches(pred: dict, tgt: dict, *, iou_threshold: float, class_aware: bool) -> list[tuple[int, int, float]]:
    pred_boxes = pred["boxes"].cpu().float()
    tgt_boxes = tgt["boxes"].cpu().float()
    if pred_boxes.numel() == 0 or tgt_boxes.numel() == 0:
        return []

    pred_labels = pred["labels"].cpu().long()
    tgt_labels = tgt["labels"].cpu().long()
    scores = pred.get("scores")
    if scores is None:
        order = torch.arange(len(pred_boxes))
    else:
        order = torch.argsort(scores.cpu().float(), descending=True)
    ious = box_iou_xyxy(pred_boxes, tgt_boxes)

    used_targets: set[int] = set()
    matches = []
    for pred_idx_t in order:
        pred_idx = int(pred_idx_t)
        candidate_ious = ious[pred_idx].clone()
        if class_aware:
            candidate_ious[tgt_labels != pred_labels[pred_idx]] = -1.0
        for target_idx in used_targets:
            candidate_ious[target_idx] = -1.0
        best_iou, target_idx_t = candidate_ious.max(dim=0)
        if float(best_iou) >= iou_threshold:
            target_idx = int(target_idx_t)
            used_targets.add(target_idx)
            matches.append((pred_idx, target_idx, float(best_iou)))
    return matches


def update_match_stats(stats: dict, preds: list[dict], targets: list[dict]) -> None:
    """Update precision/recall/F1 and matched-box localization/class stats."""
    score_threshold = float(stats["score_threshold"])
    iou_threshold = float(stats["iou_threshold"])
    for pred, tgt in zip(preds, targets):
        scores = pred.get("scores")
        if scores is not None:
            keep = scores.cpu().float() >= score_threshold
            pred = {k: v[keep] if torch.is_tensor(v) and len(v) == len(keep) else v for k, v in pred.items()}

        stats["preds"] += int(len(pred["boxes"]))
        stats["targets"] += int(len(tgt["boxes"]))

        class_matches = _greedy_matches(pred, tgt, iou_threshold=iou_threshold, class_aware=True)
        stats["class_tp"] += len(class_matches)

        loc_matches = _greedy_matches(pred, tgt, iou_threshold=iou_threshold, class_aware=False)
        stats["loc_matches"] += len(loc_matches)
        stats["loc_iou_sum"] += sum(iou for _, _, iou in loc_matches)
        for pred_idx, target_idx, _ in loc_matches:
            if int(pred["labels"][pred_idx]) == int(tgt["labels"][target_idx]):
                stats["loc_class_correct"] += 1


def summarize_match_stats(stats: dict) -> dict:
    """Return thresholded detector metrics from ``new_match_stats`` counts."""
    preds = max(int(stats["preds"]), 0)
    targets = max(int(stats["targets"]), 0)
    class_tp = max(int(stats["class_tp"]), 0)
    loc_matches = max(int(stats["loc_matches"]), 0)
    precision = class_tp / preds if preds else 0.0
    recall = class_tp / targets if targets else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    mean_iou = float(stats["loc_iou_sum"]) / loc_matches if loc_matches else 0.0
    class_acc = int(stats["loc_class_correct"]) / loc_matches if loc_matches else 0.0
    return {
        "precision_50": precision,
        "recall_50": recall,
        "f1_50": f1,
        "mean_iou_50": mean_iou,
        "matched_class_accuracy_50": class_acc,
        "match_score_threshold": float(stats["score_threshold"]),
    }


def format_per_class(summary: dict) -> str:
    pc = summary.get("per_class_ap") or {}
    lines = [f"{'class':28s} {'AP@0.5:0.95':>12s}"]
    for name, ap in sorted(pc.items(), key=lambda kv: -kv[1]):
        lines.append(f"{name:28s} {ap:12.4f}")
    return "\n".join(lines)
