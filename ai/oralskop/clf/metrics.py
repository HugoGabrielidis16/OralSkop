"""Multi-label classification metrics.

Threshold-free ranking metrics (per-class average precision, macro-mAP, micro-AP)
plus thresholded precision/recall/F1/accuracy. Classes with no positive ground-truth
in the evaluated split get ``nan`` AP and are excluded from the macro AP/F1 means
(so an absent class doesn't drag the score).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


def multilabel_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: list[str],
    *,
    threshold: float = 0.5,
) -> dict:
    """Compute ranking + thresholded multi-label metrics.

    ``y_true`` and ``y_score`` are ``[N, C]`` (binary targets / sigmoid scores).
    """
    n, c = y_true.shape
    support = y_true.sum(axis=0).astype(int)

    per_class_ap = np.full(c, np.nan, dtype=np.float64)
    for k in range(c):
        if support[k] > 0:
            per_class_ap[k] = average_precision_score(y_true[:, k], y_score[:, k])
    macro_map = float(np.nanmean(per_class_ap)) if np.isfinite(per_class_ap).any() else float("nan")

    present = support > 0
    micro_ap = (
        float(average_precision_score(y_true[:, present], y_score[:, present], average="micro"))
        if present.any()
        else float("nan")
    )

    y_pred = (y_score >= threshold).astype(int)
    per_class_accuracy = (y_pred == y_true).mean(axis=0)
    macro_accuracy = float(np.mean(per_class_accuracy[present])) if present.any() else float("nan")
    micro_accuracy = float((y_pred == y_true).mean())
    exact_match_accuracy = float((y_pred == y_true).all(axis=1).mean()) if n else float("nan")

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=list(range(c))
    )
    macro_f1 = float(np.mean(f1[present])) if present.any() else float("nan")
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )

    return {
        "num_samples": int(n),
        "macro_map": macro_map,
        "micro_ap": micro_ap,
        "macro_f1": macro_f1,
        "micro_f1": float(micro_f1),
        "macro_accuracy": macro_accuracy,
        "micro_accuracy": micro_accuracy,
        "exact_match_accuracy": exact_match_accuracy,
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "threshold": threshold,
        "class_names": class_names,
        "support": support.tolist(),
        "per_class_ap": per_class_ap.tolist(),
        "per_class_precision": p.tolist(),
        "per_class_recall": r.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_accuracy": per_class_accuracy.tolist(),
    }


def format_per_class(m: dict) -> str:
    """One line per class: name, support, AP, P/R/F1/accuracy (skips absent classes)."""
    lines = [
        f"{'class':28s} {'support':>8s} {'AP':>6s} {'P':>6s} {'R':>6s} "
        f"{'F1':>6s} {'Acc':>6s}"
    ]
    for i, name in enumerate(m["class_names"]):
        if m["support"][i] == 0:
            continue
        ap = m["per_class_ap"][i]
        ap_s = f"{ap:6.3f}" if ap == ap else "   nan"  # nan-check
        lines.append(
            f"{name:28s} {m['support'][i]:8d} {ap_s} "
            f"{m['per_class_precision'][i]:6.3f} {m['per_class_recall'][i]:6.3f} "
            f"{m['per_class_f1'][i]:6.3f} {m['per_class_accuracy'][i]:6.3f}"
        )
    return "\n".join(lines)
