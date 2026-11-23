"""Evaluation metrics for DR grading.

Primary metric is Quadratic Weighted Kappa (QWK), the standard for ordinal DR
grading. We also provide accuracy, macro-F1, per-class recall, the confusion
matrix, and bootstrap 95% confidence intervals for any metric.
"""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    recall_score,
)

from .config import NUM_CLASSES


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=list(range(NUM_CLASSES))))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    labels = list(range(NUM_CLASSES))
    per_class_recall = recall_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    return {
        "qwk": quadratic_weighted_kappa(y_true, y_pred),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class_recall": {c: float(per_class_recall[c]) for c in labels},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "n": int(len(y_true)),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] = quadratic_weighted_kappa,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict:
    """Percentile bootstrap CI for a metric over (y_true, y_pred) pairs."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = metric_fn(y_true, y_pred)
    if n == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_boot": 0}
    stats: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        # Skip degenerate resamples (single class) for kappa stability.
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            stats.append(metric_fn(y_true[idx], y_pred[idx]))
        except Exception:
            continue
    if not stats:
        return {"point": float(point), "lo": float("nan"), "hi": float("nan"), "n_boot": 0}
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": float(point), "lo": float(lo), "hi": float(hi), "n_boot": len(stats)}


def paired_bootstrap_qwk_diff(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict:
    """Paired bootstrap of QWK(b) - QWK(a) on a *shared* test set.

    Both arms are evaluated on the identical ground truth (same Aizawl images),
    so each bootstrap resample uses the SAME indices for both arms -- this pairs
    the comparison and removes test-set sampling variance. Returns the observed
    difference, its 95% CI, and a two-sided bootstrap p-value (H0: diff == 0).
    Positive ``diff`` means arm ``b`` beats arm ``a``.
    """
    y_true = np.asarray(y_true).astype(int)
    pred_a = np.asarray(pred_a).astype(int)
    pred_b = np.asarray(pred_b).astype(int)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diff_point = quadratic_weighted_kappa(y_true, pred_b) - quadratic_weighted_kappa(y_true, pred_a)
    diffs: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            d = (quadratic_weighted_kappa(y_true[idx], pred_b[idx])
                 - quadratic_weighted_kappa(y_true[idx], pred_a[idx]))
            diffs.append(d)
        except Exception:
            continue
    if not diffs:
        return {"diff": float(diff_point), "lo": float("nan"), "hi": float("nan"),
                "p_value": float("nan"), "n_boot": 0}
    arr = np.asarray(diffs)
    lo, hi = np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # Two-sided bootstrap p-value: 2x the smaller tail mass beyond 0.
    p = 2.0 * min((arr <= 0).mean(), (arr >= 0).mean())
    return {"diff": float(diff_point), "lo": float(lo), "hi": float(hi),
            "p_value": float(min(1.0, p)), "n_boot": len(diffs)}
