"""EvoPool: multi-label metrics (paper-convention abstain-aware variants).

Sample-wise / micro / macro / per-class scores via sklearn.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)


def per_class_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> List[Dict[str, float]]:
    """For each class c independently, compute binary precision/recall/F1.

    Args:
        y_true: (N, K) binary
        y_pred: (N, K) binary
    Returns:
        list of K dicts with prec/rec/f1/n_pred/n_gold/tp.
    """
    K = y_true.shape[1]
    out = []
    for c in range(K):
        yt = y_true[:, c]
        yp = y_pred[:, c]
        tp = int(((yt == 1) & (yp == 1)).sum())
        n_pred = int((yp == 1).sum())
        n_gold = int((yt == 1).sum())
        prec = tp / n_pred if n_pred > 0 else 0.0
        rec = tp / n_gold if n_gold > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        out.append({
            "class": c,
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "n_pred": n_pred,
            "n_gold": n_gold,
            "tp": tp,
        })
    return out


def multilabel_summary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Headline multi-label metrics.

    macro_f1   = mean of per-class binary F1
    micro_f1   = global F1 over all (sample, class) pairs
    sample_f1  = mean of per-sample F1 over the K-dim binary vectors
    weighted_f1= per-class F1 weighted by support
    subset_acc = exact-match accuracy (predicted set == true set)
    hamming    = average wrong-bit fraction
    coverage_anyone = fraction of samples where >=1 class predicted positive
    avg_pred_card  = mean number of positive predictions per sample
    avg_true_card  = mean number of true labels per sample
    """
    return {
        "macro_f1":   float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1":   float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "sample_f1":  float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall":    float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "subset_acc": float((y_true == y_pred).all(axis=1).mean()),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "coverage_anyone": float((y_pred.sum(axis=1) > 0).mean()),
        "avg_pred_card": float(y_pred.sum(axis=1).mean()),
        "avg_true_card": float(y_true.sum(axis=1).mean()),
        "n_total": int(y_true.shape[0]),
        "n_classes": int(y_true.shape[1]),
    }


def per_class_f1_for_orchestrator(y_true: np.ndarray, y_pred: np.ndarray,
                                  n_classes: int) -> Dict[int, float]:
    """Format expected by orchestrator's _per_class_f1_from_iter callers:
    {class_idx: f1} dict over all K classes."""
    pc = per_class_binary_metrics(y_true, y_pred)
    return {m["class"]: m["f1"] for m in pc}
