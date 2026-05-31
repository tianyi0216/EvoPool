"""EvoPool: shared compute_metrics helpers for single-label and multi-label downstream training."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


# ── single-label ──────────────────────────────────────────────────────


def compute_metrics(eval_pred) -> Dict[str, Any]:
    """Standard single-label classification metrics with per-class F1.

    Compatible with HuggingFace Trainer.compute_metrics signature.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
    }
    unique_labels = sorted(set(labels))
    per_class_f1 = f1_score(labels, preds, labels=unique_labels, average=None, zero_division=0)
    for lab, f1_val in zip(unique_labels, per_class_f1):
        metrics[f"f1_class_{lab}"] = float(f1_val)
    return metrics


# ── multi-label ───────────────────────────────────────────────────────


def compute_ml_metrics(pred_logits: np.ndarray, y_multihot: np.ndarray,
                       threshold: float = 0.5,
                       logits_are_probs: bool = True) -> Dict[str, Any]:
    """Paper-conv multi-label metrics: per-class binary F1, macro/micro/weighted/sample F1.

    If ``logits_are_probs`` is False, applies sigmoid before thresholding.
    """
    if logits_are_probs:
        probs = pred_logits
    else:
        probs = 1.0 / (1.0 + np.exp(-pred_logits))
    P = (probs >= threshold).astype(np.int8)
    Y = y_multihot.astype(np.int8)
    N, K = Y.shape
    tp = ((P == 1) & (Y == 1)).sum(axis=0).astype(float)
    fp = ((P == 1) & (Y == 0)).sum(axis=0).astype(float)
    fn = ((P == 0) & (Y == 1)).sum(axis=0).astype(float)
    prec_c = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec_c = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1_c = np.divide(2 * prec_c * rec_c, prec_c + rec_c,
                     out=np.zeros_like(tp), where=(prec_c + rec_c) > 0)
    macro_f1 = float(f1_c.mean())
    tp_s, fp_s, fn_s = float(tp.sum()), float(fp.sum()), float(fn.sum())
    mp = tp_s / max(1e-12, tp_s + fp_s)
    mr = tp_s / max(1e-12, tp_s + fn_s)
    micro_f1 = 2 * mp * mr / max(1e-12, mp + mr)
    sup = Y.sum(axis=0).astype(float)
    w_f1 = float((f1_c * sup).sum() / sup.sum()) if sup.sum() > 0 else 0.0
    cov = float((P.sum(axis=1) > 0).mean())
    exact = float((P == Y).all(axis=1).mean())
    s_f1s = []
    for i in range(N):
        yt = set(np.where(Y[i] == 1)[0].tolist())
        yp = set(np.where(P[i] == 1)[0].tolist())
        if not yt and not yp:
            s_f1s.append(1.0)
        elif not yp or not yt:
            s_f1s.append(0.0)
        else:
            inter = len(yt & yp)
            p_ = inter / len(yp); r_ = inter / len(yt)
            s_f1s.append(2 * p_ * r_ / (p_ + r_) if (p_ + r_) > 0 else 0.0)
    sample_f1 = float(np.mean(s_f1s)) if s_f1s else 0.0
    return {
        "accuracy": exact,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": w_f1,
        "sample_f1": sample_f1,
        "coverage": cov,
        "f1_macro": macro_f1,         # alias for downstream compatibility
        "f1_weighted": w_f1,          # alias for downstream compatibility
        "per_class_f1": [round(float(x), 4) for x in f1_c.tolist()],
    }


def hf_compute_metrics_ml(eval_pred) -> Dict[str, Any]:
    """HuggingFace Trainer-compatible wrapper that applies sigmoid first."""
    logits, labels = eval_pred
    return compute_ml_metrics(np.asarray(logits), np.asarray(labels),
                              threshold=0.5, logits_are_probs=False)
