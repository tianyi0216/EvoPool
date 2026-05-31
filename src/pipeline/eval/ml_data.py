"""EvoPool: multi-label data loading + per-class-OR aggregation utilities.

Each row has `true_labels: List[int]` (set of positive class indices). The
internal binary-vector representation is a K-dim numpy {0, 1} array per row,
indexed by class id.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def load_multilabel_jsonl(path: Path, n_classes: int,
                          label_key: str = "true_labels") -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Read JSONL with multi-label rows.

    Each row must contain `text` and `label_key` (list of int OR list of str
    class-names that we don't auto-resolve — caller resolves first if needed).

    Returns:
        rows: list of dicts (original row + parsed `true_labels` field)
        Y:    (N, K) binary numpy array — Y[i, c] = 1 iff class c in row i
    """
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            # Normalize label field
            raw = r.get(label_key)
            if raw is None:
                # Allow `true_label` legacy (single int → wrap as list)
                lt = r.get("true_label")
                if lt is None:
                    raise KeyError(f"row {r.get('id', '?')} missing both "
                                   f"{label_key} and true_label")
                raw = [int(lt)]
            elif not isinstance(raw, (list, tuple)):
                raw = [raw]
            labels = sorted({int(x) for x in raw})
            r["true_labels"] = labels
            rows.append(r)
    Y = np.zeros((len(rows), n_classes), dtype=np.int8)
    for i, r in enumerate(rows):
        for c in r["true_labels"]:
            if 0 <= c < n_classes:
                Y[i, c] = 1
    return rows, Y


def label_distribution(Y: np.ndarray) -> Dict[int, int]:
    """Per-class positive count."""
    return {int(c): int(v) for c, v in enumerate(Y.sum(axis=0))}


def cardinality(Y: np.ndarray) -> float:
    """Average number of labels per example."""
    return float(Y.sum() / max(1, Y.shape[0]))


def density(Y: np.ndarray) -> float:
    """Average fraction of classes assigned per example."""
    K = max(1, Y.shape[1])
    return float(Y.sum() / max(1, Y.shape[0]) / K)


def votes_to_binary_matrix(votes_list: List[Dict[str, int]],
                            lf_names: List[str],
                            n_classes: int) -> np.ndarray:
    """Convert list of {lf_name: class_idx_or_ABSTAIN} into (N, L, K) tensor
    of binary indicators where T[i, l, c] = 1 iff LF l fired class c on row i.

    For multi-label aggregation, the standard 'predict c if any LF voted c'
    reduces to T.any(axis=1) → (N, K).
    """
    ABSTAIN = -1
    name_to_idx = {n: i for i, n in enumerate(lf_names)}
    N = len(votes_list)
    L = len(lf_names)
    T = np.zeros((N, L, n_classes), dtype=np.int8)
    for i, votes in enumerate(votes_list):
        for lf_name, v in (votes or {}).items():
            if v is None:
                continue
            try:
                vi = int(v)
            except (TypeError, ValueError):
                continue
            if vi == ABSTAIN or vi < 0 or vi >= n_classes:
                continue
            li = name_to_idx.get(lf_name)
            if li is None:
                continue
            T[i, li, vi] = 1
    return T


def aggregate_majority_or(T: np.ndarray) -> np.ndarray:
    """Multi-label MV-OR: predict c iff at least 1 LF voted c.

    Args:
        T: (N, L, K) binary tensor from votes_to_binary_matrix.
    Returns:
        Y_pred: (N, K) binary array.
    """
    return T.any(axis=1).astype(np.int8)


def aggregate_majority_threshold(T: np.ndarray, threshold: int = 1) -> np.ndarray:
    """Predict c iff >= threshold LFs voted c."""
    counts = T.sum(axis=1)  # (N, K)
    return (counts >= threshold).astype(np.int8)


def collect_lf_names_from_votes(splits_votes: Dict[str, List[Dict[str, int]]]) -> List[str]:
    """Union of LF names across all splits, sorted."""
    names = set()
    for vlist in splits_votes.values():
        for v in vlist or []:
            if v:
                names.update(v.keys())
    return sorted(names)
