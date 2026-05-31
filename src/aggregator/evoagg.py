"""EvoPool: EvoAgg — learned text-aware annotator aggregator.

Combines sentence-BERT text embeddings with per-annotator vote one-hot features and
trains a LogisticRegressionCV on the val split (ground truth). Honest 5-fold
StratifiedKFold OOF predictions are used for the val split itself to avoid
in-sample inflation; train/test get predictions from a model fit on the full val.

This is the paper headline aggregator (A1 in the ablation table). Pseudo-labels
are written into ``aggregated_labels.majority_vote`` (overwriting the raw MV) so
any downstream trainer that consumes that field uses EvoAgg labels unchanged.

Multi-label tasks (``multi_label=True``) train one binary LR per class with the
same text+votes featurization; the dumped row carries
``aggregated_labels.evoagg_multi`` (List[int]) and a positive-class mask.

CLI:
    python -m src.aggregator.evoagg \\
        --task chemprot \\
        --eval_dir runs/chemprot/.../iter_12/eval \\
        --emb_path data/embeddings/chemprot_minilm_l6_v2.npz \\
        --dump_dir runs/chemprot/.../evoagg_labels
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold

from src.tasks.configs import get_task_config
from src.utils.eval import ABSTAIN, read_jsonl


# ---------------------------------------------------------------------------
# Featurization
# ---------------------------------------------------------------------------
def _build_vote_features(
    rows: List[Dict[str, Any]],
    annotator_names: List[str],
    num_classes: int,
) -> sparse.csr_matrix:
    """One-hot sparse matrix: row per example, columns = annotator x class.

    Abstain votes (value -1) and unknown annotators are silently skipped.
    """
    n = len(rows)
    k = len(annotator_names)
    name_to_idx = {n_: i for i, n_ in enumerate(annotator_names)}
    rows_idx: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    for i, r in enumerate(rows):
        votes = r.get("votes", {}) or r.get("annotator_votes", {}) or {}
        for name, vote in votes.items():
            j = name_to_idx.get(name)
            if j is None:
                continue
            try:
                v = int(vote)
            except (TypeError, ValueError):
                continue
            if v < 0 or v >= num_classes:
                continue
            rows_idx.append(i)
            cols.append(j * num_classes + v)
            data.append(1.0)
    return sparse.csr_matrix(
        (data, (rows_idx, cols)),
        shape=(n, k * num_classes),
        dtype=np.float32,
    )


def _collect_annotator_names(rows: List[Dict[str, Any]]) -> List[str]:
    names = set()
    for r in rows:
        votes = r.get("votes", {}) or r.get("annotator_votes", {}) or {}
        names.update(votes.keys())
    return sorted(names)


def _load_embeddings(
    emb_path: Optional[Path],
) -> Tuple[Optional[Dict[str, np.ndarray]], int]:
    """Load precomputed sentence-BERT embeddings. Returns (id->vec map, emb_dim)."""
    if emb_path is None or not emb_path.exists():
        return None, 0
    npz = np.load(emb_path, allow_pickle=True)
    ids = npz["ids"]
    embs = npz["embs"]
    id_to_emb = {str(i): e for i, e in zip(ids, embs)}
    return id_to_emb, int(embs.shape[1])


def _lookup_embs(
    rows: List[Dict[str, Any]],
    id_to_emb: Dict[str, np.ndarray],
    emb_dim: int,
) -> np.ndarray:
    out = np.zeros((len(rows), emb_dim), dtype=np.float32)
    for i, r in enumerate(rows):
        rid = str(r.get("id"))
        v = id_to_emb.get(rid)
        if v is not None:
            out[i] = v
    return out


def _stack_features(
    rows: List[Dict[str, Any]],
    annotator_names: List[str],
    num_classes: int,
    id_to_emb: Optional[Dict[str, np.ndarray]],
    emb_dim: int,
) -> np.ndarray:
    """Concatenate dense sentence-BERT embeddings (if available) with the
    densified sparse vote one-hot."""
    votes = _build_vote_features(rows, annotator_names, num_classes).toarray()
    if id_to_emb is None or emb_dim == 0:
        return votes
    embs = _lookup_embs(rows, id_to_emb, emb_dim)
    return np.concatenate([embs, votes], axis=1)


def _pad_proba(proba: np.ndarray, classes: np.ndarray, num_classes: int) -> np.ndarray:
    """Pad sklearn's predict_proba output to the full ``num_classes`` schema."""
    if list(classes) == list(range(num_classes)):
        return proba
    out = np.zeros((proba.shape[0], num_classes), dtype=proba.dtype)
    for j, c in enumerate(classes):
        c = int(c)
        if 0 <= c < num_classes:
            out[:, c] = proba[:, j]
    return out


# ---------------------------------------------------------------------------
# Single-label EvoAgg
# ---------------------------------------------------------------------------
def fit_predict_evoagg(
    train_rows: List[Dict[str, Any]],
    val_rows: List[Dict[str, Any]],
    test_rows: List[Dict[str, Any]],
    num_classes: int,
    emb_path: Optional[Path] = None,
    n_folds: int = 5,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Fit EvoAgg on val ground truth, predict on (train, val, test).

    Val predictions are honest out-of-fold; train and test use a model
    fit on the full val.

    Returns
    -------
    dict with keys "train", "val", "test" -> proba arrays of shape (N, num_classes).
    """
    annotator_names = _collect_annotator_names(val_rows)
    print(f"[evoagg] {len(annotator_names)} annotators discovered on val")

    id_to_emb, emb_dim = _load_embeddings(emb_path)
    if id_to_emb is None:
        print("[evoagg] no embeddings provided; using votes-only features")
    else:
        print(f"[evoagg] loaded {len(id_to_emb)} embeddings, dim={emb_dim}")

    X_val = _stack_features(val_rows, annotator_names, num_classes, id_to_emb, emb_dim)
    X_test = _stack_features(test_rows, annotator_names, num_classes, id_to_emb, emb_dim)
    X_train = _stack_features(train_rows, annotator_names, num_classes, id_to_emb, emb_dim) \
        if train_rows else np.zeros((0, X_val.shape[1]), dtype=np.float32)

    y_val = np.array([int(r.get("true_label", -1)) for r in val_rows])

    # --- 5-fold OOF for val (R094 fix; in-sample val inflated by ~25pp) ---
    counts = np.bincount(y_val[y_val >= 0])
    effective_folds = int(min(n_folds, max(2, counts.min() if len(counts) > 0 else 2)))
    oof_val: Optional[np.ndarray] = None
    if effective_folds >= 2 and len(y_val) >= 2 * effective_folds:
        skf = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=seed)
        oof_val = np.zeros((len(y_val), num_classes), dtype=np.float32)
        print(f"[evoagg] computing OOF val proba via {effective_folds}-fold StratifiedKFold")
        for fi, (tr_idx, te_idx) in enumerate(skf.split(X_val, y_val), 1):
            # Inner CV for LR-CV's C selection must also fit the fold's class support.
            inner_counts = np.bincount(y_val[tr_idx][y_val[tr_idx] >= 0])
            inner_cv = int(min(5, max(2, inner_counts.min() if len(inner_counts) > 0 else 2)))
            lr_fold = LogisticRegressionCV(
                Cs=[0.01, 0.1, 1.0, 10.0],
                cv=inner_cv,
                scoring="f1_macro",
                class_weight="balanced",
                solver="lbfgs",
                max_iter=2000,
                random_state=seed,
            )
            lr_fold.fit(X_val[tr_idx], y_val[tr_idx])
            oof_val[te_idx] = _pad_proba(
                lr_fold.predict_proba(X_val[te_idx]), lr_fold.classes_, num_classes
            )

    print(f"[evoagg] fitting final LogisticRegressionCV on full val (n={len(val_rows)})")
    full_counts = np.bincount(y_val[y_val >= 0])
    full_cv = int(min(5, max(2, full_counts.min() if len(full_counts) > 0 else 2)))
    lr = LogisticRegressionCV(
        Cs=[0.01, 0.1, 1.0, 10.0],
        cv=full_cv,
        scoring="f1_macro",
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
        random_state=seed,
    )
    lr.fit(X_val, y_val)

    proba = {
        "val": oof_val if oof_val is not None else _pad_proba(lr.predict_proba(X_val), lr.classes_, num_classes),
        "test": _pad_proba(lr.predict_proba(X_test), lr.classes_, num_classes),
    }
    if train_rows:
        proba["train"] = _pad_proba(lr.predict_proba(X_train), lr.classes_, num_classes)
    return proba


# ---------------------------------------------------------------------------
# Multi-label EvoAgg (one binary LR per class)
# ---------------------------------------------------------------------------
def fit_predict_evoagg_multilabel(
    train_rows: List[Dict[str, Any]],
    val_rows: List[Dict[str, Any]],
    test_rows: List[Dict[str, Any]],
    num_classes: int,
    emb_path: Optional[Path] = None,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Per-class binary LR aggregation for multi-label tasks.

    Returns
    -------
    dict with keys "train", "val", "test" -> proba arrays of shape (N, num_classes)
    where each column is the per-class positive probability.
    """
    annotator_names = _collect_annotator_names(val_rows)
    print(f"[evoagg-ml] {len(annotator_names)} annotators discovered on val")

    id_to_emb, emb_dim = _load_embeddings(emb_path)

    X_val = _stack_features(val_rows, annotator_names, num_classes, id_to_emb, emb_dim)
    X_test = _stack_features(test_rows, annotator_names, num_classes, id_to_emb, emb_dim)
    X_train = _stack_features(train_rows, annotator_names, num_classes, id_to_emb, emb_dim) \
        if train_rows else np.zeros((0, X_val.shape[1]), dtype=np.float32)

    def _y_multi(rows: List[Dict[str, Any]]) -> np.ndarray:
        Y = np.zeros((len(rows), num_classes), dtype=np.int8)
        for i, r in enumerate(rows):
            labels = r.get("true_labels", []) or []
            for c in labels:
                if 0 <= int(c) < num_classes:
                    Y[i, int(c)] = 1
        return Y

    Y_val = _y_multi(val_rows)
    P_val = np.zeros((len(val_rows), num_classes), dtype=np.float32)
    P_test = np.zeros((len(test_rows), num_classes), dtype=np.float32)
    P_train = np.zeros((len(train_rows), num_classes), dtype=np.float32) if train_rows else None

    for c in range(num_classes):
        y_bin = Y_val[:, c]
        if y_bin.sum() < 2 or y_bin.sum() > len(y_bin) - 2:
            # degenerate single-class — fall back to base-rate
            base = float(y_bin.mean()) if len(y_bin) > 0 else 0.0
            P_val[:, c] = base
            P_test[:, c] = base
            if P_train is not None:
                P_train[:, c] = base
            continue
        lr = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=2000,
            random_state=seed,
        )
        lr.fit(X_val, y_bin)
        pos_idx = list(lr.classes_).index(1) if 1 in lr.classes_ else -1
        if pos_idx < 0:
            continue
        P_val[:, c] = lr.predict_proba(X_val)[:, pos_idx]
        P_test[:, c] = lr.predict_proba(X_test)[:, pos_idx]
        if P_train is not None:
            P_train[:, c] = lr.predict_proba(X_train)[:, pos_idx]

    out = {"val": P_val, "test": P_test}
    if P_train is not None:
        out["train"] = P_train
    return out


# ---------------------------------------------------------------------------
# Dumping pseudo-labeled JSONL for downstream training
# ---------------------------------------------------------------------------
def _argmax_classes(proba: np.ndarray) -> np.ndarray:
    return np.argmax(proba, axis=1).astype(int)


def dump_evoagg_labels(
    rows_by_split: Dict[str, List[Dict[str, Any]]],
    proba_by_split: Dict[str, np.ndarray],
    dump_dir: Path,
    multi_label: bool = False,
    multi_label_threshold: float = 0.5,
) -> None:
    """Write {split}_labeled.jsonl files under ``dump_dir`` with EvoAgg labels.

    Single-label: overwrites ``aggregated_labels.majority_vote`` and
    ``weighted_vote`` with the EvoAgg argmax, and also writes ``evoagg`` field.

    Multi-label: writes ``aggregated_labels.evoagg_multi`` (list of positive
    class indices above the threshold) and ``evoagg_proba`` (the per-class
    probability vector, rounded to 6 decimals).
    """
    dump_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in rows_by_split.items():
        if split not in proba_by_split:
            continue
        proba = proba_by_split[split]
        out_path = dump_dir / f"{split}_labeled.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            if multi_label:
                for r, p in zip(rows, proba):
                    rr = dict(r)
                    agg = dict(rr.get("aggregated_labels") or {})
                    positives = [int(c) for c, pc in enumerate(p) if pc >= multi_label_threshold]
                    agg["evoagg_multi"] = positives
                    agg["evoagg_proba"] = [round(float(x), 6) for x in p]
                    rr["aggregated_labels"] = agg
                    f.write(json.dumps(rr, ensure_ascii=False) + "\n")
            else:
                preds = _argmax_classes(proba)
                for r, pred in zip(rows, preds):
                    rr = dict(r)
                    agg = dict(rr.get("aggregated_labels") or {})
                    agg["majority_vote"] = int(pred)
                    agg["weighted_vote"] = int(pred)
                    agg["evoagg"] = int(pred)
                    rr["aggregated_labels"] = agg
                    f.write(json.dumps(rr, ensure_ascii=False) + "\n")
        print(f"[dump] {out_path} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Lightweight evaluation
# ---------------------------------------------------------------------------
def _macro_f1(proba: np.ndarray, y_true: np.ndarray, num_classes: int) -> Tuple[float, List[float]]:
    pred = _argmax_classes(proba)
    tp = {c: 0 for c in range(num_classes)}
    fp = {c: 0 for c in range(num_classes)}
    fn = {c: 0 for c in range(num_classes)}
    for p_, y_ in zip(pred, y_true):
        if y_ < 0:
            continue
        if p_ == y_:
            tp[p_] += 1
        else:
            fp[p_] += 1
            fn[y_] += 1
    f1s = []
    for c in range(num_classes):
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return sum(f1s) / num_classes, f1s


def _print_eval(proba_by_split, rows_by_split, num_classes):
    for split, proba in proba_by_split.items():
        y = np.array([int(r.get("true_label", -1)) for r in rows_by_split[split]])
        macro, per = _macro_f1(proba, y, num_classes)
        print(f"[evoagg] {split} macro F1 = {macro:.4f}  per-class={[round(x, 3) for x in per]}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True, help="task name (must exist in src.tasks.configs)")
    p.add_argument("--eval_dir", type=Path, required=True,
                   help="iter_XX/eval directory containing {train,val,test}_labeled.jsonl")
    p.add_argument("--emb_path", type=Path, default=None,
                   help="Precomputed sentence-BERT embeddings .npz. If absent, falls back to "
                        "data/embeddings/<task>_minilm_l6_v2.npz; if that also is missing, "
                        "EvoAgg runs votes-only.")
    p.add_argument("--dump_dir", type=Path, default=None,
                   help="If set, write {train,val,test}_labeled.jsonl into this dir with "
                        "EvoAgg pseudo-labels. Consumed by downstream trainers.")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    task = get_task_config(args.task)
    num_classes = task.num_classes
    multi_label = bool(getattr(task, "multi_label", False))

    train_path = args.eval_dir / "train_labeled.jsonl"
    val_path = args.eval_dir / "val_labeled.jsonl"
    test_path = args.eval_dir / "test_labeled.jsonl"
    val_rows = read_jsonl(val_path)
    test_rows = read_jsonl(test_path)
    train_rows = read_jsonl(train_path) if train_path.exists() else []
    print(f"[evoagg] task={args.task} num_classes={num_classes} multi_label={multi_label}")
    print(f"  train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")

    # Resolve embedding path
    emb_path: Optional[Path] = args.emb_path
    if emb_path is None:
        candidate = Path(f"data/embeddings/{args.task}_minilm_l6_v2.npz")
        emb_path = candidate if candidate.exists() else None

    if multi_label:
        proba = fit_predict_evoagg_multilabel(
            train_rows, val_rows, test_rows, num_classes,
            emb_path=emb_path, seed=args.seed,
        )
    else:
        proba = fit_predict_evoagg(
            train_rows, val_rows, test_rows, num_classes,
            emb_path=emb_path, n_folds=args.n_folds, seed=args.seed,
        )
        _print_eval(proba, {"val": val_rows, "test": test_rows,
                            **({"train": train_rows} if train_rows else {})}, num_classes)

    if args.dump_dir is not None:
        rows_by_split = {"val": val_rows, "test": test_rows}
        if train_rows:
            rows_by_split["train"] = train_rows
        dump_evoagg_labels(
            rows_by_split, proba, args.dump_dir, multi_label=multi_label,
        )


if __name__ == "__main__":
    main()
