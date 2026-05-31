"""EvoPool: Aggregator ablation suite (A1-A6).

Runs the six aggregator variants used in the EvoPool paper ablation table on a
single iter_XX/eval directory and reports per-variant val + test macro F1:

  baseline_hybrid      class-balanced precision-weighted hybrid (no learning)
  v2_ensemble          3-model ensemble (LR-CV + GBDT + LR-L1) on votes + dense feats
  A1_textaware         sentence-BERT(text) plus vote one-hot -> LR-CV (EvoAgg)
  A2_perclass_OvR      C binary one-vs-rest LRs
  A3_CV_stacking       5-fold OOF predictions on v2 base learners then meta-LR
  A4_naivebayes_LM     Snorkel-style P(LF_i | Y) naive Bayes (no snorkel dep)
  A5_temp_scaled_v2    v2 ensemble plus scalar temperature calibration on val
  A6_LF_clustered_v2   cluster annotators by vote correlation, keep top-precision
                       per cluster, then re-fit v2

This script is NOT invoked by the default ``run_evopool.sh`` launcher; users run
it only when they want to reproduce the aggregator ablation cells.

CLI:
    python -m src.aggregator.ablation_suite \\
        --task chemprot \\
        --eval_dir runs/chemprot/.../iter_12/eval \\
        --out_dir  runs/chemprot/.../aggregator_ablation \\
        [--skip_a1]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from src.tasks.configs import get_task_config
from src.utils.eval import ABSTAIN, read_jsonl

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


# ---------------------------------------------------------------------------
# Shared featurization
# ---------------------------------------------------------------------------
def _build_annotator_vocab(rows_list: List[List[Dict]]) -> List[str]:
    names = set()
    for rs in rows_list:
        for r in rs:
            v = r.get("votes", {}) or {}
            if isinstance(v, dict):
                names.update(v.keys())
    return sorted(names)


def _per_annotator_precision(val_rows, annotator_names, num_classes):
    out = {}
    for n in annotator_names:
        tp, fires = 0, 0
        for r in val_rows:
            v = (r.get("votes") or {}).get(n, ABSTAIN)
            try:
                v = int(v)
            except (TypeError, ValueError):
                v = ABSTAIN
            if v == ABSTAIN:
                continue
            fires += 1
            if v == int(r.get("true_label", -2)):
                tp += 1
        out[n] = tp / fires if fires else 0.0
    return out


def _build_features(rows, annotator_names, num_classes, ann_prec):
    """Returns sparse votes one-hot (N, K*(C+1)) plus dense aggregate (N, 2C+2)."""
    n, k = len(rows), len(annotator_names)
    c1 = num_classes + 1
    name_to_idx = {n_: i for i, n_ in enumerate(annotator_names)}
    row_idx, col_idx = [], []
    vote_count = np.zeros((n, num_classes), dtype=np.float32)
    prec_score = np.zeros((n, num_classes), dtype=np.float32)
    abstain_rate = np.zeros(n, dtype=np.float32)
    vote_entropy = np.zeros(n, dtype=np.float32)
    for ri, r in enumerate(rows):
        votes = r.get("votes") or {}
        n_abst, n_total = 0, 0
        for name, v in votes.items():
            i = name_to_idx.get(name)
            if i is None:
                continue
            try:
                v = int(v)
            except (TypeError, ValueError):
                continue
            n_total += 1
            if v == ABSTAIN:
                n_abst += 1
                row_idx.append(ri)
                col_idx.append(i * c1 + 0)
            elif 0 <= v < num_classes:
                row_idx.append(ri)
                col_idx.append(i * c1 + 1 + v)
                vote_count[ri, v] += 1
                prec_score[ri, v] += ann_prec.get(name, 0.0)
        if n_total > 0:
            abstain_rate[ri] = n_abst / n_total
        s = vote_count[ri].sum()
        if s > 0:
            p = vote_count[ri] / s
            with np.errstate(divide="ignore", invalid="ignore"):
                vote_entropy[ri] = -np.nansum(p * np.log(p + 1e-12))
    Xs = sparse.csr_matrix(
        (np.ones(len(row_idx), dtype=np.float32), (row_idx, col_idx)),
        shape=(n, k * c1),
    )
    Xd = np.concatenate(
        [vote_count, prec_score, abstain_rate.reshape(-1, 1), vote_entropy.reshape(-1, 1)],
        axis=1,
    )
    return Xs, Xd


def _true_labels(rows):
    return np.array([int(r.get("true_label", -1)) for r in rows], dtype=int)


def _example_ids(rows):
    return [str(r.get("id", i)) for i, r in enumerate(rows)]


def _macro_f1(P, y, num_classes):
    valid = y >= 0
    if not valid.any():
        return None
    preds = np.argmax(P, axis=1)
    return f1_score(
        y[valid], preds[valid], average="macro",
        labels=list(range(num_classes)), zero_division=0,
    )


def _pad_proba(P, classes, num_classes):
    if list(classes) == list(range(num_classes)):
        return P
    out = np.zeros((P.shape[0], num_classes), dtype=P.dtype)
    for j, c in enumerate(classes):
        c = int(c)
        if 0 <= c < num_classes:
            out[:, c] = P[:, j]
    return out


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------
def _baseline_hybrid(rows_train, rows_val, rows_test, num_classes, ann_prec):
    def predict(rows):
        n = len(rows)
        proba = np.zeros((n, num_classes), dtype=np.float32)
        for ri, r in enumerate(rows):
            scores = np.zeros(num_classes, dtype=np.float32)
            for name, v in (r.get("votes") or {}).items():
                try:
                    v = int(v)
                except (TypeError, ValueError):
                    continue
                if 0 <= v < num_classes:
                    scores[v] += ann_prec.get(name, 0.0)
            s = scores.sum()
            proba[ri] = scores / s if s > 0 else 1.0 / num_classes
        return proba
    return {sp: predict(rs) for sp, rs in zip(
        ["train", "val", "test"], [rows_train, rows_val, rows_test])}


def _v2_ensemble(Xs, Xd, y_val, splits, num_classes, seed=42):
    Xc_train = sparse.hstack([Xs["val"], sparse.csr_matrix(Xd["val"])]).tocsr()
    Xc_splits = {
        sp: sparse.hstack([Xs[sp], sparse.csr_matrix(Xd[sp])]).tocsr() for sp in splits
    }
    n_folds = min(5, max(2, np.bincount(y_val).min()))
    lrcv = LogisticRegressionCV(
        Cs=[0.01, 0.1, 1.0, 10.0], cv=n_folds, scoring="f1_macro",
        class_weight="balanced", solver="lbfgs", max_iter=2000, random_state=seed,
    )
    lrcv.fit(Xc_train, y_val)
    Pl = {sp: _pad_proba(lrcv.predict_proba(Xc_splits[sp]), lrcv.classes_, num_classes) for sp in splits}
    gbdt = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=8,
        class_weight="balanced", random_state=seed,
        early_stopping=True, validation_fraction=0.15,
    )
    gbdt.fit(Xc_train.toarray(), y_val)
    Pg = {sp: _pad_proba(gbdt.predict_proba(Xc_splits[sp].toarray()), gbdt.classes_, num_classes) for sp in splits}
    lrl1 = LogisticRegression(
        C=0.1, penalty="l1", solver="saga", class_weight="balanced",
        max_iter=1000, random_state=seed, tol=1e-3,
    )
    lrl1.fit(Xc_train, y_val)
    Pll = {sp: _pad_proba(lrl1.predict_proba(Xc_splits[sp]), lrl1.classes_, num_classes) for sp in splits}
    return {sp: (Pl[sp] + Pg[sp] + Pll[sp]) / 3.0 for sp in splits}


def _a1_textaware(rows_train, rows_val, rows_test, Xs, Xd, y_val, splits, num_classes,
                  seed=42, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  [A1] sentence-transformers not installed; skipping")
        return None
    print(f"  [A1] encoding text with {model_name} ...")
    t0 = time.time()
    model = SentenceTransformer(model_name)
    emb_train = model.encode([r.get("text", "") for r in rows_train], batch_size=64,
                              show_progress_bar=False, convert_to_numpy=True,
                              normalize_embeddings=True)
    emb_val = model.encode([r.get("text", "") for r in rows_val], batch_size=64,
                            show_progress_bar=False, convert_to_numpy=True,
                            normalize_embeddings=True)
    emb_test = model.encode([r.get("text", "") for r in rows_test], batch_size=64,
                             show_progress_bar=False, convert_to_numpy=True,
                             normalize_embeddings=True)
    print(f"  [A1] encoded in {time.time() - t0:.1f}s, dim={emb_val.shape[1]}")

    Xa_train_full = np.concatenate([emb_val, Xd["val"]], axis=1)
    Xa_splits = {
        "train": np.concatenate([emb_train, Xd["train"]], axis=1),
        "val":   np.concatenate([emb_val,   Xd["val"]],   axis=1),
        "test":  np.concatenate([emb_test,  Xd["test"]],  axis=1),
    }
    n_folds = min(5, max(2, int(np.bincount(y_val).min())))
    oof_val = None
    if n_folds >= 2:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        oof_val = np.zeros((len(y_val), num_classes), dtype=np.float32)
        print(f"  [A1] computing OOF val proba via {n_folds}-fold StratifiedKFold")
        for fi, (tr_idx, te_idx) in enumerate(skf.split(Xa_train_full, y_val), 1):
            lr_fold = LogisticRegressionCV(
                Cs=[0.01, 0.1, 1.0, 10.0], cv=5, scoring="f1_macro",
                class_weight="balanced", solver="lbfgs", max_iter=2000, random_state=seed,
            )
            lr_fold.fit(Xa_train_full[tr_idx], y_val[tr_idx])
            oof_val[te_idx] = _pad_proba(
                lr_fold.predict_proba(Xa_train_full[te_idx]), lr_fold.classes_, num_classes
            )
    lr = LogisticRegressionCV(
        Cs=[0.01, 0.1, 1.0, 10.0], cv=5, scoring="f1_macro",
        class_weight="balanced", solver="lbfgs", max_iter=2000, random_state=seed,
    )
    lr.fit(Xa_train_full, y_val)
    out = {sp: _pad_proba(lr.predict_proba(Xa_splits[sp]), lr.classes_, num_classes) for sp in splits}
    if oof_val is not None and "val" in out:
        out["val"] = oof_val
    return out


def _a2_perclass_ovr(Xs, Xd, y_val, splits, num_classes, seed=42):
    Xc_train = sparse.hstack([Xs["val"], sparse.csr_matrix(Xd["val"])]).tocsr()
    Xc_splits = {sp: sparse.hstack([Xs[sp], sparse.csr_matrix(Xd[sp])]).tocsr() for sp in splits}
    P_class = {sp: np.zeros((Xc_splits[sp].shape[0], num_classes), dtype=np.float32) for sp in splits}
    for c in range(num_classes):
        y_bin = (y_val == c).astype(int)
        if y_bin.sum() < 2:
            continue
        lr = LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs",
                                 max_iter=2000, random_state=seed)
        lr.fit(Xc_train, y_bin)
        for sp in splits:
            p = lr.predict_proba(Xc_splits[sp])
            pos_idx = list(lr.classes_).index(1) if 1 in lr.classes_ else -1
            if pos_idx >= 0:
                P_class[sp][:, c] = p[:, pos_idx]
    for sp in splits:
        s = P_class[sp].sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        P_class[sp] = P_class[sp] / s
    return P_class


def _a3_cv_stacking(Xs, Xd, y_val, splits, num_classes, seed=42):
    Xc_train = sparse.hstack([Xs["val"], sparse.csr_matrix(Xd["val"])]).tocsr()
    Xc_splits = {sp: sparse.hstack([Xs[sp], sparse.csr_matrix(Xd[sp])]).tocsr() for sp in splits}
    n_val = Xc_train.shape[0]
    n_folds = min(5, max(2, np.bincount(y_val).min()))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    base_models = ["lrcv", "gbdt", "lrl1"]
    OOF = {b: np.zeros((n_val, num_classes), dtype=np.float32) for b in base_models}
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(n_val), y_val)):
        Xtr = Xc_train[tr_idx]; ytr = y_val[tr_idx]; Xva = Xc_train[va_idx]
        m = LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs",
                                max_iter=2000, random_state=seed).fit(Xtr, ytr)
        OOF["lrcv"][va_idx] = _pad_proba(m.predict_proba(Xva), m.classes_, num_classes)
        m = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_depth=6,
            class_weight="balanced", random_state=seed,
            early_stopping=True, validation_fraction=0.15,
        ).fit(Xtr.toarray(), ytr)
        OOF["gbdt"][va_idx] = _pad_proba(m.predict_proba(Xva.toarray()), m.classes_, num_classes)
        m = LogisticRegression(
            C=0.1, penalty="l1", solver="saga", class_weight="balanced",
            max_iter=1000, random_state=seed, tol=1e-3,
        ).fit(Xtr, ytr)
        OOF["lrl1"][va_idx] = _pad_proba(m.predict_proba(Xva), m.classes_, num_classes)
    OOF_stack = np.concatenate([OOF[b] for b in base_models], axis=1)
    meta = LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs",
                                max_iter=2000, random_state=seed).fit(OOF_stack, y_val)
    base_full = {
        "lrcv": LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs",
                                    max_iter=2000, random_state=seed).fit(Xc_train, y_val),
        "gbdt": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_depth=6,
            class_weight="balanced", random_state=seed,
            early_stopping=True, validation_fraction=0.15,
        ).fit(Xc_train.toarray(), y_val),
        "lrl1": LogisticRegression(
            C=0.1, penalty="l1", solver="saga", class_weight="balanced",
            max_iter=1000, random_state=seed, tol=1e-3,
        ).fit(Xc_train, y_val),
    }
    P_out = {}
    for sp in splits:
        P_split_stack = np.concatenate([
            _pad_proba(base_full["lrcv"].predict_proba(Xc_splits[sp]), base_full["lrcv"].classes_, num_classes),
            _pad_proba(base_full["gbdt"].predict_proba(Xc_splits[sp].toarray()), base_full["gbdt"].classes_, num_classes),
            _pad_proba(base_full["lrl1"].predict_proba(Xc_splits[sp]), base_full["lrl1"].classes_, num_classes),
        ], axis=1)
        P_out[sp] = _pad_proba(meta.predict_proba(P_split_stack), meta.classes_, num_classes)
    return P_out


def _a4_naivebayes_lm(rows_train, rows_val, rows_test, annotator_names, num_classes, eps=1.0):
    y_val_arr = np.array([int(r.get("true_label", -1)) for r in rows_val])
    valid = y_val_arr >= 0
    pri = np.zeros(num_classes)
    for c in range(num_classes):
        pri[c] = (y_val_arr[valid] == c).sum() + 1
    pri /= pri.sum()
    log_pri = np.log(pri)
    n_LF = len(annotator_names)
    counts = np.full((n_LF, num_classes + 1, num_classes), eps, dtype=np.float64)
    name_to_idx = {n: i for i, n in enumerate(annotator_names)}
    for r in rows_val:
        y = int(r.get("true_label", -1))
        if y < 0 or y >= num_classes:
            continue
        votes = r.get("votes") or {}
        for n in annotator_names:
            v = votes.get(n, ABSTAIN)
            try:
                v = int(v)
            except (TypeError, ValueError):
                v = ABSTAIN
            v_idx = 0 if v == ABSTAIN else (1 + v if 0 <= v < num_classes else 0)
            counts[name_to_idx[n], v_idx, y] += 1
    log_p_vote_given_y = np.log(counts / counts.sum(axis=1, keepdims=True))

    def predict(rows):
        n = len(rows)
        proba = np.zeros((n, num_classes), dtype=np.float32)
        for ri, r in enumerate(rows):
            log_post = log_pri.copy()
            votes = r.get("votes") or {}
            for name in annotator_names:
                v = votes.get(name, ABSTAIN)
                try:
                    v = int(v)
                except (TypeError, ValueError):
                    v = ABSTAIN
                v_idx = 0 if v == ABSTAIN else (1 + v if 0 <= v < num_classes else 0)
                log_post += log_p_vote_given_y[name_to_idx[name], v_idx, :]
            log_post -= log_post.max()
            p = np.exp(log_post)
            proba[ri] = p / p.sum()
        return proba

    return {sp: predict(rs) for sp, rs in zip(
        ["train", "val", "test"], [rows_train, rows_val, rows_test])}


def _a5_temperature_scaled(P_v2, y_val, splits, num_classes):
    P_val_log = np.log(np.clip(P_v2["val"], 1e-12, 1.0))
    valid = y_val >= 0
    best_T, best_nll = 1.0, float("inf")
    for T in [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0, 3.0]:
        p_T = np.exp(P_val_log / T)
        p_T = p_T / p_T.sum(axis=1, keepdims=True)
        ll = np.log(np.clip(p_T[np.arange(len(y_val))[valid], y_val[valid]], 1e-12, 1.0)).sum()
        nll = -ll
        if nll < best_nll:
            best_nll = nll
            best_T = T
    print(f"  [A5] best T = {best_T}")
    out = {}
    for sp in splits:
        p_log = np.log(np.clip(P_v2[sp], 1e-12, 1.0))
        p_T = np.exp(p_log / best_T)
        out[sp] = p_T / p_T.sum(axis=1, keepdims=True)
    return out


def _a6_annotator_clustered_v2(rows_train, rows_val, rows_test, annotator_names,
                               ann_prec, num_classes, splits,
                               n_clusters_target=50, seed=42):
    if len(annotator_names) <= n_clusters_target:
        print(f"  [A6] only {len(annotator_names)} annotators, skipping clustering")
        return None
    name_to_idx = {n: i for i, n in enumerate(annotator_names)}
    n_val = len(rows_val)
    vote_matrix = np.zeros((len(annotator_names), n_val), dtype=np.int8)
    for j, r in enumerate(rows_val):
        for n, v in (r.get("votes") or {}).items():
            i = name_to_idx.get(n)
            if i is None:
                continue
            try:
                v = int(v)
            except (TypeError, ValueError):
                continue
            vote_matrix[i, j] = v + 2  # offset so abstain doesn't collide
    print(f"  [A6] clustering {len(annotator_names)} annotators into {n_clusters_target}")
    cl = AgglomerativeClustering(n_clusters=n_clusters_target, linkage="average", metric="hamming")
    cluster_id = cl.fit_predict(vote_matrix)
    keep = []
    for cid in range(n_clusters_target):
        members = [annotator_names[i] for i in range(len(annotator_names)) if cluster_id[i] == cid]
        if not members:
            continue
        members.sort(key=lambda n: ann_prec.get(n, 0.0), reverse=True)
        keep.append(members[0])
    print(f"  [A6] kept {len(keep)} representative annotators")
    Xs_red, Xd_red = {}, {}
    for sp, rs in zip(["train", "val", "test"], [rows_train, rows_val, rows_test]):
        Xs_red[sp], Xd_red[sp] = _build_features(rs, keep, num_classes, ann_prec)
    y_val = _true_labels(rows_val)
    return _v2_ensemble(Xs_red, Xd_red, y_val, splits, num_classes, seed=seed)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True)
    p.add_argument("--eval_dir", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_a1", action="store_true",
                   help="skip text-aware (no torch / sentence-transformers)")
    p.add_argument("--st_model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = p.parse_args()

    task = get_task_config(args.task)
    num_classes = task.num_classes

    out_dir = args.out_dir if args.out_dir else args.eval_dir.parent / "aggregator_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "val", "test"]
    rows: Dict[str, List[Dict]] = {}
    for sp in splits:
        f = args.eval_dir / f"{sp}_labeled.jsonl"
        if not f.exists():
            print(f"[error] {f} not found")
            sys.exit(2)
        rows[sp] = read_jsonl(f)
        print(f"[load] {sp}: {len(rows[sp])} rows")

    annotator_names = _build_annotator_vocab(list(rows.values()))
    print(f"[vocab] {len(annotator_names)} annotators")
    ann_prec = _per_annotator_precision(rows["val"], annotator_names, num_classes)

    print("[featurize] building (sparse + dense) features for all splits")
    Xs, Xd = {}, {}
    for sp in splits:
        Xs[sp], Xd[sp] = _build_features(rows[sp], annotator_names, num_classes, ann_prec)
    y = {sp: _true_labels(rows[sp]) for sp in splits}
    ids = {sp: _example_ids(rows[sp]) for sp in splits}

    results: Dict[str, Dict[str, Optional[float]]] = {}

    print("\n=== variant: baseline_hybrid ===")
    P_base = _baseline_hybrid(rows["train"], rows["val"], rows["test"], num_classes, ann_prec)
    results["baseline_hybrid"] = {sp: _macro_f1(P_base[sp], y[sp], num_classes) for sp in splits}

    print("\n=== variant: v2_ensemble ===")
    t0 = time.time()
    P_v2 = _v2_ensemble(Xs, Xd, y["val"], splits, num_classes, seed=args.seed)
    print(f"  v2 fit time: {time.time() - t0:.1f}s")
    results["v2_ensemble"] = {sp: _macro_f1(P_v2[sp], y[sp], num_classes) for sp in splits}

    P_a1 = None
    if not args.skip_a1:
        print("\n=== variant: A1_textaware ===")
        try:
            P_a1 = _a1_textaware(rows["train"], rows["val"], rows["test"], Xs, Xd,
                                  y["val"], splits, num_classes,
                                  seed=args.seed, model_name=args.st_model)
            if P_a1 is not None:
                results["A1_textaware"] = {sp: _macro_f1(P_a1[sp], y[sp], num_classes) for sp in splits}
        except Exception as e:
            print(f"  [A1 error] {e}")
            results["A1_textaware"] = {sp: None for sp in splits}
    else:
        results["A1_textaware"] = {sp: None for sp in splits}

    print("\n=== variant: A2_perclass_OvR ===")
    P_a2 = _a2_perclass_ovr(Xs, Xd, y["val"], splits, num_classes, seed=args.seed)
    results["A2_perclass_OvR"] = {sp: _macro_f1(P_a2[sp], y[sp], num_classes) for sp in splits}

    print("\n=== variant: A3_CV_stacking ===")
    P_a3 = _a3_cv_stacking(Xs, Xd, y["val"], splits, num_classes, seed=args.seed)
    results["A3_CV_stacking"] = {sp: _macro_f1(P_a3[sp], y[sp], num_classes) for sp in splits}

    print("\n=== variant: A4_naivebayes_LM ===")
    P_a4 = _a4_naivebayes_lm(rows["train"], rows["val"], rows["test"], annotator_names, num_classes)
    results["A4_naivebayes_LM"] = {sp: _macro_f1(P_a4[sp], y[sp], num_classes) for sp in splits}

    print("\n=== variant: A5_temp_scaled_v2 ===")
    P_a5 = _a5_temperature_scaled(P_v2, y["val"], splits, num_classes)
    results["A5_temp_scaled_v2"] = {sp: _macro_f1(P_a5[sp], y[sp], num_classes) for sp in splits}

    print("\n=== variant: A6_LF_clustered_v2 ===")
    P_a6 = _a6_annotator_clustered_v2(rows["train"], rows["val"], rows["test"],
                                       annotator_names, ann_prec, num_classes, splits,
                                       n_clusters_target=50, seed=args.seed)
    if P_a6 is not None:
        results["A6_LF_clustered_v2"] = {sp: _macro_f1(P_a6[sp], y[sp], num_classes) for sp in splits}
    else:
        results["A6_LF_clustered_v2"] = {sp: None for sp in splits}

    def _save(P, tag):
        for sp in splits:
            with open(out_dir / f"{tag}_{sp}_proba.jsonl", "w") as f:
                for i, _id in enumerate(ids[sp]):
                    f.write(json.dumps({
                        "id": _id,
                        "true_label": int(y[sp][i]) if y[sp][i] >= 0 else None,
                        "soft_label": int(np.argmax(P[sp][i])),
                        "proba": [round(float(x), 6) for x in P[sp][i]],
                    }) + "\n")

    _save(P_base, "baseline_hybrid")
    _save(P_v2, "v2_ensemble")
    if P_a1 is not None:
        _save(P_a1, "A1_textaware")
    _save(P_a2, "A2_perclass_OvR")
    _save(P_a3, "A3_CV_stacking")
    _save(P_a4, "A4_naivebayes_LM")
    _save(P_a5, "A5_temp_scaled_v2")
    if P_a6 is not None:
        _save(P_a6, "A6_LF_clustered_v2")

    print("\n" + "=" * 75)
    print(f"{'variant':<28} {'val_F1':>8} {'test_F1':>8}  delta vs hybrid (test)")
    print("-" * 75)
    base_test = results["baseline_hybrid"].get("test")
    for v, m in results.items():
        v_v = m.get("val")
        t_v = m.get("test")
        delta = (t_v - base_test) if (t_v is not None and base_test is not None) else None
        v_str = f"{v_v:.4f}" if v_v is not None else "  -- "
        t_str = f"{t_v:.4f}" if t_v is not None else "  -- "
        d_str = f"{delta:+.4f}" if delta is not None else "  -- "
        print(f"{v:<28} {v_str:>8} {t_str:>8}  {d_str}")

    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "val_F1", "test_F1"])
        for v, m in results.items():
            w.writerow([v, m.get("val"), m.get("test")])

    with open(out_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[done] outputs in {out_dir}")


if __name__ == "__main__":
    main()
