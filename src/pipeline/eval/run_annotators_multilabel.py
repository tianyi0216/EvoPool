"""EvoPool: multi-label annotator pool evaluator (per-class OR aggregation).

Multi-label counterpart to run_annotators.py: aggregation is per-class OR
(predict c if any annotator voted c); true labels are sets / binary vectors.

Usage:
  python -m src.pipeline.eval.run_annotators_multilabel \
    --pool_module runs/pubmed/baselines/.../pool.py \
    --splits data/processed/pubmed/{train,val,test}.jsonl \
    --n_classes 14 \
    --out_dir runs/pubmed/baselines/.../eval
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

# Allow direct execution by adding sibling dir to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from multilabel_data import (
    aggregate_majority_or,
    aggregate_majority_threshold,
    load_multilabel_jsonl,
    votes_to_binary_matrix,
)
from multilabel_metrics import multilabel_summary, per_class_binary_metrics

ABSTAIN: int = -1


# ---------------------------------------------------------------------------
# Pool loading (identical to single-label)
# ---------------------------------------------------------------------------
def load_pool_module(module_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("pool_module", str(module_path.resolve()))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_annotators(mod: Any) -> List[Tuple[str, Callable]]:
    annotators = getattr(mod, "ANNOTATORS", None)
    if annotators is None:
        raise AttributeError("Pool module has no ANNOTATORS list.")
    return [(fn.__name__, fn) for fn in annotators]


# ---------------------------------------------------------------------------
# annotator execution
# ---------------------------------------------------------------------------
def run_single_lf(lf: Callable, examples: Sequence[Dict[str, Any]]) -> List[int]:
    votes: List[int] = []
    for ex in examples:
        try:
            v = lf(ex)
        except Exception:
            v = ABSTAIN
        if not isinstance(v, int) or (v != ABSTAIN and v < 0):
            v = ABSTAIN
        votes.append(v)
    return votes


def run_pool(annotators: List[Tuple[str, Callable]],
             examples: Sequence[Dict[str, Any]],
             n_workers: int = 8) -> List[List[int]]:
    """Returns vote_matrix[lf_idx][ex_idx]."""
    fns = [fn for (_, fn) in annotators]
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(lambda f: run_single_lf(f, examples), fns))
    return results


# ---------------------------------------------------------------------------
# Per-annotator metrics on multi-label gold
# ---------------------------------------------------------------------------
def per_lf_metrics(votes_matrix: List[List[int]],
                   lf_names: List[str],
                   Y: np.ndarray,
                   n_classes: int) -> List[Dict[str, Any]]:
    """For each annotator: fires, coverage, polarity (set of classes it votes for),
    and per-class hit-rate (precision per class it targets)."""
    N, K = Y.shape
    out = []
    for li, (name, votes_col) in enumerate(zip(lf_names, votes_matrix)):
        v = np.array(votes_col)
        fires = int((v != ABSTAIN).sum())
        cov = fires / N if N > 0 else 0.0
        # Polarity = classes this annotator ever voted
        polarity = sorted({int(x) for x in v if x != ABSTAIN and 0 <= int(x) < K})
        # Per-class hit-rate: when annotator votes c, what fraction of those samples
        # actually have class c positive? (annotator-precision under multi-label OR)
        hit_per_class: Dict[int, Dict[str, float]] = {}
        for c in polarity:
            mask = (v == c)
            n_fires_c = int(mask.sum())
            if n_fires_c == 0:
                continue
            n_correct_c = int(Y[mask, c].sum())
            hit_per_class[c] = {
                "fires": n_fires_c,
                "hits": n_correct_c,
                "precision": n_correct_c / n_fires_c,
            }
        out.append({
            "name": name,
            "fires": fires,
            "coverage": float(cov),
            "polarity": polarity,
            "per_class_hit_rate": hit_per_class,
        })
    return out


# ---------------------------------------------------------------------------
# Aggregation + report
# ---------------------------------------------------------------------------
def evaluate_split(split_name: str,
                   rows: List[Dict[str, Any]],
                   Y: np.ndarray,
                   annotators: List[Tuple[str, Callable]],
                   n_classes: int,
                   n_workers: int = 8) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run annotator pool on split, aggregate, return (report_block, labeled_rows)."""
    if not annotators:
        # Empty pool — emit zero predictions
        Y_pred = np.zeros_like(Y)
        votes_per_ex = [{} for _ in rows]
        lf_names = []
        lf_metrics = []
    else:
        votes_matrix = run_pool(annotators, rows, n_workers=n_workers)  # (L, N)
        lf_names = [n for (n, _) in annotators]
        lf_metrics = per_lf_metrics(votes_matrix, lf_names, Y, n_classes)
        # Per-example votes dict
        votes_per_ex: List[Dict[str, int]] = []
        for i in range(len(rows)):
            d = {}
            for li, name in enumerate(lf_names):
                vi = int(votes_matrix[li][i])
                if vi != ABSTAIN:
                    d[name] = vi
            votes_per_ex.append(d)
        # Aggregate
        T = votes_to_binary_matrix(votes_per_ex, lf_names, n_classes)  # (N, L, K)
        Y_pred = aggregate_majority_or(T)  # (N, K)

    metrics = multilabel_summary(Y, Y_pred)
    per_class = per_class_binary_metrics(Y, Y_pred)
    agg_block = {
        "strategy": "or_aggregation",
        **metrics,
        "per_class": {str(m["class"]): m for m in per_class},
    }

    labeled_rows: List[Dict[str, Any]] = []
    for i, r in enumerate(rows):
        new_r = dict(r)
        new_r["votes"] = votes_per_ex[i]
        # Predicted positive set
        pred_pos = [int(c) for c in range(n_classes) if Y_pred[i, c] == 1]
        new_r["aggregated_labels"] = {
            "majority_vote": pred_pos,  # list of int (positive class indices)
            "majority_vote_binary": Y_pred[i].tolist(),
        }
        labeled_rows.append(new_r)

    report_block = {
        "split": split_name,
        "n_examples": len(rows),
        "lf_metrics": lf_metrics,
        "aggregation": {"majority_vote": agg_block},
    }
    return report_block, labeled_rows


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool_module", type=Path, required=True,
                   help="Path to pool.py with ANNOTATORS list")
    p.add_argument("--splits", type=Path, nargs="+", required=True,
                   help="JSONL paths (typically train, val, test in that order)")
    p.add_argument("--n_classes", type=int, required=True)
    p.add_argument("--label_key", default="true_labels")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--n_workers", type=int, default=8)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    mod = load_pool_module(args.pool_module)
    annotators = get_annotators(mod)
    print(f"[ml-run] loaded {len(annotators)} annotators from {args.pool_module}")

    # Infer split names from file stems (train_labeled.jsonl → train; val.jsonl → val)
    split_names = []
    for p_in in args.splits:
        stem = p_in.stem
        for tag in ("train", "val", "test", "dev"):
            if tag in stem:
                split_names.append(tag)
                break
        else:
            split_names.append(stem)

    splits_report: List[Dict[str, Any]] = []
    for name, p_in in zip(split_names, args.splits):
        rows, Y = load_multilabel_jsonl(p_in, args.n_classes, label_key=args.label_key)
        print(f"[ml-run] {name}: n={len(rows)}, "
              f"avg_card={Y.sum() / max(1, len(rows)):.2f}, "
              f"per_class_pos={Y.sum(axis=0).tolist()}")
        block, labeled = evaluate_split(name, rows, Y, annotators, args.n_classes,
                                         n_workers=args.n_workers)
        mv = block["aggregation"]["majority_vote"]
        print(f"  [{name}] macro_f1={mv['macro_f1']:.4f} micro_f1={mv['micro_f1']:.4f} "
              f"cov_any={mv['coverage_anyone']:.3f} avg_pred_card={mv['avg_pred_card']:.2f}")
        splits_report.append(block)
        out_jsonl = args.out_dir / f"{name}_labeled.jsonl"
        with out_jsonl.open("w") as f:
            for r in labeled:
                f.write(json.dumps(r) + "\n")
        print(f"  wrote {out_jsonl}")

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pool_module": str(args.pool_module),
        "num_annotators": len(annotators),
        "annotator_names": [n for (n, _) in annotators],
        "n_classes": args.n_classes,
        "splits": splits_report,
    }
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nWrote combined report: {args.out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
