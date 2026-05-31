#!/usr/bin/env python3
"""EvoPool: single-label annotator pool evaluator (MV / WV aggregation).

Executes an annotator pool on dataset splits, aggregates votes, and evaluates.

Outputs per split:
  - labeled JSONL  (each row gets `votes`, `programmatic_label`, `agg_details`)
  - per-annotator metrics (precision, coverage, fires on this split)
  - aggregated metrics (accuracy, F1, precision, recall, coverage)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


ABSTAIN = -1


# ---------------------------------------------------------------------------
# Data I/O
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


# ---------------------------------------------------------------------------
# Load annotator pool module dynamically
# ---------------------------------------------------------------------------


def load_pool_module(module_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("pool_module", str(module_path.resolve()))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def get_annotators(mod: Any) -> List[Tuple[str, Callable]]:
    annotators = getattr(mod, "ANNOTATORS", None)
    if annotators is None:
        raise AttributeError("Pool module has no ANNOTATORS list.")
    return [(fn.__name__, fn) for fn in annotators]


# ---------------------------------------------------------------------------
# Execute LFs
# ---------------------------------------------------------------------------


def run_single_lf(
    lf: Callable,
    examples: Sequence[Dict[str, Any]],
) -> List[int]:
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


def run_pool(
    annotators: List[Tuple[str, Callable]],
    examples: Sequence[Dict[str, Any]],
    n_workers: Optional[int] = None,
) -> List[List[int]]:
    """Returns vote_matrix[lf_idx][ex_idx].

    Executes each annotator in a thread pool.  Python-level annotator code is
    mostly ``re`` / ``str`` ops that release the GIL, so threading gives a
    meaningful speedup on large pools (300+ annotators × 10k+ examples) without
    having to pickle annotator callables (which breaks for importlib-loaded
    module-level functions).
    """
    if n_workers is None:
        n_workers = int(os.environ.get("RUN_ANNOTATORS_PARALLEL", "4"))
    if n_workers <= 1 or len(annotators) <= 1:
        return [run_single_lf(lf, examples) for _, lf in annotators]

    from concurrent.futures import ThreadPoolExecutor
    out: List[Optional[List[int]]] = [None] * len(annotators)

    def _job(i_lf):
        i, (_name, lf) = i_lf
        return i, run_single_lf(lf, examples)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for i, votes in pool.map(_job, list(enumerate(annotators))):
            out[i] = votes
    return out  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Aggregation strategies
# ---------------------------------------------------------------------------


def aggregate_majority_vote(votes: List[int]) -> int:
    active = [v for v in votes if v != ABSTAIN]
    if not active:
        return ABSTAIN
    c = Counter(active)
    top = c.most_common()
    if len(top) >= 2 and top[0][1] == top[1][1]:
        return ABSTAIN  # tie → abstain
    return top[0][0]


def aggregate_weighted_vote(
    votes: List[int],
    weights: List[float],
) -> int:
    score: Dict[int, float] = {}
    for v, w in zip(votes, weights):
        if v == ABSTAIN:
            continue
        score[v] = score.get(v, 0.0) + w
    if not score:
        return ABSTAIN
    best = max(score, key=lambda k: score[k])
    vals = sorted(score.values(), reverse=True)
    if len(vals) >= 2 and abs(vals[0] - vals[1]) < 1e-9:
        return ABSTAIN  # tie
    return best


def aggregate_at_least_one(votes: List[int], target_label: int = 1) -> int:
    """If any LF votes target_label, return it; else abstain."""
    for v in votes:
        if v == target_label:
            return target_label
    return ABSTAIN


AGGREGATORS: Dict[str, str] = {
    "majority_vote": "Simple majority vote (ties → abstain)",
    "weighted_vote": "Precision-weighted vote (ties → abstain)",
}


# ---------------------------------------------------------------------------
# Per-LF metrics on a given split
# ---------------------------------------------------------------------------


@dataclass
class LFSplitMetrics:
    name: str
    fires: int = 0
    coverage: float = 0.0
    correct: int = 0
    incorrect: int = 0
    precision: Optional[float] = None
    predicted_labels: Dict[int, int] = field(default_factory=dict)


def compute_lf_metrics(
    lf_name: str,
    lf_votes: List[int],
    examples: Sequence[Dict[str, Any]],
    label_key: str,
) -> LFSplitMetrics:
    fires = 0
    correct = 0
    incorrect = 0
    label_counts: Dict[int, int] = {}
    n_labeled = 0

    for vote, ex in zip(lf_votes, examples):
        true_y = ex.get(label_key)
        if true_y is None:
            continue
        true_y = int(true_y)
        n_labeled += 1
        if vote == ABSTAIN:
            continue
        fires += 1
        label_counts[vote] = label_counts.get(vote, 0) + 1
        if vote == true_y:
            correct += 1
        else:
            incorrect += 1

    coverage = (fires / n_labeled) if n_labeled else 0.0
    precision = (correct / fires) if fires else None

    return LFSplitMetrics(
        name=lf_name,
        fires=fires,
        coverage=coverage,
        correct=correct,
        incorrect=incorrect,
        precision=precision,
        predicted_labels=label_counts,
    )


# ---------------------------------------------------------------------------
# Aggregated evaluation
# ---------------------------------------------------------------------------


@dataclass
class AggMetrics:
    strategy: str
    n_total: int = 0
    n_labeled: int = 0
    n_covered: int = 0
    coverage: float = 0.0
    n_correct: int = 0
    n_incorrect: int = 0
    accuracy_on_covered: Optional[float] = None
    accuracy_on_all: Optional[float] = None
    # Per-class metrics: {class_int: {"precision": ..., "recall": ..., "f1": ...}}
    per_class: Dict[int, Dict[str, Optional[float]]] = field(default_factory=dict)
    label_distribution: Dict[str, int] = field(default_factory=dict)
    # Backward-compat convenience (populated for binary tasks)
    precision_spam: Optional[float] = None
    recall_spam: Optional[float] = None
    f1_spam: Optional[float] = None
    precision_ham: Optional[float] = None
    recall_ham: Optional[float] = None
    f1_ham: Optional[float] = None


def _safe_div(a: int, b: int) -> Optional[float]:
    return (a / b) if b else None


def _f1(p: Optional[float], r: Optional[float]) -> Optional[float]:
    if p is None or r is None or (p + r) == 0:
        return None
    return 2 * p * r / (p + r)


def evaluate_aggregated(
    agg_labels: List[int],
    examples: Sequence[Dict[str, Any]],
    label_key: str,
    strategy_name: str,
    label_names: Optional[Dict[int, str]] = None,
) -> AggMetrics:
    n_correct = n_incorrect = n_covered = n_labeled = 0
    dist: Dict[str, int] = {"abstain": 0}

    # Collect all true labels and predictions
    all_classes: set = set()
    tp: Dict[int, int] = {}
    fp: Dict[int, int] = {}
    fn: Dict[int, int] = {}

    for pred, ex in zip(agg_labels, examples):
        true_y = ex.get(label_key)
        if true_y is None:
            continue
        true_y = int(true_y)
        all_classes.add(true_y)
        n_labeled += 1

        if pred == ABSTAIN:
            dist["abstain"] += 1
            fn[true_y] = fn.get(true_y, 0) + 1
            continue

        all_classes.add(pred)
        n_covered += 1
        cls_name = str(pred)
        if label_names and pred in label_names:
            cls_name = label_names[pred]
        dist[cls_name] = dist.get(cls_name, 0) + 1

        if pred == true_y:
            n_correct += 1
            tp[pred] = tp.get(pred, 0) + 1
        else:
            n_incorrect += 1
            fp[pred] = fp.get(pred, 0) + 1
            fn[true_y] = fn.get(true_y, 0) + 1

    # Per-class precision/recall/F1
    per_class: Dict[int, Dict[str, Optional[float]]] = {}
    for c in sorted(all_classes):
        p = _safe_div(tp.get(c, 0), tp.get(c, 0) + fp.get(c, 0))
        r = _safe_div(tp.get(c, 0), tp.get(c, 0) + fn.get(c, 0))
        per_class[c] = {"precision": p, "recall": r, "f1": _f1(p, r)}

    result = AggMetrics(
        strategy=strategy_name,
        n_total=len(examples),
        n_labeled=n_labeled,
        n_covered=n_covered,
        coverage=(n_covered / n_labeled) if n_labeled else 0.0,
        n_correct=n_correct,
        n_incorrect=n_incorrect,
        accuracy_on_covered=_safe_div(n_correct, n_covered),
        accuracy_on_all=_safe_div(n_correct, n_labeled),
        per_class=per_class,
        label_distribution=dist,
    )

    # Backward compat: populate spam/ham fields for binary tasks
    if 1 in per_class:
        result.precision_spam = per_class[1].get("precision")
        result.recall_spam = per_class[1].get("recall")
        result.f1_spam = per_class[1].get("f1")
    if 0 in per_class:
        result.precision_ham = per_class[0].get("precision")
        result.recall_ham = per_class[0].get("recall")
        result.f1_ham = per_class[0].get("f1")

    return result


# ---------------------------------------------------------------------------
# Build labeled output rows
# ---------------------------------------------------------------------------


def build_labeled_rows(
    examples: List[Dict[str, Any]],
    annotators: List[Tuple[str, Callable]],
    vote_matrix: List[List[int]],
    agg_results: Dict[str, List[int]],
    label_key: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    n_lfs = len(annotators)

    for i, ex in enumerate(examples):
        votes = {annotators[j][0]: vote_matrix[j][i] for j in range(n_lfs)}
        agg = {strategy: labels[i] for strategy, labels in agg_results.items()}

        row = {
            "id": ex.get("id"),
            "text": ex.get("text"),
            "true_label": ex.get(label_key),
            "votes": votes,
            "aggregated_labels": agg,
        }
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Process one split
# ---------------------------------------------------------------------------


def process_split(
    split_name: str,
    examples: List[Dict[str, Any]],
    annotators: List[Tuple[str, Callable]],
    lf_weights: List[float],
    label_key: str,
    out_dir: Path,
) -> Dict[str, Any]:
    print(f"\n--- {split_name} ({len(examples)} examples) ---")

    vote_matrix = run_pool(annotators, examples)

    # Per-LF metrics
    lf_metrics: List[Dict[str, Any]] = []
    for j, (name, _) in enumerate(annotators):
        m = compute_lf_metrics(name, vote_matrix[j], examples, label_key)
        lf_metrics.append(asdict(m))
        tag = f"  {name}: fires={m.fires} cov={m.coverage:.3f} prec={m.precision:.3f}" if m.precision is not None else f"  {name}: fires={m.fires} cov={m.coverage:.3f} prec=N/A"
        print(tag)

    # Aggregation
    n = len(examples)
    agg_results: Dict[str, List[int]] = {}

    # majority vote
    agg_results["majority_vote"] = [
        aggregate_majority_vote([vote_matrix[j][i] for j in range(len(annotators))])
        for i in range(n)
    ]

    # weighted vote (use precision from ANNOTATOR_METRICS as weights)
    agg_results["weighted_vote"] = [
        aggregate_weighted_vote(
            [vote_matrix[j][i] for j in range(len(annotators))],
            lf_weights,
        )
        for i in range(n)
    ]

    # Evaluate each strategy
    agg_eval: Dict[str, Dict[str, Any]] = {}
    for strategy_name, labels in agg_results.items():
        m = evaluate_aggregated(labels, examples, label_key, strategy_name)
        agg_eval[strategy_name] = asdict(m)
        def _fmt(v: Optional[float]) -> str:
            return f"{v:.3f}" if v is not None else "N/A"

        per_class_str = " ".join(
            f"f1_{c}={_fmt(m.per_class.get(c, {}).get('f1'))}"
            for c in sorted(m.per_class.keys())
        )
        print(
            f"  [{strategy_name}] coverage={m.coverage:.3f} "
            f"acc_covered={_fmt(m.accuracy_on_covered)} "
            f"acc_all={_fmt(m.accuracy_on_all)} "
            f"{per_class_str}"
        )

    # Write labeled JSONL
    labeled_rows = build_labeled_rows(examples, annotators, vote_matrix, agg_results, label_key)
    labeled_path = out_dir / f"{split_name}_labeled.jsonl"
    write_jsonl(labeled_path, labeled_rows)
    print(f"  Wrote {labeled_path}")

    return {
        "split": split_name,
        "n_examples": len(examples),
        "lf_metrics": lf_metrics,
        "aggregation": agg_eval,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run annotator pool on dataset splits, aggregate, and evaluate.")
    p.add_argument(
        "--pool_module",
        type=Path,
        default=Path("annotators/youtube_spam/initial_pool.py"),
        help="Path to the annotator pool Python module.",
    )
    p.add_argument(
        "--splits",
        type=Path,
        nargs="+",
        default=[
            Path("data/processed/youtube_spam_collection/train.jsonl"),
            Path("data/processed/youtube_spam_collection/val.jsonl"),
            Path("data/processed/youtube_spam_collection/test.jsonl"),
        ],
        help="Paths to JSONL split files.",
    )
    p.add_argument(
        "--label_key",
        type=str,
        default="true_label",
        help="Key for ground-truth label.",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("runs/youtube_spam/initial"),
        help="Output directory for labeled data and reports.",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    # Load pool
    mod = load_pool_module(args.pool_module)
    annotators = get_annotators(mod)
    print(f"Loaded {len(annotators)} annotators from {args.pool_module}")
    for name, _ in annotators:
        print(f"  - {name}")

    # Build weights from ANNOTATOR_METRICS (fall back to 1.0)
    metrics_dict = getattr(mod, "ANNOTATOR_METRICS", {}) or {}
    lf_weights: List[float] = []
    for name, _ in annotators:
        m = metrics_dict.get(name, {})
        prec = m.get("precision")
        lf_weights.append(float(prec) if prec is not None else 1.0)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_split_reports: List[Dict[str, Any]] = []
    for split_path in args.splits:
        if not split_path.exists():
            print(f"WARNING: {split_path} not found, skipping.", file=sys.stderr)
            continue
        split_name = split_path.stem  # e.g. "train", "val", "test"
        examples = read_jsonl(split_path)
        report = process_split(
            split_name=split_name,
            examples=examples,
            annotators=annotators,
            lf_weights=lf_weights,
            label_key=args.label_key,
            out_dir=args.out_dir,
        )
        all_split_reports.append(report)

    # Write combined report
    full_report = {
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "pool_module": str(args.pool_module),
        "num_annotators": len(annotators),
        "annotator_names": [name for name, _ in annotators],
        "splits": all_split_reports,
    }
    report_path = args.out_dir / "report.json"
    write_json(report_path, full_report)
    print(f"\nWrote combined report: {report_path}")


if __name__ == "__main__":
    main()
