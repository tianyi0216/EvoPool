"""EvoPool: Majority-vote (MV) aggregator.

The pipeline eval step (``src.pipeline.eval.run_annotators``) already writes
``aggregated_labels.majority_vote`` and ``aggregated_labels.weighted_vote`` into
each row of ``{train,val,test}_labeled.jsonl``. This module is a thin shim that
re-emits those files into a separate dump dir so that the downstream trainers
can consume MV labels via the same ``--hard_label_path`` interface they use for
EvoAgg.

Selected when ``config.aggregator.method == 'mv'`` in the project config.yaml.
CLI:
    python -m src.aggregator.majority_vote \\
        --eval_dir runs/chemprot/.../iter_12/eval \\
        --dump_dir runs/chemprot/.../mv_labels
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from src.tasks.configs import get_task_config
from src.utils.eval import ABSTAIN, read_jsonl


def _majority_vote_label(
    votes: Dict[str, Any], num_classes: int
) -> int:
    """Plain majority vote over per-annotator votes. Returns -1 if abstain-only."""
    counts: Counter = Counter()
    for v in votes.values():
        try:
            v_int = int(v)
        except (TypeError, ValueError):
            continue
        if 0 <= v_int < num_classes:
            counts[v_int] += 1
    if not counts:
        return ABSTAIN
    # Ties broken by lowest class id for determinism.
    top = max(counts.values())
    winners = sorted([c for c, k in counts.items() if k == top])
    return int(winners[0])


def _multilabel_or(
    votes: Dict[str, Any], num_classes: int
) -> List[int]:
    """Per-class OR aggregation for multi-label tasks: a class is positive if
    at least one annotator voted for it."""
    pos: set = set()
    for v in votes.values():
        try:
            v_int = int(v)
        except (TypeError, ValueError):
            continue
        if 0 <= v_int < num_classes:
            pos.add(v_int)
    return sorted(pos)


def dump_majority_vote_labels(
    rows_by_split: Dict[str, List[Dict[str, Any]]],
    dump_dir: Path,
    num_classes: int,
    multi_label: bool = False,
) -> None:
    """Write ``{split}_labeled.jsonl`` files under ``dump_dir`` with MV labels.

    For single-label tasks, the existing ``aggregated_labels.majority_vote`` is
    trusted if present and otherwise recomputed from ``votes``. For multi-label
    tasks, ``aggregated_labels.mv_multi`` is the per-class OR of votes.
    """
    dump_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in rows_by_split.items():
        out_path = dump_dir / f"{split}_labeled.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for r in rows:
                rr = dict(r)
                votes = rr.get("votes", {}) or rr.get("annotator_votes", {}) or {}
                agg = dict(rr.get("aggregated_labels") or {})
                if multi_label:
                    agg["mv_multi"] = _multilabel_or(votes, num_classes)
                else:
                    if "majority_vote" not in agg or agg["majority_vote"] is None:
                        agg["majority_vote"] = _majority_vote_label(votes, num_classes)
                    # Mirror weighted_vote so downstream sees the same key.
                    if "weighted_vote" not in agg or agg["weighted_vote"] is None:
                        agg["weighted_vote"] = agg["majority_vote"]
                rr["aggregated_labels"] = agg
                f.write(json.dumps(rr, ensure_ascii=False) + "\n")
        print(f"[dump-mv] {out_path} ({len(rows)} rows)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True)
    p.add_argument("--eval_dir", type=Path, required=True,
                   help="iter_XX/eval directory containing {train,val,test}_labeled.jsonl")
    p.add_argument("--dump_dir", type=Path, required=True)
    args = p.parse_args()

    task = get_task_config(args.task)
    multi_label = bool(getattr(task, "multi_label", False))
    rows_by_split: Dict[str, List[Dict[str, Any]]] = {}
    for split in ("train", "val", "test"):
        path = args.eval_dir / f"{split}_labeled.jsonl"
        if path.exists():
            rows_by_split[split] = read_jsonl(path)
            print(f"[load-mv] {split}: {len(rows_by_split[split])} rows")

    dump_majority_vote_labels(
        rows_by_split, args.dump_dir,
        num_classes=task.num_classes, multi_label=multi_label,
    )


if __name__ == "__main__":
    main()
