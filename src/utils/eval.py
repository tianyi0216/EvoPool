"""EvoPool: shared eval utilities used by the aggregator and downstream.

This module contains the minimal I/O primitives needed by ``src.aggregator.*``.
The full annotator-evaluation logic (which actually executes annotator pools on
dataset splits, writes ``{split}_labeled.jsonl``, and computes per-annotator
metrics) lives in ``src.pipeline.eval.run_annotators`` and
``src.pipeline.eval.analyze_errors`` (owned by the pipeline agent).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


ABSTAIN: int = -1


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts. Blank lines are skipped."""
    rows: List[Dict[str, Any]] = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # Be lenient: skip malformed lines rather than aborting.
                continue
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write a list of dicts to JSONL. Parent dirs are created if missing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
