"""EvoPool: shared eval utilities used by the aggregator and downstream.

Minimal I/O primitives needed by ``src.aggregator.*``. Full annotator-evaluation
logic lives in ``src.pipeline.eval.*``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


ABSTAIN: int = -1


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
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
                # Skip malformed lines rather than aborting.
                continue
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
