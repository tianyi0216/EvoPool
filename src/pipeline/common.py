"""EvoPool: shared pipeline utilities for both single-label and multi-label task variants.

Dispatch keys:
  - read_jsonl: returns rows with BOTH `true_label` (int) and `true_labels`
    (List[int]) populated when the source contains either, so single-label
    and multi-label code read the same row without branching.
  - run_eval_subprocess / run_analyze_subprocess: route to single-label vs
    multi-label evaluator/analyzer based on `task_type` (src.tasks.configs).

Also provides: AST helper-prefix renamer (collision avoidance), ListPurger
(pool-prune safety), pool I/O, per-annotator evaluation, and filter chains
(threshold / Jaccard / train↔val consistency). Set EVOPOOL_FAST_EVAL=1 to
skip the train split during iter-loop eval.
"""
from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

# Re-export the LLM call + annotator parsing utilities from src.prompts so agents can
# `from src.pipeline.common import call_llm, ...` without knowing the layout.
from src.prompts.utils import (
    parse_lfs_from_response,
    build_callable_permissive,
    call_llm,
    stratified_seed_text,
)
from src.tasks.configs import get_task_config


# ────────────────────────────────────────────────────────────────────────
#  AST-level rename: prefix all top-level helpers + Assign constants with
#  a unique tag so blocks from different agents/iters don't collide.
# ────────────────────────────────────────────────────────────────────────

def prefix_helpers_in_block(block_src: str, tag: str) -> str:
    """Prefix all top-level helpers (NOT lf_*) and Assign constants with `_<tag>_`.

    Rewrites references within the block. Prevents helper-name collisions
    across agents/iters even when both blocks define `has_inhibition_keywords`.
    """
    try:
        tree = ast.parse(block_src)
    except SyntaxError:
        return block_src

    rename_map: Dict[str, str] = {}
    SKIP = {"ABSTAIN", "ANNOTATORS", "ANNOTATOR_METRICS", "POOL_METADATA"}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if node.name.startswith("lf_"):
                continue
            if node.name in SKIP:
                continue
            rename_map[node.name] = f"_{tag}_{node.name}"
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id not in SKIP:
                    rename_map[t.id] = f"_{tag}_{t.id}"

    if not rename_map:
        return block_src

    class _Renamer(ast.NodeTransformer):
        def visit_Name(self, n):
            if n.id in rename_map:
                return ast.copy_location(ast.Name(id=rename_map[n.id], ctx=n.ctx), n)
            return n

        def visit_FunctionDef(self, n):
            self.generic_visit(n)
            if n.name in rename_map:
                n.name = rename_map[n.name]
            return n

    new_tree = _Renamer().visit(tree)
    ast.fix_missing_locations(new_tree)
    return ast.unparse(new_tree)


class ListPurger(ast.NodeTransformer):
    """AST transformer dropping named identifiers from every list literal.

    Orchestrator uses this to clean ANNOTATORS / bundled-annotator lists after pool
    prune so the rewritten pool.py imports without NameError.
    """

    def __init__(self, drop_names: Set[str]):
        super().__init__()
        self.drop_names = set(drop_names)

    def visit_List(self, node):  # noqa: N802
        node.elts = [
            e for e in node.elts
            if not (isinstance(e, ast.Name) and e.id in self.drop_names)
        ]
        return node


# ────────────────────────────────────────────────────────────────────────
#  Pool I/O
# ────────────────────────────────────────────────────────────────────────

POOL_HEADER = """\
# Auto-generated annotator pool (EvoPool)
# {comment}

from __future__ import annotations
import re
import math
import collections
import itertools
from typing import Any, Dict, List

ABSTAIN: int = -1
"""


def write_pool(out_path: Path, blocks: List[str], lf_names: List[str], comment: str = ""):
    """Write pool.py from code blocks + lf_* names (caller orders blocks)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen_blocks: Set[str] = set()
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(POOL_HEADER.format(comment=comment) + "\n")
        for blk in blocks:
            blk = (blk or "").rstrip()
            if not blk or blk in seen_blocks:
                continue
            seen_blocks.add(blk)
            f.write(blk + "\n\n")
        f.write("\nANNOTATORS: List[Any] = [\n")
        for n in lf_names:
            if not n.startswith("lf_"):
                continue
            f.write(f"    {n},\n")
        f.write("]\n")


def read_pool_blocks_and_lfs(pool_path: Path) -> Tuple[List[str], List[str]]:
    src = pool_path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    lf_names = [n.name for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name.startswith("lf_")]
    return [src], lf_names


def extract_lf_sources_per_class(pool_path: Path) -> List[Dict[str, Any]]:
    """For Refiner: per-lf_* source segment extracted from pool.py."""
    src = pool_path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    out = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("lf_"):
            seg = "".join(lines[node.lineno - 1: node.end_lineno or node.lineno])
            out.append({"name": node.name, "src": seg.rstrip()})
    return out


# ────────────────────────────────────────────────────────────────────────
#  Build + evaluate callables from a pool.py
# ────────────────────────────────────────────────────────────────────────

def load_pool_callables(pool_path: Path) -> List[Tuple[str, Callable]]:
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"evopool_pool_{pool_path.parent.name}", pool_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return [(fn.__name__, fn) for fn in getattr(mod, "ANNOTATORS", [])]


def evaluate_lf_on_split(lf: Callable, examples: Sequence[Dict[str, Any]],
                          label_key: str) -> Dict[str, Any]:
    """Run lf on examples; return {fires, correct, precision, fired_ids, predicted_label, coverage}."""
    fires = 0
    correct = 0
    fired_ids: List[str] = []
    pred_labels: Set[int] = set()
    for ex in examples:
        try:
            v = lf(ex)
        except Exception:
            v = -1
        if v == -1:
            continue
        fires += 1
        fired_ids.append(str(ex.get("id", len(fired_ids))))
        try:
            pred_labels.add(int(v))
        except Exception:
            pass
        try:
            true_v = ex.get(label_key, -2)
            if isinstance(true_v, list):
                # Multi-label: correct if predicted label is in the gold set
                if int(v) in {int(x) for x in true_v if x is not None}:
                    correct += 1
            else:
                if v == int(true_v):
                    correct += 1
        except Exception:
            pass
    prec = (correct / fires) if fires else 0.0
    pred = next(iter(pred_labels)) if len(pred_labels) == 1 else None
    return {"fires": fires, "correct": correct, "precision": prec,
            "predicted_label": pred, "fired_ids": fired_ids,
            "coverage": fires / max(len(examples), 1)}


def filter_lfs_by_threshold(lfs_with_metrics: List[Tuple[str, str, Dict]],
                              min_precision: float = 0.25,
                              min_fires: int = 5) -> List[Tuple[str, str, Dict]]:
    return [(n, src, m) for n, src, m in lfs_with_metrics
            if (m.get("precision") or 0) >= min_precision
            and (m.get("fires") or 0) >= min_fires]


def collect_existing_fired_ids(pool_path: Path, val_rows: List[Dict[str, Any]],
                                 label_key: str = "true_label") -> Dict[str, set]:
    """{lf_name: set(fired_ids)} for every lf_* in pool when run on val."""
    out: Dict[str, set] = {}
    try:
        for name, fn in load_pool_callables(pool_path):
            fired: set = set()
            for ex in val_rows:
                try:
                    v = fn(ex)
                except Exception:
                    v = -1
                if v != -1:
                    fired.add(str(ex.get("id", id(ex))))
            out[name] = fired
    except Exception as e:
        print(f"  [collect_existing_fired_ids] WARN: {e}; returning empty (no dedup)")
    return out


def jaccard_dedup_against_pool(
    new_lfs: List[Tuple[str, str, Dict]],
    existing_fired_sets: Dict[str, set],
    max_overlap: float = 0.95,
) -> Tuple[List[Tuple[str, str, Dict]], List[Tuple[str, str]]]:
    """Drop new annotators whose fired_ids overlap >max_overlap with ANY existing one."""
    kept: List[Tuple[str, str, Dict]] = []
    dropped: List[Tuple[str, str]] = []
    for name, src, m in new_lfs:
        new_fired = set(m.get("fired_ids") or [])
        if not new_fired:
            dropped.append((name, "no fires"))
            continue
        max_jac = 0.0
        max_partner = ""
        for other_name, other_fired in existing_fired_sets.items():
            if not other_fired:
                continue
            inter = len(new_fired & other_fired)
            union = len(new_fired | other_fired)
            if union == 0:
                continue
            j = inter / union
            if j > max_jac:
                max_jac = j
                max_partner = other_name
        if max_jac > max_overlap:
            dropped.append((name, f"jaccard_dup of {max_partner} = {max_jac:.2f}"))
            continue
        kept.append((name, src, m))
        existing_fired_sets[name] = new_fired
    return kept, dropped


def consistency_filter_train_val(
    lfs_with_metrics: List[Tuple[str, str, Dict]],
    train_rows: List[Dict[str, Any]],
    label_key: str = "true_label",
    max_train_val_prec_gap: float = 0.30,
) -> Tuple[List[Tuple[str, str, Dict]], List[Tuple[str, str]]]:
    """Drop val-overfit annotators (val_prec - train_prec > gap)."""
    kept: List[Tuple[str, str, Dict]] = []
    dropped: List[Tuple[str, str]] = []
    for name, src, m_val in lfs_with_metrics:
        val_prec = m_val.get("precision") or 0
        try:
            lf = build_callable_permissive(src, name)
        except Exception:
            kept.append((name, src, m_val))
            continue
        m_train = evaluate_lf_on_split(lf, train_rows, label_key)
        train_prec = m_train.get("precision") or 0
        if val_prec - train_prec > max_train_val_prec_gap:
            dropped.append((name, f"val_prec={val_prec:.2f} train_prec={train_prec:.2f} gap>{max_train_val_prec_gap:.2f}"))
            continue
        m_val["train_precision"] = train_prec
        m_val["train_fires"] = m_train.get("fires") or 0
        kept.append((name, src, m_val))
    return kept, dropped


# ────────────────────────────────────────────────────────────────────────
#  Subprocess wrappers — route to single-label vs multi-label evaluators
# ────────────────────────────────────────────────────────────────────────

def _resolve_python() -> str:
    """Pick the Python executable for subprocesses.

    Uses the current interpreter when scikit-learn is importable. Set
    EVOPOOL_PYTHON to point at another interpreter if you keep scikit-learn in
    a separate environment.
    """
    try:
        import sklearn  # noqa: F401
        return sys.executable
    except ImportError:
        pass

    override = os.environ.get("EVOPOOL_PYTHON")
    if override and Path(override).exists():
        import subprocess as _sp
        try:
            r = _sp.run([override, "-c", "import sklearn"], capture_output=True, timeout=10)
            if r.returncode == 0:
                print(f"  [_resolve_python] using EVOPOOL_PYTHON={override}")
                return override
        except Exception:
            pass

    print(f"  [_resolve_python] WARN: scikit-learn is not importable from "
          f"{sys.executable}. Install it with 'pip install scikit-learn', or set "
          f"EVOPOOL_PYTHON to an interpreter that has it.")
    return sys.executable


_EVAL_DIR = Path(__file__).resolve().parent / "eval"


def _is_multilabel(task: str) -> bool:
    try:
        cfg = get_task_config(task)
        return bool(getattr(cfg, "task_type", "single_label") == "multi_label")
    except Exception:
        return False


def _label_key_for(task: str) -> str:
    return "true_labels" if _is_multilabel(task) else "true_label"


def _n_classes_for(task: str, default: int = 10) -> int:
    try:
        cfg = get_task_config(task)
        n = getattr(cfg, "n_classes", None)
        if n is None and hasattr(cfg, "label_names"):
            n = len(cfg.label_names)
        return int(n or default)
    except Exception:
        return default


def run_eval_subprocess(pool_module: Path, train: Path, val: Path, test: Path,
                          out_dir: Path, label_key: Optional[str] = None,
                          task: str = "chemprot") -> Path:
    """Evaluate pool on train/val/test, dispatching by task_type.

    Set EVOPOOL_FAST_EVAL=1 to skip the train split during iter-loop eval
    (train labels are only needed once at the final iter).
    """
    import subprocess as _sp
    out_dir.mkdir(parents=True, exist_ok=True)

    multilabel = _is_multilabel(task)
    if label_key is None:
        label_key = "true_labels" if multilabel else "true_label"

    if os.environ.get("EVOPOOL_FAST_EVAL") == "1":
        splits = [str(val), str(test)]
    else:
        splits = [str(train), str(val), str(test)]

    script = _EVAL_DIR / ("run_annotators_multilabel.py" if multilabel else "run_annotators.py")
    cmd = [
        _resolve_python(), str(script),
        "--pool_module", str(pool_module),
        "--splits", *splits,
        "--label_key", label_key,
        "--out_dir", str(out_dir),
    ]
    if multilabel:
        cmd += ["--n_classes", str(_n_classes_for(task))]
    _sp.run(cmd, check=True)
    return out_dir / "report.json"


def run_analyze_subprocess(pool_module: Path, train_path: Path, brief_path: Path,
                            task_name: str = "chemprot",
                            label_key: Optional[str] = None,
                            query_selection_method: str = "batchbald",
                            query_k_min: int = 20, query_k_max: int = 60,
                            class_balance_floor: int = 2, seed: int = 42) -> Path:
    """Run analyzer → improvement_brief.json, dispatching by task_type."""
    import subprocess as _sp
    brief_path.parent.mkdir(parents=True, exist_ok=True)

    multilabel = _is_multilabel(task_name)
    if label_key is None:
        # Analyzer always samples by a single-label projection; multi-label
        # uses the first true_label for cluster targeting.
        label_key = "true_label"

    script = _EVAL_DIR / ("analyze_errors_multilabel.py" if multilabel else "analyze_errors.py")
    cmd = [
        _resolve_python(), str(script),
        "--task_name", task_name,
        "--pool_module", str(pool_module),
        "--data_path", str(train_path),
        "--label_key", label_key,
        "--out_path", str(brief_path),
        "--query_selection_method", query_selection_method,
        "--query_k_min", str(query_k_min),
        "--query_k_max", str(query_k_max),
        "--class_balance_floor", str(class_balance_floor),
        "--seed", str(seed),
    ]
    _sp.run(cmd, check=True)
    return brief_path


# ────────────────────────────────────────────────────────────────────────
#  JSONL I/O — task_type-aware
# ────────────────────────────────────────────────────────────────────────

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read JSONL; populate BOTH `true_label` (int) and `true_labels` (List[int])
    so single-label and multi-label code share the same row."""
    rows: List[Dict[str, Any]] = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if "true_labels" in r and r["true_labels"] is not None:
            tl = r["true_labels"]
            if not isinstance(tl, (list, tuple)):
                tl = [tl]
            tl = sorted({int(x) for x in tl if x is not None})
            r["true_labels"] = tl
            if "true_label" not in r:
                r["true_label"] = tl[0] if tl else -1
        elif "true_label" in r:
            try:
                tl_int = int(r["true_label"])
                r["true_labels"] = [tl_int] if tl_int >= 0 else []
            except (TypeError, ValueError):
                r["true_labels"] = []
        else:
            r["true_labels"] = []
            r["true_label"] = -1
        rows.append(r)
    return rows


def load_task_data(task: str) -> Dict[str, List[Dict[str, Any]]]:
    base = Path(f"data/processed/{task}")
    return {
        "train": read_jsonl(base / "train.jsonl"),
        "val":   read_jsonl(base / "val.jsonl"),
        "test":  read_jsonl(base / "test.jsonl"),
    }
