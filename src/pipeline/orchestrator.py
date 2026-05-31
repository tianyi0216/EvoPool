#!/usr/bin/env python3
"""EvoPool: Orchestrator — runs Generator → loop(analyze + Improver + Refiner + eval).

Production defaults: gpt-4o-mini, T=0.5, n_iters=12, seed=42. See config.yaml
for full hyperparameters.

Auto-dispatches single-label vs multi-label via task_type field on the task
config (src.tasks.configs.get_task_config(task).task_type).

Output structure:
  <out_root>/
    iter_00/
      pool.py                  ← Generator output
      eval/report.json
      analysis/improvement_brief.json
    iter_01/
      improver/pool_fragment.py
      refiner/pool_fragment.py
      pool.py                  ← merged: parent + improver + refiner
      eval/report.json
    ...
    summary.json (trajectory across iters)
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from src.pipeline.common import (
    write_pool, run_eval_subprocess, run_analyze_subprocess,
    load_pool_callables, evaluate_lf_on_split, read_jsonl,
    build_callable_permissive, _resolve_python, get_task_config,
    ListPurger,
)


def _macro_full(per_lf_votes_train: list, true_train: list, n_classes: int) -> float:
    """Weighted-vote macro_full F1 for a candidate-removal subset (used by post-iter ablation)."""
    import numpy as _np
    if not per_lf_votes_train:
        return 0.0
    V = _np.array(per_lf_votes_train, dtype=int).T  # (n_examples, n_lfs)
    y = _np.array(true_train, dtype=int)
    preds = _np.full(V.shape[0], -1, dtype=int)
    for i in range(V.shape[0]):
        votes = V[i]
        non_abstain = votes[votes >= 0]
        if non_abstain.size == 0:
            continue
        vals, cnts = _np.unique(non_abstain, return_counts=True)
        preds[i] = vals[_np.argmax(cnts)]
    f1s = []
    for c in range(n_classes):
        mask_y = (y == c)
        mask_p = (preds == c)
        tp = int((mask_y & mask_p).sum())
        fp = int((~mask_y & mask_p).sum())
        fn = int((mask_y & ~mask_p).sum())
        if tp + fp == 0 or tp + fn == 0:
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        if prec + rec == 0:
            continue
        f1s.append(2 * prec * rec / (prec + rec))
    return sum(f1s) / max(n_classes, 1)


def post_iter_ablation(parent_pool_path: Path, fragment_lf_names: list,
                        merged_pool_path: Path, val_path: Path, label_key: str,
                        n_classes: int, drop_threshold: float = 0.001) -> list:
    """Greedily drop new annotators that hurt val macro (MV-scored).

    Returns the subset of fragment_lf_names to KEEP. Deterministic local search
    that lets Improver/Refiner propose freely without polluting the pool.
    """
    if not fragment_lf_names:
        return []
    val = read_jsonl(val_path)
    true_y = [int(r.get(label_key, -1)) for r in val]

    name_to_lf = dict(load_pool_callables(merged_pool_path))
    all_lf_names = list(name_to_lf.keys())

    per_lf_votes = {}
    for nm, fn in name_to_lf.items():
        votes = []
        for ex in val:
            try:
                v = fn(ex)
            except Exception:
                v = -1
            votes.append(int(v) if isinstance(v, int) or (hasattr(v, "__int__")) else -1)
        per_lf_votes[nm] = votes

    parent_lf_names = [n for n in all_lf_names if n not in set(fragment_lf_names)]

    def _macro_with(active_names):
        votes_list = [per_lf_votes[n] for n in active_names if n in per_lf_votes]
        return _macro_full(votes_list, true_y, n_classes)

    base_macro = _macro_with(all_lf_names)
    print(f"  [ablation] starting val macro: {base_macro:.4f} "
          f"({len(fragment_lf_names)} new annotators to consider)")

    keep_fragment = list(fragment_lf_names)
    dropped = []
    improved = True
    iteration = 0
    while improved and len(keep_fragment) > 0:
        improved = False
        iteration += 1
        best_drop = None
        best_macro = base_macro
        for nm in list(keep_fragment):
            trial = parent_lf_names + [n for n in keep_fragment if n != nm]
            m = _macro_with(trial)
            if m > best_macro + drop_threshold:
                best_macro = m
                best_drop = nm
        if best_drop is not None:
            keep_fragment.remove(best_drop)
            dropped.append((best_drop, best_macro))
            print(f"  [ablation r{iteration}] dropped {best_drop} → val macro "
                  f"{base_macro:.4f} → {best_macro:.4f}")
            base_macro = best_macro
            improved = True

    print(f"  [ablation] kept {len(keep_fragment)}/{len(fragment_lf_names)} new annotators; "
          f"dropped {len(dropped)} ({[d[0] for d in dropped[:5]]})")
    return keep_fragment


def merge_blocks_into_pool(parent_pool: Path, fragments: list, out_pool: Path,
                            comment: str = ""):
    """Concat parent pool + fragments; emit unified ANNOTATORS list."""
    parent_src = parent_pool.read_text(encoding="utf-8")
    parent_tree = ast.parse(parent_src)
    parent_lfs = [n.name for n in parent_tree.body
                  if isinstance(n, ast.FunctionDef) and n.name.startswith("lf_")]

    parent_lines = parent_src.splitlines(keepends=True)
    cut_at = None
    for n in parent_tree.body:
        if isinstance(n, ast.Assign):
            tgts = [t.id for t in n.targets if isinstance(t, ast.Name)]
            if any(t in {"ANNOTATORS", "ANNOTATOR_METRICS", "POOL_METADATA"} for t in tgts):
                cut_at = n.lineno - 1 if cut_at is None else min(cut_at, n.lineno - 1)
    parent_body = "".join(parent_lines[:cut_at]) if cut_at is not None else parent_src

    all_new_lfs: list = []
    fragment_bodies: list = []
    for frag_path in fragments:
        if not frag_path.exists():
            continue
        frag_src = frag_path.read_text(encoding="utf-8")
        try:
            frag_tree = ast.parse(frag_src)
        except SyntaxError:
            continue
        new_lfs = [n.name for n in frag_tree.body
                   if isinstance(n, ast.FunctionDef) and n.name.startswith("lf_")]
        all_new_lfs.extend(new_lfs)
        fragment_bodies.append(frag_src)

    out_pool.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pool, "w", encoding="utf-8") as f:
        f.write(f"# Auto-merged annotator pool (EvoPool)\n# {comment}\n\n")
        f.write(parent_body.rstrip() + "\n\n")
        for frag_src in fragment_bodies:
            f.write("# " + "-" * 70 + "\n# Fragment\n# " + "-" * 70 + "\n")
            f.write(frag_src.rstrip() + "\n\n")
        f.write("ANNOTATORS = [\n")
        for n in parent_lfs + all_new_lfs:
            f.write(f"    {n},\n")
        f.write("]\n")
    print(f"  merged pool: {len(parent_lfs)} parent + {len(all_new_lfs)} new = "
          f"{len(parent_lfs) + len(all_new_lfs)} total → {out_pool}")


def _purge_names_from_pool(pool_path: Path, drop_set: set):
    """Strip dropped lf_* defs + clean every list literal referencing them.

    AST cleanup ensures the rewritten pool.py imports without NameError even
    when ANNOTATORS or per-class bundle lists referenced the dropped names.
    """
    if not drop_set:
        return
    src = pool_path.read_text()
    tree = ast.parse(src)
    src_lines = src.splitlines(keepends=True)
    drop_ranges = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in drop_set:
            drop_ranges.append((node.lineno - 1, node.end_lineno or node.lineno))
    keep_mask = [True] * len(src_lines)
    for s, e in drop_ranges:
        for i in range(s, e):
            if 0 <= i < len(keep_mask):
                keep_mask[i] = False
    new_src = "".join(l for l, k in zip(src_lines, keep_mask) if k)
    # AST cleanup: strip dropped names from EVERY list assignment
    try:
        tree2 = ast.parse(new_src)
        new_tree = ListPurger(drop_set).visit(tree2)
        ast.fix_missing_locations(new_tree)
        new_src = ast.unparse(new_tree) + "\n"
    except Exception as ce:
        print(f"  [purge] WARN: AST list-cleanup failed ({ce}); line-mode fallback")
        out_lines = []
        in_ann = False
        for line in new_src.splitlines(keepends=True):
            ss = line.strip()
            if ss.startswith("ANNOTATORS"):
                in_ann = True
                out_lines.append(line)
                continue
            if in_ann:
                if ss == "]":
                    in_ann = False
                    out_lines.append(line)
                    continue
                stripped_name = ss.rstrip(",").strip()
                if stripped_name in drop_set:
                    continue
            out_lines.append(line)
        new_src = "".join(out_lines)
    pool_path.write_text(new_src)


def _subsumption_prune(final_pool: Path, val_rows: List[Dict], label_key: str,
               new_lf_names: set,
               subsume_jaccard: float = 0.95,
               subsume_prec_gain: float = 0.05) -> int:
    """Delete existing annotators that are subsumed by new arrivals.

    An old annotator is dropped when a new one fires on (almost) the same
    val examples (jaccard >= subsume_jaccard) at meaningfully higher
    precision (>= subsume_prec_gain). Returns the count of pruned annotators.
    """
    callables = load_pool_callables(final_pool)
    fired = {}
    preds = {}
    precs = {}
    for name, fn in callables:
        pmap = {}
        for i, ex in enumerate(val_rows):
            try:
                v = fn(ex)
            except Exception:
                v = -1
            if v != -1:
                pmap[i] = int(v)
        fired[name] = set(pmap.keys())
        preds[name] = pmap
        if pmap:
            true_vs = [val_rows[i].get(label_key) for i in pmap]
            def _ok(predv, truev):
                if isinstance(truev, list):
                    return predv in {int(x) for x in truev if x is not None}
                try:
                    return predv == int(truev)
                except Exception:
                    return False
            correct = sum(1 for i, predv in pmap.items() if _ok(predv, val_rows[i].get(label_key)))
            precs[name] = correct / max(len(pmap), 1)
        else:
            precs[name] = 0.0

    to_delete = set()
    for nn in new_lf_names:
        if nn not in fired or not fired[nn]:
            continue
        for on, of in fired.items():
            if on == nn or on in new_lf_names or on in to_delete or not of:
                continue
            inter = len(fired[nn] & of)
            union = len(fired[nn] | of)
            if union == 0:
                continue
            jac = inter / union
            cov_of_old = inter / max(len(of), 1)
            if (jac >= subsume_jaccard or cov_of_old >= 0.95) \
               and precs[nn] >= precs[on] + subsume_prec_gain:
                to_delete.add(on)
                print(f"  [PRUNE] {nn} (prec={precs[nn]:.3f}) subsumes "
                      f"{on} (prec={precs[on]:.3f}); jac={jac:.2f} cov_old={cov_of_old:.2f}")
    if to_delete:
        _purge_names_from_pool(final_pool, to_delete)
        print(f"  [PRUNE] pruned {len(to_delete)} subsumed annotators from pool")
    else:
        print(f"  [PRUNE] no annotators subsumed this iter")
    return len(to_delete)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="chemprot")
    p.add_argument("--out_root", type=Path, required=True)
    p.add_argument("--n_iters", type=int, default=12)
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)

    # Generator
    p.add_argument("--gen_n_calls", type=int, default=18)
    p.add_argument("--gen_min_precision", type=float, default=0.30)
    p.add_argument("--gen_min_fires", type=int, default=5)

    # Improver
    p.add_argument("--imp_n_calls", type=int, default=6)
    p.add_argument("--imp_min_precision", type=float, default=0.25)
    p.add_argument("--imp_min_fires", type=int, default=5)
    p.add_argument("--imp_max_jaccard_overlap", type=float, default=0.95)
    p.add_argument("--imp_max_train_val_prec_gap", type=float, default=0.30)

    # Refiner
    p.add_argument("--ref_max_lfs", type=int, default=6)
    p.add_argument("--ref_min_prec_to_refine", type=float, default=0.55)
    p.add_argument("--ref_max_cov_to_refine", type=float, default=0.08)
    p.add_argument("--ref_filter_min_precision", type=float, default=0.25)
    p.add_argument("--ref_filter_min_fires", type=int, default=5)
    p.add_argument("--ref_max_jaccard_overlap", type=float, default=0.95)
    p.add_argument("--ref_min_iter", type=int, default=3,
                   help="Skip Refiner for iter_k where k < ref_min_iter "
                        "(avoids dilution at small pool sizes).")

    # Pipeline-level
    p.add_argument("--n_classes", type=int, default=None,
                   help="Defaults to task_config.n_classes.")
    p.add_argument("--class_dropout_after", type=int, default=99,
                   help="Blacklist classes that fail K consecutive iters.")
    p.add_argument("--do_no_harm_tolerance", type=float, default=0.020,
                   help="Revert iter if test macro drops by more than this.")
    p.add_argument("--min_iter_gain", type=float, default=0.0,
                   help="0 = only revert on regression.")

    # Post-iter ablation
    p.add_argument("--enable_post_iter_ablation", action="store_true", default=True)
    p.add_argument("--disable_post_iter_ablation", dest="enable_post_iter_ablation",
                   action="store_false")
    p.add_argument("--ablation_drop_threshold", type=float, default=0.001)

    # Subsumption pruning
    p.add_argument("--enable_subsumption_pruning", action="store_true", default=True)
    p.add_argument("--disable_subsumption_pruning", dest="enable_subsumption_pruning",
                   action="store_false")
    p.add_argument("--subsumption_jaccard", type=float, default=0.95)
    p.add_argument("--subsumption_prec_gain", type=float, default=0.05)

    # Per-class Improver budget
    p.add_argument("--enable_per_class_improver", action="store_true", default=True)
    p.add_argument("--disable_per_class_improver", dest="enable_per_class_improver",
                   action="store_false")
    p.add_argument("--per_class_f1_threshold", type=float, default=0.40)
    p.add_argument("--per_class_max_targets", type=int, default=3)
    p.add_argument("--per_class_n_pos_examples", type=int, default=8)

    # Analyzer
    p.add_argument("--query_selection_method", default="batchbald",
                   choices=["batchbald", "random", "uncertainty"])
    p.add_argument("--query_k_min", type=int, default=20)
    p.add_argument("--query_k_max", type=int, default=60)
    p.add_argument("--class_balance_floor", type=int, default=2)
    p.add_argument("--cache_root", type=Path,
                   default=Path("runs/cache/openai_responses"))

    # Memory level (production default = 0; >=2 enables Reflector)
    p.add_argument("--memory_level", type=int, default=0, choices=[0, 1, 2, 3],
                   help="0 = Darwinian production (no memory). "
                        ">=2 enables Reflector. Paper headline uses 0.")
    p.add_argument("--reflector_interval", type=int, default=3,
                   help="When Reflector is enabled, run every N iters.")
    p.add_argument("--reflector_token_budget", type=int, default=2000)
    p.add_argument("--reflector_top_k_struggling", type=int, default=3)

    args = p.parse_args()

    # Resolve n_classes from task config if not set
    cfg = get_task_config(args.task)
    if args.n_classes is None:
        args.n_classes = getattr(cfg, "n_classes", None) or len(cfg.label_names)
    task_type = getattr(cfg, "task_type", "single_label")
    label_key = "true_labels" if task_type == "multi_label" else "true_label"
    print(f"[ORCH] task={args.task} task_type={task_type} n_classes={args.n_classes} "
          f"label_key={label_key}")

    args.out_root.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    train = Path(f"data/processed/{args.task}/train.jsonl")
    val = Path(f"data/processed/{args.task}/val.jsonl")
    test = Path(f"data/processed/{args.task}/test.jsonl")

    # ─── iter_00: Generator ───
    iter0 = args.out_root / "iter_00"
    pool0 = iter0 / "pool.py"
    if not pool0.exists():
        print(f"\n=== iter_00: Generator ===")
        gen_cmd = [
            _resolve_python(), str(here / "generator.py"),
            "--task", args.task,
            "--out_dir", str(args.out_root),
            "--n_calls", str(args.gen_n_calls),
            "--model", args.model,
            "--temperature", str(args.temperature),
            "--cache_dir", str(args.cache_root / "evopool_gen"),
            "--min_precision", str(args.gen_min_precision),
            "--min_fires", str(args.gen_min_fires),
            "--seed", str(args.seed),
        ]
        subprocess.run(gen_cmd, check=True)
    else:
        print(f"[skip iter_00] {pool0} exists")

    # Eval iter_00
    if not (iter0 / "eval" / "report.json").exists():
        print(f"\n--- iter_00 eval ---")
        run_eval_subprocess(pool0, train, val, test, iter0 / "eval",
                              label_key=label_key, task=args.task)

    summary = {"iters": []}

    def _summary_for(it_dir: Path) -> dict:
        rep = json.loads((it_dir / "eval" / "report.json").read_text())
        for s in rep.get("splits", []):
            if s.get("split") != "test":
                continue
            wv = (s.get("aggregation") or {}).get("weighted_vote", {})
            pc = wv.get("per_class", {})
            f1s = [pc.get(str(c), {}).get("f1") for c in range(args.n_classes) if str(c) in pc]
            n = rep.get("num_annotators", "?")
            return {"iter": it_dir.name, "n_lf": n,
                    "cov": wv.get("coverage"),
                    "acc": wv.get("accuracy_on_covered"),
                    "macro_full": (sum(active for active in f1s if active is not None) /
                                   max(args.n_classes, 1)) if f1s else 0,
                    "k_active": len([f for f in f1s if f is not None])}
        return {"iter": it_dir.name, "macro_full": None}

    summary["iters"].append(_summary_for(iter0))
    print(f"\n[summary iter_00] {summary['iters'][-1]}")

    # ─── iter_01..N ───
    cur_pool = pool0
    class_fail_streak: Dict[int, int] = {}

    mem_dir = args.out_root / "memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    (mem_dir / "_archive").mkdir(parents=True, exist_ok=True)
    raw_events_path = mem_dir / "raw_events.jsonl"
    lessons_path = mem_dir / "lessons.json"

    def _append_raw_events(records: list) -> None:
        if not records:
            return
        with open(raw_events_path, "a") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")

    def _classify_target_label(name_with_lf_prefix: str) -> int:
        m = re.search(r"_c(\d+)(?:b\d+)?(?:_|$)", name_with_lf_prefix)
        if m:
            return int(m.group(1))
        m = re.search(r"class(\d+)", name_with_lf_prefix)
        if m:
            return int(m.group(1))
        return -1

    def _per_class_f1_from_iter(it_dir: Path) -> Dict[int, float]:
        rep_p = it_dir / "eval" / "report.json"
        if not rep_p.exists():
            return {}
        rep = json.loads(rep_p.read_text())
        for s in rep.get("splits", []):
            if s.get("split") == "val":
                mv = (s.get("aggregation") or {}).get("majority_vote") or {}
                pc = mv.get("per_class") or {}
                out = {}
                for i in range(args.n_classes):
                    m = pc.get(str(i)) or {}
                    f1 = m.get("f1")
                    out[i] = float(f1) if f1 is not None else 0.0
                return out
        return {}

    for k in range(1, args.n_iters + 1):
        it_dir = args.out_root / f"iter_{k:02d}"
        it_dir.mkdir(parents=True, exist_ok=True)
        tag = f"i{k:02d}"

        # 1. analyze on current pool
        brief = it_dir / "analysis" / "improvement_brief.json"
        if not brief.exists():
            print(f"\n=== iter_{k:02d}: analyze ({cur_pool.name}) ===")
            run_analyze_subprocess(cur_pool, train, brief,
                                    task_name=args.task,
                                    query_selection_method=args.query_selection_method,
                                    query_k_min=args.query_k_min,
                                    query_k_max=args.query_k_max,
                                    class_balance_floor=args.class_balance_floor,
                                    seed=args.seed)

        excluded = sorted([c for c, s in class_fail_streak.items()
                           if s >= args.class_dropout_after])
        exclude_arg = ",".join(str(c) for c in excluded)
        if excluded:
            print(f"\n[iter_{k:02d}] viability blacklist: classes {excluded} "
                  f"(failed >={args.class_dropout_after} consecutive iters)")

        # Per-class Improver: target prev-iter's lowest-F1 classes
        per_class_targets_arg = ""
        if args.enable_per_class_improver:
            prev_dir = args.out_root / f"iter_{k-1:02d}"
            per_f1 = _per_class_f1_from_iter(prev_dir)
            if per_f1:
                low_f1 = sorted(per_f1.items(), key=lambda x: x[1])
                excluded_set = set(excluded) if excluded else set()
                low_f1 = [(c, f) for c, f in low_f1
                          if f < args.per_class_f1_threshold and c not in excluded_set]
                targets = [c for c, _ in low_f1[: args.per_class_max_targets]]
                if targets:
                    per_class_targets_arg = ",".join(str(c) for c in targets)
                    print(f"[PER-CLASS] iter_{k:02d} targets: {targets} "
                          f"(F1<{args.per_class_f1_threshold}; rest of {args.imp_n_calls} batches stay broad)")

        # 2. Improver
        imp_dir = it_dir / "improver"
        imp_frag = imp_dir / "pool_fragment.py"
        if not imp_frag.exists():
            print(f"\n=== iter_{k:02d}: Improver ===")
            imp_cmd = [
                _resolve_python(), str(here / "improver.py"),
                "--task", args.task,
                "--pool_module", str(cur_pool),
                "--brief_path", str(brief),
                "--out_dir", str(imp_dir),
                "--n_calls", str(args.imp_n_calls),
                "--iter_tag", tag,
                "--model", args.model,
                "--temperature", str(args.temperature),
                "--cache_dir", str(args.cache_root / f"evopool_imp_{tag}"),
                "--min_precision", str(args.imp_min_precision),
                "--min_fires", str(args.imp_min_fires),
                "--max_jaccard_overlap", str(args.imp_max_jaccard_overlap),
                "--max_train_val_prec_gap", str(args.imp_max_train_val_prec_gap),
            ]
            if exclude_arg:
                imp_cmd += ["--exclude_classes", exclude_arg]
            if per_class_targets_arg:
                imp_cmd += ["--per_class_targets", per_class_targets_arg,
                            "--per_class_n_pos_examples", str(args.per_class_n_pos_examples)]
            subprocess.run(imp_cmd, check=True)

        # Optional: Reflector raw-event tracking (only when memory_level >= 1)
        if args.memory_level >= 1:
            try:
                killed_p = imp_dir / "filter_killed.json"
                if killed_p.exists():
                    killed = json.loads(killed_p.read_text())
                    records = []
                    for entry in killed:
                        nm = entry.get("name", "?")
                        records.append({
                            "iter": k,
                            "type": "improver_dropped",
                            "name": nm,
                            "target_class": _classify_target_label(nm),
                            "reason": entry.get("reason", ""),
                            "snippet": (entry.get("snippet") or "")[:240],
                        })
                    _append_raw_events(records)
            except Exception as e:
                print(f"[REFLECTOR] WARN: improver event log failed: {e}")

        # Update class viability based on which classes the Improver succeeded for
        try:
            new_meta = json.loads((imp_dir / "new_candidates_meta.json").read_text())
            classes_with_new_lf = set()
            for entry in new_meta.get("metrics", []):
                pl = (entry.get("metrics") or {}).get("predicted_label")
                if pl is not None:
                    classes_with_new_lf.add(int(pl))
            for c in range(args.n_classes):
                if c in classes_with_new_lf:
                    class_fail_streak[c] = 0
                else:
                    class_fail_streak[c] = class_fail_streak.get(c, 0) + 1
        except Exception as e:
            print(f"[iter_{k:02d}] viability tracking warn: {e}")

        # 3. Eval pool merged with improver fragment
        merged_after_imp = it_dir / "pool_after_imp.py"
        merge_blocks_into_pool(cur_pool, [imp_frag], merged_after_imp,
                                comment=f"iter_{k:02d} after improver")

        eval_after_imp = it_dir / "eval_after_imp"
        if not (eval_after_imp / "report.json").exists():
            run_eval_subprocess(merged_after_imp, train, val, test, eval_after_imp,
                                  label_key=label_key, task=args.task)

        # 4. Refiner — gated by ref_min_iter
        ref_dir = it_dir / "refiner"
        ref_frag = ref_dir / "pool_fragment.py"
        if k < args.ref_min_iter:
            ref_dir.mkdir(parents=True, exist_ok=True)
            ref_frag.write_text(f"# Refiner gated: skipped at iter_{k:02d} "
                                f"(ref_min_iter={args.ref_min_iter})\n")
            print(f"\n=== iter_{k:02d}: Refiner SKIPPED (ref_min_iter gate) ===")
        elif not ref_frag.exists():
            print(f"\n=== iter_{k:02d}: Refiner ===")
            ref_cmd = [
                _resolve_python(), str(here / "refiner.py"),
                "--task", args.task,
                "--pool_module", str(merged_after_imp),
                "--pool_eval_dir", str(eval_after_imp),
                "--out_dir", str(ref_dir),
                "--iter_tag", f"r{k:02d}",
                "--max_lfs_to_refine", str(args.ref_max_lfs),
                "--min_precision_to_refine", str(args.ref_min_prec_to_refine),
                "--max_coverage_to_refine", str(args.ref_max_cov_to_refine),
                "--filter_min_precision", str(args.ref_filter_min_precision),
                "--filter_min_fires", str(args.ref_filter_min_fires),
                "--max_jaccard_overlap", str(args.ref_max_jaccard_overlap),
                "--model", args.model,
                "--cache_dir", str(args.cache_root / f"evopool_ref_{tag}"),
            ]
            subprocess.run(ref_cmd, check=True)

            if args.memory_level >= 1:
                try:
                    killed_p = ref_dir / "filter_killed.json"
                    if killed_p.exists():
                        killed = json.loads(killed_p.read_text())
                        records = []
                        for entry in killed:
                            nm = entry.get("name", "?")
                            records.append({
                                "iter": k,
                                "type": "refiner_dropped",
                                "name": nm,
                                "target_class": _classify_target_label(nm),
                                "reason": entry.get("reason", ""),
                                "snippet": (entry.get("snippet") or "")[:240],
                            })
                        _append_raw_events(records)
                except Exception as e:
                    print(f"[REFLECTOR] WARN: refiner event log failed: {e}")

        # 5. Final merge: parent + improver + refiner
        final_pool = it_dir / "pool.py"
        merge_blocks_into_pool(cur_pool, [imp_frag, ref_frag], final_pool,
                                comment=f"iter_{k:02d} after improver + refiner")

        # ─── Subsumption pruning: delete annotators dominated by new arrivals ───
        if args.enable_subsumption_pruning:
            print(f"\n=== iter_{k:02d}: Subsumption pruning ===")
            try:
                val_rows = list(read_jsonl(val))
                new_names = set()
                for frag_path in (imp_frag, ref_frag):
                    if not frag_path.exists():
                        continue
                    try:
                        tree = ast.parse(frag_path.read_text())
                        new_names |= {n.name for n in tree.body
                                      if isinstance(n, ast.FunctionDef)
                                      and n.name.startswith("lf_")}
                    except SyntaxError:
                        pass
                n_pruned = _subsumption_prune(final_pool, val_rows, label_key, new_names,
                                       subsume_jaccard=args.subsumption_jaccard,
                                       subsume_prec_gain=args.subsumption_prec_gain)
                if n_pruned > 0:
                    shutil.rmtree(it_dir / "eval", ignore_errors=True)
            except Exception as e:
                print(f"  [PRUNE] WARN: pruning failed: {e}")

        # 5b. Post-iter ablation
        if args.enable_post_iter_ablation:
            print(f"\n=== iter_{k:02d}: Post-iter ablation ===")
            new_lf_names = []
            for frag_path in (imp_frag, ref_frag):
                if not frag_path.exists():
                    continue
                try:
                    tree = ast.parse(frag_path.read_text())
                    new_lf_names.extend([n.name for n in tree.body
                                          if isinstance(n, ast.FunctionDef)
                                          and n.name.startswith("lf_")])
                except SyntaxError:
                    pass
            if new_lf_names:
                kept = post_iter_ablation(
                    cur_pool, new_lf_names, final_pool, val, label_key,
                    n_classes=args.n_classes,
                    drop_threshold=args.ablation_drop_threshold,
                )
                dropped_set = set(new_lf_names) - set(kept)
                if dropped_set:
                    _purge_names_from_pool(final_pool, dropped_set)
                    print(f"  [ablation] rewrote {final_pool} (dropped {len(dropped_set)} annotators)")
                    shutil.rmtree(it_dir / "eval", ignore_errors=True)

        # 6. Eval merged final pool
        if not (it_dir / "eval" / "report.json").exists():
            run_eval_subprocess(final_pool, train, val, test, it_dir / "eval",
                                  label_key=label_key, task=args.task)

        # 7. Acceptance gate
        s = _summary_for(it_dir)
        prev_macro = (summary["iters"][-1].get("macro_full") or 0) if summary["iters"] else 0
        cur_macro = s.get("macro_full") or 0
        delta = cur_macro - prev_macro
        revert_reason = None
        if -delta > args.do_no_harm_tolerance:
            revert_reason = (f"REGRESSION: dlt={delta:+.4f} exceeds do_no_harm tolerance "
                             f"-{args.do_no_harm_tolerance:.4f}")
        elif args.min_iter_gain > 0 and delta < args.min_iter_gain:
            revert_reason = (f"INSUFFICIENT GAIN: dlt={delta:+.4f} < "
                             f"min_iter_gain={args.min_iter_gain:.4f}")
        if revert_reason:
            print(f"\n[iter_{k:02d} ACCEPTANCE GATE] {revert_reason}")
            print(f"   Reverting: copying parent pool to iter_{k:02d}/pool.py and re-evaluating.")
            shutil.copy2(cur_pool, final_pool)
            shutil.rmtree(it_dir / "eval", ignore_errors=True)
            run_eval_subprocess(final_pool, train, val, test, it_dir / "eval",
                                  label_key=label_key, task=args.task)
            s = _summary_for(it_dir)
            s["reverted"] = True
            print(f"   Post-revert: {s}")

        cur_pool = final_pool
        summary["iters"].append(s)
        print(f"\n[summary iter_{k:02d}] {s}")

        # ─── Reflector invocation (only when memory_level >= 2) ───
        if args.memory_level >= 2:
            try:
                cur_per_f1 = _per_class_f1_from_iter(it_dir)
                _append_raw_events([
                    {"iter": k, "type": "class_f1", "class": c, "f1": cur_per_f1.get(c, 0.0)}
                    for c in range(args.n_classes)
                ])
            except Exception as e:
                print(f"[REFLECTOR] WARN: f1 event log failed: {e}")

            if k % args.reflector_interval == 0 or k == args.n_iters:
                print(f"\n=== iter_{k:02d}: REFLECTOR ===")
                rfl_cmd = [
                    _resolve_python(), str(here / "reflector.py"),
                    "--task", args.task,
                    "--raw_events_path", str(raw_events_path),
                    "--prev_lessons_path", str(lessons_path),
                    "--out_lessons_path", str(lessons_path),
                    "--archive_dir", str(mem_dir / "_archive"),
                    "--iter_at_reflection", str(k),
                    "--cache_dir", str(args.cache_root / "evopool_reflector"),
                    "--model", args.model,
                    "--token_budget", str(args.reflector_token_budget),
                    "--top_k_struggling", str(args.reflector_top_k_struggling),
                ]
                try:
                    subprocess.run(rfl_cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"[REFLECTOR] WARN: reflection failed at iter_{k:02d}: {e}")

    # Final summary
    print(f"\n=== TRAJECTORY ===")
    print(f"{'iter':<8} {'#lf':>4} {'k_act':>5} {'cov':>6} {'acc':>6} {'macro_full':>11}")
    for s in summary["iters"]:
        print(f"{s['iter']:<8} {str(s.get('n_lf','?')):>4} {str(s.get('k_active','?')):>5} "
              f"{(s.get('cov') or 0):>6.3f} {(s.get('acc') or 0):>6.3f} "
              f"{(s.get('macro_full') or 0):>11.4f}")

    json.dump(summary, open(args.out_root / "summary.json", "w"), indent=2)
    print(f"\n-> {args.out_root}/summary.json")


if __name__ == "__main__":
    main()
