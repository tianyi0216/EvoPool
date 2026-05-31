#!/usr/bin/env python3
"""EvoPool: Refiner agent — keyword-bank expansion of high-prec low-cov annotators.

Picks annotators with precision >= min_precision_to_refine AND coverage <=
max_coverage_to_refine, then asks the LLM for 2-3 broader variants per target
via keyword-bank expansion / relaxed conditions / structural variants.

Reads:
  --pool_module    current pool.py
  --pool_eval_dir  eval/ dir from src.pipeline.eval.run_annotators

Writes:
  <out_dir>/pool_fragment.py
  <out_dir>/refined_candidates_meta.json
  <out_dir>/filter_killed.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from src.pipeline.common import (
    parse_lfs_from_response, build_callable_permissive, call_llm,
    evaluate_lf_on_split, filter_lfs_by_threshold,
    extract_lf_sources_per_class, prefix_helpers_in_block,
    load_task_data, get_task_config,
    collect_existing_fired_ids, jaccard_dedup_against_pool,
)


REFINER_INSTRUCTION = """\
EXPAND THIS HIGH-PRECISION-LOW-COVERAGE ANNOTATOR TO BROADER VARIANTS.

Original annotator (precision={prec:.3f}, coverage={cov:.4f} on val):
```python
{src}
```
Target class: {target_class} ({target_name})

Write 2-3 BROADER variants of this annotator using THESE STRATEGIES (use a
different strategy per variant):

STRATEGY 1 — KEYWORD-BANK EXPANSION (most powerful):
  If the original checks a single keyword, replace with a BANK of synonyms
  via a helper:
    Original: if 'inhibit' in text: return 3
    Variant:  INHIB_KW = ['inhibit', 'suppress', 'block', 'reduce', 'attenuate', 'downregulate']
              def has_inhibition(text):
                  return any(k in text for k in INHIB_KW)
              def lf_inhibitor_broad(ex):
                  text = (ex.get('text') or '').lower()
                  meta = ex.get('metadata') or dict()
                  if has_inhibition(text) and ...:
                      return {target_class}
                  return ABSTAIN
  Coverage typically 3-5x of original at minimal precision cost.

STRATEGY 2 — RELAXED CONDITIONS:
  Drop one of the AND-conditions in the original to broaden recall.

STRATEGY 3 — STRUCTURAL VARIANT:
  Use entirely different features (metadata thresholds, length, regex patterns)
  to catch the same target class from a different angle.

REQUIREMENTS
- Each variant: def lf_<descriptive_unique_name>(ex), returns {target_class} or ABSTAIN.
- Use module-level helpers + permissive Python (imports re/math/collections allowed).
- Names: do NOT reuse from denylist.
- Output ONE python ```code block``` containing helpers + 2-3 lf_* variants.

DENYLIST (do NOT reuse): {denylist}

Task: {task_description}
Metadata fields: {metadata_fields}
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="chemprot")
    p.add_argument("--pool_module", type=Path, required=True)
    p.add_argument("--pool_eval_dir", type=Path, required=True,
                   help="eval/ dir containing report.json")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--iter_tag", type=str, default="r01")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--max_output_tokens", type=int, default=2000)
    p.add_argument("--cache_dir", type=Path,
                   default=Path("runs/cache/openai_responses/evopool_ref"))
    p.add_argument("--min_precision_to_refine", type=float, default=0.55,
                   help="Only refine annotators with val precision >= this.")
    p.add_argument("--max_coverage_to_refine", type=float, default=0.08,
                   help="Only refine annotators with coverage <= this (low-cov).")
    p.add_argument("--max_lfs_to_refine", type=int, default=6,
                   help="Cap how many annotators to refine in this pass.")
    p.add_argument("--filter_min_precision", type=float, default=0.25,
                   help="Refined variants must beat this precision floor.")
    p.add_argument("--filter_min_fires", type=int, default=5)
    p.add_argument("--max_jaccard_overlap", type=float, default=0.95,
                   help="Drop refined annotator if firing pattern overlaps existing one by > this.")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[REF] task={args.task} loading data + cfg")
    data = load_task_data(args.task)
    cfg = get_task_config(args.task)

    # Load eval report and pool to find refine-targets
    report = json.loads((args.pool_eval_dir / "report.json").read_text())
    val_split = next((s for s in report.get("splits", []) if s.get("split") == "val"), {})
    # NB: eval writes per-annotator stats as a LIST under "lf_metrics"
    lf_list = val_split.get("lf_metrics", []) or []
    per_lf = {item["name"]: item for item in lf_list if isinstance(item, dict) and "name" in item}

    targets = []
    for name, m in per_lf.items():
        prec = m.get("precision") or 0
        cov = m.get("coverage") or 0
        if prec >= args.min_precision_to_refine and 0 < cov <= args.max_coverage_to_refine:
            targets.append((name, prec, cov))
    targets.sort(key=lambda x: x[1], reverse=True)
    targets = targets[: args.max_lfs_to_refine]

    if not targets:
        print(f"[REF] no high-prec low-cov annotators to refine "
              f"(thresholds prec>={args.min_precision_to_refine}, cov<={args.max_coverage_to_refine})")
        (args.out_dir / "pool_fragment.py").write_text(
            f"# EvoPool Refiner ({args.iter_tag}) — no targets to refine\n")
        json.dump({"new_lf_names": []},
                  open(args.out_dir / "refined_candidates_meta.json", "w"), indent=2)
        json.dump([], open(args.out_dir / "filter_killed.json", "w"), indent=2)
        return

    print(f"[REF] refining {len(targets)} annotators:")
    for n, p_, c_ in targets:
        print(f"  {n}: prec={p_:.3f} cov={c_:.4f}")

    # Get sources for those annotators
    src_map = {it["name"]: it["src"] for it in extract_lf_sources_per_class(args.pool_module)}

    # Existing pool's lf_* names → denylist
    import importlib.util
    spec = importlib.util.spec_from_file_location("cur_pool", args.pool_module)
    cur_pool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cur_pool)
    denylist = sorted([fn.__name__ for fn in getattr(cur_pool, "ANNOTATORS", [])])

    # Per-target LLM call
    candidates = []
    seen = set(denylist)
    for ti, (name, prec, cov) in enumerate(targets, 1):
        # eval writes `predicted_labels` as a {label_str: count} dict; pick the modal class
        pred_labels = (per_lf.get(name) or {}).get("predicted_labels") or {}
        target_label = None
        if isinstance(pred_labels, dict) and pred_labels:
            target_label = int(max(pred_labels.items(), key=lambda kv: kv[1])[0])
        target_name = (cfg.label_names.get(int(target_label), str(target_label))
                       if target_label is not None else "?")
        prompt = REFINER_INSTRUCTION.format(
            src=src_map.get(name, ""),
            prec=prec, cov=cov,
            target_class=target_label,
            target_name=target_name,
            denylist=denylist[:60],
            task_description=cfg.task_description,
            metadata_fields=cfg.metadata_fields_description,
        )
        prompt_tagged = prompt + f"\n\n# refiner_iter_tag: {args.iter_tag}__{name}\n"
        try:
            text = call_llm(prompt_tagged, model=args.model, cache_dir=args.cache_dir,
                            max_output_tokens=args.max_output_tokens,
                            temperature=args.temperature)
        except Exception as e:
            print(f"  [{name}] ERROR: {e}")
            continue
        raw_pairs = parse_lfs_from_response(text, seen_names=set())
        if not raw_pairs:
            continue
        shared_block = raw_pairs[0][1]

        rename_map = {}
        for new_n, _ in raw_pairs:
            if new_n in seen:
                renamed = f"{new_n}_{args.iter_tag}_t{ti}"
                bump = 0
                while renamed in seen:
                    bump += 1
                    renamed = f"{new_n}_{args.iter_tag}_t{ti}_{bump}"
                rename_map[new_n] = renamed
        renamed_block = shared_block
        if rename_map:
            for old, new in rename_map.items():
                renamed_block = re.sub(rf"\bdef\s+{re.escape(old)}\b",
                                         f"def {new}", renamed_block)
        renamed_block = prefix_helpers_in_block(renamed_block, f"{args.iter_tag}t{ti}")

        for orig, _ in raw_pairs:
            final = rename_map.get(orig, orig)
            if final in seen:
                continue
            candidates.append((final, renamed_block, name))
            seen.add(final)
        print(f"  [{name}] +{len(raw_pairs)} raw  (total candidates={len(candidates)})")

    # Score new candidates on val
    print(f"\n[REF] scoring {len(candidates)} new variants on val")
    scored = []
    for nm, blk, src_lf in candidates:
        try:
            lf = build_callable_permissive(blk, nm)
        except Exception as e:
            print(f"  [{nm}] build_error: {e}")
            continue
        m = evaluate_lf_on_split(lf, data["val"], "true_label")
        m["refined_from"] = src_lf
        scored.append((nm, blk, m))

    after_thresh = filter_lfs_by_threshold(scored, args.filter_min_precision, args.filter_min_fires)
    print(f"[REF] precision/fires gate (>={args.filter_min_precision}, "
          f">={args.filter_min_fires}): {len(after_thresh)}/{len(scored)} pass")

    existing_fired = collect_existing_fired_ids(args.pool_module, data["val"], "true_label")
    kept, dropped = jaccard_dedup_against_pool(
        after_thresh, existing_fired, max_overlap=args.max_jaccard_overlap,
    )
    if dropped:
        print(f"[REF] Jaccard dedup (>{args.max_jaccard_overlap}): "
              f"dropped {len(dropped)} near-clones (kept {len(kept)})")
        for n, why in dropped[:5]:
            print(f"    {n}: {why}")
    print(f"[REF] FINAL: {len(kept)} variants survive")

    blocks = []
    seen_blk = set()
    for _, blk, _ in kept:
        if blk not in seen_blk:
            blocks.append(blk)
            seen_blk.add(blk)
    lf_names = [n for n, _, _ in kept]

    fragment_path = args.out_dir / "pool_fragment.py"
    with open(fragment_path, "w", encoding="utf-8") as f:
        f.write(f"# EvoPool Refiner fragment ({args.iter_tag}) — {len(lf_names)} new variants\n\n")
        for blk in blocks:
            f.write(blk.rstrip() + "\n\n")
        f.write(f"\n# New annotator names: {lf_names}\n")
    json.dump({"new_lf_names": lf_names,
               "metrics": [{"name": n, "metrics": m} for n, _, m in kept]},
              open(args.out_dir / "refined_candidates_meta.json", "w"), indent=2)

    # Emit filter_killed.json
    kept_names = {n for n, _, _ in kept}
    killed_records = []
    for nm, blk, m in scored:
        if nm in kept_names:
            continue
        try:
            mm = re.search(rf"def\s+{re.escape(nm)}\([^)]*\):.*?(?=\n\S|\Z)",
                             blk, re.S)
            snip = mm.group(0)[:240] if mm else blk[:240]
        except Exception:
            snip = blk[:240]
        prec_v = (m or {}).get("precision")
        fires_v = (m or {}).get("fires")
        if nm in {n for n, _, _ in after_thresh}:
            reason = "jaccard-dup vs existing pool"
        else:
            reason = f"failed precision/fires gate (prec={prec_v} fires={fires_v})"
        killed_records.append({"name": nm, "reason": reason, "snippet": snip})
    json.dump(killed_records, open(args.out_dir / "filter_killed.json", "w"), indent=2)
    print(f"[REF] wrote filter_killed.json ({len(killed_records)} records) → {fragment_path}")


if __name__ == "__main__":
    main()
