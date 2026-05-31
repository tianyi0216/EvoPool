#!/usr/bin/env python3
"""EvoPool: Improver agent — generate NEW compositional annotators that target
the failure clusters surfaced by the analyzer.

Production: C1_v2 per-class budget allocation. Reserve K of n_calls batches
for low-F1 classes (K classes with prev-iter F1 < f1_threshold, capped at
max_targets). Remaining batches stay broad (cluster-based). Hybrid design:
if a per-class batch over-fits, broad batches still cover the iter — avoids
the C1 starvation failure mode.

Reads:
  --pool_module    current pool.py
  --brief_path     improvement_brief.json from src.pipeline.eval.analyze_errors

Writes:
  <out_dir>/pool_fragment.py        — code block(s) of new helpers + lf_*
  <out_dir>/new_candidates_meta.json
  <out_dir>/filter_killed.json      — annotators that failed the filter chain
"""
from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import List

from src.pipeline.common import (
    parse_lfs_from_response, build_callable_permissive, call_llm,
    evaluate_lf_on_split, filter_lfs_by_threshold,
    prefix_helpers_in_block,
    load_task_data, get_task_config,
    collect_existing_fired_ids, jaccard_dedup_against_pool,
    consistency_filter_train_val,
)


# Task-aware example pattern: claim-verification tasks need metadata-comparison
# examples, not the lexical text-regex pattern.
_VERIFICATION_TASKS = {"fever", "vitaminc", "scifact", "anli"}

_CLASSIFICATION_EXAMPLE_PATTERN = """\
    import re
    INHIB_KW = ['inhibit', 'suppress', 'block', 'reduce', 'attenuate']
    def has_inhibition(text):
        return any(k in text for k in INHIB_KW)
    def has_named_pair(meta):
        e1 = (meta.get('entity1') or '').strip()
        e2 = (meta.get('entity2') or '').strip()
        return bool(e1) and bool(e2)
    def lf_downreg_compositional(ex):
        text = (ex.get('text') or '').lower()
        meta = ex.get('metadata') or dict()
        if has_inhibition(text) and has_named_pair(meta):
            if 'metabolism' not in text:
                return 3  # Downregulator
        return ABSTAIN"""

_VERIFICATION_EXAMPLE_PATTERN = """\
    # CLAIM-VERIFICATION TASKS: compose pre-computed metadata comparison features
    # (token_overlap, entity_containment, negation_mismatch, antonym_present,
    #  numbers_mismatch, and OPTIONAL nli_contradict/nli_entail/refutes_phrase_count).
    # DO NOT regex the raw "Claim: ... Evidence: ..." text — the metadata IS your primitive.
    def has_strong_negation_flip(meta):
        return bool(meta.get('negation_mismatch')) and meta.get('entity_containment', 0.0) > 0.3
    def has_nli_contradict(meta):
        return meta.get('nli_contradict', 0.0) > 0.5
    def has_high_overlap_clean(meta):
        return (meta.get('claim_in_evidence', 0.0) > 0.5 and
                not meta.get('negation_mismatch', False) and
                not meta.get('numbers_mismatch', False) and
                not meta.get('antonym_present', False))
    def has_topic_drift(meta):
        return (meta.get('token_overlap', 0.0) < 0.18 and
                meta.get('entity_containment', 0.0) < 0.2 and
                meta.get('num_shared_entities', 0) == 0)
    def lf_refutes_negation_or_contradict(ex):
        meta = ex.get('metadata') or dict()
        if has_strong_negation_flip(meta) or has_nli_contradict(meta):
            return 1  # Refutes
        return ABSTAIN
    def lf_supports_high_overlap(ex):
        meta = ex.get('metadata') or dict()
        if has_high_overlap_clean(meta) and meta.get('entity_containment', 0.0) > 0.4:
            return 0  # Supports
        return ABSTAIN
    def lf_nei_topic_drift(ex):
        meta = ex.get('metadata') or dict()
        if has_topic_drift(meta) and not has_strong_negation_flip(meta):
            return 2  # NotEnoughInfo
        return ABSTAIN"""


def _pick_example_pattern(task: str) -> str:
    try:
        cfg = get_task_config(task)
        if getattr(cfg, "task_family", None) == "verification":
            return _VERIFICATION_EXAMPLE_PATTERN
    except Exception:
        pass
    return _VERIFICATION_EXAMPLE_PATTERN if task in _VERIFICATION_TASKS else _CLASSIFICATION_EXAMPLE_PATTERN


IMPROVER_INSTRUCTION = """\
WRITE COMPOSITIONAL ANNOTATORS TARGETING THE FAILURE CLUSTERS BELOW.

The current annotator pool has weaknesses in specific clusters of training
examples. Your job: write 3-6 NEW compositional Python annotators (lf_*)
that target those specific clusters.

CRITICAL — QUALITY OVER COVERAGE:
- Each annotator must be a CONFIDENT class signal. If you cannot confidently
  distinguish the target class from competing classes, DO NOT write it.
  An ABSENT annotator is far better than a low-precision one.
- For "marginal" or rare classes (e.g. classes you are not 100% sure about
  the linguistic boundary): only write one if you can name the discriminator
  clearly. Otherwise SKIP the class.
- Each annotator you write should plausibly hit precision >= 0.50 on val.
  Anything weaker will be filtered out anyway and waste effort.
- Do NOT generate near-duplicates of the existing pool. If a similar pattern
  is already covered, write a DIFFERENT angle (different keywords, different
  metadata signal, different boundary).

PHILOSOPHY: each annotator is a *compositional Python program*. Define small
module-level helper predicates first, then compose them in lf_*(ex) via
explicit IF-ELIF-ELSE. Use whatever Python feature helps: helper functions,
regex compilation, for-loops over keyword banks, list/dict comprehensions,
imports (re, math, collections, itertools).

EXAMPLE PATTERN:
{example_pattern}

REQUIREMENTS
- Each lf_*(ex) takes one positional arg, returns int class id or ABSTAIN (=-1).
- Each lf_* should call >=2 helpers OR combine >=2 boolean conditions.
- Target the cluster's TARGET CLASS specifically.
- USE THE QUERY EXAMPLES AND DISTINCTIVE KEYWORDS BELOW to inform helper design.
- Helpers can be shared across this batch's lf_*.
- Names: do NOT reuse names from the denylist below.
- Use ONLY metadata fields enumerated in the task's metadata description.

CLUSTER FAILURE CONTEXT:
{cluster_block}

DENYLIST (do NOT reuse these names): {denylist}

Output ONE python ```code block``` containing helpers + new lf_* together.
"""


def _brief_clusters(brief: dict) -> List[dict]:
    """Locate the cluster list inside a brief.

    analyzer writes clusters under brief["query_selection"]["clusters"]; older
    consumer code looked at brief["clusters"]. Fall through both shapes.
    """
    qs = brief.get("query_selection") or {}
    return (qs.get("clusters")
            or brief.get("clusters")
            or brief.get("targets")
            or [])


def render_clusters(brief: dict, max_clusters: int = 6, max_examples_per_cluster: int = 6) -> str:
    """Render top clusters as a prompt block for the Improver."""
    clusters = _brief_clusters(brief)
    if not clusters:
        return "  (no clusters surfaced — generate diverse class-targeted annotators broadly)"
    out = []
    for i, cl in enumerate(clusters[:max_clusters], 1):
        target_class = cl.get("target_label")
        target_name = cl.get("target_label_name", "?")
        if target_class is None:
            header = f"  CLUSTER {i}: target_class=uncovered (members mostly abstain)"
        else:
            header = f"  CLUSTER {i}: target_class={target_class} ({target_name})"
        out.append(header)
        kws = cl.get("distinctive_keywords") or []
        kw_list = [k for k, _ in kws[:10]] if kws and isinstance(kws[0], (list, tuple)) else kws[:10]
        kw_str = ", ".join(str(k) for k in kw_list)
        if kw_str:
            out.append(f"    distinctive keywords: [{kw_str}]")
        examples = cl.get("top_examples") or cl.get("examples") or []
        for ex in examples[:max_examples_per_cluster]:
            txt = (ex.get("text") or "").replace("\n", " ").strip()[:300]
            out.append(f"    EXAMPLE: {txt}")
        out.append("")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="chemprot")
    p.add_argument("--pool_module", type=Path, required=True,
                   help="Current pool.py to improve upon")
    p.add_argument("--brief_path", type=Path, required=True,
                   help="improvement_brief.json from analyze_errors")
    p.add_argument("--out_dir", type=Path, required=True,
                   help="Where to write pool_fragment.py + meta")
    p.add_argument("--n_calls", type=int, default=6,
                   help="Number of LLM calls; each generates 5-8 annotators")
    p.add_argument("--iter_tag", type=str, default="i01",
                   help="Tag to namespace this iter's helpers (e.g. i01, i02)")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--max_output_tokens", type=int, default=2400)
    p.add_argument("--cache_dir", type=Path,
                   default=Path("runs/cache/openai_responses/evopool_imp"))
    p.add_argument("--min_precision", type=float, default=0.25,
                   help="D_v6 production floor.")
    p.add_argument("--min_fires", type=int, default=5)
    p.add_argument("--max_jaccard_overlap", type=float, default=0.95,
                   help="Drop new annotator if firing pattern overlaps existing one by > this.")
    p.add_argument("--max_train_val_prec_gap", type=float, default=0.30,
                   help="Drop annotators whose train precision lags val by more than this (val-overfit).")
    p.add_argument("--exclude_classes", type=str, default="",
                   help="Comma-separated class IDs to skip in cluster targeting "
                        "(passed by orchestrator from viability history).")

    # ─── C1_v2: per-class Improver budget allocation (production) ───
    p.add_argument("--c1v2_targets", type=str, default="",
                   help="Comma-sep class IDs to dedicate 1 batch each to "
                        "(e.g. '6,9,7' for low-F1 long-tail classes).")
    p.add_argument("--c1v2_n_pos_examples", type=int, default=8,
                   help="Positive train examples per per-class batch.")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[IMP] task={args.task} loading data + brief")
    data = load_task_data(args.task)
    cfg = get_task_config(args.task)

    brief = json.loads(args.brief_path.read_text())

    # Parse excluded classes
    excluded = set()
    if args.exclude_classes.strip():
        for tok in args.exclude_classes.split(","):
            tok = tok.strip()
            if tok.isdigit():
                excluded.add(int(tok))
    if excluded:
        qs = brief.get("query_selection")
        if isinstance(qs, dict) and isinstance(qs.get("clusters"), list):
            before = len(qs["clusters"])
            qs["clusters"] = [c for c in qs["clusters"]
                              if (c.get("target_label") is None
                                  or int(c.get("target_label")) not in excluded)]
            print(f"[IMP] excluded classes {sorted(excluded)} → "
                  f"clusters: {before} → {len(qs['clusters'])}")
    cluster_block = render_clusters(brief)

    # Load existing pool → denylist
    import importlib.util
    spec = importlib.util.spec_from_file_location("cur_pool", args.pool_module)
    cur_pool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cur_pool)
    denylist = sorted([fn.__name__ for fn in getattr(cur_pool, "ANNOTATORS", [])])
    if len(denylist) > 80:
        denylist = denylist[:80]
    print(f"[IMP] denylist: {len(denylist)} names")

    # ─── C1_v2: parse per-class targets ───
    c1v2_targets: list = []
    if args.c1v2_targets.strip():
        for tok in args.c1v2_targets.split(","):
            tok = tok.strip()
            if tok.isdigit():
                c1v2_targets.append(int(tok))
        c1v2_targets = c1v2_targets[: max(0, args.n_calls - 1)]

    # Build per-class context: positive train examples per target class
    c1v2_pos_by_class: dict = {}
    if c1v2_targets:
        import random as _rnd
        rng = _rnd.Random(42)
        train_by_cls: dict = {}
        for r in data["train"]:
            try:
                c = int(r.get("true_label", -1))
            except Exception:
                continue
            if c < 0:
                continue
            train_by_cls.setdefault(c, []).append(r)
        for c in c1v2_targets:
            pool = train_by_cls.get(c, [])
            n = min(args.c1v2_n_pos_examples, len(pool))
            c1v2_pos_by_class[c] = rng.sample(pool, n) if n else []
        print(f"[IMP] C1_v2: targeting low-F1 classes {c1v2_targets} "
              f"(1 dedicated batch each + {args.n_calls - len(c1v2_targets)} broad batches)")

    def _build_per_class_prompt(class_id: int, batch_idx: int, total: int) -> str:
        class_name = cfg.label_names.get(class_id, str(class_id))
        guardrail = cfg.class_guardrails.get(class_id, "(no guardrail)")
        all_class_summary = "\n".join(f"  {c} = {n}" for c, n in sorted(cfg.label_names.items()))
        other_classes = [(c, n) for c, n in sorted(cfg.label_names.items()) if c != class_id]
        confusable_block = "\n".join(
            f"  {c} = {n}: {(cfg.class_guardrails.get(c) or '')[:120]}"
            for c, n in other_classes[:6]
        )
        pos_examples = c1v2_pos_by_class.get(class_id, [])
        pos_text = "\n".join(
            f"  [{i+1}] {(r.get('text') or '').strip()[:280]}"
            for i, r in enumerate(pos_examples)
        )
        target_clusters = [c for c in _brief_clusters(brief)
                           if c.get("target_label") is not None
                           and int(c.get("target_label")) == class_id]
        sub_brief = deepcopy(brief)
        sub_brief.setdefault("query_selection", {})["clusters"] = target_clusters
        local_cluster_block = (render_clusters(sub_brief) if target_clusters
                               else "  (no clusters available for this class — use positive examples below)")

        return (
            IMPROVER_INSTRUCTION.format(
                cluster_block=local_cluster_block,
                denylist=denylist,
                example_pattern=_pick_example_pattern(args.task),
            )
            + (
                f"\n=== PER-CLASS FOCUS: target_class={class_id} ({class_name}) ===\n"
                f"\nALL CLASSES (do NOT predict these unless target):\n{all_class_summary}\n"
                f"\nTARGET CLASS DEFINITION:\n{guardrail}\n"
                f"\nCONFUSABLE CLASSES (the most common errors — be careful NOT to fire on these):\n{confusable_block}\n"
                f"\nPOSITIVE TRAIN EXAMPLES of class {class_id}={class_name}:\n{pos_text or '  (none — use cluster examples)'}\n"
                f"\n[improver batch {batch_idx+1}/{total}; PER-CLASS targeting class {class_id}={class_name}]\n"
                f"Write 5-8 compositional lf_*. Each must:\n"
                f"  - Return {class_id} on positive cases, ABSTAIN otherwise\n"
                f"  - NOT fire on the confusable classes listed above\n"
                f"  - Use >=2 helper conditions (compositional)\n"
                f"Task: {cfg.task_description}\n"
                f"Metadata fields available:\n{cfg.metadata_fields_description}\n"
            )
        )

    def _build_broad_prompt(batch_idx: int, total: int) -> str:
        return IMPROVER_INSTRUCTION.format(
            cluster_block=cluster_block,
            denylist=denylist,
            example_pattern=_pick_example_pattern(args.task),
        ) + (
            f"\n[improver batch {batch_idx+1}/{total}; vary helpers + class targets]\n"
            f"Task: {cfg.task_description}\n"
            f"Class options: {sorted(cfg.label_names.items())}\n"
            f"Metadata fields available:\n{cfg.metadata_fields_description}\n"
        )

    # Build batch plan: K per-class first, then (n_calls - K) broad
    batch_plan: list = []
    for cls_id in c1v2_targets:
        batch_plan.append((f"c{cls_id}", _build_per_class_prompt(cls_id, len(batch_plan), args.n_calls)))
    n_broad = args.n_calls - len(batch_plan)
    for j in range(n_broad):
        batch_plan.append((f"b{len(batch_plan)+1}", _build_broad_prompt(len(batch_plan), args.n_calls)))

    print(f"[IMP] generating {len(batch_plan)} batches "
          f"({len(c1v2_targets)} per-class + {n_broad} broad)")
    candidates = []
    seen = set(denylist)
    for i, (btag, prompt) in enumerate(batch_plan):
        # Cache-bust per iter so re-runs with identical targets force fresh generations
        prompt_tagged = prompt + f"\n\n# improver_iter_tag: {args.iter_tag}__b{i}\n"
        try:
            text = call_llm(prompt_tagged, model=args.model, cache_dir=args.cache_dir,
                            max_output_tokens=args.max_output_tokens,
                            temperature=args.temperature)
        except Exception as e:
            print(f"  [batch {i+1}] ERROR: {e}")
            continue
        raw_pairs = parse_lfs_from_response(text, seen_names=set())
        if not raw_pairs:
            continue
        shared_block = raw_pairs[0][1]

        rename_map = {}
        for name, _ in raw_pairs:
            if name in seen:
                new = f"{name}_{args.iter_tag}_{btag}"
                bump = 0
                while new in seen:
                    bump += 1
                    new = f"{name}_{args.iter_tag}_{btag}_{bump}"
                rename_map[name] = new

        renamed_block = shared_block
        if rename_map:
            for old, new in rename_map.items():
                renamed_block = re.sub(rf"\bdef\s+{re.escape(old)}\b",
                                         f"def {new}", renamed_block)

        renamed_block = prefix_helpers_in_block(renamed_block, f"{args.iter_tag}{btag}")

        for orig_name, _ in raw_pairs:
            final = rename_map.get(orig_name, orig_name)
            if final in seen:
                continue
            candidates.append((final, renamed_block))
            seen.add(final)
        print(f"  [batch {i+1} ({btag})] +{len(raw_pairs)} raw  (total candidates={len(candidates)})")

    # Score each candidate on val
    print(f"\n[IMP] scoring {len(candidates)} candidates on val")
    scored = []
    for name, block in candidates:
        try:
            lf = build_callable_permissive(block, name)
        except Exception as e:
            print(f"  [{name}] build_error: {e}")
            continue
        m = evaluate_lf_on_split(lf, data["val"], "true_label")
        scored.append((name, block, m))

    # Filter chain: precision/fires → train↔val consistency → Jaccard dedup
    after_thresh = filter_lfs_by_threshold(scored, args.min_precision, args.min_fires)
    print(f"[IMP] precision/fires gate (>={args.min_precision}, >={args.min_fires}): "
          f"{len(after_thresh)}/{len(scored)} pass")

    after_consist, dropped_overfit = consistency_filter_train_val(
        after_thresh, data["train"], "true_label",
        max_train_val_prec_gap=args.max_train_val_prec_gap,
    )
    if dropped_overfit:
        print(f"[IMP] train/val consistency: dropped {len(dropped_overfit)} val-overfit annotators")
        for n, why in dropped_overfit[:5]:
            print(f"    {n}: {why}")

    print(f"[IMP] computing existing pool firing patterns for Jaccard dedup")
    existing_fired = collect_existing_fired_ids(args.pool_module, data["val"], "true_label")
    print(f"[IMP]   {len(existing_fired)} existing annotators to compare against")
    kept, dropped_jaccard = jaccard_dedup_against_pool(
        after_consist, existing_fired, max_overlap=args.max_jaccard_overlap,
    )
    if dropped_jaccard:
        print(f"[IMP] Jaccard dedup (>{args.max_jaccard_overlap}): "
              f"dropped {len(dropped_jaccard)} near-clones")
        for n, why in dropped_jaccard[:5]:
            print(f"    {n}: {why}")
    print(f"[IMP] FINAL: {len(kept)} new annotators survive (was {len(scored)} raw → "
          f"{len(after_thresh)} thresh → {len(after_consist)} consist → {len(kept)} jaccard)")

    # Write fragment
    blocks = []
    seen_blk = set()
    for _, blk, _ in kept:
        if blk not in seen_blk:
            blocks.append(blk)
            seen_blk.add(blk)
    lf_names = [n for n, _, _ in kept]

    fragment_path = args.out_dir / "pool_fragment.py"
    with open(fragment_path, "w", encoding="utf-8") as f:
        f.write(f"# EvoPool Improver fragment ({args.iter_tag}) — {len(lf_names)} new annotators\n\n")
        for blk in blocks:
            f.write(blk.rstrip() + "\n\n")
        f.write(f"\n# New annotator names this fragment introduces:\n# {lf_names}\n")
    json.dump({"new_lf_names": lf_names,
               "metrics": [{"name": n, "metrics": m} for n, _, m in kept]},
              open(args.out_dir / "new_candidates_meta.json", "w"), indent=2)

    # Emit filter_killed.json (record of dropped candidates for orchestrator)
    kept_names = {n for n, _, _ in kept}
    killed_records = []
    for name, block, m in scored:
        if name in kept_names:
            continue
        snip_cap = 240
        try:
            mm = re.search(rf"def\s+{re.escape(name)}\([^)]*\):.*?(?=\n\S|\Z)",
                             block, re.S)
            snip = mm.group(0)[:snip_cap] if mm else block[:snip_cap]
        except Exception:
            snip = block[:snip_cap]
        prec = (m or {}).get("precision")
        fires = (m or {}).get("fires")
        if name in {n for n, _, _ in after_thresh}:
            if name in {n for n, _, _ in after_consist}:
                reason = "jaccard-dup vs existing pool"
            else:
                reason = "train↔val precision gap (val-overfit)"
        else:
            reason = f"failed precision/fires gate (prec={prec} fires={fires})"
        killed_records.append({"name": name, "reason": reason, "snippet": snip})
    json.dump(killed_records, open(args.out_dir / "filter_killed.json", "w"), indent=2)
    print(f"[IMP] wrote filter_killed.json ({len(killed_records)} records)")
    print(f"[IMP] done → {fragment_path}")


if __name__ == "__main__":
    main()
