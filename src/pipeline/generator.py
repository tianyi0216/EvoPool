#!/usr/bin/env python3
"""EvoPool: Generator agent — initial annotator pool from compositional prompt.

Standalone (no orchestrator integration required). Produces:
  <out_dir>/iter_00/pool.py            — code blocks + ANNOTATORS list
  <out_dir>/iter_00/candidates_meta.json
  <out_dir>/iter_00/summary.json

Auto-selects the prompt by task_family:
  - classification tasks      → LEXICAL_INSTRUCTION (lexical prompt)
  - verification/NLI tasks    → VERIFICATION_INSTRUCTION (claim-evidence variant)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from src.pipeline.common import (
    parse_lfs_from_response, build_callable_permissive, call_llm,
    evaluate_lf_on_split, filter_lfs_by_threshold,
    write_pool, prefix_helpers_in_block,
    load_task_data, get_task_config, stratified_seed_text,
)
from src.prompts.lexical import LEXICAL_INSTRUCTION
from src.prompts.verification import VERIFICATION_INSTRUCTION
from src.prompts.utils import base_prompt_classification


# Tasks where the lexical prompt's ChemProt-style example biases the LLM away
# from the right primitive (metadata comparison). For these, use the
# verification variant which pushes annotators to compose pre-computed
# comparison features instead of regexing the raw text.
_VERIFICATION_TASKS = {"fever"}


def _pick_instruction(task: str) -> str:
    """Auto-select prompt by task_family, falling back to the hard-coded set."""
    try:
        cfg = get_task_config(task)
        family = getattr(cfg, "task_family", None)
        if family == "verification":
            print(f"[GEN] task={task!r} (family=verification) → VERIFICATION_INSTRUCTION")
            return VERIFICATION_INSTRUCTION
        if family == "classification":
            return LEXICAL_INSTRUCTION
    except Exception:
        pass
    if task in _VERIFICATION_TASKS:
        print(f"[GEN] task={task!r} → VERIFICATION_INSTRUCTION (claim-evidence)")
        return VERIFICATION_INSTRUCTION
    return LEXICAL_INSTRUCTION


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="chemprot")
    p.add_argument("--out_dir", type=Path, required=True,
                   help="Output root; iter_00/pool.py is written under it.")
    p.add_argument("--n_calls", type=int, default=18,
                   help="Number of broad LLM batches.")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--max_output_tokens", type=int, default=2400)
    p.add_argument("--cache_dir", type=Path,
                   default=Path("runs/cache/openai_responses/evopool_gen"))
    p.add_argument("--min_precision", type=float, default=0.30)
    p.add_argument("--min_fires", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    iter0 = args.out_dir / "iter_00"
    iter0.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[GEN] task={args.task} loading data + cfg")
    data = load_task_data(args.task)
    cfg = get_task_config(args.task)
    seed_text = stratified_seed_text(data["val"], cfg, n_per_class=2, seed=args.seed)
    instruction = _pick_instruction(args.task)
    base = base_prompt_classification(cfg, seed_text, prompt_extra=instruction)

    # Broad batch plan
    batches = [(f"b{i+1}", base + f"\n[batch {i+1}/{args.n_calls}; vary helpers and target classes]")
               for i in range(args.n_calls)]
    print(f"[GEN] generating {len(batches)} broad batches")

    candidates = []
    seen = set()
    for bi, (bid, prompt) in enumerate(batches, 1):
        try:
            text = call_llm(prompt, model=args.model, cache_dir=args.cache_dir,
                            max_output_tokens=args.max_output_tokens,
                            temperature=args.temperature)
        except Exception as e:
            print(f"  [batch {bi} ({bid})] ERROR: {e}")
            continue
        raw_pairs = parse_lfs_from_response(text, seen_names=set())
        if not raw_pairs:
            continue
        shared_block_src = raw_pairs[0][1]

        rename_map = {}
        for name, _ in raw_pairs:
            if name in seen:
                new_name = f"{name}_{bid}"
                while new_name in seen:
                    new_name += "x"
                rename_map[name] = new_name

        renamed_block = shared_block_src
        if rename_map:
            for old, new in rename_map.items():
                renamed_block = re.sub(rf"\bdef\s+{re.escape(old)}\b",
                                         f"def {new}", renamed_block)

        # Prefix this block's helpers with batch tag → no helper collisions across batches
        renamed_block = prefix_helpers_in_block(renamed_block, bid)

        for orig_name, _ in raw_pairs:
            final_name = rename_map.get(orig_name, orig_name)
            if final_name in seen:
                continue
            candidates.append((final_name, renamed_block))
            seen.add(final_name)
        print(f"  [batch {bi} ({bid})] +{len(raw_pairs)} raw; total candidates={len(candidates)}")

    # Score each candidate on val
    print(f"\n[GEN] scoring {len(candidates)} candidates on val")
    scored = []
    for name, block_src in candidates:
        try:
            lf = build_callable_permissive(block_src, name)
        except Exception as e:
            print(f"  [{name}] build_error: {e}")
            continue
        m = evaluate_lf_on_split(lf, data["val"], "true_label")
        scored.append((name, block_src, m))

    kept = filter_lfs_by_threshold(scored, args.min_precision, args.min_fires)
    print(f"[GEN] {len(kept)}/{len(scored)} annotators survive "
          f"(min_prec={args.min_precision}, min_fires={args.min_fires})")

    # Write pool: dedupe blocks by content, list all kept lf_* names
    blocks: list = []
    seen_blk: set = set()
    for _, blk, _ in kept:
        if blk not in seen_blk:
            blocks.append(blk)
            seen_blk.add(blk)
    lf_names = [n for n, _, _ in kept]

    write_pool(iter0 / "pool.py", blocks, lf_names,
               comment=f"EvoPool Generator — {len(lf_names)} annotators across {len(blocks)} blocks")

    json.dump([{"name": n, "metrics": m} for n, _, m in kept],
              open(iter0 / "candidates_meta.json", "w"), indent=2)
    json.dump({"n_raw": len(candidates), "n_scored": len(scored), "n_kept": len(kept)},
              open(iter0 / "summary.json", "w"), indent=2)

    print(f"[GEN] done → {iter0}/pool.py")


if __name__ == "__main__":
    main()
