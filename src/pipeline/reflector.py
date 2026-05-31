#!/usr/bin/env python3
"""EvoPool: Reflector agent — Reflexion-inspired memory consolidator (OPTIONAL).

OFF by default (memory_level=0 / L0 production is purely Darwinian and beats
the Reflector-enabled R1 variant on the paper's headline aggregator A1).
Shipped for memory ablations only.

Reads:
  --raw_events_path   memory/raw_events.jsonl (append-only buffer of dropped
                      annotators + per-class F1 deltas, written by orchestrator)
  --prev_lessons_path memory/lessons.json from previous reflection (may not exist)

Writes:
  --out_lessons_path  memory/lessons.json (per-class semantic lessons + meta)
  --archive_dir       memory/_archive/iter_{NN}_raw_events.jsonl (audit log)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from src.pipeline.common import call_llm, get_task_config


REFLECTION_PROMPT = """\
You are reflecting on raw events (annotators that were dropped from the pool, plus
per-class F1 trajectory) for class {class_id} = "{class_name}" in a multi-class
classification task.

TASK CONTEXT
{task_description}

CLASS DEFINITION
{class_guardrail}

CONFUSABLE CLASSES (most likely sources of error)
{confusable_block}

PREVIOUS LESSON (from earlier reflection — may be empty):
{prev_lesson}

CURRENT PER-CLASS F1 TRAJECTORY (most recent first):
{f1_trajectory}

RAW EVENTS THIS PERIOD ({n_events} events for class {class_id}):
{events_block}

YOUR TASK
Write a 2-4 sentence LESSON for future annotators targeting class {class_id}. The
lesson should:
  1. Identify the COMMON FAILURE PATTERN across the events.
  2. State explicitly what future annotators should AVOID.
  3. State explicitly what future annotators should DO instead.

If a previous lesson exists and is still accurate, REFINE it; do not just
repeat it. If new evidence contradicts it, write a corrected lesson.

OUTPUT
Plain prose. 2-4 sentences. No bullets, no numbering, no headers, no quotes.
Just the lesson text.
"""


CONSOLIDATION_PROMPT = """\
You have {n_classes} per-class lessons that together exceed the token budget
({total_tokens} > {budget}). Consolidate them while preserving the
class-specific guidance for the {top_k} classes that are currently STRUGGLING
(lowest F1 in the latest iter).

INPUT LESSONS:
{lessons_block}

CURRENT STRUGGLING CLASSES (highest priority — keep these verbatim):
{top_k_classes}

YOUR TASK
1. Keep the lessons for the struggling classes EXACTLY as-is.
2. Combine the lessons for the OTHER classes into a single "general lessons"
   paragraph (3-5 sentences) that captures the cross-class transferable
   patterns (e.g., common pitfalls, useful disambiguators).

OUTPUT (strict JSON, no markdown fences):
{{
  "general": "<3-5 sentence cross-class summary>",
  "kept_class_ids": [<list of class ids you kept verbatim>]
}}
"""


def _estimate_tokens(s: str) -> int:
    """Rough heuristic: 1 token per ~4 chars."""
    return max(1, len(s) // 4)


def _all_lessons_token_estimate(lessons: Dict) -> int:
    return sum(_estimate_tokens(v.get("lesson", ""))
               for k, v in lessons.items()
               if not k.startswith("_") and isinstance(v, dict))


def _read_jsonl(path: Path) -> List[Dict]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="chemprot")
    p.add_argument("--raw_events_path", type=Path, required=True)
    p.add_argument("--prev_lessons_path", type=Path, required=True)
    p.add_argument("--out_lessons_path", type=Path, required=True)
    p.add_argument("--archive_dir", type=Path, required=True)
    p.add_argument("--iter_at_reflection", type=int, required=True)
    p.add_argument("--cache_dir", type=Path, required=True)
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--max_output_tokens", type=int, default=400)
    p.add_argument("--token_budget", type=int, default=2000,
                   help="Total lessons.json token cap; consolidation triggers above this.")
    p.add_argument("--top_k_struggling", type=int, default=3,
                   help="When consolidating, keep this many lowest-F1 classes verbatim.")
    args = p.parse_args()

    args.archive_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.out_lessons_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = get_task_config(args.task)
    raw = _read_jsonl(args.raw_events_path)
    prev_lessons: Dict[str, Any] = {}
    if args.prev_lessons_path.exists():
        try:
            prev_lessons = json.loads(args.prev_lessons_path.read_text())
        except Exception:
            prev_lessons = {}

    print(f"[REFLECTOR] iter_{args.iter_at_reflection} task={args.task}")
    print(f"  raw_events: {len(raw)} entries; "
          f"prev_lessons: {len([k for k in prev_lessons if not k.startswith('_')])} classes")

    by_class: Dict[int, List[Dict]] = {}
    f1_history: Dict[int, List[float]] = {}
    for ev in raw:
        et = ev.get("type", "")
        if et.endswith("_dropped"):
            cls = ev.get("target_class")
            if cls is not None:
                by_class.setdefault(int(cls), []).append(ev)
        elif et == "class_f1":
            cls = ev.get("class")
            f1 = ev.get("f1")
            if cls is not None and f1 is not None:
                f1_history.setdefault(int(cls), []).append(float(f1))

    new_lessons: Dict[str, Any] = {}

    for cls_id in sorted(by_class.keys()):
        events = by_class[cls_id]
        cls_name = cfg.label_names.get(cls_id, str(cls_id))
        guardrail = cfg.class_guardrails.get(cls_id, "(no guardrail)")
        confusable_block = "\n".join(
            f"  {c} = {n}: {(cfg.class_guardrails.get(c) or '')[:120]}"
            for c, n in sorted(cfg.label_names.items()) if c != cls_id
        )
        events_block = "\n".join(
            f"[{i+1}] Annotator '{e.get('name','?')}' (iter_{e.get('iter','?')}) dropped: "
            f"{e.get('reason','?')}\n      code: "
            f"{(e.get('snippet') or '').replace(chr(10),' ')[:240]}"
            for i, e in enumerate(events[-15:])
        )
        traj = f1_history.get(cls_id, [])
        traj_str = ", ".join(f"{x:.3f}" for x in traj[-8:]) or "(no data)"

        prev_entry = prev_lessons.get(f"class_{cls_id}", {}) or {}
        prev_lesson_text = (prev_entry.get("lesson") or "").strip()

        prompt = REFLECTION_PROMPT.format(
            class_id=cls_id, class_name=cls_name,
            task_description=cfg.task_description,
            class_guardrail=guardrail,
            confusable_block=confusable_block,
            prev_lesson=prev_lesson_text or "(none yet)",
            f1_trajectory=traj_str,
            n_events=len(events),
            events_block=events_block or "(no events this period)",
        )
        try:
            text = call_llm(prompt, model=args.model, cache_dir=args.cache_dir,
                            max_output_tokens=args.max_output_tokens,
                            temperature=args.temperature)
        except Exception as e:
            print(f"  [class {cls_id}] REFLECTOR ERROR: {e}")
            new_lessons[f"class_{cls_id}"] = prev_entry
            continue
        lesson_text = re.sub(r"^```[a-z]*\n?|\n?```$", "", text.strip(), flags=re.M).strip()
        sentences = re.split(r"(?<=[.!?])\s+", lesson_text)
        if len(sentences) > 4:
            lesson_text = " ".join(sentences[:4])
        new_lessons[f"class_{cls_id}"] = {
            "lesson": lesson_text,
            "n_events_observed": prev_entry.get("n_events_observed", 0) + len(events),
            "last_updated_iter": args.iter_at_reflection,
        }
        print(f"  [class {cls_id}] {cls_name}: lesson updated ({len(events)} events) "
              f"~{_estimate_tokens(lesson_text)} tok")

    # Carry forward classes that didn't have new events
    for k, v in prev_lessons.items():
        if k.startswith("_"):
            continue
        if k not in new_lessons:
            new_lessons[k] = v

    # Token-budget trim
    total_tokens = _all_lessons_token_estimate(new_lessons)
    consolidation_applied = False
    if total_tokens > args.token_budget:
        print(f"[REFLECTOR] token budget exceeded ({total_tokens} > {args.token_budget}); consolidating")
        latest_f1 = {c: (f1_history.get(c, [0.0])[-1] if f1_history.get(c) else 0.0)
                     for c in range(len(cfg.label_names))}
        sorted_cls = sorted(latest_f1.items(), key=lambda kv: kv[1])
        top_k_cls = [c for c, _ in sorted_cls[:args.top_k_struggling]]
        kept_keys = {f"class_{c}" for c in top_k_cls}
        lessons_block = "\n".join(
            f"Class {k.split('_')[-1]}: {v.get('lesson','')}"
            for k, v in new_lessons.items()
            if not k.startswith("_") and isinstance(v, dict)
        )
        cprompt = CONSOLIDATION_PROMPT.format(
            n_classes=len([k for k in new_lessons if not k.startswith("_")]),
            total_tokens=total_tokens, budget=args.token_budget,
            top_k=args.top_k_struggling,
            lessons_block=lessons_block,
            top_k_classes=", ".join(f"class {c} ({cfg.label_names.get(c,'?')})" for c in top_k_cls),
        )
        try:
            cresp = call_llm(cprompt, model=args.model, cache_dir=args.cache_dir,
                             max_output_tokens=600, temperature=0.2)
            cresp_clean = re.sub(r"^```[a-z]*\n?|\n?```$", "", cresp.strip(), flags=re.M).strip()
            cdata = json.loads(cresp_clean)
            general_summary = (cdata.get("general") or "").strip()
            kept_returned = set(f"class_{c}" for c in (cdata.get("kept_class_ids") or top_k_cls))
            consolidated: Dict[str, Any] = {"_general": general_summary}
            for k, v in new_lessons.items():
                if k.startswith("_"):
                    continue
                if k in kept_returned or k in kept_keys:
                    consolidated[k] = v
            new_lessons = consolidated
            consolidation_applied = True
            print(f"  [REFLECTOR] consolidated. Kept {len(kept_returned | kept_keys)} verbatim + 1 general.")
        except Exception as e:
            print(f"  [REFLECTOR] consolidation failed ({e}); keeping per-class as-is.")

    new_lessons["_meta"] = {
        "iter_at_reflection": args.iter_at_reflection,
        "n_events_this_period": len(raw),
        "tokens_estimate": _all_lessons_token_estimate(new_lessons),
        "consolidation_applied": consolidation_applied,
    }

    args.out_lessons_path.write_text(json.dumps(new_lessons, indent=2))
    archive_path = args.archive_dir / f"iter_{args.iter_at_reflection:02d}_raw_events.jsonl"
    if args.raw_events_path.exists():
        archive_path.write_text(args.raw_events_path.read_text())
        args.raw_events_path.write_text("")
    print(f"[REFLECTOR] done → {args.out_lessons_path} (archived {archive_path.name})")


if __name__ == "__main__":
    main()
