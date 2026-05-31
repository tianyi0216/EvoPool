"""EvoPool: Prompt-side helpers shared by all agents (Generator/Improver/Refiner).

Exposes:
- ``call_llm(prompt, model, cache_dir, ...)`` -- thin wrapper around
  ``llm_client.openai_call`` that returns the assistant text directly.
- ``parse_lfs_from_response(text, seen_names)`` -- pulls ``lf_*`` candidates
  from a fenced code block; PERMISSIVE (allows imports + helpers + loops).
- ``build_callable_permissive(full_src, fn_name)`` -- exec the full source
  in a fresh namespace with full Python builtins + ``re`` + ``ABSTAIN``
  pre-bound; returns a callable. Trade-off: not safe-sandboxed, but
  acceptable here because annotators are LLM-authored for our own research
  code, not user-supplied input.
- ``filter_by_precision(...)`` -- run candidates on val, keep those above
  a precision/fires floor.
- ``write_pool_py(...)`` -- emit a ``pool.py`` compatible with
  ``src.pipeline.eval.run_annotators``.
- ``stratified_seed_text(...)`` -- per-class val examples formatted for prompts.
- ``base_prompt_classification(...)`` -- shared classification preamble that
  per-task scripts wrap with ``LEXICAL_INSTRUCTION`` or
  ``VERIFICATION_INSTRUCTION``.
"""
from __future__ import annotations

import ast as _ast
import random
import re as _module_re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.prompts.llm_client import (
    OpenAIRequest,
    _extract_response_text,
    extract_code_block,
    openai_call,
)


def call_llm(
    prompt: str,
    model: str,
    cache_dir: Path,
    max_output_tokens: int = 2000,
    temperature: float = 0.5,
) -> str:
    """Send a single prompt to the LLM and return the response text."""
    req = OpenAIRequest(
        model=model,
        input_text=prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    resp = openai_call(req, cache_dir=cache_dir)
    return _extract_response_text(resp)


def _normalize_negative_literals(code: str) -> str:
    """The pipeline's AST safety filter rejects ``ast.USub`` (unary minus), so
    e.g. ``return -1`` is silently dropped even though ``ABSTAIN = -1`` is the
    convention. Rewrite the most common forms to use the bound name instead.
    Conservative: only replace at exact word boundaries, not inside numbers
    like ``> -0.5``.
    """
    out = code
    out = _module_re.sub(r"\breturn\s+-\s*1\b(?!\.)", "return ABSTAIN", out)
    out = _module_re.sub(r"==\s*-\s*1\b(?!\.)", "== ABSTAIN", out)
    out = _module_re.sub(r"!=\s*-\s*1\b(?!\.)", "!= ABSTAIN", out)
    return out


def parse_lfs_from_response(
    response_text: str,
    seen_names: Optional[set] = None,
) -> List[Tuple[str, str]]:
    """Pull (name, src) pairs from a code block. PERMISSIVE: allows imports,
    helper functions, for/while loops, etc. We only require the candidate
    annotator to be named ``lf_*`` and to take a single positional argument.

    Source for each candidate annotator is the FULL code block (so any
    helpers and imports in the same block come along when
    ``build_callable_permissive`` runs).
    """
    seen_names = seen_names if seen_names is not None else set()
    code = extract_code_block(response_text)
    if not code:
        return []
    code = _normalize_negative_literals(code)

    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return []
    out = []
    for node in tree.body:
        if not isinstance(node, _ast.FunctionDef):
            continue
        if not node.name.startswith("lf_"):
            continue
        if node.name in seen_names:
            continue
        if len(node.args.args) != 1:
            continue
        out.append((node.name, code))
        seen_names.add(node.name)
    return out


def build_callable_permissive(full_src: str, fn_name: str):
    """Exec ``full_src`` in a fresh namespace with FULL Python builtins +
    ``ABSTAIN`` and ``re`` pre-bound, then return the function named ``fn_name``.

    Allows imports, helper functions, for/while loops, regex compilation, etc.
    Trade-off: not safe-sandboxed. Acceptable here because annotators are
    LLM-authored for our own research code, not user-supplied input.
    """
    import builtins as _builtins
    env: Dict[str, Any] = {
        "__builtins__": dict(vars(_builtins)),
        "re": _module_re,
        "ABSTAIN": -1,
    }
    code_obj = compile(full_src, filename=f"<{fn_name}>", mode="exec")
    exec(code_obj, env, env)
    fn = env.get(fn_name)
    if not callable(fn):
        raise RuntimeError(f"Failed to build callable for {fn_name}")
    return fn


def code_for_fn(full_code: str, fn) -> str:
    """Extract the source of a single function (and any helper funcs at module level)."""
    lines = full_code.split("\n")
    return "\n".join(lines[fn.lineno - 1: fn.end_lineno])


def filter_by_precision(
    candidates: List[Tuple[str, str]],
    val_rows: List[Dict[str, Any]],
    label_key: str,
    min_precision: float = 0.20,
    min_fires: int = 5,
) -> List[Tuple[str, str, Dict[str, float]]]:
    """Build callable (PERMISSIVE), run on val, keep if
    ``precision >= min_precision`` and ``fires >= min_fires``.
    Returns list of (name, src, metrics).
    """
    kept = []
    for name, src in candidates:
        try:
            lf = build_callable_permissive(src, name)
        except Exception as e:
            if len(kept) < 3:
                print(f"    [filter] {name} build error: {e}", flush=True)
            continue
        fires = 0
        correct = 0
        for r in val_rows:
            try:
                v = lf(r)
            except Exception:
                v = -1
            if v == -1:
                continue
            fires += 1
            if v == int(r.get(label_key, -2)):
                correct += 1
        if fires < min_fires:
            continue
        prec = correct / fires
        if prec < min_precision:
            continue
        kept.append((name, src, {"fires": fires, "correct": correct, "precision": prec}))
    return kept


def write_pool_py(
    kept_lfs: List[Tuple[str, str, Dict]],
    out_path: Path,
    header_comment: str = "",
):
    """Write ``pool.py`` compatible with ``src.pipeline.eval.run_annotators``.

    Each kept annotator carries ``src`` = the FULL code block from its LLM
    response (with imports + helpers). We dedupe by ``src`` content and emit
    each unique block once, then list all kept ``lf_*`` names in ``ANNOTATORS``.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen_src: List[str] = []
    for _, src, _ in kept_lfs:
        if src not in seen_src:
            seen_src.append(src)
    with open(out_path, "w") as f:
        f.write("# Auto-generated EvoPool annotator pool (PERMISSIVE)\n")
        f.write(f"# {header_comment}\n\n")
        f.write("from typing import Any, List\n")
        f.write("ABSTAIN = -1\n\n")
        for src in seen_src:
            f.write(src.strip() + "\n\n")
        f.write("# run_annotators expects this exact name + bare function refs\n")
        f.write("ANNOTATORS: List[Any] = [\n")
        for name, _, _ in kept_lfs:
            f.write(f"    {name},\n")
        f.write("]\n")
    print(f"  wrote {out_path}  ({len(kept_lfs)} annotators across {len(seen_src)} blocks)", flush=True)


def stratified_seed_text(
    val_rows: List[Dict[str, Any]],
    task_cfg,
    n_per_class: int = 2,
    label_key: str = "true_label",
    seed: int = 42,
) -> str:
    """Pull a few val examples per class, format as
    ``[Class X=label_name] <text>`` for prompts.
    """
    by_cls: Dict[int, List[Dict]] = {}
    for r in val_rows:
        c = r.get(label_key)
        if c is None:
            continue
        try:
            c = int(c)
        except Exception:
            continue
        by_cls.setdefault(c, []).append(r)
    rng = random.Random(seed)
    chunks = []
    for c in sorted(by_cls.keys()):
        sampled = rng.sample(by_cls[c], min(n_per_class, len(by_cls[c])))
        cname = task_cfg.label_names.get(c, str(c))
        for r in sampled:
            txt = (r.get("text") or "")[:300]
            chunks.append(f"  [Class {c}={cname}] {txt}")
    return "\n".join(chunks)


def base_prompt_classification(task_cfg, seed_examples_text: str, prompt_extra: str = "") -> str:
    """Shared classification preamble. Per-task scripts append either
    ``LEXICAL_INSTRUCTION`` or ``VERIFICATION_INSTRUCTION`` via ``prompt_extra``.
    """
    label_lines = "\n".join(f"  {k} = {v}" for k, v in sorted(task_cfg.label_names.items()))
    guardrails = "\n".join(
        f"\n[{task_cfg.label_names[c]}]\n{txt}"
        for c, txt in task_cfg.class_guardrails.items()
    )
    return (
        f"You are an expert at writing executable Python annotators for "
        f"{task_cfg.task_description}\n\n"
        f"Class options:\n{label_lines}\n\n"
        f"Per-class guidance:\n{guardrails}\n\n"
        f"Each annotator takes a dict ex with fields:\n"
        f"{task_cfg.metadata_fields_description}\n\n"
        f"Return ABSTAIN (= -1) when the function should not predict.\n"
        f"Return an integer class label otherwise.\n\n"
        f"Seed examples (val):\n{seed_examples_text}\n\n"
        f"{prompt_extra}\n"
    )
