"""EvoPool: prompts package.

Production Generator prompts are auto-selected by the orchestrator from
``task_config.task_family``:

- ``classification`` -> ``LEXICAL_INSTRUCTION`` (lexical, no-CoT). Used for
  ChemProt and PubMed (multi-label).
- ``verification``   -> ``VERIFICATION_INSTRUCTION`` (metadata-comparison).
  Used for FEVER.

Helpers (``call_llm``, ``parse_lfs_from_response``,
``build_callable_permissive``, ``filter_by_precision``, ``write_pool_py``,
``stratified_seed_text``, ``base_prompt_classification``) are re-exported
from ``utils.py``. The HTTP/cache layer lives in ``llm_client.py``.
"""
from src.prompts.lexical import LEXICAL_INSTRUCTION
from src.prompts.verification import VERIFICATION_INSTRUCTION
from src.prompts.utils import (
    base_prompt_classification,
    build_callable_permissive,
    call_llm,
    code_for_fn,
    filter_by_precision,
    parse_lfs_from_response,
    stratified_seed_text,
    write_pool_py,
)
from src.prompts.llm_client import (
    ABSTAIN,
    OpenAIRequest,
    extract_code_block,
    openai_call,
)

__all__ = [
    "LEXICAL_INSTRUCTION",
    "VERIFICATION_INSTRUCTION",
    "call_llm",
    "parse_lfs_from_response",
    "build_callable_permissive",
    "filter_by_precision",
    "write_pool_py",
    "stratified_seed_text",
    "base_prompt_classification",
    "code_for_fn",
    "OpenAIRequest",
    "openai_call",
    "extract_code_block",
    "ABSTAIN",
]
