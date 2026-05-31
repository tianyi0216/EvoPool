"""EvoPool: shared config loader. Reads YAML, resolves ${path.x} refs, returns dict."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import yaml
except ImportError as e:
    raise SystemExit("PyYAML required. Install with: pip install pyyaml") from e


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load config.yaml and resolve ${dotted.path} interpolations."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"Config file not found: {path}")
    with path.open("r") as f:
        cfg = yaml.safe_load(f) or {}
    return _resolve_refs(cfg, cfg)


def get(cfg: Dict[str, Any], path: str, default: Optional[Any] = None) -> Any:
    """Read a nested key with dotted notation, e.g. get(cfg, 'pipeline.generator.n_calls')."""
    cur: Any = cfg
    for key in path.split("."):
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def _resolve_refs(value: Any, root: Dict[str, Any]) -> Any:
    """Recursively resolve ${path.to.key} placeholders inside string values."""
    if isinstance(value, str):
        pattern = re.compile(r"\$\{([^}]+)\}")

        def _sub(match: re.Match) -> str:
            ref = get(root, match.group(1))
            return str(ref) if ref is not None else match.group(0)

        prev: Optional[str] = None
        cur = value
        while cur != prev:
            prev = cur
            cur = pattern.sub(_sub, cur)
        return cur
    if isinstance(value, dict):
        return {k: _resolve_refs(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_refs(v, root) for v in value]
    return value
