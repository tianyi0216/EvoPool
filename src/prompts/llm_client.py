"""EvoPool: OpenAI Responses/Chat API client with on-disk response cache.

Provides ``openai_call`` (cache-aware HTTP wrapper) and
``_extract_response_text`` (cross-shape text extraction) used by every agent
that talks to an LLM. All responses are cached under ``cache_dir`` keyed by
the SHA-256 of the request payload, so repeated runs hit the cache and stay
deterministic.

Environment variables:
  - OPENAI_API_KEY    (required)
  - OPENAI_BASE_URL   (optional; defaults to https://api.openai.com)
  - OPENAI_API_TYPE   (optional; "vllm" / "chat" / "local" routes to
                       /v1/chat/completions instead of /v1/responses)
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


ABSTAIN = -1


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# -----------------------------
# OpenAI Responses API (stdlib)
# -----------------------------


@dataclass(frozen=True)
class OpenAIRequest:
    model: str
    input_text: str
    temperature: float = 0.2
    max_output_tokens: int = 2000


def _openai_base_url() -> str:
    return os.environ.get("OPENAI_BASE_URL", "https://api.openai.com").rstrip("/")


def _use_chat_completions() -> bool:
    """Use /v1/chat/completions instead of /v1/responses when serving via vLLM or similar."""
    return os.environ.get("OPENAI_API_TYPE", "").lower() in ("vllm", "chat", "local")


def _extract_response_text(resp: Dict[str, Any]) -> str:
    """Extract assistant text from either Responses API or Chat Completions shape."""
    if isinstance(resp.get("output_text"), str) and resp["output_text"]:
        return resp["output_text"]

    texts = []
    out = resp.get("output")
    if isinstance(out, list):
        for item in out:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for c in content:
                    if not isinstance(c, dict):
                        continue
                    t = c.get("text")
                    if isinstance(t, str) and t:
                        texts.append(t)
            t2 = item.get("text")
            if isinstance(t2, str) and t2:
                texts.append(t2)
    if texts:
        return "".join(texts)

    # Chat Completions fallback
    choices = resp.get("choices")
    if isinstance(choices, list) and choices:
        ch0 = choices[0]
        if isinstance(ch0, dict):
            msg = ch0.get("message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return msg["content"]
            if isinstance(ch0.get("text"), str):
                return ch0["text"]
    return ""


def extract_code_block(text: str) -> str:
    """Extract the first fenced code block if present; otherwise return text."""
    fence = "```"
    if fence not in text:
        return text.strip()

    m = re.search(r"```python\s*\n([\s\S]*?)```", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    m2 = re.search(r"```\s*\n([\s\S]*?)```", text)
    if m2:
        return m2.group(1).strip()
    return text.strip()


def openai_call(
    req: OpenAIRequest,
    cache_dir: Path,
    timeout_s: int = 60,
    max_retries: int = 12,
) -> Dict[str, Any]:
    """Send a request to OpenAI, caching responses by payload SHA-256.

    Handles reasoning-model quirks (gpt-5/o1/o3/o4, gemini-3, deepseek-v4/r1)
    via per-family payload tweaks. Retries on transient HTTP errors with
    backoff that respects Retry-After and rate-limit-reset headers.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    m = req.model.lower()
    is_reasoning_openai = m.startswith(("gpt-5", "o1", "o3", "o4"))
    is_reasoning_gemini = (m.startswith("gemini-3") or "preview" in m
                           or (m.startswith("gemini-2.5") and "pro" in m))
    is_reasoning_deepseek = m.startswith(("deepseek-v4", "deepseek-r1", "deepseek-reasoner"))

    if _use_chat_completions():
        payload = {
            "model": req.model,
            "messages": [{"role": "user", "content": req.input_text}],
            "max_tokens": req.max_output_tokens,
        }
        if is_reasoning_openai:
            payload["max_tokens"] = max(req.max_output_tokens * 8, 8192)
            payload["reasoning_effort"] = "minimal"
        elif is_reasoning_deepseek:
            payload["max_tokens"] = max(req.max_output_tokens * 32, 16384)
            payload["temperature"] = req.temperature
        elif is_reasoning_gemini:
            payload["max_tokens"] = max(req.max_output_tokens * 8, 8192)
            payload["temperature"] = req.temperature
        else:
            payload["temperature"] = req.temperature
        endpoint = "/v1/chat/completions"
    else:
        payload = {
            "model": req.model,
            "input": req.input_text,
        }
        if is_reasoning_openai:
            payload["max_output_tokens"] = max(req.max_output_tokens * 8, 8192)
            payload["reasoning"] = {"effort": "minimal"}
        else:
            payload["temperature"] = req.temperature
            payload["max_output_tokens"] = req.max_output_tokens
        endpoint = "/v1/responses"

    payload_str = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    cache_key = _sha256_hex(payload_str)
    cache_path = cache_dir / f"{cache_key}.json"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    url = f"{_openai_base_url()}{endpoint}"
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    def _parse_seconds(s: str) -> Optional[float]:
        s = (s or "").strip().lower()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            pass
        mm = re.match(r"^(\d+(\.\d+)?)(ms|s|m|h)$", s)
        if not mm:
            return None
        val = float(mm.group(1))
        unit = mm.group(3)
        if unit == "ms":
            return val / 1000.0
        if unit == "s":
            return val
        if unit == "m":
            return val * 60.0
        if unit == "h":
            return val * 3600.0
        return None

    def _extract_err_info(body_text: str) -> Dict[str, Optional[str]]:
        try:
            obj = json.loads(body_text)
        except Exception:
            return {"message": body_text.strip()[:500] or None, "type": None, "code": None}
        if isinstance(obj, dict):
            err = obj.get("error")
            if isinstance(err, dict):
                msg = err.get("message")
                et = err.get("type")
                code = err.get("code")
                return {
                    "message": str(msg) if msg is not None else None,
                    "type": str(et) if et is not None else None,
                    "code": str(code) if code is not None else None,
                }
        return {"message": body_text.strip()[:500] or None, "type": None, "code": None}

    def _suggested_sleep_s_from_message(msg: Optional[str]) -> Optional[float]:
        if not msg:
            return None
        mm = re.search(r"try again in\s+(\d+(\.\d+)?)\s*s", msg, flags=re.IGNORECASE)
        if mm:
            return float(mm.group(1))
        m2 = re.search(r"retry after\s+(\d+(\.\d+)?)\s*s", msg, flags=re.IGNORECASE)
        if m2:
            return float(m2.group(1))
        return None

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            http_req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(http_req, timeout=timeout_s) as resp:
                body = resp.read().decode("utf-8")
                obj = json.loads(body)
                with cache_path.open("w", encoding="utf-8") as f:
                    json.dump(obj, f, ensure_ascii=False, indent=2)
                    f.write("\n")
                return obj
        except urllib.error.HTTPError as e:
            last_err = e
            status = getattr(e, "code", None)
            body_bytes = b""
            try:
                body_bytes = e.read()  # type: ignore[assignment]
            except Exception:
                body_bytes = b""
            body_text = body_bytes.decode("utf-8", errors="replace") if body_bytes else ""
            err_info = _extract_err_info(body_text)
            msg_lower = (err_info.get("message") or "").lower()
            err_code = (err_info.get("code") or "").lower()
            err_type = (err_info.get("type") or "").lower()

            req_id = None
            try:
                req_id = (e.headers.get("x-request-id") or e.headers.get("x-request-id".upper()))  # type: ignore[attr-defined]
            except Exception:
                req_id = None

            if status in (401, 403):
                raise RuntimeError(
                    f"OpenAI auth error (HTTP {status}). "
                    f"message={err_info.get('message')!r} code={err_info.get('code')!r} type={err_info.get('type')!r} "
                    f"request_id={req_id!r}"
                ) from e

            if status == 429 and (
                "insufficient_quota" in err_code
                or "billing" in msg_lower
                or "quota" in msg_lower
                or "exceeded your current quota" in msg_lower
            ):
                raise RuntimeError(
                    "OpenAI returned HTTP 429 quota/billing limit. "
                    f"message={err_info.get('message')!r} code={err_info.get('code')!r} type={err_info.get('type')!r} "
                    f"request_id={req_id!r}"
                ) from e

            retryable = status in (408, 409, 429, 500, 502, 503, 504) or status is None
            if not retryable:
                raise RuntimeError(
                    f"OpenAI HTTP error {status}. "
                    f"message={err_info.get('message')!r} code={err_info.get('code')!r} type={err_info.get('type')!r} "
                    f"request_id={req_id!r}"
                ) from e

            retry_after_s = None
            try:
                retry_after_s = _parse_seconds(e.headers.get("Retry-After", ""))  # type: ignore[attr-defined]
            except Exception:
                retry_after_s = None

            reset_s: Optional[float] = None
            try:
                reset_req = e.headers.get("x-ratelimit-reset-requests", "")  # type: ignore[attr-defined]
                reset_tok = e.headers.get("x-ratelimit-reset-tokens", "")  # type: ignore[attr-defined]
                reset_s = _parse_seconds(reset_req) or _parse_seconds(reset_tok)
            except Exception:
                reset_s = None

            msg_sleep = _suggested_sleep_s_from_message(err_info.get("message"))

            base_backoff = min(60.0, 1.0 * (2 ** (attempt - 1)))
            sleep_s = max(
                x for x in [
                    0.0,
                    base_backoff,
                    retry_after_s or 0.0,
                    reset_s or 0.0,
                    msg_sleep or 0.0,
                ]
            )
            sleep_cap = 90.0
            sleep_s = min(sleep_cap, sleep_s)
            jitter = random.random() * min(1.0, sleep_s * 0.1)
            sleep_s = min(sleep_cap, sleep_s + jitter)

            print(
                f"[openai] HTTP {status} attempt {attempt}/{max_retries} "
                f"(type={err_type or None}, code={err_code or None}); sleeping {sleep_s:.1f}s",
                file=sys.stderr, flush=True,
            )
            time.sleep(sleep_s)
            continue
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last_err = e
            base_backoff = min(60.0, 1.0 * (2 ** (attempt - 1)))
            jitter = random.random() * 0.25
            sleep_s = min(60.0, base_backoff + jitter)
            print(
                f"[openai] transient error attempt {attempt}/{max_retries}; sleeping {sleep_s:.1f}s",
                file=sys.stderr, flush=True,
            )
            time.sleep(sleep_s)
            continue
    raise RuntimeError(f"OpenAI API call failed after {max_retries} retries: {last_err}") from last_err
