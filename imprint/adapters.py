"""Format adapters — Imprint is MODEL-AGNOSTIC.

Normalizes request/response logs from ANY major LLM API shape into Imprint's
internal Pair schema, and renders responses back in the caller's native format.

Supported ingress shapes (auto-detected):
  - OpenAI ChatCompletion : {"model", "messages":[{"role","content"}], ...}
  - Anthropic Messages    : {"model", "system", "messages":[{"role","content"}], ...}
  - Google Gemini-ish     : {"contents":[{"role","parts":[{"text"}]}]}
  - Raw completion        : {"prompt": "...", "completion"/"response": "..."}
  - Router log records    : A3M/LiteLLM/OpenRouter JSONL export (see README)
"""

from __future__ import annotations

import re
from typing import Any, Optional

PII_PATTERNS = [
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "[EMAIL]"),
    (re.compile(r"\b(?:\d[ -]*?){13,16}\b"), "[CARD]"),
    (re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"), "[KEY]"),
]


def redact(text: str) -> str:
    for pat, sub in PII_PATTERNS:
        text = pat.sub(sub, text)
    return text


def _flatten_content(content: Any) -> str:
    """Content may be a string, a list of parts (OpenAI vision / Gemini), or None."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict):
                if "text" in part:
                    texts.append(str(part["text"]))
                elif part.get("type") == "text":
                    texts.append(str(part.get("text", "")))
            elif isinstance(part, str):
                texts.append(part)
        return "\n".join(texts)
    return str(content)


def _messages_to_prompt(messages: list[dict]) -> str:
    """Flatten a role/content message list into one transcript string."""
    lines = []
    for m in messages or []:
        role = m.get("role", "user")
        content = _flatten_content(m.get("content"))
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def detect_format(record: dict) -> str:
    if isinstance(record.get("messages"), list):
        if record.get("system") is not None and any(
            m.get("role") == "assistant" for m in record["messages"]
        ):
            # anthropic puts system top-level; openai puts it as a message
            return "anthropic"
        return "openai"
    if isinstance(record.get("contents"), list):
        return "gemini"
    if "prompt" in record:
        return "raw"
    return "unknown"


# ---------------------------------------------------------------- egress ----


def render_response(
    fmt: str, text: str, model: str, extra_headers: Optional[dict] = None
) -> dict:
    """Wrap `text` back into the caller's native response shape."""
    meta = {
        "imprint_signature": extra_headers.get("X-Imprint-Signature"),
        "imprint_version": extra_headers.get("X-Imprint-Version"),
        "imprint_escalate": extra_headers.get("X-Imprint-Escalate") == "true",
    }
    if fmt == "openai":
        return {
            "id": f"imprint-{meta['imprint_version'] or 'dev'}",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            **{k: v for k, v in meta.items() if v not in (None, False)},
        }
    if fmt == "anthropic":
        return {
            "id": f"imprint_{meta['imprint_version'] or 'dev'}",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            **{"imprint": {k: v for k, v in meta.items() if v not in (None, False)}},
        }
    if fmt == "gemini":
        return {
            "candidates": [
                {
                    "content": {"parts": [{"text": text}], "role": "model"},
                    "finishReason": "STOP",
                }
            ],
            **{"imprint": {k: v for k, v in meta.items() if v not in (None, False)}},
        }
    # raw / unknown
    return {
        "response": text,
        **{k: v for k, v in meta.items() if v not in (None, False)},
    }


# --------------------------------------------------------------- ingress ----


def normalize_record(record: dict) -> Optional[dict]:
    """Normalize any supported log/request shape into an Imprint pair dict.

    Returns keys: prompt, response, model, provider (+ passthrough economics).
    Returns None when no usable prompt can be extracted.
    """
    fmt = detect_format(record)

    if fmt == "openai" or fmt == "anthropic":
        prompt = _messages_to_prompt(record.get("messages"))
        response = _flatten_content(
            (record.get("response_body") or {}).get("content")
            if isinstance(record.get("response_body"), dict)
            else record.get("response", "")
        ) or str(record.get("response", ""))
    elif fmt == "gemini":
        contents = []
        for c in record.get("contents") or []:
            role = c.get("role", "user")
            for part in c.get("parts") or []:
                t = part.get("text") if isinstance(part, dict) else str(part)
                if t:
                    contents.append(f"{role}: {t}")
        prompt = "\n".join(contents)
        response = str(record.get("response", ""))
    else:  # raw / router log
        prompt = str(record.get("prompt", ""))
        response = str(record.get("response") or record.get("completion") or "")

    if not prompt.strip():
        return None

    usage = record.get("usage") or {}
    cost = record.get("cost_usd")
    if cost is None and usage:
        cost = usage.get("total_cost") or usage.get("cost")

    return {
        "fmt": fmt,
        "prompt": redact(prompt),
        "response": redact(response),
        "model": record.get("model"),
        "provider": record.get("provider"),
        "cost_usd": float(cost or 0),
        "latency_ms": int(record.get("latency_ms") or 0),
        "cache_hit": bool(record.get("cache_hit")),
    }
