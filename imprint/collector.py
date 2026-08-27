"""Collector: ingest MODEL-AGNOSTIC request logs into the Imprint store.

Auto-detects and normalizes: OpenAI ChatCompletion, Anthropic Messages,
Gemini-style contents, raw prompt/completion, and router JSONL exports
(A3M / LiteLLM / OpenRouter). PII redacted at ingest.
"""

import json
import sys
from typing import Any

import hashlib
import sqlite3

from .adapters import (  # noqa: F401
    normalize_record,
    redact,
    render_response,
)
from .store import connect, now_ms


def pair_id(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


def ingest_record(conn: sqlite3.Connection, record: dict[str, Any]) -> tuple[str, int]:
    """Returns ("inserted"|"duplicate"|"skipped", count=1)."""
    rec = normalize_record(record)
    if rec is None:
        return "skipped", 0
    pid = pair_id(rec["prompt"])
    try:
        conn.execute(
            "INSERT INTO pairs (id, ts, prompt, response, model, provider,"
            " cost_usd, latency_ms, cache_hit) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                pid,
                record.get("ts") or now_ms(),
                rec["prompt"],
                rec["response"],
                rec["model"],
                rec["provider"],
                rec["cost_usd"],
                rec["latency_ms"],
                int(rec["cache_hit"]),
            ),
        )
        return "inserted", 1
    except sqlite3.IntegrityError:
        return "duplicate", 0


def run(input_path: str, db_path: str = "data/imprint.db") -> dict[str, Any]:
    conn = connect(db_path)
    stats = {"inserted": 0, "duplicate": 0, "skipped": 0}
    with open(input_path) as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats["skipped"] += 1
                continue
            kind, n = ingest_record(conn, record)
            stats[kind] += n
    conn.commit()
    print(
        f"collected {stats['inserted']} new pairs "
        f"({stats['duplicate']} duplicates, {stats['skipped']} skipped) -> {db_path}"
    )
    return stats


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "/dev/stdin")
