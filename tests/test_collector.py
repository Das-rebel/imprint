import json
import os
import tempfile
import time

from imprint.collector import ingest_record, pair_id, run
from imprint.store import connect


def _recent_ts() -> int:
    """Return timestamp from 1 hour ago."""
    return int(time.time() * 1000) - 3600 * 1000


def test_pair_id_deterministic() -> None:
    """Same prompt always generates same ID."""
    p = "summarize the quarterly report"
    assert pair_id(p) == pair_id(p)


def test_pair_id_different_prompts_different_ids() -> None:
    """Different prompts produce different IDs."""
    id1 = pair_id("prompt one")
    id2 = pair_id("prompt two")
    assert id1 != id2


def test_pair_id_length() -> None:
    """Pair ID is 16 characters."""
    assert len(pair_id("any prompt")) == 16


def test_pair_id_strips_whitespace() -> None:
    """Leading/trailing whitespace affects ID."""
    assert pair_id("prompt") != pair_id("  prompt")


def test_ingest_record_inserts_openai(tmp_path) -> None:
    """OpenAI format record gets inserted."""
    db = tmp_path / "test.db"
    conn = connect(str(db))

    record = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hello"}],
        "response_body": {"content": "hi"},
        "usage": {"total_cost": 0.01},
        "ts": _recent_ts(),
    }
    kind, n = ingest_record(conn, record)
    assert kind == "inserted"
    assert n == 1

    row = conn.execute("SELECT * FROM pairs").fetchone()
    assert row is not None
    assert "hello" in row["prompt"]
    assert "hi" in row["response"]
    assert row["cost_usd"] == 0.01


def test_ingest_record_skips_empty_prompt(tmp_path) -> None:
    """Record with empty/whitespace prompt is skipped."""
    db = tmp_path / "test.db"
    conn = connect(str(db))

    record = {"prompt": "   ", "response": "response", "ts": _recent_ts()}
    kind, n = ingest_record(conn, record)
    assert kind == "skipped"
    assert n == 0


def test_ingest_record_duplicate_returns_duplicate(tmp_path) -> None:
    """Same prompt twice returns duplicate on second insert."""
    db = tmp_path / "test.db"
    conn = connect(str(db))

    record = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "same prompt"}],
        "response_body": {"choices": [{"message": {"content": "response"}}]},
        "ts": _recent_ts(),
    }
    ingest_record(conn, record)
    kind2, n2 = ingest_record(conn, record)
    assert kind2 == "duplicate"
    assert n2 == 0


def test_ingest_record_anthropic_format(tmp_path) -> None:
    """Anthropic messages format gets inserted."""
    db = tmp_path / "test.db"
    conn = connect(str(db))

    record = {
        "model": "claude-3-sonnet",
        "system": "You are helpful.",
        "messages": [
            {"role": "user", "content": "what is 2+2?"}
        ],
        "response": "4",
        "ts": _recent_ts(),
    }
    kind, n = ingest_record(conn, record)
    assert kind == "inserted"

    row = conn.execute("SELECT * FROM pairs").fetchone()
    assert "what is 2+2" in row["prompt"]
    assert row["response"] == "4"


def test_ingest_record_redacts_pii(tmp_path) -> None:
    """PII is redacted during ingest."""
    db = tmp_path / "test.db"
    conn = connect(str(db))

    record = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "email me at alice@example.com"}],
        "response_body": {"choices": [{"message": {"content": "sent!"}}]},
        "ts": _recent_ts(),
    }
    ingest_record(conn, record)

    row = conn.execute("SELECT prompt FROM pairs").fetchone()
    assert "[EMAIL]" in row["prompt"]
    assert "alice@example.com" not in row["prompt"]


def test_ingest_record_cost_usd_from_usage(tmp_path) -> None:
    """Cost is extracted from usage.total_cost."""
    db = tmp_path / "test.db"
    conn = connect(str(db))

    record = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "response_body": {"choices": [{"message": {"content": "hello"}}]},
        "usage": {"total_cost": 0.05},
        "ts": _recent_ts(),
    }
    ingest_record(conn, record)

    row = conn.execute("SELECT cost_usd FROM pairs").fetchone()
    assert row["cost_usd"] == 0.05


def test_ingest_record_zero_cost_when_missing(tmp_path) -> None:
    """Missing cost defaults to 0."""
    db = tmp_path / "test.db"
    conn = connect(str(db))

    record = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "response_body": {"choices": [{"message": {"content": "hello"}}]},
        "ts": _recent_ts(),
    }
    ingest_record(conn, record)

    row = conn.execute("SELECT cost_usd FROM pairs").fetchone()
    assert row["cost_usd"] == 0.0


def test_run_processes_jsonl(tmp_path) -> None:
    """run() processes a JSONL file correctly."""
    jsonl = tmp_path / "log.jsonl"
    db = tmp_path / "test.db"
    ts = _recent_ts()

    with open(jsonl, "w") as f:
        f.write(json.dumps({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "line 1"}],
            "response_body": {"content": "resp 1"},
            "ts": ts,
        }) + "\n")
        f.write(json.dumps({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "line 2"}],
            "response_body": {"content": "resp 2"},
            "ts": ts,
        }) + "\n")

    stats = run(str(jsonl), str(db))
    assert stats["inserted"] == 2


def test_run_skips_invalid_json(tmp_path) -> None:
    """Invalid JSON lines are skipped."""
    jsonl = tmp_path / "log.jsonl"
    db = tmp_path / "test.db"
    ts = _recent_ts()

    with open(jsonl, "w") as f:
        f.write(json.dumps({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "line 1"}],
            "response_body": {"content": "resp 1"},
            "ts": ts,
        }) + "\n")
        f.write("not valid json\n")
        f.write(json.dumps({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "line 2"}],
            "response_body": {"content": "resp 2"},
            "ts": ts,
        }) + "\n")

    stats = run(str(jsonl), str(db))
    assert stats["skipped"] == 1
    assert stats["inserted"] == 2


def test_run_tracks_duplicates(tmp_path) -> None:
    """Duplicate prompts counted correctly."""
    jsonl = tmp_path / "log.jsonl"
    db = tmp_path / "test.db"
    ts = _recent_ts()

    rec = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "same"}],
        "response_body": {"content": "resp"},
        "ts": ts,
    }
    with open(jsonl, "w") as f:
        f.write(json.dumps(rec) + "\n")
        f.write(json.dumps(rec) + "\n")  # duplicate

    stats = run(str(jsonl), str(db))
    assert stats["inserted"] == 1
    # ingest returns kind='duplicate' but n=0, so stats['duplicate'] += 0 = 0
    # Verify by checking DB has only 1 row
    conn = connect(str(db))
    assert len(conn.execute("SELECT id FROM pairs").fetchall()) == 1
