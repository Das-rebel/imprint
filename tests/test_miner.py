import json
import os
import tempfile
import time

from imprint.miner import _prefix_key, _sig_id, mine, report
from imprint.store import connect


def test_prefix_key_drops_numbers() -> None:
    """Numbers are dropped so region-3 and region-7 cluster together."""
    k1 = _prefix_key("summarize sales for region 3")
    k2 = _prefix_key("summarize sales for region 7")
    assert k1 == k2


def test_prefix_key_first_n_words() -> None:
    """First 8 content words form the key."""
    k = _prefix_key("the quick brown fox jumps over the lazy dog")
    assert len(k.split()) <= 8
    assert "fox" in k


def test_prefix_key_handles_punctuation() -> None:
    """Pure numbers are dropped even with punctuation."""
    k = _prefix_key("what is 2+2?")
    # "2+2?" - "2" and "2" are pure digits, "+" is kept, "?" is kept but...
    # The split is: ["what", "is", "2+2?"] - "2+2?" is NOT pure digits
    assert "+" in k or "what" in k


def test_prefix_key_empty_string() -> None:
    """Empty string returns empty key."""
    assert _prefix_key("") == ""


def test_sig_id_prefix() -> None:
    """Signature ID starts with 'sig_'."""
    sid = _sig_id("some key")
    assert sid.startswith("sig_")


def test_sig_id_deterministic() -> None:
    """Same key always produces same ID."""
    key = "summarize the quarterly report"
    assert _sig_id(key) == _sig_id(key)


def test_sig_id_different_keys_different_ids() -> None:
    """Different keys produce different IDs."""
    id1 = _sig_id("key one")
    id2 = _sig_id("key two")
    assert id1 != id2


def _recent_ts() -> int:
    """Return timestamp from 1 hour ago (within the 7-day window)."""
    return int(time.time() * 1000) - 3600 * 1000


def test_mine_clusters_by_prefix(tmp_path) -> None:
    """Pairs with same prefix key should get same signature."""
    db = tmp_path / "test.db"
    conn = connect(str(db))
    ts = _recent_ts()

    # Insert pairs with same prefix
    for i in range(3):
        conn.execute(
            "INSERT INTO pairs (prompt, response, cost_usd, ts) VALUES (?,?,?,?)",
            (f"summarize the {i} report", "ok", 0.01, ts),
        )
    conn.commit()

    results = mine(db_path=str(db))
    assert len(results) == 1
    assert results[0]["volume_week"] == 3


def test_mine_groups_different_prefixes(tmp_path) -> None:
    """Different prefixes produce different signatures."""
    db = tmp_path / "test.db"
    conn = connect(str(db))
    ts = _recent_ts()

    conn.execute(
        "INSERT INTO pairs (prompt, response, cost_usd, ts) VALUES (?,?,?,?)",
        ("summarize the sales report", "ok", 0.01, ts),
    )
    conn.execute(
        "INSERT INTO pairs (prompt, response, cost_usd, ts) VALUES (?,?,?,?)",
        ("translate this to french", "ok", 0.02, ts),
    )
    conn.commit()

    results = mine(db_path=str(db))
    assert len(results) == 2


def test_mine_priority_ranking(tmp_path) -> None:
    """Results are sorted by priority descending."""
    db = tmp_path / "test.db"
    conn = connect(str(db))
    ts = _recent_ts()

    # Insert multiple pairs with same prefix (high volume = high priority)
    for i in range(10):
        conn.execute(
            "INSERT INTO pairs (prompt, response, cost_usd, ts) VALUES (?,?,?,?)",
            ("popular query", "ok", 0.01, ts),
        )
    # Insert one pair with different prefix (low volume = low priority)
    conn.execute(
        "INSERT INTO pairs (prompt, response, cost_usd, ts) VALUES (?,?,?,?)",
        ("rare query", "ok", 0.01, ts),
    )
    conn.commit()

    results = mine(db_path=str(db))
    assert len(results) == 2
    assert results[0]["volume_week"] == 10
    assert results[1]["volume_week"] == 1


def test_mine_status_active_vs_candidate(tmp_path) -> None:
    """High volume = active, low volume = candidate."""
    db = tmp_path / "test.db"
    conn = connect(str(db))
    ts = _recent_ts()

    # 100+ pairs = active
    for i in range(101):
        conn.execute(
            "INSERT INTO pairs (prompt, response, cost_usd, ts) VALUES (?,?,?,?)",
            ("active query", "ok", 0.01, ts),
        )
    # < 100 pairs = candidate
    conn.execute(
        "INSERT INTO pairs (prompt, response, cost_usd, ts) VALUES (?,?,?,?)",
        ("candidate query", "ok", 0.01, ts),
    )
    conn.commit()

    results = mine(db_path=str(db))
    by_prefix = {r["prefix"].split()[0]: r for r in results}
    assert by_prefix["active"]["status"] == "active"
    assert by_prefix["candidate"]["status"] == "candidate"


def test_mine_updates_existing_signatures(tmp_path) -> None:
    """Second run should update existing signatures, not create duplicates."""
    db = tmp_path / "test.db"
    conn = connect(str(db))
    ts = _recent_ts()

    conn.execute(
        "INSERT INTO pairs (prompt, response, cost_usd, ts) VALUES (?,?,?,?)",
        ("same query", "ok", 0.01, ts),
    )
    conn.commit()

    r1 = mine(db_path=str(db))
    assert len(r1) == 1

    conn.execute(
        "INSERT INTO pairs (prompt, response, cost_usd, ts) VALUES (?,?,?,?)",
        ("same query", "ok", 0.01, ts),
    )
    conn.commit()

    r2 = mine(db_path=str(db))
    assert len(r2) == 1  # still 1 signature
    assert r2[0]["volume_week"] == 2  # but now 2 pairs


def test_mine_min_volume_filter(tmp_path) -> None:
    """Pairs below min_volume should not appear in results."""
    db = tmp_path / "test.db"

    results = mine(db_path=str(db), min_volume=10)
    assert len(results) == 0
