"""Tests for semantic cache.

Lightweight tests that work without sentence-transformers installed.
"""
import json
import tempfile
from pathlib import Path

import pytest


def test_redact_pii():
    """Test PII redaction in semantic cache."""
    from imprint.semantic_cache import _redact

    text = "Send to user@example.com card 4111-1111-1111-1111 key sk-abc123def456ghi789jkl"
    redacted = _redact(text)
    assert "[EMAIL]" in redacted
    assert "[CARD]" in redacted
    assert "[KEY]" in redacted
    assert "user@example.com" not in redacted
    assert "4111-1111-1111-1111" not in redacted


def test_cache_entry_serialization():
    """Test CacheEntry to_dict/from_dict roundtrip."""
    from imprint.semantic_cache import CacheEntry

    entry = CacheEntry(
        prompt="test prompt",
        response="test response",
        signature_id="sig-1",
        embedding=[0.1, 0.2, 0.3],
        ttl=3600,
    )
    d = entry.to_dict()
    assert d["prompt"] == "test prompt"
    assert d["signature_id"] == "sig-1"
    assert d["embedding"] == [0.1, 0.2, 0.3]
    assert d["ttl"] == 3600

    restored = CacheEntry.from_dict(d)
    assert restored.prompt == entry.prompt
    assert restored.response == entry.response
    assert restored.embedding == entry.embedding


def test_cache_entry_expiration():
    """Test TTL expiration logic."""
    from imprint.semantic_cache import CacheEntry
    import time

    # Very short TTL
    entry = CacheEntry(
        prompt="test",
        response="resp",
        signature_id=None,
        embedding=[],
        ttl=0.01,  # 10ms
    )
    assert not entry.is_expired()
    time.sleep(0.05)
    assert entry.is_expired()


def test_cache_stats_initial():
    """Test initial stats are zero."""
    with tempfile.TemporaryDirectory() as tmpdir:
        from imprint.semantic_cache import SemanticCache

        cache = SemanticCache(tmpdir)
        stats = cache.stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0
        assert stats["entries"] == 0


def test_cache_clear():
    """Test clear() removes all entries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        from imprint.semantic_cache import SemanticCache

        cache = SemanticCache(tmpdir)
        # Manually add an entry to avoid model download
        from imprint.semantic_cache import CacheEntry
        cache._entries["test_key"] = CacheEntry(
            prompt="test",
            response="resp",
            signature_id=None,
            embedding=[],
        )
        assert cache.stats()["entries"] == 1

        count = cache.clear()
        assert count == 1
        assert cache.stats()["entries"] == 0


def test_cache_key_consistency():
    """Test that same text produces same key."""
    from imprint.semantic_cache import SemanticCache

    key1 = SemanticCache._make_key("test prompt")
    key2 = SemanticCache._make_key("test prompt")
    assert key1 == key2
    assert len(key1) == 16

    key3 = SemanticCache._make_key("different prompt")
    assert key1 != key3
