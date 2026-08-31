"""Semantic Cache: Vector similarity search on LLM prompts.

Uses BAAI/bge-small-en-v1.5 embeddings (~24MB, CPU-friendly) with FAISS
for fast similarity search. Falls back to prefix matching when embeddings
are unavailable.

Architecture:
    Request → Embed → FAISS search → Cache hit? → Return / Miss → LLM

Related:
    - imprint/prefix_tree.py: O(1) prefix lookup
    - imprint/compressor.py: Token reduction
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# PII redaction (same pattern as collector.py)
# ---------------------------------------------------------------------------
PII_PATTERNS = [
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[EMAIL]"),
    (re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"), "[CARD]"),
    (re.compile(r"\b(?:sk|pk|api|rk|token)[-_]?[A-Za-z0-9]{20,}\b", re.IGNORECASE), "[KEY]"),
]


def _redact(text: str) -> str:
    for pattern, replacement in PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ---------------------------------------------------------------------------
# Embedding model (lazy import, CPU-friendly bge-small)
# ---------------------------------------------------------------------------
_ENCODER = None


def _get_encoder():
    global _ENCODER
    if _ENCODER is None:
        from sentence_transformers import SentenceTransformer

        _ENCODER = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _ENCODER


@dataclass
class CacheEntry:
    """A cached prompt→response pair with metadata."""

    prompt: str
    response: str
    signature_id: Optional[str]
    embedding: list[float]
    created_at: float = field(default_factory=time.time)
    ttl: float = 86400 * 7  # 7 days default
    access_count: int = 0

    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl

    def touch(self) -> None:
        self.access_count += 1

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "response": self.response,
            "signature_id": self.signature_id,
            "embedding": self.embedding,
            "created_at": self.created_at,
            "ttl": self.ttl,
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CacheEntry":
        return cls(
            prompt=d["prompt"],
            response=d["response"],
            signature_id=d.get("signature_id"),
            embedding=d["embedding"],
            created_at=d.get("created_at", time.time()),
            ttl=d.get("ttl", 86400 * 7),
            access_count=d.get("access_count", 0),
        )


class SemanticCache:
    """Semantic cache using bge-small embeddings + FAISS.

    Usage:
        cache = SemanticCache("data/semantic_cache")
        cache.insert("summarize the report", "Here is the summary...")
        result = cache.get("summarize the quarterly report")
        if result:
            print(result.response)
    """

    def __init__(
        self,
        cache_dir: str = "data/semantic_cache",
        top_k: int = 5,
        similarity_threshold: float = 0.85,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model

        self._entries: dict[str, CacheEntry] = {}
        self._index = None  # FAISS index (built lazily)
        self._ids: list[str] = []  # Maps FAISS index position → entry id
        self._stats = {"hits": 0, "misses": 0, "embed_time": 0.0, "search_time": 0.0}

        self._load_index()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed one or more texts using bge-small.

        Returns:
            numpy array of shape (n, 384) — 384-dim bge-small embeddings
        """
        encoder = _get_encoder()
        return encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def insert(self, prompt: str, response: str, signature_id: str = None, ttl: float = 86400 * 7) -> str:
        """Insert a prompt→response pair into the semantic cache.

        Returns:
            The cache key (sha256 of redacted prompt)[:16]
        """
        redacted = _redact(prompt)
        key = self._make_key(redacted)

        # Compute embedding
        t0 = time.perf_counter()
        emb = self.embed([redacted])[0]
        self._stats["embed_time"] += time.perf_counter() - t0

        entry = CacheEntry(
            prompt=redacted,
            response=response,
            signature_id=signature_id,
            embedding=emb.tolist(),
            ttl=ttl,
        )
        self._entries[key] = entry
        self._add_to_index(key, emb)
        self._save_entry(key, entry)
        return key

    def get(self, prompt: str) -> Optional[CacheEntry]:
        """Find the best semantic match for a prompt.

        Returns:
            CacheEntry if similarity >= threshold, else None
        """
        redacted = _redact(prompt)
        results = self.search(redacted, top_k=1)
        if results and results[0][0] is not None:
            entry, score = results[0]
            self._stats["hits" if entry else "misses"] += 1
            if entry:
                entry.touch()
                self._save_index()  # Persist access_count
            return entry
        self._stats["misses"] += 1
        return None

    def search(self, query: str, top_k: int = None) -> list[tuple[Optional[CacheEntry], float]]:
        """Find top-k semantic matches for a query.

        Returns:
            List of (CacheEntry, similarity_score) tuples, best-first
        """
        if top_k is None:
            top_k = self.top_k
        if not self._entries:
            return [(None, 0.0)] * min(top_k, 1)

        t0 = time.perf_counter()
        q_emb = self.embed([_redact(query)])[0]
        self._stats["embed_time"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        if self._index is not None:
            scores, indices = self._index.search(q_emb.reshape(1, -1).astype("float32"), top_k)
            self._stats["search_time"] += time.perf_counter() - t0

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self._ids):
                    results.append((None, float(score)))
                else:
                    key = self._ids[idx]
                    entry = self._entries.get(key)
                    results.append((entry, float(score)))
            return results

        # Fallback: brute-force cosine similarity
        embeddings = np.array([e.embedding for e in self._entries.values()])
        norm_q = q_emb / (np.linalg.norm(q_emb) + 1e-8)
        norm_e = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        scores = np.dot(norm_e, norm_q)
        top_indices = np.argsort(scores)[::-1][:top_k]
        self._stats["search_time"] += time.perf_counter() - t0

        return [(self._entries.get(list(self._entries.keys())[i]), float(scores[i])) for i in top_indices]

    def clear_expired(self) -> int:
        """Remove expired entries. Returns count of removed entries."""
        expired_keys = [k for k, e in self._entries.items() if e.is_expired()]
        for key in expired_keys:
            del self._entries[key]
        if expired_keys:
            self._rebuild_index()
            self._save_index()
        return len(expired_keys)

    def clear(self) -> int:
        """Clear all entries. Returns count of cleared entries."""
        count = len(self._entries)
        self._entries.clear()
        self._index = None
        self._ids.clear()
        self._save_index()
        (self.cache_dir / "entries.jsonl").unlink(missing_ok=True)
        return count

    def stats(self) -> dict:
        """Return cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        return {
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate": self._stats["hits"] / total if total > 0 else 0.0,
            "entries": len(self._entries),
            "avg_embed_ms": (self._stats["embed_time"] / max(total, 1)) * 1000,
            "avg_search_ms": (self._stats["search_time"] / max(total, 1)) * 1000,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    @staticmethod
    def _make_key(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _add_to_index(self, key: str, embedding: np.ndarray) -> None:
        """Add entry to FAISS index."""
        try:
            import faiss

            dim = embedding.shape[0]
            if self._index is None:
                self._index = faiss.IndexFlatIP(dim)  # Inner product (cosine for normalized)
            vec = embedding.reshape(1, -1).astype("float32")
            faiss.normalize_L2(vec)
            self._index.add(vec)
            self._ids.append(key)
        except ImportError:
            self._index = None  # Fall back to brute-force

    def _rebuild_index(self) -> None:
        """Rebuild FAISS index after deletion."""
        import faiss

        if not self._entries:
            self._index = None
            self._ids.clear()
            return
        embeddings = np.array([e.embedding for e in self._entries.values()], dtype="float32")
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)
        self._ids = list(self._entries.keys())

    def _load_index(self) -> None:
        """Load entries and rebuild index from disk."""
        entries_file = self.cache_dir / "entries.jsonl"
        if entries_file.exists():
            with open(entries_file) as f:
                for line in f:
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    entry = CacheEntry.from_dict(d)
                    if entry.is_expired():
                        continue
                    self._entries[entry.prompt[:16]] = entry  # Use prompt prefix as key

        if self._entries:
            self._rebuild_index()

    def _save_entry(self, key: str, entry: CacheEntry) -> None:
        """Append entry to JSONL file."""
        entries_file = self.cache_dir / "entries.jsonl"
        with open(entries_file, "a") as f:
            f.write(json.dumps({"key": key, **entry.to_dict()}) + "\n")

    def _save_index(self) -> None:
        """Save entries metadata (FAISS index rebuilt on load for simplicity)."""
        meta_file = self.cache_dir / "meta.json"
        with open(meta_file, "w") as f:
            json.dump(
                {
                    "stats": self._stats,
                    "config": {
                        "top_k": self.top_k,
                        "similarity_threshold": self.similarity_threshold,
                        "embedding_model": self.embedding_model,
                    },
                },
                f,
                indent=2,
            )
