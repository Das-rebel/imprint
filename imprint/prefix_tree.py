"""Prefix Tree: RadixAttention-style O(1) prefix lookup with KV reuse.

Unlike semantic cache (similarity search), prefix tree matches exact prefix
patterns. This enables:
1. O(1) prefix lookup vs O(n) linear scan
2. Shared KV cache for common prefixes
3. Automatic prefix extraction and merging

Architecture (inspired by SGLang's RadixAttention):
    Request → Trie lookup (longest prefix) → KV cache hit? → Serve / Compute

Related:
    - imprint/semantic_cache.py: Similarity-based cache
    - imprint/compressor.py: Token reduction
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any


@dataclass
class TrieNode:
    """A single node in the compressed trie."""

    children: dict[str, TrieNode] = field(default_factory=dict)
    is_end: bool = False
    value: Optional[dict] = None  # {response, metadata, depth}
    prefix: str = ""  # The string fragment this node represents


class PrefixTree:
    """Radix-style compressed trie for O(1) prefix lookups.

    Uses path compression: a sequence of single-child nodes is collapsed
    into one node with a combined prefix string. This reduces memory and
    speeds up traversal.

    Usage:
        tree = PrefixTree()
        tree.insert("summarize the", {"response": "...", "tokens": 50})
        tree.insert("summarize the report", {"response": "...", "tokens": 120})

        result = tree.lookup("summarize the quarterly report")
        # Returns value for "summarize the" (longest matching prefix)

        prefixes = tree.get_prefixes("summarize the meeting notes")
        # Returns [("summarize the", ...), ("summarize the report", ...)]
    """

    def __init__(self):
        self._root = TrieNode()
        self._stats = {"inserts": 0, "lookups": 0, "hits": 0, "misses": 0}
        self._path_stats = defaultdict(int)  # prefix → hit count

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def insert(self, text: str, value: dict) -> str:
        """Insert text with associated value.

        The value dict should contain:
            - response: str — the cached response
            - metadata: dict — any additional data (signature_id, tokens, etc.)

        Returns:
            The key (sha256 of text)[:16]
        """
        self._stats["inserts"] += 1
        key = self._make_key(text)

        node = self._root
        i = 0
        while i < len(text):
            char = text[i]

            if char not in node.children:
                # No child matching this char — create new leaf path
                remaining = text[i:]
                node.children[char] = TrieNode(prefix=remaining, is_end=True, value=value)
                break

            child = node.children[char]
            common_len = self._common_prefix_len(child.prefix, text[i:])

            if common_len == len(child.prefix):
                # Full match of child prefix, descend
                i += common_len
                node = child
            else:
                # Partial match — need to split this node
                # child.prefix = common_prefix + remainder
                # Create split node
                split_node = TrieNode(prefix=child.prefix[:common_len])
                remainder_node = TrieNode(
                    prefix=child.prefix[common_len:],
                    children=child.children,
                    is_end=child.is_end,
                    value=child.value,
                )
                child.prefix = split_node.prefix
                child.children = {remainder_node.prefix[0]: remainder_node}
                child.is_end = False
                child.value = None

                if i + common_len == len(text):
                    # We're at the end — make split node the end
                    split_node.is_end = True
                    split_node.value = value
                    node.children[char] = split_node
                else:
                    # Continue inserting remainder
                    remainder = text[i + common_len :]
                    new_leaf = TrieNode(prefix=remainder, is_end=True, value=value)
                    split_node.children[remainder[0]] = new_leaf
                    node.children[char] = split_node
                break
        else:
            # Fell through — we're at the end of an existing path
            node.is_end = True
            node.value = value

        return key

    def lookup(self, text: str) -> Optional[dict]:
        """Find the value for the longest matching prefix.

        Returns:
            The value dict associated with the longest prefix match,
            or None if no prefix matches.
        """
        self._stats["lookups"] += 1

        node = self._root
        i = 0
        last_value = None
        last_end = False

        while i < len(text):
            char = text[i]

            if char not in node.children:
                break

            child = node.children[char]

            if child.prefix.startswith(char):
                # Check if text matches this node's prefix
                remaining_text = text[i : i + len(child.prefix)]
                if remaining_text == child.prefix[: len(remaining_text)]:
                    i += len(child.prefix)
                    if child.is_end:
                        last_value = child.value
                        last_end = True
                    node = child
                    continue
                elif child.prefix.startswith(text[i:]):
                    # Text ends mid-prefix — no complete match
                    break
                else:
                    # Partial prefix match at this node
                    match_len = self._common_prefix_len(child.prefix, text[i:])
                    if match_len > 0 and child.is_end:
                        last_value = child.value
                        last_end = False
                    break

        if last_value is not None:
            self._stats["hits"] += 1
            if last_value.get("response"):
                self._path_stats[last_value.get("signature_id", "_default")] += 1
            return last_value

        self._stats["misses"] += 1
        return None

    def get_prefixes(self, text: str) -> list[tuple[str, dict]]:
        """Get all matching prefixes for text, longest first.

        Returns:
            List of (prefix_string, value_dict) tuples, longest-first
        """
        results = []
        node = self._root
        i = 0
        current_prefix = ""

        while i < len(text):
            char = text[i]

            if char not in node.children:
                break

            child = node.children[char]
            remaining_text = text[i:]

            if child.prefix.startswith(char):
                match_len = self._common_prefix_len(child.prefix, remaining_text)
                if match_len > 0:
                    current_prefix += child.prefix[:match_len]
                    if child.is_end:
                        results.append((current_prefix, child.value))
                    if match_len == len(child.prefix):
                        i += len(child.prefix)
                        node = child
                    else:
                        break
                else:
                    break
            else:
                break

        # Sort by prefix length (longest first)
        results.sort(key=lambda x: len(x[0]), reverse=True)
        return results

    def get_common_prefix(self, text1: str, text2: str) -> str:
        """Find common prefix between two strings."""
        return text1[: self._common_prefix_len(text1, text2)]

    def save(self, path: str) -> None:
        """Serialize the tree to a JSON file."""
        obj = {
            "stats": dict(self._stats),
            "path_stats": dict(self._path_stats),
            "tree": self._node_to_dict(self._root),
        }
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)

    def load(self, path: str) -> None:
        """Deserialize the tree from a JSON file."""
        with open(path) as f:
            obj = json.load(f)
        self._stats = dict(obj.get("stats", {}))
        self._path_stats = defaultdict(int, obj.get("path_stats", {}))
        self._root = self._dict_to_node(obj.get("tree", {}))

    def stats(self) -> dict:
        """Return tree statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        return {
            "inserts": self._stats["inserts"],
            "lookups": self._stats["lookups"],
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate": self._stats["hits"] / total if total > 0 else 0.0,
            "nodes": self._count_nodes(self._root),
            "max_depth": self._max_depth(self._root),
        }

    def clear(self) -> int:
        """Clear all entries. Returns count of cleared entries."""
        count = self._stats["inserts"]
        self._root = TrieNode()
        self._stats = {"inserts": 0, "lookups": 0, "hits": 0, "misses": 0}
        self._path_stats.clear()
        return count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _make_key(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    @staticmethod
    def _common_prefix_len(s1: str, s2: str) -> int:
        """Length of common prefix between two strings."""
        i = 0
        while i < len(s1) and i < len(s2) and s1[i] == s2[i]:
            i += 1
        return i

    def _node_to_dict(self, node: TrieNode) -> dict:
        """Serialize a trie node to dict."""
        return {
            "children": {k: self._node_to_dict(v) for k, v in node.children.items()},
            "is_end": node.is_end,
            "value": node.value,
            "prefix": node.prefix,
        }

    def _dict_to_node(self, d: dict) -> TrieNode:
        """Deserialize a dict to TrieNode."""
        return TrieNode(
            children={k: self._dict_to_node(v) for k, v in d.get("children", {}).items()},
            is_end=d.get("is_end", False),
            value=d.get("value"),
            prefix=d.get("prefix", ""),
        )

    def _count_nodes(self, node: TrieNode) -> int:
        """Count total nodes in tree."""
        return 1 + sum(self._count_nodes(v) for v in node.children.values())

    def _max_depth(self, node: TrieNode, depth: int = 0) -> int:
        """Find maximum depth of tree."""
        if not node.children:
            return depth
        return max(self._max_depth(v, depth + 1) for v in node.children.values())
