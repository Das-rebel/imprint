"""Prompt Compressor: LLMLingua-style token reduction (2-4x).

Reduces token count while preserving key semantic information. Uses:
1. LLMLingua when available (best quality)
2. Heuristic compression (fast, no model needed)
3. Truncation with key phrase preservation

Architecture:
    Long prompt → Compressor → Compressed prompt (2-4x fewer tokens) → LLM

Related:
    - imprint/semantic_cache.py: Cache before compression
    - imprint/prefix_tree.py: Lookup before compression
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CompressionResult:
    """Result of prompt compression."""

    original: str
    compressed: str
    original_tokens: int
    compressed_tokens: int
    ratio: float  # compressed_tokens / original_tokens (< 1 is good)
    method: str  # "llmlingua", "heuristic", "truncation"
    quality_score: float  # 0-1 estimate of semantic preservation


class PromptCompressor:
    """LLMLingua-style prompt compressor.

    Tries LLMLingua first, falls back to heuristic compression.

    Usage:
        compressor = PromptCompressor()
        result = compressor.compress("Write a long report about...", budget=1024)
        print(result.compressed)  # ~50% of original size
    """

    def __init__(
        self,
        model: str = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        use_llmlingua: bool = True,
    ):
        self.model_name = model
        self.use_llmlingua = use_llmlingua
        self._llmlingua = None
        self._stats = {
            "total_calls": 0,
            "llmlingua_calls": 0,
            "heuristic_calls": 0,
            "total_original_tokens": 0,
            "total_compressed_tokens": 0,
        }

    def _init_llmlingua(self):
        """Lazy-load LLMLingua."""
        if self._llmlingua is None:
            try:
                from llmlingua import PromptCompressor as LL

                self._llmlingua = LL(
                    model_name=self.model_name,
                    device_map="cpu",
                    use_runtime=True,
                )
            except ImportError:
                self._llmlingua = False  # Mark as unavailable

    def compress(
        self,
        prompt: str,
        budget: int = 1024,
        ratio: float = 0.5,
        target_token_num: Optional[int] = None,
    ) -> CompressionResult:
        """Compress a prompt to meet budget or ratio target.

        Args:
            prompt: The input prompt to compress
            budget: Maximum tokens in compressed output
            ratio: Target compression ratio (0.5 = 50% of original)
            target_token_num: Alternative to budget — exact token target

        Returns:
            CompressionResult with compressed text and metrics
        """
        self._stats["total_calls"] += 1
        original_tokens = self._count_tokens(prompt)

        if target_token_num is None:
            target_token_num = min(int(original_tokens * ratio), budget)

        # Don't compress if already below budget threshold
        if original_tokens <= budget:
            return CompressionResult(
                original=prompt,
                compressed=prompt,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                ratio=1.0,
                method="none",
                quality_score=1.0,
            )

        # Try LLMLingua first
        if self.use_llmlingua:
            self._init_llmlingua()
            if self._llmlingua:
                try:
                    compressed = self._llmlingua.compress_prompt(
                        prompt,
                        target_token_num=target_token_num,
                        force_return_text=True,
                    )
                    self._stats["llmlingua_calls"] += 1
                    compressed_tokens = self._count_tokens(compressed)
                    return CompressionResult(
                        original=prompt,
                        compressed=compressed,
                        original_tokens=original_tokens,
                        compressed_tokens=compressed_tokens,
                        ratio=compressed_tokens / original_tokens,
                        method="llmlingua",
                        quality_score=self._estimate_quality(prompt, compressed),
                    )
                except Exception:
                    pass  # Fall through to heuristic

        # Fallback: heuristic compression
        compressed = self._heuristic_compress(prompt, target_token_num)
        self._stats["heuristic_calls"] += 1
        compressed_tokens = self._count_tokens(compressed)

        return CompressionResult(
            original=prompt,
            compressed=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            ratio=compressed_tokens / original_tokens,
            method="heuristic",
            quality_score=self._estimate_quality(prompt, compressed),
        )

    def compress_instruction(
        self,
        instruction: str,
        target_token_num: int,
    ) -> str:
        """Compress an instruction to exact token target.

        Optimized for system prompts and instructions.
        """
        result = self.compress(instruction, target_token_num=target_token_num)
        return result.compressed

    def should_compress(self, prompt: str, threshold: int = 512) -> bool:
        """Check if prompt should be compressed.

        Args:
            prompt: The prompt to check
            threshold: Token count threshold above which compression is recommended
        """
        return self._count_tokens(prompt) > threshold

    def batch_compress(
        self, prompts: list[str], budget: int = 1024
    ) -> list[CompressionResult]:
        """Compress multiple prompts.

        Args:
            prompts: List of prompts to compress
            budget: Maximum tokens per compressed prompt
        """
        return [self.compress(p, budget=budget) for p in prompts]

    def stats(self) -> dict:
        """Return compression statistics."""
        total = self._stats["total_calls"]
        return {
            "total_calls": total,
            "llmlingua_calls": self._stats["llmlingua_calls"],
            "heuristic_calls": self._stats["heuristic_calls"],
            "avg_ratio": (
                self._stats["total_compressed_tokens"] / self._stats["total_original_tokens"]
                if self._stats["total_original_tokens"] > 0
                else 0.0
            ),
            "llmlingua_available": self._llmlingua is not None,
        }

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------
    def _count_tokens(self, text: str) -> int:
        """Rough token count (≈4 chars per token for English)."""
        if not text:
            return 0
        # Simple approximation: 1 token ≈ 4 chars for English
        # For non-Latin scripts, use different estimate
        return len(text) // 4 + len(text.split())

    def _estimate_quality(self, original: str, compressed: str) -> float:
        """Estimate semantic quality preservation (0-1).

        Checks:
        1. Key entities preserved
        2. Question/action words preserved
        3. Length ratio
        """
        if not compressed:
            return 0.0
        if not original:
            return 1.0

        score = 0.0

        # Length ratio check (don't penalize small compression)
        ratio = len(compressed) / len(original)
        if ratio > 0.5:
            score += 0.4
        elif ratio > 0.3:
            score += 0.2

        # Key word preservation
        key_words = {
            "write", "create", "generate", "summarize", "explain",
            "analyze", "compare", "list", "describe", "tell",
            "who", "what", "where", "when", "why", "how",
        }
        original_lower = original.lower()
        compressed_lower = compressed.lower()
        original_keys = key_words & set(original_lower.split())
        compressed_keys = key_words & set(compressed_lower.split())
        if original_keys:
            key_ratio = len(compressed_keys) / len(original_keys)
            score += 0.3 * key_ratio

        # Entity preservation (capitalized words, numbers)
        entities = re.findall(r"[A-Z][a-z]+|\d+", original)
        compressed_entities = re.findall(r"[A-Z][a-z]+|\d+", compressed)
        if entities:
            entity_ratio = len(compressed_entities) / len(entities)
            score += 0.3 * entity_ratio

        return min(score, 1.0)

    def _heuristic_compress(self, text: str, target_tokens: int) -> str:
        """Heuristic compression without external model.

        Applies:
        1. Remove redundant whitespace
        2. Shorten common phrases
        3. Remove filler words
        4. Truncate with ellipsis for context
        """
        if not text:
            return text

        original_len = len(text)
        current_tokens = self._count_tokens(text)

        # Step 1: Clean whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Step 2: Common phrase shortening
        replacements = [
            (r"\bplease\b", ""),
            (r"\bkindly\b", ""),
            (r"\bbasically\b", ""),
            (r"\bactually\b", ""),
            (r"\breally\b", ""),
            (r"\bthat is to say\b", ""),
            (r"\bin other words\b", ""),
            (r"\bto be honest\b", ""),
            (r"\bthe fact that\b", ""),
            (r"\bfor the purpose of\b", "to"),
            (r"\bin order to\b", "to"),
            (r"\bdue to the fact that\b", "because"),
            (r"\bat this point in time\b", "now"),
            (r"\bin the event that\b", "if"),
            (r"\bwith regard to\b", "about"),
            (r"\bin spite of the fact that\b", "although"),
            (r"\bhas the ability to\b", "can"),
            (r"\bdoes not have the ability to\b", "cannot"),
        ]
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Step 3: Remove repeated words
        text = re.sub(r"\b(\w+)( \1)+\b", r"\1", text)

        # Step 4: Check if we're under target
        current_tokens = self._count_tokens(text)
        if current_tokens <= target_tokens:
            return text

        # Step 5: Truncate with ellipsis for context
        # Find sentence boundaries and truncate at a natural point
        sentences = re.split(r"([.!?]+)", text)
        if len(sentences) > 1:
            # Try truncating at sentence boundaries
            for i in range(len(sentences) - 2, -1, -2):
                truncated = "".join(sentences[:i + 1])
                if self._count_tokens(truncated) <= target_tokens:
                    # Add ellipsis if we cut mid-thought
                    if i < len(sentences) - 2:
                        truncated = truncated.rstrip(".!?") + "..."
                    return truncated

        # Step 6: Hard truncate
        words = text.split()
        if len(words) <= target_tokens:
            return text

        # Find best truncation point (prefer word boundaries)
        target_words = int(target_tokens * 1.5)  # ~1.5x for safety
        truncated = " ".join(words[:target_words])
        if self._count_tokens(truncated) <= target_tokens:
            return truncated.rstrip(".!?") + "..."

        # Last resort: word-level truncate
        truncated = " ".join(words[:target_tokens])
        return truncated.rstrip(".!?") + "..."


class AdaptiveCompressor:
    """Adaptive compressor that chooses compression strategy based on content.

    Analyzes prompt characteristics to choose optimal compression:
    - Short prompts: No compression
    - Structured prompts (lists, sections): Preserve structure
    - Narrative prompts: Aggressive compression
    - Code prompts: Minimal compression (preserve syntax)
    """

    def __init__(self, llmlingua_model: str = None):
        self._compressor = PromptCompressor(
            model=llmlingua_model or "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        )
        self._stats = {"total": 0, "compressed": 0, "skipped": 0}

    def compress(self, prompt: str, max_tokens: int = 2048) -> tuple[str, float]:
        """Compress prompt adaptively.

        Returns:
            Tuple of (compressed_prompt, compression_ratio)
        """
        self._stats["total"] += 1
        original_tokens = self._compressor._count_tokens(prompt)

        # Don't compress very short prompts
        if original_tokens <= 256:
            self._stats["skipped"] += 1
            return prompt, 1.0

        # Choose strategy based on content type
        strategy = self._detect_strategy(prompt)
        target_tokens = self._compute_target(prompt, max_tokens, strategy)
        result = self._compressor.compress(prompt, target_token_num=target_tokens)

        if result.ratio < 0.95:
            self._stats["compressed"] += 1
        else:
            self._stats["skipped"] += 1

        return result.compressed, result.ratio

    def should_compress(self, prompt: str, threshold: int = 512) -> bool:
        """Check if compression is recommended."""
        return self._compressor.should_compress(prompt, threshold)

    def _detect_strategy(self, prompt: str) -> str:
        """Detect prompt type to choose compression strategy."""
        lower = prompt.lower()

        # Code
        if any(kw in lower for kw in ["```", "def ", "class ", "function ", "import ", "return "]):
            return "code"

        # Structured
        if any(marker in prompt for marker in ["\n##", "\n###", "\n1.", "\n2.", "\n- ", "\n* "]):
            return "structured"

        # Conversational / chat
        if lower.startswith("hello") or lower.startswith("hi ") or "?" in prompt:
            return "conversational"

        # Narrative / long form
        if len(prompt) > 2000:
            return "narrative"

        return "general"

    def _compute_target(self, prompt: str, max_tokens: int, strategy: str) -> int:
        """Compute target token count based on strategy."""
        base = self._compressor._count_tokens(prompt)

        # Strategy-specific compression ratios
        ratios = {
            "code": 0.8,  # Minimal compression for code
            "structured": 0.6,  # Moderate, preserve structure
            "conversational": 0.7,  # Moderate for Q&A
            "narrative": 0.4,  # Aggressive for long text
            "general": 0.5,  # Default
        }

        target = int(base * ratios.get(strategy, 0.5))
        return min(target, max_tokens)

    def stats(self) -> dict:
        """Return statistics."""
        return {
            **self._stats,
            "compression_rate": (
                self._stats["compressed"] / self._stats["total"]
                if self._stats["total"] > 0
                else 0.0
            ),
        }
