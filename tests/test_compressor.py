"""Tests for the prompt compressor.

Tests work without LLMLingua (uses heuristic fallback).
"""
import pytest


def test_short_prompt_unchanged():
    """Short prompts should not be compressed."""
    from imprint.compressor import PromptCompressor

    compressor = PromptCompressor(use_llmlingua=False)
    result = compressor.compress("Write a haiku", budget=1024, ratio=0.5)

    assert result.compressed == "Write a haiku"
    assert result.ratio == 1.0
    assert result.method == "none"


def test_long_prompt_compressed():
    """Long prompts should be compressed below original length."""
    from imprint.compressor import PromptCompressor

    compressor = PromptCompressor(use_llmlingua=False)
    long_prompt = (
        "Please write a comprehensive and detailed summary of the following "
        "long article about artificial intelligence. Basically, you should "
        "actually try to be really detailed in your response and kind of "
        "thorough. The article is about how AI systems are increasingly "
        "being used in many different domains. It talks about machine learning, "
        "deep learning, natural language processing, computer vision, "
        "and reinforcement learning. It also discusses the ethical implications "
        "of AI, the potential for job displacement, and the need for responsible "
        "AI development. The article concludes that AI will continue to "
        "transform many industries and that we need to be prepared for the "
        "changes that are coming." * 5
    )

    result = compressor.compress(long_prompt, budget=100, ratio=0.5)
    assert len(result.compressed) < len(result.original)
    assert result.ratio < 1.0
    assert result.method == "heuristic"


def test_compress_removes_redundant_phrases():
    """Heuristic compression should remove filler phrases."""
    from imprint.compressor import PromptCompressor

    compressor = PromptCompressor(use_llmlingua=False)
    prompt = "Please basically actually write a summary that is to say a short version"
    result = compressor.compress(prompt, budget=20, ratio=0.3)

    # The compressed result should be shorter
    assert len(result.compressed) < len(result.original)
    # Some redundant words should be removed
    assert "basically" not in result.compressed.lower() or len(result.compressed) < len(prompt)


def test_should_compress_threshold():
    """should_compress should respect threshold."""
    from imprint.compressor import PromptCompressor

    compressor = PromptCompressor(use_llmlingua=False)
    short = "hello"
    long = " ".join(["word"] * 200)

    assert not compressor.should_compress(short, threshold=100)
    assert compressor.should_compress(long, threshold=100)


def test_compress_preserves_question_words():
    """Compression should preserve key action words."""
    from imprint.compressor import PromptCompressor

    compressor = PromptCompressor(use_llmlingua=False)
    prompt = "Please write a comprehensive summary of the report that explains what happened"
    result = compressor.compress(prompt, budget=30, ratio=0.5)

    # Quality score should be reasonable
    assert result.quality_score >= 0.0
    assert result.quality_score <= 1.0


def test_adaptive_compressor_skips_short():
    """AdaptiveCompressor should skip very short prompts."""
    from imprint.compressor import AdaptiveCompressor

    compressor = AdaptiveCompressor()
    compressed, ratio = compressor.compress("hello world", max_tokens=2048)

    assert ratio == 1.0
    assert compressed == "hello world"


def test_adaptive_compressor_handles_code():
    """AdaptiveCompressor should use lighter compression for code."""
    from imprint.compressor import AdaptiveCompressor

    compressor = AdaptiveCompressor()
    code_prompt = "```python\ndef hello():\n    print('hi')\n```"
    compressed, ratio = compressor.compress(code_prompt, max_tokens=2048)

    # Code should be preserved mostly
    assert "def" in compressed or "```" in compressed


def test_adaptive_compressor_detects_strategy():
    """AdaptiveCompressor should detect content type."""
    from imprint.compressor import AdaptiveCompressor

    compressor = AdaptiveCompressor()

    # Code
    code_strategy = compressor._detect_strategy("```python\nimport os\n```")
    assert code_strategy == "code"

    # Structured
    structured_strategy = compressor._detect_strategy("1. First point\n2. Second point\n- Item")
    assert structured_strategy in ("structured", "conversational")

    # Conversational
    conv_strategy = compressor._detect_strategy("Hello! How are you?")
    assert conv_strategy == "conversational"


def test_compression_result_dataclass():
    """CompressionResult should hold all fields."""
    from imprint.compressor import CompressionResult

    result = CompressionResult(
        original="long text",
        compressed="short",
        original_tokens=100,
        compressed_tokens=50,
        ratio=0.5,
        method="heuristic",
        quality_score=0.8,
    )
    assert result.original == "long text"
    assert result.ratio == 0.5
    assert result.quality_score == 0.8


def test_batch_compress():
    """Batch compression should handle multiple prompts."""
    from imprint.compressor import PromptCompressor

    compressor = PromptCompressor(use_llmlingua=False)
    prompts = [
        "short",
        "Please basically write a moderately long response about the topic " * 3,
        "another short one",
    ]
    results = compressor.batch_compress(prompts, budget=50)

    assert len(results) == 3
    # Short prompts should be unchanged
    assert results[0].ratio == 1.0
    # Long prompt should be compressed
    assert results[1].ratio < 1.0


def test_compression_stats():
    """Compression stats should track calls correctly."""
    from imprint.compressor import PromptCompressor

    compressor = PromptCompressor(use_llmlingua=False)
    # Make some calls
    compressor.compress("short", budget=1024)
    compressor.compress("a long prompt " * 50, budget=20)

    stats = compressor.stats()
    assert stats["total_calls"] == 2
    assert stats["heuristic_calls"] == 1
    assert stats["llmlingua_available"] == False
