"""Tests for imprint CLI route command."""

import subprocess
import sys
from pathlib import Path

import pytest


def test_route_short_prompt():
    """Route command should recommend small model for short prompts."""
    result = subprocess.run(
        [sys.executable, "-m", "imprint", "route", "Sort a list"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    assert result.returncode == 0
    assert "SMALL" in result.stdout
    assert "Qwen2.5-3B-Instruct" in result.stdout


def test_route_medium_prompt():
    """Route command should recommend large model for long prompts."""
    # Use enough words to exceed 20-token threshold
    prompt = " ".join(["word"] * 30)
    result = subprocess.run(
        [sys.executable, "-m", "imprint", "route", prompt],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    assert result.returncode == 0
    # Should be medium (30 tokens = moderate complexity)
    assert "MEDIUM" in result.stdout or "LARGE" in result.stdout


def test_route_long_prompt():
    """Route command should recommend large model for long prompts."""
    prompt = " ".join(["word"] * 150)
    result = subprocess.run(
        [sys.executable, "-m", "imprint", "route", prompt],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    assert result.returncode == 0
    assert "LARGE" in result.stdout


def test_route_no_args():
    """Route command without args should show usage."""
    result = subprocess.run(
        [sys.executable, "-m", "imprint", "route"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    assert result.returncode == 1
    assert "usage" in result.stdout


def test_route_shows_cost_estimate():
    """Route command should show cost estimate."""
    result = subprocess.run(
        [sys.executable, "-m", "imprint", "route", "Test prompt"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    assert result.returncode == 0
    assert "Cost Estimate" in result.stdout
    assert "Per-request" in result.stdout


def test_route_shows_compression_info():
    """Route command should show compression info."""
    result = subprocess.run(
        [sys.executable, "-m", "imprint", "route", "Test prompt"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    assert result.returncode == 0
    assert "Compression" in result.stdout
    assert "Recommended" in result.stdout


def test_route_shows_optimization_pipeline():
    """Route command should show full optimization pipeline."""
    result = subprocess.run(
        [sys.executable, "-m", "imprint", "route", "Test prompt"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    assert result.returncode == 0
    assert "Semantic Cache" in result.stdout
    assert "Prefix Tree" in result.stdout
    assert "llmlingua" in result.stdout


def test_status_verbose_command():
    """Status-verbose command should show models config."""
    result = subprocess.run(
        [sys.executable, "-m", "imprint", "status-verbose"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    assert result.returncode == 0
    # Should show models config or be empty
    assert "models config" in result.stdout or "signatures" in result.stdout
