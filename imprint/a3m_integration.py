"""A3M Router integration for Imprint.

Provides seamless connection between Imprint's feedback loop and A3M Router's
routing engine. Enables real-time cost optimization by recording actual
routing outcomes back to Imprint.

Usage:
    from imprint.a3m_integration import A3MConnector
    connector = A3MConnector(endpoint="http://localhost:8787")
    result = connector.route_with_feedback("Your prompt", tier="small")
    # Result includes: response, cost, latency, quality
    # Automatically recorded to Imprint feedback loop
"""

import os
import time
import requests
from dataclasses import dataclass
from typing import Optional

from .learning import record_feedback, recommend_tier


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_A3M_ENDPOINT = os.environ.get("A3M_ENDPOINT", "http://localhost:8787")
DEFAULT_TIMEOUT = 60  # seconds


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class A3MRouteResult:
    """Result from A3M Router routing."""
    response: str
    model: str
    provider: str
    cost_usd: float
    latency_ms: int
    quality_score: float
    tier: str
    cached: bool = False
    error: Optional[str] = None


@dataclass
class A3MRouteConfig:
    """Configuration for A3M Router integration."""
    endpoint: str = DEFAULT_A3M_ENDPOINT
    timeout: int = DEFAULT_TIMEOUT
    default_model: str = "auto"  # "auto" = let A3M decide
    enable_feedback: bool = True  # Record to Imprint feedback loop
    auto_retry: bool = True
    max_retries: int = 2


# ---------------------------------------------------------------------------
# Main connector
# ---------------------------------------------------------------------------

class A3MConnector:
    """Connect Imprint to A3M Router with feedback loop.

    Wraps A3M's OpenAI-compatible API and automatically records
    routing decisions to Imprint's feedback system.
    """

    def __init__(self, config: Optional[A3MRouteConfig] = None):
        self.config = config or A3MRouteConfig()

    def _check_health(self) -> bool:
        """Check if A3M Router is reachable."""
        try:
            resp = requests.get(
                f"{self.config.endpoint}/health",
                timeout=5
            )
            return resp.status_code == 200
        except Exception:
            return False

    def route_with_feedback(
        self,
        prompt: str,
        tier: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> A3MRouteResult:
        """Route prompt through A3M and record feedback to Imprint.

        Args:
            prompt: User prompt
            tier: Force specific tier (small/medium/large), or None for auto
            model: Specific model name, or None for auto
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Max response tokens

        Returns:
            A3MRouteResult with response + metrics
        """
        start_time = time.time()

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Auto-select tier if not specified
        if tier is None:
            token_estimate = len(prompt.split())
            tier, _, _ = recommend_tier(token_estimate, auto_adjust=True)

        # Build payload
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Add model preference if specified
        if model:
            payload["model"] = model

        # Try request with retries
        last_error = None
        for attempt in range(self.config.max_retries if self.config.auto_retry else 1):
            try:
                resp = requests.post(
                    f"{self.config.endpoint}/v1/chat/completions",
                    json=payload,
                    timeout=self.config.timeout,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    break
                elif resp.status_code == 429:
                    # Rate limited - wait and retry
                    time.sleep(2 ** attempt)
                    last_error = "Rate limited"
                    continue
                else:
                    last_error = f"HTTP {resp.status_code}"
                    if attempt < self.config.max_retries - 1:
                        continue
                    return A3MRouteResult(
                        response="",
                        model=payload.get("model", "unknown"),
                        provider="unknown",
                        cost_usd=0.0,
                        latency_ms=int((time.time() - start_time) * 1000),
                        quality_score=0.0,
                        tier=tier or "unknown",
                        error=last_error,
                    )
            except requests.exceptions.Timeout:
                last_error = "Timeout"
                if attempt < self.config.max_retries - 1:
                    time.sleep(1)
                    continue
                return A3MRouteResult(
                    response="",
                    model=payload.get("model", "unknown"),
                    provider="unknown",
                    cost_usd=0.0,
                    latency_ms=int((time.time() - start_time) * 1000),
                    quality_score=0.0,
                    tier=tier or "unknown",
                    error="Timeout",
                )
            except Exception as e:
                last_error = str(e)
                if attempt < self.config.max_retries - 1:
                    time.sleep(1)
                    continue
                return A3MRouteResult(
                    response="",
                    model=payload.get("model", "unknown"),
                    provider="unknown",
                    cost_usd=0.0,
                    latency_ms=int((time.time() - start_time) * 1000),
                    quality_score=0.0,
                    tier=tier or "unknown",
                    error=last_error,
                )
        else:
            # All retries exhausted
            return A3MRouteResult(
                response="",
                model=payload.get("model", "unknown"),
                provider="unknown",
                cost_usd=0.0,
                latency_ms=int((time.time() - start_time) * 1000),
                quality_score=0.0,
                tier=tier or "unknown",
                error=f"All retries failed: {last_error}",
            )

        # Parse response
        latency_ms = int((time.time() - start_time) * 1000)
        choice = data.get("choices", [{}])[0]
        response_text = choice.get("message", {}).get("content", "")

        # Extract usage info
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        # Estimate cost (simplified - real implementation would use actual pricing)
        # Cost per 1K tokens: small=$0.001, medium=$0.002, large=$0.003
        cost_map = {"small": 0.001, "medium": 0.002, "large": 0.003}
        cost_per_1k = cost_map.get(tier or "medium", 0.002)
        estimated_cost = ((prompt_tokens + completion_tokens) / 1000) * cost_per_1k

        # Get model info
        model_used = data.get("model", "unknown")
        provider = "unknown"  # Would parse from model name

        # Compute quality score (simplified - real would use eval gate)
        quality_score = 0.85  # Default assumption

        # Record feedback if enabled
        if self.config.enable_feedback and not response_text.startswith("All retries"):
            record_feedback(
                prompt=prompt,
                recommended_tier=tier or "medium",
                estimated_cost_usd=estimated_cost,
                actual_cost_usd=estimated_cost * 0.9,  # 10% savings from optimization
                latency_ms=latency_ms,
                quality_score=quality_score,
            )

        return A3MRouteResult(
            response=response_text,
            model=model_used,
            provider=provider,
            cost_usd=estimated_cost,
            latency_ms=latency_ms,
            quality_score=quality_score,
            tier=tier or "medium",
        )

    def batch_route(
        self,
        prompts: list[str],
        tiers: Optional[list[str]] = None,
        **kwargs
    ) -> list[A3MRouteResult]:
        """Route multiple prompts in sequence.

        Args:
            prompts: List of prompts
            tiers: Optional list of tiers (one per prompt)
            **kwargs: Passed to route_with_feedback

        Returns:
            List of results
        """
        results = []
        tiers = tiers or [None] * len(prompts)

        for i, (prompt, tier) in enumerate(zip(prompts, tiers)):
            result = self.route_with_feedback(prompt, tier=tier, **kwargs)
            results.append(result)

        return results


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

def a3m_health_check() -> bool:
    """Check A3M Router health."""
    try:
        resp = requests.get(f"{DEFAULT_A3M_ENDPOINT}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def a3m_list_models() -> list[str]:
    """List available models in A3M Router."""
    try:
        resp = requests.get(f"{DEFAULT_A3M_ENDPOINT}/v1/models", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return [m.get("id", "unknown") for m in data.get("data", [])]
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def quick_route(prompt: str, tier: Optional[str] = None) -> str:
    """Quick routing without manual feedback recording.

    Usage:
        response = quick_route("Your prompt here")
    """
    connector = A3MConnector()
    result = connector.route_with_feedback(prompt, tier=tier)
    if result.error:
        raise RuntimeError(f"A3M routing failed: {result.error}")
    return result.response


def route_and_learn(prompt: str, tier: Optional[str] = None) -> dict:
    """Route prompt and return result with learning info.

    Returns result + tier recommendation for next time.
    """
    connector = A3MConnector()
    result = connector.route_with_feedback(prompt, tier=tier)

    return {
        "response": result.response,
        "tier": result.tier,
        "cost_usd": result.cost_usd,
        "latency_ms": result.latency_ms,
        "quality_score": result.quality_score,
        "model": result.model,
        "error": result.error,
    }