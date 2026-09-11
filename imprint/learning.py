"""Learning algorithm for Imprint routing optimization.

Computes savings, updates tier preferences, and enables continuous
learner feedback loop from production routing decisions.

Phase 1: Production feedback loop + Phase 2: Enhanced automated learning
"""

from datetime import datetime, timedelta
from typing import Optional, List

from .store import connect


# ---------------------------------------------------------------------------
# Core algorithms
# ---------------------------------------------------------------------------

def compute_savings_pct(estimated_cost: float, actual_cost: float) -> float:
    """Compute percentage savings from routing decision.

    Savings = (estimated - actual) / estimated
    Returns 0 if estimated is 0 or negative.
    """
    if estimated_cost is None or estimated_cost <= 0:
        return 0.0
    savings = (estimated_cost - actual_cost) / estimated_cost
    return max(0.0, savings * 100.0)


def compute_tier_efficiency_score(
    avg_savings_pct: float,
    avg_latency_ms: float,
    avg_quality: float,
    weight_savings: float = 0.5,
    weight_latency: float = 0.3,
    weight_quality: float = 0.2,
) -> float:
    """Compute efficiency score for a tier.

    Higher score = better tier recommendation.
    Combines savings, latency, and quality into single metric.
    """
    savings_score = avg_savings_pct / 100.0

    latency_score = max(0.0, 1.0 - min(avg_latency_ms, 500) / 500.0)
    quality_score = avg_quality

    return (
        weight_savings * savings_score +
        weight_latency * latency_score +
        weight_quality * quality_score
    )


def compute_cost_efficiency_ratio(
    total_savings_usd: float,
    total_actual_cost_usd: float,
    total_estimated_cost_usd: float
) -> float:
    """Compute overall cost efficiency of the system.

    Returns ratio of savings to actual cost.
    """
    if total_actual_cost_usd == 0:
        return 0.0
    return total_savings_usd / total_actual_cost_usd


def record_feedback(
    prompt: str,
    recommended_tier: str,
    estimated_cost_usd: float,
    actual_cost_usd: float,
    latency_ms: int,
    quality_score: Optional[float] = None,
    signature_id: Optional[str] = None,
    top_k_results: Optional[List[float]] = None,
) -> dict:
    """Record routing feedback for continuous learning.

    Args:
        prompt: The original prompt
        recommended_tier: Tier that was recommended (small/medium/large)
        estimated_cost_usd: Cost estimated by Imprint
        actual_cost_usd: Actual cost from the model provider
        latency_ms: Actual latency in milliseconds
        quality_score: Optional quality score (0-1)
        signature_id: Optional signature ID for clustering
        top_k_results: Optional list of top-k match scores from cache

    Returns:
        Dict with recorded data and computed savings percentage
    """
    conn = connect()
    ts = int(datetime.utcnow().timestamp() * 1000)

    savings_pct = compute_savings_pct(estimated_cost_usd, actual_cost_usd)

    conn.execute(
        """INSERT INTO routing_feedback
        (ts, prompt_text, recommended_tier, selected_model,
         estimated_cost_usd, actual_cost_usd, latency_ms, quality_score,
         signature_id, top_k_match_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            ts,
            prompt,
            recommended_tier,
            f"{recommended_tier}-model",
            estimated_cost_usd,
            actual_cost_usd,
            latency_ms,
            quality_score or 0.5,
            signature_id,
            top_k_results[0] if top_k_results else 0.0,
        ),
    )
    conn.commit()

    return {
        "ts": ts,
        "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
        "tier": recommended_tier,
        "estimated_cost": estimated_cost_usd,
        "actual_cost": actual_cost_usd,
        "savings_pct": savings_pct,
        "latency_ms": latency_ms,
        "quality_score": quality_score or 0.5,
        "signature_id": signature_id,
    }


def get_tier_rankings(
    days: int = 30,
    min_records: int = 5,
    exclude_tiers: Optional[List[str]] = None,
    use_semantic_cache: bool = True,
) -> List[dict]:
    """Rank tiers by performance over specified period.

    Args:
        days: Number of days to look back
        min_records: Minimum records required for tier to be ranked
        exclude_tiers: Tiers to exclude from ranking
        use_semantic_cache: Whether to consider semantic cache hit rate

    Returns:
        List of tier ranking dicts sorted by efficiency score
    """
    conn = connect()
    since = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)

    exclude_tiers = exclude_tiers or []

    placeholders = ",".join(["?"] * len(exclude_tiers))

    rows = conn.execute(
        f"""SELECT
          recommended_tier,
          COUNT(*) as cnt,
          AVG(actual_cost_usd) as avg_cost,
          AVG(latency_ms) as avg_latency,
          AVG(quality_score) as avg_quality,
          AVG(estimated_cost_usd) as avg_estimated,
          AVG((estimated_cost_usd - actual_cost_usd) / NULLIF(estimated_cost_usd, 0) * 100) as avg_savings_pct,
          AVG(top_k_match_score) as avg_cache_score
          FROM routing_feedback
          WHERE ts >= ?
          AND recommended_tier NOT IN ({placeholders})
          GROUP BY recommended_tier
          HAVING COUNT(*) >= ?
          ORDER BY avg_savings_pct DESC, avg_latency ASC""",
        [since] + exclude_tiers + [min_records],
    ).fetchall()

    rankings = []
    for row in rows:
        tier = row[0]
        savings_pct = row[6] or 0.0
        latency = row[3] or 0.0
        quality = row[4] or 0.5
        cache_score = row[8] or 0.0

        # Enhanced efficiency score with cache consideration
        base_score = compute_tier_efficiency_score(savings_pct, latency, quality)
        if use_semantic_cache and cache_score > 0:
            # Boost score for high cache hit rate
            base_score = base_score * (1 + cache_score * 0.5)  # Up to 50% boost

        rankings.append({
            "tier": tier,
            "count": row[1],
            "avg_cost": row[2],
            "avg_latency": row[3],
            "avg_quality": row[4],
            "avg_estimated": row[5],
            "avg_savings_pct": savings_pct,
            "avg_cache_score": cache_score,
            "efficiency_score": base_score,
        })

    rankings.sort(key=lambda x: x["efficiency_score"], reverse=True)
    return rankings


def get_tier_thresholds(
    days: int = 30,
    min_records: int = 10,
) -> dict:
    """Compute adaptive thresholds based on tier performance.

    Returns recommended tier for different prompt complexities.
    """
    rankings = get_tier_rankings(days=days, min_records=min_records)

    if not rankings:
        return {"small": 10, "medium": 50, "large": 100}

    result = {}
    for tier_data in rankings:
        tier = tier_data["tier"]
        quality_latency_ratio = tier_data["avg_quality"] / max(tier_data["avg_latency"], 1)
        cache_boost = tier_data.get("avg_cache_score", 0.0) * 0.2  # Cache bonus

        if tier == "small":
            result[tier] = int(10 + 20 * (quality_latency_ratio + cache_boost) * 10)
        elif tier == "medium":
            result[tier] = int(50 + 50 * (quality_latency_ratio + cache_boost) * 10)
        else:
            result[tier] = int(100 + 100 * (quality_latency_ratio + cache_boost) * 10)

    result.setdefault("small", 10)
    result.setdefault("medium", 50)
    result.setdefault("large", 100)

    return result


def get_feedback_stats(
    days: Optional[int] = None,
    include_trends: bool = True,
    include_a3m_metrics: bool = True,
) -> dict:
    """Get overall feedback statistics.

    Args:
        days: Number of days to include (None = all time)
        include_trends: Whether to include trend analysis
        include_a3m_metrics: Whether to include A3M Router specific metrics

    Returns:
        Dict with statistics
    """
    conn = connect()

    if days:
        since = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
        filter_clause = f"WHERE ts >= {since}"
    else:
        filter_clause = ""

    total = conn.execute(
        f"SELECT COUNT(*) FROM routing_feedback {filter_clause}"
    ).fetchone()[0]

    total_cost = conn.execute(
        f"SELECT SUM(actual_cost_usd) FROM routing_feedback {filter_clause}"
    ).fetchone()[0] or 0.0

    total_estimated = conn.execute(
        f"SELECT SUM(estimated_cost_usd) FROM routing_feedback {filter_clause}"
    ).fetchone()[0] or 0.0

    total_savings_money = total_estimated - total_cost
    total_savings_pct = (total_savings_money / total_estimated * 100) if total_estimated > 0 else 0

    cost_efficiency = compute_cost_efficiency_ratio(
        total_savings_money, total_actual_cost_usd=total_cost, 
        total_estimated_cost_usd=total_estimated
    )

    tier_performance = conn.execute(
        f"""SELECT recommended_tier,
          COUNT(*) as cnt,
          AVG(actual_cost_usd) as avg_cost,
          AVG(latency_ms) as avg_latency,
          AVG(quality_score) as avg_quality,
          AVG(estimated_cost_usd) as avg_estimated,
          AVG((estimated_cost_usd - actual_cost_usd) / NULLIF(estimated_cost_usd, 0) * 100) as avg_savings_pct,
          AVG(top_k_match_score) as avg_cache_score
          FROM routing_feedback {filter_clause}
          GROUP BY recommended_tier""",
    ).fetchall()

    result = {
        "total_records": total,
        "total_cost_usd": total_cost,
        "total_estimated_usd": total_estimated,
        "total_savings_usd": total_savings_money,
        "total_savings_pct": total_savings_pct,
        "cost_efficiency_ratio": cost_efficiency,
        "tier_performance": [
            {
                "tier": row[0],
                "count": row[1],
                "avg_cost": row[2],
                "avg_latency": row[3],
                "avg_quality": row[4],
                "avg_estimated": row[5],
                "avg_savings_pct": row[6] or 0.0,
                "avg_cache_score": row[7] or 0.0,
            }
            for row in tier_performance
        ],
    }

    if include_trends:
        # Daily trends
        daily_trends = conn.execute(
            f"""SELECT DATE(ts/1000, 'unixepoch') as day,
                          COUNT(*) as records,
                          AVG(actual_cost_usd) as avg_cost,
                          AVG(latency_ms) as avg_latency,
                          AVG(top_k_match_score) as avg_cache_score
               FROM routing_feedback {filter_clause}
               GROUP BY day
               ORDER BY day DESC
               LIMIT 30""",
        ).fetchall()

        result["daily_trends"] = [
            {"day": row[0], "records": row[1], "avg_cost": row[2], 
             "avg_latency": row[3], "avg_cache_score": row[4] or 0.0}
            for row in daily_trends
        ]

        # Hourly trends
        hourly_trends = conn.execute(
            f"""SELECT strftime('%H', ts/1000, 'unixepoch') as hour,
                            COUNT(*) as records,
                            AVG(actual_cost_usd) as avg_cost,
                            AVG(latency_ms) as avg_latency
               FROM routing_feedback {filter_clause}
               GROUP BY hour
               ORDER BY hour""",
        ).fetchall()

        result["hourly_trends"] = [
            {"hour": row[0], "records": row[1], "avg_cost": row[2], "avg_latency": row[3]}
            for row in hourly_trends
        ]

    if include_a3m_metrics:
        # A3M-specific metrics
        a3m_stats = get_a3m_metrics(days=days)
        result["a3m_metrics"] = a3m_stats

    return result


def get_a3m_metrics(days: Optional[int] = None) -> dict:
    """Get A3M Router specific performance metrics.

    Args:
        days: Number of days to include (None = all time)

    Returns:
        Dict with A3M Router metrics
    """
    # Lazy import to avoid circular dependency
    try:
        from .a3m_integration import a3m_health_check, a3m_list_models
        health = a3m_health_check()
        models = a3m_list_models()
    except Exception:
        health = False
        models = []

    return {
        "a3m_healthy": health,
        "available_models": len(models),
        "model_list": models[:10],  # First 10 models
        "health_check_passed": health,
    }


def recommend_tier(
    token_estimate: int,
    auto_adjust: bool = True,
    days: int = 30,
) -> tuple[str, float, str]:
    """Recommend optimal tier for a given token count.

    Args:
        token_estimate: Estimated token count
        auto_adjust: Whether to use learned thresholds
        days: Days to consider for learned thresholds

    Returns:
        Tuple of (tier, estimated_cost_usd, reason)
    """
    if auto_adjust:
        thresholds = get_tier_thresholds(days=days)

        if token_estimate <= thresholds.get("small", 10):
            tier = "small"
            cost_per_1k = 0.001
            reason = f"Token count {token_estimate} ≤ small threshold ({thresholds['small']})"
        elif token_estimate <= thresholds.get("medium", 50):
            tier = "medium"
            cost_per_1k = 0.002
            reason = f"Token count {token_estimate} ≤ medium threshold ({thresholds['medium']})"
        else:
            tier = "large"
            cost_per_1k = 0.003
            reason = f"Token count {token_estimate} > medium threshold ({thresholds['medium']})"
    else:
        # Default thresholds
        if token_estimate < 20:
            tier = "small"
            cost_per_1k = 0.001
            reason = f"Default threshold for {token_estimate} tokens"
        elif token_estimate < 100:
            tier = "medium"
            cost_per_1k = 0.002
            reason = f"Default threshold for {token_estimate} tokens"
        else:
            tier = "large"
            cost_per_1k = 0.003
            reason = f"Default threshold for {token_estimate} tokens"

    estimated_cost = (token_estimate / 1000) * cost_per_1k
    return tier, estimated_cost, reason


def auto_learn_and_update(
    min_feedback: int = 10,
    learning_rate: float = 0.1,
) -> dict:
    """Perform automated learning cycle.

    Analyzes recent feedback and suggests tier preference updates.

    Args:
        min_feedback: Minimum feedback records to trigger learning
        learning_rate: How much to adjust preferences (0-1)

    Returns:
        Dict with learning results and recommendations
    """
    # Get recent feedback
    stats = get_feedback_stats(days=7)  # Last week
    total_records = stats["total_records"]

    if total_records < min_feedback:
        return {
            "status": "insufficient_data",
            "message": f"Need {min_feedback} feedback records, got {total_records}",
            "recommendations": []
        }

    # Get tier rankings
    rankings = get_tier_rankings(days=30, min_records=5)

    # Compute suggested adjustments
    adjustments = []
    if rankings:
        best_tier = rankings[0]["tier"]
        best_score = rankings[0]["efficiency_score"]
        
        for tier_data in rankings:
            tier = tier_data["tier"]
            current_efficiency = tier_data["efficiency_score"]
            score_diff = best_score - current_efficiency
            
            if score_diff > 0.1:  # Significant difference
                adjustment = min(learning_rate * score_diff, 0.3)  # Max 30% adjustment
                adjustments.append({
                    "tier": tier,
                    "current_efficiency": current_efficiency,
                    "best_efficiency": best_score,
                    "suggested_adjustment": adjustment,
                    "reason": f"Tier {best_tier} performs better by {score_diff:.2f}"
                })

    return {
        "status": "learning_complete",
        "total_feedback": total_records,
        "best_tier": rankings[0]["tier"] if rankings else None,
        "best_score": rankings[0]["efficiency_score"] if rankings else None,
        "adjustments": adjustments,
        "recommendations": [
            f"Consider using '{rankings[0]['tier']}' tier for optimal performance"
            if rankings else "Insufficient data for recommendations"
        ]
    }


def test_learning():
    """Test the learning module."""
    print("Testing learning module...")

    # Test savings computation
    assert compute_savings_pct(0.001, 0.0008) == 20.0
    assert compute_savings_pct(0.001, 0.001) == 0.0
    assert compute_savings_pct(0, 0.001) == 0.0

    # Test efficiency score
    score1 = compute_tier_efficiency_score(0.1, 100, 0.5)
    score2 = compute_tier_efficiency_score(0.2, 50, 0.6)
    assert score2 > score1  # Better savings + lower latency + higher quality

    print("✅ All tests passed!")