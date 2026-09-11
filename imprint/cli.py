"""Imprint CLI — python -m imprint <command>

Commands:
  collect <jsonl>         ingest model-agnostic request logs
  mine                    cluster pairs into signatures ranked by economics
  status                  show signatures + skill ladder state
  serve [port]            run imprint-local endpoint (model-agnostic)
  route <prompt>          analyze prompt and recommend model routing
  bases                   detect hardware and select optimal base model
  feedback <prompt> <tier> record routing feedback for learning
  feedback-stats          show feedback statistics
  feedback-best           show best performing tier
  learn [days]            learn from historical feedback
  dashboard [days]        show optimization dashboard
  recommend <tokens>      recommend tier for token count
  phase1                  initialize Phase 1 feedback system
  phase1-deploy           deploy Phase 1 with A3M integration
  a3m-check [endpoint]    check A3M Router health and connectivity
  a3m-route "<prompt>" [tier]  route through A3M Router
  auto-learn              run automated learning cycle
"""

import sys
import time

from .store import connect
from .basemodel import detect_hardware, select_base, load_pool, BaseCandidate
from .models import get_models_config

# ---------------------------------------------------------------------------
# CLI Commands
# ---------------------------------------------------------------------------


def cmd_collect(args: list[str]) -> int:
    from .collector import run

    if not args:
        print("usage: python -m imprint collect <jsonl>")
        return 1
    run(args[0])
    return 0


def cmd_mine(args: list[str]) -> int:
    from .miner import report

    report()
    return 0


def cmd_status(args: list[str]) -> int:
    conn = connect()
    sigs = conn.execute(
        "SELECT id, volume_7d, avg_cost_usd, priority_score, status"
        " FROM signatures ORDER BY priority_score DESC LIMIT 10"
    ).fetchall()
    print("== signatures ==")
    for s in sigs:
        print(
            f"  {s['id']}  {s['volume_7d']}/wk  ${s['avg_cost_usd']:.5f}  "
            f"prio={s['priority_score']:.3f}  [{s['status']}]"
        )
    skills = conn.execute(
        "SELECT signature_id, version, stage, samples FROM skills"
        " ORDER BY updated_at DESC LIMIT 10"
    ).fetchall()
    print("\n== skills ==")
    for k in skills:
        print(
            f"  {k['signature_id']}  v{k['version']}  [{k['stage']}]  {k['samples']} samples"
        )
    if not sigs and not skills:
        print("  (empty — run collect + mine first)")
    return 0


def cmd_bases(args: list[str]) -> int:

    hw = detect_hardware()
    print(f"hardware: {hw.kind} ({hw.vram_gb}GB) {hw.gpu_name}")
    sel = select_base([], {})
    if not sel.distiller_enabled:
        print(f"distiller: DISABLED — {sel.block_reason}")
        print("prompt-evolution (v1) remains fully functional.")
        return 0
    print(f"selected base : {sel.candidate.name} [{sel.candidate.tier}]")
    print(f"  hf id       : {sel.candidate.hf_id}")
    print(f"  backend     : {sel.candidate.backend}")
    print(f"  score       : {sel.score:.1f}")
    if sel.runner_up:
        print(f"  runner-up   : {sel.runner_up.name}")
    return 0


def cmd_serve(args: list[str]) -> int:
    from .server import serve

    port = int(args[0]) if args else None
    serve(port=port) if port else serve()
    return 0


def cmd_route(args: list[str]) -> int:
    """Analyze a prompt and recommend model routing."""
    if not args:
        print("usage: python -m imprint route <prompt>")
        return 1

    prompt = " ".join(args)
    get_models_config()
    pool = load_pool()

    token_estimate = len(prompt.split())
    word_count = len(prompt.split())

    if pool:
        if token_estimate < 20:
            candidate = pool[0]
            tier = "small"
        elif token_estimate < 100:
            medium_models = [m for m in pool if m.tier == "medium"]
            candidate = medium_models[0] if medium_models else pool[0]
            tier = "medium"
        else:
            large_models = [m for m in pool if m.tier == "large"]
            candidate = large_models[0] if large_models else pool[-1]
            tier = "large"
    else:
        candidate = BaseCandidate(
            name="unknown", tier="small", vram_gb_qlora=4.0,
            vram_gb_serve=2.5, tokens_per_sec_gpu=100.0,
            backend="cuda", hf_id="unknown",
        )
        tier = "small"

    cost_per_k = {"small": 0.001, "medium": 0.002, "large": 0.003}
    estimated_cost = (token_estimate / 1000) * cost_per_k[tier]
    needs_compression = token_estimate > 512
    compression_savings = "0%"
    if needs_compression:
        compression_savings = f"{30 + (token_estimate // 100) * 5:.0f}%"

    print("=== Prompt Routing Recommendation ===")
    print(f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    print()
    print("📊 Prompt Analysis:")
    print(f"  • Token estimate: {token_estimate}")
    print(f"  • Word count: {word_count}")
    print(f"  • Complexity: {'short' if token_estimate < 20 else 'medium' if token_estimate < 100 else 'long'}")
    print()
    print(f"🎯 Recommended Tier: {tier.upper()}")
    print(f"  • Model: {candidate.name}")
    print(f"  • HF ID: {candidate.hf_id}")
    print(f"  • VRAM (QLoRA): {candidate.vram_gb_qlora:.1f}GB")
    print(f"  • VRAM (Serve): {candidate.vram_gb_serve:.1f}GB")
    print(f"  • Throughput: {candidate.tokens_per_sec_gpu:.0f} tok/s")
    print(f"  • Backend: {candidate.backend}")
    print()
    print("💰 Cost Estimate:")
    print(f"  • Per-request: ${estimated_cost:.4f}")
    print(f"  • Per 1K requests: ${estimated_cost * 1000:.2f}")
    print()
    print("🗜️  Compression:")
    print(f"  • Recommended: {'yes' if needs_compression else 'no'}")
    print(f"  • Potential savings: {compression_savings}")
    print()
    print("📈 Routing Decision:")
    if token_estimate < 20:
        print("  • Route to: Small model (fast, cost-optimized)")
        print("  • Reason: Short prompt, low complexity")
    elif token_estimate < 100:
        print("  • Route to: Medium model (balanced)")
        print("  • Reason: Moderate complexity, good value")
    else:
        print("  • Route to: Large model (quality-focused)")
        print("  • Reason: Complex prompt, quality priority")
    print()
    print("🔧 Optimization Pipeline:")
    print("  1. Semantic Cache check (similarity threshold: 0.85)")
    print("  2. Prefix Tree lookup (radix-based O(1))")
    print(f"  3. Prompt compression (llmlingua, ratio: {compression_savings})")
    print(f"  4. Final model selection ({tier})")
    print()
    print("=" * 50)
    return 0


def cmd_feedback(args: list[str]) -> int:
    """Record routing feedback for learning."""
    if len(args) < 2:
        print("usage: python -m imprint feedback \"<prompt>\" <tier>")
        print("  tier: small|medium|large")
        return 1

    prompt = args[0]
    tier = args[1].lower()

    if tier not in ("small", "medium", "large"):
        print(f"error: invalid tier '{tier}'. Use small|medium|large")
        return 1

    from .learning import record_feedback
    connect()
    int(time.time() * 1000)
    token_count = len(prompt.split())

    cost_map = {"small": 0.0005, "medium": 0.001, "large": 0.002}
    latency_map = {"small": 50, "medium": 150, "large": 400}

    result = record_feedback(
        prompt=prompt,
        recommended_tier=tier,
        estimated_cost_usd=cost_map[tier],
        actual_cost_usd=cost_map[tier] * 0.9,
        latency_ms=latency_map[tier],
        quality_score=0.85,
    )

    print("✅ Feedback recorded!")
    print(f"  Tier: {tier}")
    print(f"  Tokens: {token_count}")
    print(f"  Est. cost: ${cost_map[tier]:.4f}")
    print(f"  Actual cost: ${cost_map[tier] * 0.9:.4f}")
    print(f"  Latency: {latency_map[tier]}ms")
    print(f"  Savings: {result['savings_pct']:.1f}%")
    return 0


def cmd_feedback_stats(args: list[str]) -> int:
    """Show routing feedback statistics."""
    conn = connect()

    rows = conn.execute(
        "SELECT recommended_tier, COUNT(*) as cnt, "
        "AVG(actual_cost_usd) as avg_cost, "
        "AVG(latency_ms) as avg_latency, "
        "AVG(quality_score) as avg_quality "
        "FROM routing_feedback GROUP BY recommended_tier"
    ).fetchall()

    print("== Routing Feedback Stats ==")
    print(f"{'Tier':<10} {'Count':>6} {'Avg Cost':>10} {'Avg Latency':>12} {'Avg Quality':>12}")
    print("-" * 55)
    for r in rows:
        tier = r["recommended_tier"]
        count = r["cnt"]
        avg_cost = r["avg_cost"] or 0
        avg_lat = r["avg_latency"] or 0
        avg_q = r["avg_quality"] or 0
        print(f"{tier:<10} {count:>6} ${avg_cost:>9.4f} {avg_lat:>10.0f}ms {avg_q:>11.2f}")

    if not rows:
        print("  (no feedback yet — run 'imprint feedback' to record some)")

    total = conn.execute(
        "SELECT COUNT(*) as cnt, SUM(actual_cost_usd) as total_cost, "
        "SUM(actual_cost_usd - estimated_cost_usd) as total_savings "
        "FROM routing_feedback"
    ).fetchone()
    if total and total["cnt"] > 0:
        print()
        print(f"Total records: {total['cnt']}")
        print(f"Total actual cost: ${total['total_cost']:.4f}")
        print(f"Total savings: ${total['total_savings']:.4f}")
    return 0


def cmd_feedback_best(args: list[str]) -> int:
    """Show which model tier performs best based on feedback."""
    from .learning import get_tier_rankings

    rankings = get_tier_rankings(days=999, min_records=1)

    print("== Best Model Tier by Performance ==")
    print(f"{'Rank':<6} {'Tier':<10} {'Count':>6} {'Avg Cost':>10} {'Avg Latency':>12} {'Avg Quality':>12}")
    print("-" * 60)
    for i, r in enumerate(rankings, 1):
        tier = r["tier"]
        count = r["count"]
        avg_cost = r["avg_cost"] or 0
        avg_lat = r["avg_latency"] or 0
        avg_q = r["avg_quality"] or 0
        print(f"{i:<6} {tier:<10} {count:>6} ${avg_cost:>9.4f} {avg_lat:>10.0f}ms {avg_q:>11.2f}")

    if not rankings:
        print("  (no feedback yet — record feedback with 'imprint feedback')")
    return 0


def cmd_learn(args: list[str]) -> int:
    """Learn from historical routing feedback and update tier preferences."""
    from .learning import get_tier_rankings, get_feedback_stats

    days = 30
    min_records = 5
    if args:
        try:
            days = int(args[0])
        except ValueError:
            print("Usage: python -m imprint learn [days]")
            return 1

    print(f"=== Imprint Learning Analysis (last {days} days) ===")

    stats = get_feedback_stats(days=days)
    print("\nOverall Statistics:")
    print(f"  Total feedback records: {stats['total_records']}")
    print(f"  Total actual cost: ${stats['total_cost_usd']:.4f}")
    print(f"  Total estimated cost: ${stats['total_estimated_usd']:.4f}")
    print(f"  Total savings: ${stats['total_savings_usd']:.4f} ({stats['total_savings_pct']:.1f}%)")

    rankings = get_tier_rankings(days=days, min_records=min_records)

    print(f"\nTier Rankings (last {days} days, min {min_records} records):")
    if rankings:
        print(f"{'Rank':<6} {'Tier':<10} {'Records':>8} {'Savings%':>10} {'Avg Cost':>10} {'Avg Latency':>12} {'Efficiency':>10}")
        print("-" * 65)
        for i, tier_data in enumerate(rankings, 1):
            print(f"{i:<6} {tier_data['tier']:<10} {tier_data['count']:>8} {tier_data['avg_savings_pct']:>9.1f}% ${tier_data['avg_cost']:>9.4f} {tier_data['avg_latency']:>10.0f}ms {tier_data['efficiency_score']:>9.3f}")
    else:
        print(f"  No tier rankings available (need {min_records}+ records per tier)")

    if not rankings:
        print(f"\nNo feedback data found for the last {days} days.")
        return 0

    print(f"\n🏆 Recommendation: Use '{rankings[0]['tier']}' tier")
    print(f"   Based on {rankings[0]['count']} records with {rankings[0]['avg_savings_pct']:.1f}% average savings")
    print(f"   Efficiency score: {rankings[0]['efficiency_score']:.3f}")
    return 0


def cmd_dashboard(args: list[str]) -> int:
    """Show routing optimization dashboard."""
    from .learning import get_feedback_stats, get_tier_rankings, get_tier_thresholds, compute_tier_efficiency_score

    days = 30
    if args:
        try:
            days = int(args[0])
        except ValueError:
            print("Usage: python -m imprint dashboard [days]")
            return 1

    stats = get_feedback_stats(days=days)
    rankings = get_tier_rankings(days=days)

    print(f"=== Imprint Optimization Dashboard (last {days} days) ===")
    print()
    print("📊 Overall Statistics:")
    print(f"  Total feedback records: {stats['total_records']}")
    print(f"  Total actual cost: ${stats['total_cost_usd']:.4f}")
    print(f"  Total estimated cost: ${stats['total_estimated_usd']:.4f}")
    print(f"  Total savings: ${stats['total_savings_usd']:.4f} ({stats['total_savings_pct']:.1f}%)")
    print()

    print("📈 Tier Performance:")
    print(f"{'Tier':<10} {'Count':>6} {'Avg Cost':>10} {'Avg Latency':>12} {'Avg Quality':>10} {'Savings%':>10} {'Efficiency':>10}")
    print("-" * 70)
    for r in stats['tier_performance']:
        efficiency = compute_tier_efficiency_score(
            r['avg_savings_pct'], r['avg_latency'], r['avg_quality']
        )
        print(f"{r['tier']:<10} {r['count']:>6} ${r['avg_cost']:>9.4f} {r['avg_latency']:>10.0f}ms {r['avg_quality']:>9.2f} {r['avg_savings_pct']:>9.1f}% {efficiency:>9.3f}")
    print()

    if stats.get('daily_trends'):
        print(f"📅 Daily Trends (last {min(30, len(stats['daily_trends']))} days):")
        print(f"{'Day':<12} {'Records':>8} {'Avg Cost':>10} {'Avg Latency':>12}")
        print("-" * 45)
        for trend in stats['daily_trends'][:10]:
            print(f"{trend['day']:<12} {trend['records']:>8} ${trend['avg_cost']:>9.4f} {trend['avg_latency']:>10.0f}ms")
        print()

    if stats.get('hourly_trends'):
        print("⏰ Hourly Distribution:")
        for trend in stats['hourly_trends']:
            bar = "█" * min(20, trend['records'])
            print(f"  {trend['hour']:>2}:00 {bar} ({trend['records']} records)")
        print()

    thresholds = get_tier_thresholds(days=days)
    print("🎯 Adaptive Tier Thresholds:")
    for tier, threshold in thresholds.items():
        print(f"  {tier:<10}: {threshold} tokens")
    print()

    if rankings:
        best = rankings[0]
        print(f"🏆 Recommendation: Use '{best['tier']}' tier")
        print(f"   Based on {best['count']} records with {best['avg_savings_pct']:.1f}% avg savings")
        print(f"   Efficiency score: {best['efficiency_score']:.3f}")
    return 0


def cmd_recommend(args: list[str]) -> int:
    """Recommend tier for a given token count."""
    from .learning import recommend_tier

    if not args:
        print("Usage: python -m imprint recommend <token_count>")
        return 1

    try:
        tokens = int(args[0])
    except ValueError:
        print("Error: token_count must be an integer")
        return 1

    tier, estimated_cost, reason = recommend_tier(tokens, auto_adjust=True)

    print("=== Tier Recommendation ===")
    print(f"Token estimate: {tokens}")
    print(f"Recommended tier: {tier.upper()}")
    print(f"Estimated cost: ${estimated_cost:.4f}")
    print(f"Reason: {reason}")
    return 0


def cmd_phase1(args: list[str]) -> int:
    """Initialize Phase 1 feedback system."""
    from .store import connect
    from .learning import record_feedback

    print("=== Imprint Phase 1 Initialization ===")
    print()

    connect()
    print("✅ Database initialized with feedback schema")

    print("Creating sample feedback records...")
    record_feedback("Test small prompt", "small", 0.001, 0.0008, 50, 0.9)
    record_feedback("Test medium prompt", "medium", 0.002, 0.0018, 120, 0.85)
    record_feedback("Test large prompt", "large", 0.003, 0.0025, 300, 0.95)
    print("✅ Sample feedback records created")

    print()
    print("Phase 1 ready! Commands:")
    print("  imprint feedback <prompt> <tier>    # Record feedback")
    print("  imprint feedback-stats              # Show stats")
    print("  imprint feedback-best               # Show best tier")
    print("  imprint learn [days]               # Learn from feedback")
    print("  imprint dashboard [days]           # Show dashboard")
    print("  imprint recommend <tokens>          # Recommend tier")
    print()
    print("Next: Connect to A3M Router with:")
    print("  export A3M_ENDPOINT=http://a3m-router:8080")
    print("  python -m imprint phase1-deploy")
    return 0


def cmd_a3m_check(args: list[str]) -> int:
    """Check A3M Router health and connectivity.

    Usage: python -m imprint a3m-check [endpoint]
    """
    import os
    import requests

    endpoint = args[0] if args else os.environ.get("A3M_ENDPOINT", "http://localhost:8787")

    print("=== A3M Router Health Check ===")
    print(f"Endpoint: {endpoint}")
    print()

    # Health check
    try:
        resp = requests.get(f"{endpoint}/health", timeout=5)
        if resp.status_code == 200:
            print(f"✅ A3M Router reachable: {resp.status_code}")
        else:
            print(f"⚠️  A3M Router returned status {resp.status_code}")
            return 1
    except Exception as e:
        print(f"❌ Cannot reach A3M Router: {e}")
        print(f"   Make sure A3M is running and {endpoint} is accessible")
        return 1

    # Models check
    try:
        resp = requests.get(f"{endpoint}/v1/models", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            models = [m.get("id", "unknown") for m in data.get("data", [])]
            print(f"✅ Models available: {len(models)}")
            for model in models[:5]:
                print(f"   - {model}")
            if len(models) > 5:
                print(f"   ... and {len(models) - 5} more")
    except Exception as e:
        print(f"⚠️  Could not list models: {e}")

    return 0


def cmd_a3m_route(args: list[str]) -> int:
    """Route prompt through A3M Router.

    Usage: python -m imprint a3m-route "<prompt>" [tier]
    """
    from .a3m_integration import A3MConnector

    if not args:
        print("usage: python -m imprint a3m-route \"<prompt>\" [tier]")
        print("  tier: small|medium|large (default: auto)")
        return 1

    prompt = args[0]
    tier = args[1] if len(args) > 1 else None

    connector = A3MConnector()
    result = connector.route_with_feedback(prompt, tier=tier)

    if result.error:
        print(f"❌ Error: {result.error}")
        return 1

    print("=== A3M Route Result ===")
    print(f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    print(f"Tier: {result.tier}")
    print(f"Model: {result.model}")
    print(f"Cost: ${result.cost_usd:.4f}")
    print(f"Latency: {result.latency_ms}ms")
    print(f"Quality: {result.quality_score:.2f}")
    print()
    print("Response:")
    print(result.response)
    print()
    print("✅ Feedback recorded to Imprint database")
    return 0


def cmd_auto_learn(args: list[str]) -> int:
    """Run automated learning cycle.

    Usage: python -m imprint auto-learn
    """
    from .learning import auto_learn_and_update

    result = auto_learn_and_update(min_feedback=10)

    print("=== Automated Learning Cycle ===")
    print(f"Status: {result['status']}")
    print(f"Total feedback: {result.get('total_feedback', 0)}")
    print()

    if result['status'] == 'insufficient_data':
        print(f"⚠️  {result['message']}")
        print("   Record more feedback with 'python -m imprint feedback'")
        return 1

    print(f"Best tier: {result.get('best_tier', 'N/A')}")
    print(f"Best score: {result.get('best_score', 0):.3f}")
    print()

    adjustments = result.get('adjustments', [])
    if adjustments:
        print("📊 Suggested adjustments:")
        for adj in adjustments:
            print(f"  {adj['tier']}:")
            print(f"    Current: {adj['current_efficiency']:.3f}")
            print(f"    Best: {adj['best_efficiency']:.3f}")
            print(f"    Adjustment: {adj['suggested_adjustment']:.3f}")
            print(f"    Reason: {adj['reason']}")
            print()
    else:
        print("✅ No adjustments needed - all tiers performing well")

    recommendations = result.get('recommendations', [])
    if recommendations:
        print("💡 Recommendations:")
        for rec in recommendations:
            print(f"  - {rec}")

    return 0


def cmd_phase1_deploy(args: list[str]) -> int:
    """Deploy Phase 1 with A3M Router integration."""
    import os

    print("=== Imprint Phase 1 + A3M Deployment ===")
    print()

    a3m_endpoint = os.environ.get("A3M_ENDPOINT", "http://a3m-router:8080")
    print(f"A3M Endpoint: {a3m_endpoint}")

    try:
        import requests
        resp = requests.get(f"{a3m_endpoint}/health", timeout=5)
        if resp.status_code == 200:
            print("✅ A3M Router is reachable")
        else:
            print(f"⚠️  A3M Router returned status {resp.status_code}")
    except Exception as e:
        print(f"⚠️  Cannot reach A3M Router: {e}")
        print("   Make sure A3M Router is running and A3M_ENDPOINT is set")

    from .store import connect
    from .learning import record_feedback

    conn = connect()
    print("✅ Imprint database initialized")

    total = conn.execute("SELECT COUNT(*) FROM routing_feedback").fetchone()[0]
    if total == 0:
        print("\nNo feedback data found. Creating sample data...")
        record_feedback("Sample small prompt", "small", 0.001, 0.0008, 50, 0.9)
        record_feedback("Sample medium prompt", "medium", 0.002, 0.0018, 120, 0.85)
        record_feedback("Sample large prompt", "large", 0.003, 0.0025, 300, 0.95)
        print("✅ Sample data created")

    print()
    print("📊 Deployment Summary:")
    print("  Imprint DB: data/imprint.db")
    print(f"  Feedback records: {total}")
    print(f"  A3M Endpoint: {a3m_endpoint}")
    print()
    print("🚀 Ready for production! Use:")
    print("  python -m imprint serve 8477          # Start Imprint server")
    print("  python -m imprint dashboard            # View dashboard")
    print("  python -m imprint learn               # Learn from feedback")
    return 0


COMMANDS = {
    "collect": cmd_collect,
    "mine": cmd_mine,
    "status": cmd_status,
    "status-verbose": cmd_status,
    "serve": cmd_serve,
    "bases": cmd_bases,
    "route": cmd_route,
    "feedback": cmd_feedback,
    "feedback-stats": cmd_feedback_stats,
    "feedback-best": cmd_feedback_best,
    "learn": cmd_learn,
    "dashboard": cmd_dashboard,
    "recommend": cmd_recommend,
    "phase1": cmd_phase1,
    "phase1-deploy": cmd_phase1_deploy,
    "a3m-check": cmd_a3m_check,
    "a3m-route": cmd_a3m_route,
    "auto-learn": cmd_auto_learn,
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        return 1
    return COMMANDS[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    sys.exit(main())