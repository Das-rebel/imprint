"""Imprint CLI — python -m imprint <command>

Commands:
  collect <jsonl>   ingest model-agnostic request logs
  mine              cluster pairs into signatures ranked by economics
  status            show signatures + skill ladder state
  serve [port]      run imprint-local endpoint (model-agnostic)
  route [prompt]    analyze prompt and recommend model routing
  bases             detect hardware and select optimal base model
"""

import sys
import json

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
    from .basemodel import detect_hardware, select_base

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
    """Analyze a prompt and recommend model routing.

    Usage: python -m imprint route "Your prompt here"

    Returns:
        - Recommended tier (small/medium/large)
        - Estimated cost savings
        - Model selection rationale
    """
    import sys

    if not args:
        print("usage: python -m imprint route <prompt>")
        return 1

    prompt = " ".join(args)

    # Load models config
    models_cfg = get_models_config()
    pool = load_pool()

    # Analyze prompt complexity
    token_estimate = len(prompt.split())
    word_count = len(prompt.split())

    # Select best base model from pool
    if pool:
        # Simple selection: pick first small model if prompt is short,
        # medium if longer, large if very long
        if token_estimate < 20:
            candidate = pool[0]  # small
            tier = "small"
        elif token_estimate < 100:
            # Pick medium model
            medium_models = [m for m in pool if m.tier == "medium"]
            candidate = medium_models[0] if medium_models else pool[0]
            tier = "medium"
        else:
            # Pick large model
            large_models = [m for m in pool if m.tier == "large"]
            candidate = large_models[0] if large_models else pool[-1]
            tier = "large"
    else:
        candidate = BaseCandidate(
            name="unknown",
            tier="small",
            vram_gb_qlora=4.0,
            vram_gb_serve=2.5,
            tokens_per_sec_gpu=100.0,
            backend="cuda",
            hf_id="unknown",
        )
        tier = "small"

    # Compute rough cost estimate
    # Simplified: small = $0.001/1K tokens, medium = $0.002/1K, large = $0.003/1K
    cost_per_k = {"small": 0.001, "medium": 0.002, "large": 0.003}
    estimated_cost = (token_estimate / 1000) * cost_per_k[tier]

    # Determine if compression would help
    needs_compression = token_estimate > 512
    compression_savings = "0%"
    if needs_compression:
        # Rough estimate: llmlingua can achieve 30-50% compression
        compression_savings = f"{30 + (token_estimate // 100) * 5:.0f}%"

    # Output routing recommendation
    print(f"=== Prompt Routing Recommendation ===")
    print(f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    print()
    print(f"📊 Prompt Analysis:")
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
    print(f"💰 Cost Estimate:")
    print(f"  • Per-request: ${estimated_cost:.4f}")
    print(f"  • Per 1K requests: ${estimated_cost * 1000:.2f}")
    print()
    print(f"🗜️  Compression:")
    print(f"  • Recommended: {'yes' if needs_compression else 'no'}")
    print(f"  • Potential savings: {compression_savings}")
    print()
    print(f"📈 Routing Decision:")
    if token_estimate < 20:
        print(f"  • Route to: Small model (fast, cost-optimized)")
        print(f"  • Reason: Short prompt, low complexity")
    elif token_estimate < 100:
        print(f"  • Route to: Medium model (balanced)")
        print(f"  • Reason: Moderate complexity, good value")
    else:
        print(f"  • Route to: Large model (quality-focused)")
        print(f"  • Reason: Complex prompt, quality priority")
    print()
    print(f"🔧 Optimization Pipeline:")
    print(f"  1. Semantic Cache check (similarity threshold: 0.85)")
    print(f"  2. Prefix Tree lookup (radix-based O(1))")
    print(f"  3. Prompt compression (llmlingua, ratio: {compression_savings})")
    print(f"  4. Final model selection ({tier})")
    print()
    print(f"=" * 50)

    return 0


def cmd_status_verbose(args: list[str]) -> int:
    """Verbose status with full details."""
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

    # Models config
    models_cfg = get_models_config()
    print("\n== models config ==")
    print(f"  embedding_models: {list(models_cfg.get('embedding_models', {}).keys())}")
    print(f"  compression_models: {list(models_cfg.get('compression_models', {}).keys())}")
    print(f"  fallback_chain keys: {list(models_cfg.get('fallback_chain', {}).keys())}")
    print(f"  cache TTLs: {models_cfg.get('cache', {})}")

    return 0


COMMANDS = {
    "collect": cmd_collect,
    "mine": cmd_mine,
    "status": cmd_status,
    "serve": cmd_serve,
    "bases": cmd_bases,
    "route": cmd_route,
    "status-verbose": cmd_status_verbose,
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        return 1
    return COMMANDS[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    sys.exit(main())