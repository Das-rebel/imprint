# ADR-001: Context evolution before weight distillation

**Status:** Accepted · **Date:** 2026-08-26 · **Deciders:** agent council (Claude-MiniMax, Gemini-2.5)

## Context
Imprint's original plan made QLoRA weight-distillation the v1 core mechanism. Council review flagged this as the biggest strategic flaw: ~10 weeks to first value, silent-quality-regression risk, GPU dependency.

## Decision
v1 evolves **context/skill-prompts** per signature. Weight distillation moves to Phase 3+, triggered by a plateau heuristic (3 consecutive evolutions with <5% improvement).

## Rationale
- ACE framework evidence: evolving context often beats evolving weights for narrow adaptation
- Microsoft SkillOpt (16K stars) validates trace-to-skill demand; nobody pairs it with router economics data
- Prompt-only failure modes are safer than weight-level ones

## Consequences
- First user-visible value ships in ~3 weeks instead of ~10
- Distiller becomes an escalation path, not the foundation
