# Imprint

> *Every request leaves an imprint. Eventually, the imprints become instinct.*

**Self-learning task handler for [A3M Router](https://github.com/Das-rebel/a3m-router).**
Imprint watches router traffic, detects repeatable task patterns, and progressively optimizes
them — v1 by evolving context/skill-prompts, v2 by distilling weights into self-managed local
adapters. Recurring tasks trend toward zero marginal cost.

```text
Your AI bill should decay over time.
```

## How it works

```
A3M Router ──telemetry──▶ Collector ──▶ Signature Miner ──▶ Skill Evolver (v1)
   ▲                                          │                    │
   │                                          ▼                    ▼
   └────────── OpenAI-compatible ◀──── Promotion Ladder ◀──── Eval Gate
                endpoint: imprint-local      (shadow→canary→preferred→pinned)
                                                     │
                                     Phase 3+: Distiller (QLoRA via LoRAX)
```

- **Signatures**: clusters of semantically-similar recurring requests
- **Skill Evolver**: per-signature optimized prompts + few-shot packs (v1)
- **Promotion ladder**: `shadow → canary → preferred → pinned` with drift-triggered auto-demote
- **Escalation-on-uncertainty**: low confidence returns `X-Imprint-Escalate: true` → A3M routes live

## Status

🚧 **Phase 0** — validating the core hypothesis on real A3M traffic.
See [PLAN.md](PLAN.md) for the full council-reviewed roadmap.

## Install (when shipped)

```bash
pip install imprint-router
```

## License

MIT © Subhajit Das
