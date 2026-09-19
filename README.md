# Imprint

> *Every request leaves an imprint. Eventually, the imprints become instinct.*

**Self-learning cost optimizer for [A3M Router](https://github.com/Das-rebel/a3m-router).**
Imprint watches router traffic, gates its fallback chain with a one-pass
**System One (Jev) decision head**, caches repeatable task patterns, compresses
prompts, and progressively optimizes them — v0.5 with **40-60% token savings**.

```text
Your AI bill should decay over time.
```

![Imprint Banner](https://img.shields.io/badge/v0.5-System%20One%20gating-brightgreen)
![License: MIT](https://img.shields.io/badge/License-MIT-blue)
![Python](https://img.shields.io/badge/Python-3.12%2B-blue)

## Architecture

```mermaid
flowchart LR
    A3M["A3M Router"] -->|telemetry| C["Collector"]
    C --> M["Signature Miner"]
    M --> E["Skill Evolver (v1)"]
    E --> G["Eval Gate"]
    G --> L["Promotion Ladder<br/>shadow→canary→preferred→pinned"]
    L -->|imprint-local endpoint| A3M
    E -.->|"Phase 3+ (plateau)"| D["Distiller (QLoRA / LoRAX)"]
    
    subgraph v0.5 ["Cost Optimization Layer"]
        J["Jev Decision Head<br/>System One · ~0.5ms"]
        SC["Semantic Cache<br/>FAISS + bge-small"]
        PT["Prefix Tree<br/>RadixAttention-style"]
        CP["Prompt Compressor<br/>llmlingua + heuristic"]
        
        J -->|gates| SC
        J -->|gates| PT
        J -->|gates| CP
    end
    
    A3M -->|fallback chain| J
    SC -->|miss| PT
    PT -->|miss| CP
    CP -->|compressed| A3M
```

## The Fallback Chain (now with System One gating)

Every request flows through an **adaptive three-stage pipeline** — now gated
by a **Jev decision head** that predicts the winning stage in a single pass
and skips doomed lookups:

| Stage | Method | Latency | Hit Rate | Savings |
|-------|--------|---------|----------|---------|
| 🥇 **Semantic Cache** | FAISS vector similarity (bge-small) | ~1ms | High | Full |
| 🥈 **Prefix Tree** | O(1) radix lookup | ~0.1µs | Medium | Partial |
| 🥉 **Prompt Compressor** | llmlingua / heuristic / truncation | ~5ms | Low | 30-60% |

**Intelligent routing**: Exact prefix? Prefix Tree. Similar prompt? Semantic Cache.
Long context? Compressor. Best of all worlds, zero config.

## System One Decisions (Jev)

> *The cascade doesn't try every stage anymore — it predicts which one wins.*

Imprint ships a tiny **option-attention decision model** implementing the open
[Jev / System One](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
interface pattern (see [harshatheg/Qwen-2.5-1B-RLCD](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD),
[TheoLeeCJ/SemIf](https://github.com/TheoLeeCJ/SemIf),
[vinnylarouge/jevlike](https://github.com/vinnylarouge/jevlike)).
One forward pass (~0.5ms, numpy-only, 64-dim) answers:

| Question | Type | Output |
|----------|------|--------|
| Which stage wins? | **choice** | `semantic_cache \| prefix_tree \| compressor \| passthrough` + probs |
| Will caching pay off? | **noul** | bool + calibrated p |
| Safe to compress? | **noul** | bool + calibrated p |
| Prompt weight | **score** | complexity ∈ [0,1] |

**What it changes:**

- Fresh prompts **skip both cache lookups** (FAISS + radix) when the head
  predicts they'll miss — the common case for new traffic
- Sub-threshold prompts get compressed when `compress_safe` and complexity
  are both high; risky-looking prompts (code blocks, structured docs) don't
- Every decision is observable: `X-Imprint-Jev-Stage`, `X-Imprint-Jev-Conf`,
  `X-Imprint-Jev-Cacheable`, `X-Imprint-Jev-Compress-Safe`, `X-Imprint-Jev-Ms`
- The head **never answers** — it only gates. Every guarantee of the cascade
  still holds; `IMPRINT_JEV=0` restores the ungated pipeline

**The RLCD loop** — the head retrains from real outcomes:

```bash
python -m imprint jev-train   # uses SQLite telemetry (>=40 rows) or bootstrap
python -m imprint jev-status  # live decision samples
```

`jev-train` learns from `pairs.cache_hit` and token economics — decide,
observe outcome, recalibrate. Day-one installs fall back to bootstrap weights
distilled from the heuristic cascade.

**Cross-compatible with [a3m-router](https://github.com/Das-rebel/adaptive-memory-multi-model-router)**:
same FNV-1a trigram tokenizer, same forward math, same weights JSON schema.
Train in Imprint, serve in a3m's `model="jev-auto"` — or vice versa.

## What's New in v0.5

- ✅ **Jev Decision Head** — System One gating of the fallback chain: one-pass
  stage prediction, cacheable/compress-safe noul decisions, complexity score;
  retrains from telemetry (`jev-train`); ~0.5ms/decision, numpy-only
- ✅ **Semantic Cache** — vector similarity search on LLM prompts using BAAI/bge-small embeddings + FAISS
- ✅ **Prefix Tree** — O(1) radix tree lookup for prefix-based routing (RadixAttention-style)
- ✅ **Prompt Compressor** — adaptive token compression with llmlingua, heuristic, and truncation methods
- ✅ **models.json** — configurable embedding models, compression settings, and fallback chain
- ✅ **GET /health** — health check endpoint for containerized deployment
- ✅ **Docker** — Dockerfile + docker-compose for one-command deployment
- ✅ **Railway** — railway.json for PaaS deployment
- ✅ **164/164 tests passing** — full test coverage across all components

## Model-agnostic

Imprint speaks **every major LLM API shape** — OpenAI ChatCompletion, Anthropic Messages,
Gemini-style contents, and raw completions — auto-detected on both ingest and serving.
It normalizes anything into its internal pair schema (with PII redaction) and responds
in the caller's native format. Works with A3M, LiteLLM, OpenRouter exports, or any gateway.

## CLI

```bash
python -m imprint collect /path/to/requests.jsonl   # ingest (any format)
python -m imprint mine                              # cluster + rank by economics
python -m imprint status                            # signatures + skill ladder
python -m imprint serve 8477                        # imprint-local endpoint
python -m imprint jev-train                         # retrain decision head from telemetry
python -m imprint jev-status                        # inspect live Jev decisions
```

## Quickstart

```bash
git clone https://github.com/Das-rebel/imprint && cd imprint
pip install -e ".[dev]"
make test

# Phase 0 — point at your A3M request log (JSONL):
make collect LOG=/path/to/a3m-requests.jsonl
make mine   # prints signatures ranked by $ savings potential
```

## Local Dev

```bash
# Run tests
make test

# Start local server
python -m imprint serve 8477

# Health check
curl http://localhost:8477/

# Docker
docker-compose up -d
```

## The Bill Decay Curve (goal)

```text
monthly AI cost
 │▇▇▇
 │▇▇▇▇▁▁
 │▇▇▇▇▇▇▇▁▁▁          ← as Imprint promotes skills:
 │▇▇▇▇▇▇▇▇▇▇▀▁▁▁        recurring tasks trend to ~$0 marginal
 └──────────────────▶ time
   wk1  wk2  wk4  wk8
```

Every promoted skill must beat the routed baseline on **both** cost and quality
(enforced in code — see `imprint/ladder.py`).

## Roadmap

| Component | Status | Description |
|-----------|--------|-------------|
| Jev Decision Head | ✅ Complete | System One gating: one-pass stage/cache/compress decisions |
| Semantic Cache | ✅ Complete | FAISS-based vector similarity search |
| Prefix Tree | ✅ Complete | RadixAttention-style O(1) prefix lookup |
| Prompt Compressor | ✅ Complete | Adaptive token reduction (llmlingua) |
| Skill Promotion via Jev | 📋 P1 | Calibrated noul for shadow→canary→preferred |
| LangChain Integration | 📋 Planned | `from langchain.llms import Imprint` |
| Speculative Decoding | 📋 P1 | Medusa-style 2-3x inference speedup |
| Production Feedback | 📋 P1 | Auto-evaluate outputs, route to cheaper models |
| Model Merging | 📋 P2 | T-Switch for storage-efficient routing |

Full roadmap: [PLAN.md](PLAN.md)

## Competitive Landscape

| Project | Approach | Imprint Advantage |
|---------|----------|-------------------|
| GPTCache | Static semantic cache | **Learn + optimize** (not static) |
| SGLang | Inference runtime optimization | **Application-level** (not infra) |
| RadixAttention | KV cache prefix sharing | **Full pipeline** (cache + prefix + compress) |
| LoRAX | Model serving | **Cost optimization** (not serving) |
| LiteLLM | Router + cache | **Self-learning** (not static rules) |

Imprint combines **semantic caching**, **prefix routing**, and **prompt compression**
into a single self-learning pipeline — the first to do so.

## Research Foundation

- **Semantic Caching**: [arXiv:2303.12711](https://arxiv.org/abs/2303.12711) — Semantic Cache for LLM
- **RadixAttention**: [arXiv:2312.07104](https://arxiv.org/abs/2312.07104) — SGLang prefix caching
- **Prompt Compression**: [arXiv:2404.08245](https://arxiv.org/abs/2404.08245) — LLMLingua
- **Speculative Decoding**: [arXiv:2302.01318](https://arxiv.org/abs/2302.01318) — Medusa
- **Route Optimization**: [arXiv:2501.08795](https://arxiv.org/abs/2501.08795) — RouteLLM
- **System One Decisions**: [TypeSafe/Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) —
  non-autoregressive decision models; open reproductions:
  [Qwen-2.5-RLCD](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD) (parallel constrained decoding),
  [SemIf](https://github.com/TheoLeeCJ/SemIf), [jevlike](https://github.com/vinnylarouge/jevlike) (option attention)

## Status: Phase 0

🚧 Validating the core hypothesis on real A3M traffic.
Roadmap + council-reviewed decisions: [PLAN.md](PLAN.md) · [ADRs](docs/adr/)

**Help wanted:** run the collector on your traffic and [share a Phase 0 report](.github/ISSUE_TEMPLATE/phase0-report.md).

## License

MIT © Subhajit Das
