# Imprint Phase 2: A3M Integration + Automated Learning

> *Every request leaves an imprint. Eventually, the imprints become instinct.*

**Phase 2** connects Imprint's feedback loop to A3M Router for production-grade cost optimization.

## Overview

```
Imprint v0.4 (Phase 1) → Phase 2 → A3M Router
       ↓                      ↓
  Semantic Cache         Real-time Feedback
  Prefix Tree          Auto-Learning
  Compressor           Production Routing
```

## Quick Start

### 1. Deploy Phase 2

```bash
# Full Phase 2 deployment
bash deploy_phase1.sh deploy

# Start Imprint server
python -m imprint serve 8477

# Set A3M endpoint
export A3M_ENDPOINT=http://localhost:8787
```

### 2. Record Feedback

```bash
# Simple feedback recording
python -m imprint feedback "Your prompt here" small

# Check feedback stats
python -m imprint feedback-stats

# View dashboard
python -m imprint dashboard
```

### 3. Connect to A3M Router

```python
from imprint.a3m_integration import A3MConnector

connector = A3MConnector()
result = connector.route_with_feedback(
    prompt="Your prompt here",
    tier="small"  # or None for auto
)

print(f"Response: {result.response}")
print(f"Cost: ${result.cost_usd:.4f}")
print(f"Latency: {result.latency_ms}ms")
# Feedback automatically recorded to Imprint
```

## Features

### A3M Router Integration

| Feature | Description |
|---------|-------------|
| **OpenAI-compatible API** | Works with any OpenAI-compatible client |
| **47+ providers** | Automatic failover across providers |
| **Health monitoring** | Real-time provider health tracking |
| **Cost optimization** | Route to cheapest capable model |
| **Feedback loop** | Every response recorded to Imprint |

### Automated Learning

| Feature | Description |
|---------|-------------|
| **Tier ranking** | Learn best tier per workload type |
| **Adaptive thresholds** | Auto-adjust token thresholds |
| **Quality scoring** | Track response quality over time |
| **Cost efficiency** | Optimize for cost vs quality |

### Production Deployment

```bash
# Deploy with A3M
export A3M_ENDPOINT=http://localhost:8787
python -m imprint serve 8477

# Check status
bash deploy_phase1.sh status

# View dashboard
python -m imprint dashboard 30
```

## CLI Commands

### Feedback Collection

```bash
# Record feedback for a prompt
python -m imprint feedback "<prompt>" <tier>

# Show feedback statistics
python -m imprint feedback-stats

# Show best performing tier
python -m imprint feedback-best
```

### Learning & Analysis

```bash
# Learn from historical feedback
python -m imprint learn [days]

# Show optimization dashboard
python -m imprint dashboard [days]

# Recommend tier for token count
python -m imprint recommend <tokens>
```

### A3M Integration

```bash
# Phase 1 initialization
python -m imprint phase1

# Deploy with A3M
python -m imprint phase1-deploy

# Health check A3M
curl http://localhost:8787/health
```

## API Reference

### Python SDK

```python
from imprint.a3m_integration import A3MConnector, Quick_route
from imprint.learning import record_feedback, get_tier_rankings

# Quick route (auto-feedback)
response = Quick_route("Your prompt")

# Full control
connector = A3MConnector(
    endpoint="http://localhost:8787",
    enable_feedback=True
)
result = connector.route_with_feedback(
    prompt="Your prompt",
    tier="small",  # or None for auto
    temperature=0.7
)

# Manual feedback recording
record_feedback(
    prompt="Your prompt",
    recommended_tier="small",
    estimated_cost_usd=0.001,
    actual_cost_usd=0.0008,
    latency_ms=50,
    quality_score=0.9
)

# Get tier rankings
rankings = get_tier_rankings(days=30)
print(f"Best tier: {rankings[0]['tier']}")
```

### Response Format

```python
@dataclass
class A3MRouteResult:
    response: str           # Model response text
    model: str             # Model used
    provider: str          # Provider name
    cost_usd: float       # Actual cost in USD
    latency_ms: int        # Response latency
    quality_score: float   # Quality estimate (0-1)
    tier: str             # Tier used (small/medium/large)
    cached: bool = False   # Whether response was cached
    error: Optional[str] = None  # Error message if failed
```

## Architecture

### Feedback Loop

```
User Prompt → A3M Router → Imprint Feedback → Learning Algorithm
     ↓              ↓              ↓                  ↓
  Response    Cost+Latency   Record to DB      Update Rankings
     ↓              ↓              ↓                  ↓
  Quality     Tier Selection  Feedback Stats    Tier Preferences
```

### Tier Selection

| Tier | Model Size | Latency | Cost | Best For |
|------|------------|---------|------|----------|
| **small** | 3B params | ~50ms | $0.001/1K | Simple tasks, code completion |
| **medium** | 7B params | ~150ms | $0.002/1K | Balanced tasks,写作 |
| **large** | 24B+ params | ~400ms | $0.003/1K | Complex reasoning |

### Learning Algorithm

```python
# Efficiency score = weighted combination of metrics
efficiency = (
    0.5 * savings_score +      # Cost savings
    0.3 * latency_score +       # Response speed
    0.2 * quality_score        # Output quality
)
```

## Dashboard Metrics

| Metric | Description |
|---------|-------------|
| **Total Records** | Number of feedback entries |
| **Cost Savings** | USD saved vs estimated cost |
| **Tier Performance** | Efficiency per tier |
| **Daily Trends** | Cost/savings over time |
| **Hourly Distribution** | Request patterns |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `A3M_ENDPOINT` | `http://localhost:8787` | A3M Router endpoint |
| `IMPRINT_DB_PATH` | `data/imprint.db` | SQLite database path |
| `IMPRINT_FEEDBACK_AUTO` | `true` | Auto-record feedback |
| `LEARNING_DAYS` | `30` | Days for learning window |
| `LEARNING_MIN_RECORDS` | `5` | Min records for ranking |

## Troubleshooting

### A3M Router Not Reachable

```bash
# Check A3M health
curl http://localhost:8787/health

# Set correct endpoint
export A3M_ENDPOINT=http://your-a3m:8787

# Verify connection
python -c "from imprint.a3m_integration import a3m_health_check; print(a3m_health_check())"
```

### No Feedback Data

```bash
# Initialize with sample data
python -m imprint phase1

# Record sample feedback
python -m imprint feedback "Test prompt" small

# Check database
sqlite3 data/imprint.db "SELECT * FROM routing_feedback LIMIT 5;"
```

### Learning Not Updating

```bash
# Check feedback count
python -m imprint feedback-stats

# Force learning cycle
python -m imprint learn 7  # Last 7 days

# Check tier rankings
python -m imprint feedback-best
```

## Examples

### Example 1: Simple Routing

```python
from imprint.a3m_integration import Quick_route

response = Quick_route("Write a hello world in Python")
print(response)
# Feedback automatically recorded
```

### Example 2: Tier-Specific Routing

```python
from imprint.a3m_integration import A3MConnector

connector = A3MConnector()

# Force small tier for simple task
result = connector.route_with_feedback(
    prompt="What is 2+2?",
    tier="small"  # Force small model
)
print(f"Response: {result.response}")
print(f"Tier: {result.tier}")
```

### Example 3: Batch Processing

```python
from imprint.a3m_integration import A3MConnector

connector = A3MConnector()
prompts = [
    "Hello world in Python",
    "Binary search algorithm",
    "Explain quantum computing",
]

results = connector.batch_route(prompts)
for prompt, result in zip(prompts, results):
    print(f"Q: {prompt[:30]}...")
    print(f"A: {result.response[:50]}...")
    print(f"Cost: ${result.cost_usd:.4f}")
    print()
```

### Example 4: Learning Analysis

```python
from imprint.learning import get_feedback_stats, get_tier_rankings

# Get overall stats
stats = get_feedback_stats(days=30)
print(f"Total records: {stats['total_records']}")
print(f"Savings: ${stats['total_savings_usd']:.4f}")

# Get tier rankings
rankings = get_tier_rankings(days=30)
for tier in rankings:
    print(f"{tier['tier']}: {tier['efficiency_score']:.3f}")
```

## Performance Benchmarks

| Operation | Latency | Throughput |
|-----------|---------|------------|
| Semantic Cache Lookup | ~1ms | 1.3M lookups/sec |
| Prefix Tree Lookup | ~0.1µs | 1.3M lookups/sec |
| Feedback Recording | ~5ms | 200 records/sec |
| Learning Update | ~50ms | 20 updates/sec |
| A3M Route (cached) | ~50ms | 20 req/sec |
| A3M Route (uncached) | ~200ms | 5 req/sec |

## Next Steps

1. **Connect to production A3M** → Real-time feedback loop
2. **Add quality scoring** → User ratings for responses
3. **Implement adaptive thresholds** → Learn from feedback
4. **Multi-provider routing** → Distribute across providers
5. **Set up monitoring** → Alert on anomalies

## License

MIT © Subhajit Das
