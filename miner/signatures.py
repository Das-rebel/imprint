"""Phase 0: cluster collected pairs into candidate task signatures.

Uses cheap embeddings + HDBSCAN-style clustering to find recurring patterns,
then scores them by volume x avg-cost (the economics-first ranking).
Requires: pip install sentence-transformers numpy
"""
import json
from collections import defaultdict
from pathlib import Path

def load_pairs(path="data/pairs.jsonl"):
    return [json.loads(l) for l in open(path)] if Path(path).exists() else []

def rank_by_economics(pairs, min_volume=5):
    """Group by exact-prefix heuristic first (Phase 0 has no embeddings dep);
    rank clusters by total monthly cost — the Imprint differentiator."""
    groups = defaultdict(list)
    for p in pairs:
        key = " ".join(p["prompt"].split()[:8])  # crude prefix key; embeddings come Phase 1
        groups[key].append(p)
    rows = []
    for key, items in groups.items():
        if len(items) < min_volume:
            continue
        rows.append({
            "signature_prefix": key,
            "volume": len(items),
            "avg_cost_usd": sum(i["cost_usd"] or 0 for i in items) / len(items),
            "est_monthly_savings_target": sum(i["cost_usd"] or 0 for i in items),
        })
    return sorted(rows, key=lambda r: -r["est_monthly_savings_target"])

if __name__ == "__main__":
    for r in rank_by_economics(load_pairs())[:10]:
        print(f"{r['volume']:>6}x  ${r['avg_cost_usd']:.4f}/req  target=${r['est_monthly_savings_target']:.2f}  {r['signature_prefix'][:60]}")
