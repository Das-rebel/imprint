# Phase 0 — Validate the Hypothesis (1 week)

**Claim to refute:** real A3M traffic contains task signatures where prompt optimization cuts cost ≥30%.

## Steps
1. Point collector at A3M request logs:
   `python collector/collector.py /path/to/a3m-requests.jsonl`
2. Mine signatures ranked by economics (not frequency):
   `python miner/signatures.py`
3. Take the #1 signature by savings target. Manually write an optimized skill-prompt.
4. Shadow-test optimized prompt vs original on that signature's traffic.
5. **Exit:** ≥30% cost reduction on one signature → proceed to Phase 1. Otherwise → revisit.

## Guardrails active from day one
- PII redaction in collector (emails, cards, keys)
- <100 occurrences/week signatures excluded
- Never train/serve on unverified outputs
