"""Phase 0: capture A3M router traffic and persist normalized request/response pairs.

Consumes A3M request logs (JSONL or webhook) and writes redacted, embedding-ready
records to data/pairs.jsonl for signature mining.
"""
import json, hashlib, re, sys
from pathlib import Path

PII_PATTERNS = [
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "[EMAIL]"),
    (re.compile(r"\b(?:\d[ -]*?){13,16}\b"), "[CARD]"),
    (re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"), "[KEY]"),
]

def redact(text: str) -> str:
    for pat, sub in PII_PATTERNS:
        text = pat.sub(sub, text)
    return text

def normalize(record: dict) -> dict:
    prompt = redact(record.get("prompt", ""))
    response = redact(record.get("response", ""))
    return {
        "id": hashlib.sha1(prompt.encode()).hexdigest()[:16],
        "prompt": prompt,
        "response": response,
        "model": record.get("model"),
        "provider": record.get("provider"),
        "cost_usd": record.get("cost_usd"),
        "latency_ms": record.get("latency_ms"),
        "cache_hit": record.get("cache_hit"),
        "ts": record.get("ts"),
    }

def run(input_path: str, output_path: str = "data/pairs.jsonl") -> int:
    out = Path(output_path); out.parent.mkdir(exist_ok=True)
    n = 0
    with open(input_path) as f, open(out, "a") as w:
        for line in f:
            try:
                rec = normalize(json.loads(line))
                if rec["prompt"]:
                    w.write(json.dumps(rec) + "\n"); n += 1
            except json.JSONDecodeError:
                continue
    print(f"collected {n} pairs -> {out}")
    return n

if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "/dev/stdin")
