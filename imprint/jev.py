"""Imprint Jev decision head — System One decisions for the fallback chain.

Implements the open Jev interface pattern (TypeSafe "System One"; see
harshatheg/Qwen-2.5-1B-RLCD, TheoLeeCJ/SemIf, vinnylarouge/jevlike) as a
tiny option-attention model. One forward pass answers:

  - stage choice:   semantic_cache | prefix_tree | compressor | passthrough
  - cacheable noul: will a cache/prefix hit pay off? (bool + calibrated p)
  - compress_safe noul: is this prompt safe to compress? (bool + p)
  - complexity score: normalized prompt weight in [0,1]

The engine is byte-compatible with a3m-router's TypeScript engine
(src/routing/jev/optionAttention.ts): identical FNV-1a trigram hashing and
forward math, same weights JSON schema. Train in Imprint, serve in a3m —
or vice versa.

Two training modes:
  1. bootstrap  — distill Imprint's current heuristic cascade (day-one use)
  2. telemetry  — the RLCD loop: learn from real outcomes in SQLite
                  (`pairs.cache_hit`, `routing_feedback.quality_score`)
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Tokenizer — must stay byte-identical to a3m-router optionAttention.ts
# ---------------------------------------------------------------------------

VOCAB = 2048
CTX_CAP = 160

WEIGHTS_PATH = Path(__file__).parent / "weights" / "jev-imprint-weights.json"

STAGES = ("semantic_cache", "prefix_tree", "compressor", "passthrough")
COMPRESS_RISKY_MARKERS = (
    "```", "def ", "class ", "function ", "import ", "return ",
    "\n##", "\n###", "\n1.", "\n2.", "\n- ", "\n* ",
)


def trigram_hash(b0: int, b1: int, b2: int) -> int:
    h = 0x811C9DC5
    for c in (b0 & 0xFF, b1 & 0xFF, b2 & 0xFF):
        h = ((h ^ c) * 0x01000193) & 0xFFFFFFFF
    return h % VOCAB


def tokenize(text: str, cap: int = CTX_CAP) -> list[int]:
    s = " " + text + " "
    out: list[int] = []
    for i in range(len(s) - 2):
        out.append(trigram_hash(ord(s[i]), ord(s[i + 1]), ord(s[i + 2])))
        if len(out) >= cap:
            break
    return out or [trigram_hash(32, 32, 32)]


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(e.sum(axis=axis, keepdims=True), 1e-9)


# ---------------------------------------------------------------------------
# Decision model
# ---------------------------------------------------------------------------

@dataclass
class JevDecision:
    stage: str
    stage_probs: dict[str, float]
    cacheable: bool
    cacheable_prob: float
    compress_safe: bool
    compress_safe_prob: float
    complexity: float
    elapsed_ms: float
    backend: str = "local-weights"

    @property
    def confidence(self) -> float:
        return max(self.stage_probs.values())


class JevHead:
    """Option-attention decision head (trainable, numpy-only)."""

    def __init__(self, dim: int = 64, seed: int = 7):
        rng = np.random.default_rng(seed)
        self.dim = dim
        self.emb = rng.normal(0, 0.40, (VOCAB, dim))
        self.wq = rng.normal(0, 0.40, (dim, dim))
        self.bq = np.zeros(dim)
        self.w = rng.normal(0, 1.2, dim)
        self.b = np.zeros(1)
        self.ws = rng.normal(0, 0.30, dim)
        self.bs = np.array([0.0])
        self.temperature = 1.0

    # -- persistence ------------------------------------------------------

    def save(self, path: str | os.PathLike) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "emb": np.round(self.emb, 4).tolist(),
            "wq": np.round(self.wq, 4).tolist(),
            "bq": np.round(self.bq, 4).tolist(),
            "w": np.round(self.w, 4).tolist(),
            "b": np.round(self.b, 4).tolist(),
            "ws": np.round(self.ws, 4).tolist(),
            "bs": np.round(self.bs, 4).tolist(),
            "temperature": round(float(self.temperature), 4),
            "dim": self.dim,
        }
        Path(path).write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: str | os.PathLike) -> "JevHead":
        raw = json.loads(Path(path).read_text())
        head = cls(dim=raw["dim"])
        head.emb = np.array(raw["emb"], dtype=float)
        head.wq = np.array(raw["wq"], dtype=float)
        head.bq = np.array(raw["bq"], dtype=float)
        head.w = np.array(raw["w"], dtype=float)
        head.b = np.array(raw["b"], dtype=float)
        head.ws = np.array(raw["ws"], dtype=float)
        head.bs = np.array(raw["bs"], dtype=float)
        head.temperature = float(raw["temperature"])
        return head

    # -- forward ------------------------------------------------------------

    def _option_scores(self, ctx_tokens: np.ndarray, option_texts: list[str]):
        E = self.emb[ctx_tokens]                                   # (T, D)
        cache = {t: None for t in set(option_texts)}

        def q_for(text: str) -> np.ndarray:
            if cache[text] is None:
                h = self.emb[np.array(tokenize(text))].mean(axis=0)
                cache[text] = np.tanh(self.wq @ h + self.bq)
            return cache[text]

        scale = 1.0 / math.sqrt(self.dim)
        scores = []
        for text in option_texts:
            q = q_for(text)
            att = softmax((E @ q) * scale)
            c = att @ E
            scores.append(float(self.w @ (q * c) + self.b[0]))
        return np.array(scores)

    def predict(self, prompt: str) -> JevDecision:
        t0 = time.perf_counter()
        ctx = np.array(tokenize(prompt))
        stage_scores = self._option_scores(ctx, list(STAGES))
        stage_probs = softmax(stage_scores / self.temperature)
        stage = STAGES[int(np.argmax(stage_probs))]

        noul = self._option_scores(ctx, ["cache payoff — yes", "cache payoff — no"])
        cp = softmax(noul)
        cacheable, cacheable_prob = bool(cp[0] >= 0.5), float(max(cp[0], 1 - cp[0]))

        cs = self._option_scores(ctx, ["safe to compress — yes", "safe to compress — no"])
        csp = softmax(cs)
        compress_safe = bool(csp[0] >= 0.5)
        compress_safe_prob = float(max(csp[0], 1 - csp[0]))

        p = self.emb[ctx].mean(axis=0)
        complexity = float(1 / (1 + math.exp(-float(self.ws @ p + self.bs[0]))))

        return JevDecision(
            stage=stage,
            stage_probs={s: float(p_) for s, p_ in zip(STAGES, stage_probs)},
            cacheable=cacheable,
            cacheable_prob=cacheable_prob,
            compress_safe=compress_safe,
            compress_safe_prob=compress_safe_prob,
            complexity=complexity,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )

    # -- training -----------------------------------------------------------

    def fit(
        self,
        rows: list[dict],
        epochs: int = 200,
        lr: float = 0.5,
        verbose: bool = True,
    ) -> dict:
        """rows: {context, stage, cacheable, compress_safe, complexity}."""
        rng = np.random.default_rng(42)
        D, lr_c, lr_s = self.dim, lr, 0.20
        scale = 1.0 / math.sqrt(D)
        y_stage = np.array([STAGES.index(r["stage"]) for r in rows])
        y_cache = np.array([1.0 if r["cacheable"] else 0.0 for r in rows])
        y_comp = np.array([1.0 if r["compress_safe"] else 0.0 for r in rows])
        y_cplx = np.array([r["complexity"] for r in rows])
        ctx_tok = [np.array(tokenize(r["context"])) for r in rows]
        stage_opt = [np.array(tokenize(s)) for s in STAGES]
        nq_c = [np.array(tokenize("cache payoff — yes")), np.array(tokenize("cache payoff — no"))]
        nq_s = [np.array(tokenize("safe to compress — yes")), np.array(tokenize("safe to compress — no"))]
        n = len(rows)
        if n == 0:
            raise ValueError("no rows")

        def forward(i, opt_tokens):
            E = self.emb[ctx_tok[i]]
            Ho = np.stack([self.emb[t].mean(axis=0) for t in opt_tokens])
            Q = np.tanh(Ho @ self.wq.T + self.bq)
            A = softmax((Q @ E.T) * scale, axis=1)
            C = A @ E
            scores = (Q * C) @ self.w + self.b[0]
            return E, Ho, Q, A, C, scores

        for epoch in range(epochs):
            ce = correct = 0
            order = rng.permutation(n)
            for i in order:
                E, Ho, Q, A, C, scores = forward(i, stage_opt)
                probs = softmax(scores / self.temperature)
                ce -= math.log(max(probs[y_stage[i]], 1e-9))
                correct += int(np.argmax(probs) == y_stage[i])

                ds = probs.copy()
                ds[y_stage[i]] -= 1.0
                ds /= self.temperature
                self.w -= lr_c * ((Q * C) * ds[:, None]).sum(0)
                self.b -= lr_c * 0.10 * ds.sum()
                dQ = ds[:, None] * (self.w * C) * (1 - Q * Q)
                self.wq -= lr_c * 0.10 * dQ.T @ Ho
                self.bq -= lr_c * 0.10 * dQ.sum(0)
                dHo = dQ @ self.wq
                flat = np.concatenate(stage_opt)
                rep = np.concatenate([np.full(len(t), m) for m, t in enumerate(stage_opt)])
                lens = np.array([len(t) for t in stage_opt], dtype=float)
                np.add.at(self.emb, flat, -lr_c * 0.10 * dHo[rep] / lens[rep, None])
                dC = ds[:, None] * (self.w * Q)
                np.add.at(self.emb, ctx_tok[i], -lr_c * 0.10 * (A.T @ dC))

                # score head
                p = E.mean(axis=0)
                sig = 1 / (1 + math.exp(-float(self.ws @ p + self.bs[0])))
                gz = (sig - y_cplx[i]) * sig * (1 - sig)
                self.ws -= lr_s * gz * p
                self.bs -= lr_s * 0.10 * gz

                # noul heads
                for opt_tokens, yv in ((nq_c, y_cache[i]), (nq_s, y_comp[i])):
                    Hn = np.stack([self.emb[t].mean(axis=0) for t in opt_tokens])
                    Qn = np.tanh(Hn @ self.wq.T + self.bq)
                    An = softmax((Qn @ E.T) * scale, axis=1)
                    Cn = An @ E
                    ns = softmax((Qn * Cn) @ self.w + self.b[0])
                    yn = 0 if yv > 0.5 else 1
                    gn = ns.copy()
                    gn[yn] -= 1.0
                    dQn = gn[:, None] * (self.w * Cn) * (1 - Qn * Qn)
                    self.wq -= lr_c * 0.05 * dQn.T @ Hn
                    self.bq -= lr_c * 0.05 * dQn.sum(0)

            if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
                print(f"epoch {epoch:4d}  stageCE={ce/n:.4f} (chance={math.log(4):.3f})  top1={correct/n:.3f}")

        # temperature calibration → mean top-prob ≈ accuracy (capped at 0.95)
        acc = correct / n
        target = min(max(acc, 0.5), 0.95)
        lo, hi = 0.1, 8.0
        for _ in range(40):
            mid = (lo + hi) / 2
            ps = [softmax(forward(i, stage_opt)[5] / mid) for i in range(n)]
            mean_top = float(np.mean([p.max() for p in ps]))
            lo, hi = (mid, hi) if mean_top > target else (lo, mid)
        self.temperature = (lo + hi) / 2
        return {"rows": n, "top1": acc, "temperature": float(self.temperature)}


# ---------------------------------------------------------------------------
# Training corpora
# ---------------------------------------------------------------------------

def _token_count_approx(text: str) -> int:
    return max(1, len(text) // 4)


def bootstrap_rows() -> list[dict]:
    """Distill the current heuristic cascade — day-one weights."""
    rows: list[dict] = []

    def add(ctx: str, stage: str):
        rows.append({
            "context": ctx,
            "stage": stage,
            "cacheable": stage in ("semantic_cache", "prefix_tree"),
            "compress_safe": not any(m in ctx.lower() for m in COMPRESS_RISKY_MARKERS),
            "complexity": min(1.0, _token_count_approx(ctx) / 1200),
        })

    # semantic-cache class: recurring paraphrased prompts
    recurring = [
        "what is the status of the deployment",
        "check the status of my deployment please",
        "give me deployment status update",
        "summarize today's standup notes",
        "summarize the standup notes from today",
        "can you summarize today's standup",
    ]
    for _ in range(3):
        for r in recurring:
            add(r, "semantic_cache")

    # prefix-tree class: shared long prefixes
    prefixes = [
        "You are a senior kotlin architect. Given the following architecture decision record, evaluate:",
        "INSTRUCTIONS FOR SUPPORT TRIAGE: classify the ticket, then route it. Ticket:",
        "According to the DPDP Act 2023 section 4, analyze this data request:",
    ]
    for p in prefixes:
        for suffix in (" case one", " case two", " another matter", " the next item"):
            add(p + suffix, "prefix_tree")

    # compressor class: long prompts, compressible
    long_safe = (
        "Explain the history of distributed systems in detail. " * 30,
        "Write a detailed summary of the quarterly business review covering all regions. " * 25,
        "Provide an overview of the European regulatory landscape for fintech. " * 28,
    )
    for t in long_safe:
        add(t.strip(), "compressor")

    # passthrough class: fresh short unique prompts
    fresh = [
        "invent a name for a mars rover",
        "what is the airspeed velocity of an unladen swallow",
        "draft an apology letter to my neighbor about the fence",
        "explain quantum entanglement to a five year old",
        "brainstorm uses for surplus concrete",
        "translate this contract clause into plain english: indemnitee shall hold harmless",
        "rate this pun out of ten: i used to hate math, then i realized decimals have a point",
    ]
    for f in fresh:
        add(f, "passthrough")

    # long-but-risky (code/structured) → still compressor but compress_safe=false
    risky = "```python\ndef fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)\n```\n\nReview this code. " * 6
    add(risky, "compressor")

    return rows


def telemetry_rows(db_path: str = "data/imprint.db", limit: int = 5000) -> list[dict]:
    """RLCD loop: build training rows from real outcomes in SQLite."""
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows: list[dict] = []
    try:
        cur = conn.execute(
            "SELECT prompt, cache_hit, latency_ms, cost_usd FROM pairs "
            "WHERE prompt IS NOT NULL AND prompt != '' ORDER BY ts DESC LIMIT ?",
            (limit,),
        )
        for r in cur:
            prompt = r["prompt"]
            hit = bool(r["cache_hit"])
            tok = _token_count_approx(prompt)
            if hit:
                stage = "semantic_cache"
            elif tok > 512:
                stage = "compressor"
            else:
                stage = "passthrough"
            rows.append({
                "context": prompt,
                "stage": stage,
                "cacheable": hit,
                "compress_safe": not any(m in prompt.lower() for m in COMPRESS_RISKY_MARKERS),
                "complexity": min(1.0, tok / 1200),
            })
    finally:
        conn.close()
    return rows


# ---------------------------------------------------------------------------
# Default head
# ---------------------------------------------------------------------------

_default: JevHead | None | bool | None = None  # None=unset, True/False=loaded/failed


def get_default_head() -> JevHead | None:
    """Load shipped weights once; None if absent (features degrade gracefully)."""
    global _default
    if _default is None:
        if WEIGHTS_PATH.exists():
            try:
                _default = JevHead.load(WEIGHTS_PATH)
            except Exception:
                _default = False
        else:
            _default = False
    return _default if _default else None


def reset_default_head() -> None:
    global _default
    _default = None
