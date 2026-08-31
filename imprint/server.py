"""Imprint serving endpoint — MODEL-AGNOSTIC.

Speaks every format the caller speaks: OpenAI ChatCompletion, Anthropic
Messages, Gemini-style contents, and raw completion. Detects the ingress
format from the request body and responds in-kind.

Enhanced pipeline (v4):
    Request → Semantic Cache (embed + FAISS search) → Cache hit?
                ↓ miss
    Request → Prefix Tree (longest prefix match) → Hit?
                ↓ miss
    Request → Prompt Compressor (if > threshold tokens) → Compressed prompt
                ↓
    Skill lookup → Eval gate → Route

Escalation contract (any format): response carries `X-Imprint-Escalate: true`
semantics — for JSON bodies an `imprint.escalate: true` field; callers should
retry via their normal router path.
"""

from __future__ import annotations

import json
import os
import sqlite3
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable

from .adapters import detect_format, normalize_record, render_response
from .compressor import AdaptiveCompressor, CompressionResult
from .evalgate import EvalGate
from .prefix_tree import PrefixTree
from .semantic_cache import SemanticCache
from .skills import Skill

DEFAULT_PORT = 8477
CONFIDENCE_THRESHOLD = 0.55
COMPRESSION_THRESHOLD_TOKENS = 1024


def _find_skill(
    conn: sqlite3.Connection, prompt: str, match_fn: Callable[[str, Skill], bool]
) -> tuple[Skill | None, sqlite3.Row | None]:
    """Find the highest-stage active skill whose template matches the signature."""
    best = None
    stage_rank = {"shadow": 0, "canary": 1, "preferred": 2, "pinned": 3}
    for row in conn.execute(
        "SELECT * FROM skills ORDER BY CASE stage"
        " WHEN 'pinned' THEN 0 WHEN 'preferred' THEN 1"
        " WHEN 'canary' THEN 2 ELSE 3 END"
    ):
        skill = Skill(
            signature_id=row["signature_id"],
            template=row["template"],
            few_shot_pack=json.loads(row["few_shot_json"] or "[]"),
        )
        if match_fn(prompt, skill):
            rank = stage_rank.get(row["stage"], -1)
            if best is None or rank > best[2]:
                best = (skill, row, rank)
    return (best[0], best[1]) if best else (None, None)


class ImprintServer:
    """Server with integrated semantic cache, prefix tree, and compressor.

    Usage:
        server = ImprintServer("data/imprint.db", port=8477)
        server.serve()
    """

    def __init__(
        self,
        db_path: str = "data/imprint.db",
        cache_dir: str = "data/imprint_cache",
        matcher: Callable[[str, Skill], bool] | None = None,
        compression_threshold: int = COMPRESSION_THRESHOLD_TOKENS,
    ):
        self.db_path = db_path
        self.cache_dir = cache_dir
        self.matcher = matcher or (lambda prompt, skill: True)
        self.compression_threshold = compression_threshold

        # Initialize components
        self.semantic_cache = SemanticCache(cache_dir)
        self.prefix_tree = PrefixTree()
        self.compressor = AdaptiveCompressor()

        # Load saved state
        prefix_path = os.path.join(cache_dir, "prefix_tree.json")
        if os.path.exists(prefix_path):
            self.prefix_tree.load(prefix_path)

    def handle_request(self, body: dict) -> tuple[dict, dict[str, str]]:
        """Process a request through the full pipeline.

        Returns:
            Tuple of (response_dict, headers_dict)
        """
        fmt = detect_format(body)
        rec = normalize_record({**body, "response": ""})
        if rec is None:
            return ({"error": "could not extract prompt"}, {})

        prompt = rec["prompt"]
        model = rec["model"] or "imprint"

        # ── 1. Semantic Cache Lookup ──────────────────────────────────
        cache_hit = self.semantic_cache.get(prompt)
        if cache_hit and cache_hit.response:
            self.semantic_cache.clear_expired()  # Lazy cleanup
            return (
                render_response(
                    fmt,
                    cache_hit.response,
                    model,
                    {
                        "X-Imprint-Signature": cache_hit.signature_id or "",
                        "X-Imprint-Version": "cache-v1",
                        "X-Imprint-Cache-Hit": "semantic",
                        "X-Imprint-Escalate": "false",
                        "X-Imprint-Compression-Ratio": "1.0",
                    },
                ),
                {},
            )

        # ── 2. Prefix Tree Lookup ─────────────────────────────────────
        prefix_hit = self.prefix_tree.lookup(prompt)
        if prefix_hit and prefix_hit.get("response"):
            return (
                render_response(
                    fmt,
                    _safe_response(prefix_hit.get("response", "")),
                    model,
                    {
                        "X-Imprint-Signature": prefix_hit.get("signature_id", "") or "",
                        "X-Imprint-Version": "prefix-v1",
                        "X-Imprint-Cache-Hit": "prefix",
                        "X-Imprint-Escalate": "false",
                        "X-Imprint-Compression-Ratio": "1.0",
                    },
                ),
                {},
            )

        # ── 3. Prompt Compression ─────────────────────────────────────
        compression_ratio = 1.0
        compressed_prompt = prompt
        needs_compression = (
            self.compressor.should_compress(prompt, self.compression_threshold)
        )
        if needs_compression:
            compressed, ratio = self.compressor.compress(
                prompt,
                max_tokens=self.compression_threshold,
            )
            compressed_prompt = compressed
            compression_ratio = ratio

        # ── 4. Skill Lookup ───────────────────────────────────────────
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        skill, row = _find_skill(conn, compressed_prompt, self.matcher)

        if skill is None or (row is not None and row["stage"] == "shadow"):
            # No skill or shadow-only → escalate
            payload = render_response(
                fmt,
                "",
                model,
                {
                    "X-Imprint-Signature": row["signature_id"] if row else "",
                    "X-Imprint-Version": skill.prompt_hash if skill else "",
                    "X-Imprint-Cache-Hit": "skill",
                    "X-Imprint-Escalate": "true",
                    "X-Imprint-Compression-Ratio": f"{compression_ratio:.2f}",
                },
            )
            return (payload, {})

        # ── 5. Eval gate + serve ──────────────────────────────────────
        gate = EvalGate()
        result = gate.evaluate(prompt, "", "")  # baseline unknown at serve-time
        escalate = not result.passed or result.agreement < CONFIDENCE_THRESHOLD

        rendered = skill.render(compressed_prompt)
        assert row is not None
        headers = {
            "X-Imprint-Signature": row["signature_id"],
            "X-Imprint-Version": skill.prompt_hash,
            "X-Imprint-Cache-Hit": "skill",
            "X-Imprint-Escalate": "true" if escalate else "false",
            "X-Imprint-Compression-Ratio": f"{compression_ratio:.2f}",
        }
        payload = render_response(fmt, rendered, model, headers)
        payload.setdefault("imprint", {})["escalate"] = escalate
        return (payload, {})

    def save_state(self) -> None:
        """Save cache and prefix tree state."""
        os.makedirs(self.cache_dir, exist_ok=True)
        prefix_path = os.path.join(self.cache_dir, "prefix_tree.json")
        self.prefix_tree.save(prefix_path)
        self.semantic_cache._save_index()

    def stats(self) -> dict:
        """Return combined statistics."""
        return {
            "semantic_cache": self.semantic_cache.stats(),
            "prefix_tree": self.prefix_tree.stats(),
            "compressor": self.compressor.stats(),
        }


def _safe_response(value: Any) -> str:
    """Safely convert a value to a response string."""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value) if value else ""


class ImprintHandler(BaseHTTPRequestHandler):
    """HTTP handler wrapping the ImprintServer pipeline."""

    server_instance: ImprintServer = None  # Set by factory

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            self._json(200, {
                "status": "ok",
                "service": "imprint",
                "version": "0.4.0",
                "components": ["semantic_cache", "prefix_tree", "compressor", "skill_evolution"],
            })
        elif self.path == "/cache/stats":
            self._json(200, self.server_instance.stats() if self.server_instance else {"error": "not ready"})
        elif self.path == "/health":
            self._json(200, {"status": "healthy"})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid json"})
            return

        if self.path == "/cache/clear":
            count = self.server_instance.semantic_cache.clear() if self.server_instance else 0
            self._json(200, {"cleared": count})
            return

        if self.path == "/compress":
            result = self.server_instance.compressor.compress(body.get("prompt", "")) if self.server_instance else None
            if result:
                self._json(200, {
                    "original": result.original,
                    "compressed": result.compressed,
                    "ratio": result.ratio,
                    "method": result.method,
                    "quality_score": result.quality_score,
                    "original_tokens": result.original_tokens,
                    "compressed_tokens": result.compressed_tokens,
                })
            else:
                self._json(400, {"error": "server not ready"})
            return

        if self.server_instance is None:
            self._json(500, {"error": "server not initialized"})
            return

        payload, extra_headers = self.server_instance.handle_request(body)
        self._json(200, payload, extra_headers)

    def _json(
        self,
        code: int,
        obj: dict[str, Any],
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for k, v in (extra_headers or {}).items():
            if v:
                self.send_header(k, str(v))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: Any) -> None:  # quiet
        pass


def make_handler(server_instance: ImprintServer):
    """Create a handler bound to a specific server instance."""
    return type("BoundHandler", (ImprintHandler,), {"server_instance": server_instance})


def serve(
    db_path: str = "data/imprint.db",
    cache_dir: str = "data/imprint_cache",
    port: int = DEFAULT_PORT,
    matcher: Callable[[str, Skill], bool] | None = None,
) -> None:
    """Start the Imprint server."""
    os.makedirs(db_path.rsplit("/", 1)[0] if "/" in db_path else ".", exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    server = ImprintServer(
        db_path=db_path,
        cache_dir=cache_dir,
        matcher=matcher,
    )
    handler = make_handler(server)

    print(
        f"imprint v0.4.0 listening on 0.0.0.0:{port} "
        f"(model-agnostic | semantic-cache | prefix-tree | compressor)"
    )
    try:
        HTTPServer(("", port), handler).serve_forever()
    except KeyboardInterrupt:
        server.save_state()
        print("\nSaved state. Bye.")


if __name__ == "__main__":
    import sys

    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    db_path = sys.argv[2] if len(sys.argv) > 2 else "data/imprint.db"
    cache_dir = sys.argv[3] if len(sys.argv) > 3 else "data/imprint_cache"
    serve(db_path=db_path, cache_dir=cache_dir, port=port)
