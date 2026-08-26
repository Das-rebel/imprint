"""Imprint serving endpoint — MODEL-AGNOSTIC.

Speaks every format the caller speaks: OpenAI ChatCompletion, Anthropic
Messages, Gemini-style contents, and raw completion. Detects the ingress
format from the request body and responds in-kind.

Escalation contract (any format): response carries `X-Imprint-Escalate: true`
semantics — for JSON bodies an `imprint.escalate: true` field; callers should
retry via their normal router path.
"""

from __future__ import annotations

import json
import sqlite3
from http.server import BaseHTTPRequestHandler, HTTPServer

from .adapters import detect_format, normalize_record, render_response
from .evalgate import EvalGate
from .skills import Skill

DEFAULT_PORT = 8477
CONFIDENCE_THRESHOLD = 0.55


def _find_skill(
    conn: sqlite3.Connection, prompt: str, match_fn
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


class ImprintHandler(BaseHTTPRequestHandler):
    """Stdlib HTTP handler — zero required dependencies beyond Python."""

    db_path: str = "data/imprint.db"
    matcher = staticmethod(lambda prompt, skill: True)  # pluggable matching

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid json"})
            return

        fmt = detect_format(body)
        rec = normalize_record({**body, "response": ""})
        if rec is None:
            self._json(400, {"error": "could not extract prompt"})
            return

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        skill, row = _find_skill(conn, rec["prompt"], self.matcher)

        if skill is None or row["stage"] == "shadow":
            # shadow mode never serves; always escalate
            payload = render_response(
                fmt,
                "",
                rec["model"] or "imprint",
                {
                    "X-Imprint-Signature": row["signature_id"] if row else "",
                    "X-Imprint-Version": skill.prompt_hash if skill else "",
                    "X-Imprint-Escalate": "true",
                },
            )
            self._json(200, payload)
            return

        # serve the rendered skill prompt (caller executes against its own model
        # in Phase 1; in Phase 2+ Imprint serves locally via configured backend)
        gate = EvalGate()
        result = gate.evaluate(rec["prompt"], "", "")  # baseline unknown at serve-time
        escalate = not result.passed or result.agreement < CONFIDENCE_THRESHOLD

        rendered = skill.render(rec["prompt"])
        headers = {
            "X-Imprint-Signature": row["signature_id"],
            "X-Imprint-Version": skill.prompt_hash,
            "X-Imprint-Escalate": "true" if escalate else "false",
        }
        payload = render_response(fmt, rendered, rec["model"] or "imprint", headers)
        payload.setdefault("imprint", {})["escalate"] = escalate
        self._json(200, payload, headers)

    def _json(self, code: int, obj: dict, extra_headers: dict | None = None):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for k, v in (extra_headers or {}).items():
            if v:
                self.send_header(k, str(v))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):  # quiet
        pass


def serve(db_path: str = "data/imprint.db", port: int = DEFAULT_PORT):
    handler = type("BoundHandler", (ImprintHandler,), {"db_path": db_path})
    print(
        f"imprint-local listening on :{port} (model-agnostic: openai|anthropic|gemini|raw)"
    )
    HTTPServer(("127.0.0.1", port), handler).serve_forever()


if __name__ == "__main__":
    import sys

    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    serve(port=port)
