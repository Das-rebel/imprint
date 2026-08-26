from imprint.adapters import detect_format, normalize_record, render_response


def test_detect_openai():
    rec = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    assert detect_format(rec) == "openai"


def test_detect_anthropic():
    rec = {
        "model": "claude-x",
        "system": "be nice",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
    }
    assert detect_format(rec) == "anthropic"


def test_detect_gemini():
    rec = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}
    assert detect_format(rec) == "gemini"


def test_normalize_openai_messages_flatten():
    rec = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You help."},
            {"role": "user", "content": "Summarize this"},
        ],
        "response_body": {"content": "Sure thing"},
        "usage": {"total_cost": 0.01},
    }
    pair = normalize_record(rec)
    assert "system: You help." in pair["prompt"]
    assert "user: Summarize this" in pair["prompt"]
    assert pair["response"] == "Sure thing"
    assert pair["cost_usd"] == 0.01
    assert pair["fmt"] == "openai"


def test_normalize_anthropic_content_parts():
    rec = {
        "model": "claude-x",
        "system": "s",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "describe"}]}
        ],
        "response": "ok",
    }
    pair = normalize_record(rec)
    assert "describe" in pair["prompt"]


def test_redaction_applies_to_prompts():
    from imprint.collector import ingest_record

    # redaction happens at collector layer; adapter output feeds it
    rec = {"prompt": "email me at bob@x.com", "response": "ok"}
    # redaction is applied inside normalize_record (single ingress path)
    assert "[EMAIL]" in normalize_record(rec)["prompt"]
    import tempfile
    import os
    from imprint.store import connect

    db = tempfile.mktemp(suffix=".db")
    conn = connect(db)
    kind, _ = ingest_record(conn, rec)
    row = conn.execute("SELECT prompt FROM pairs").fetchone()
    assert "[EMAIL]" in row["prompt"]
    os.remove(db)


def test_render_response_each_format():
    hdr = {
        "X-Imprint-Signature": "sig_1",
        "X-Imprint-Version": "abc123",
        "X-Imprint-Escalate": "true",
    }
    o = render_response("openai", "hi", "m", hdr)
    assert o["choices"][0]["message"]["content"] == "hi"
    a = render_response("anthropic", "hi", "m", hdr)
    assert a["content"][0]["text"] == "hi"
    g = render_response("gemini", "hi", "m", hdr)
    assert g["candidates"][0]["content"]["parts"][0]["text"] == "hi"
    r = render_response("raw", "hi", "m", hdr)
    assert r["response"] == "hi"
