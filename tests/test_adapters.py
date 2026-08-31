from imprint.adapters import detect_format, normalize_record, render_response


def test_detect_openai() -> None:
    rec = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    assert detect_format(rec) == "openai"


def test_detect_anthropic() -> None:
    rec = {
        "model": "claude-x",
        "system": "be nice",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
    }
    assert detect_format(rec) == "anthropic"


def test_detect_gemini() -> None:
    rec = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}
    assert detect_format(rec) == "gemini"


def test_normalize_openai_messages_flatten() -> None:
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
    assert pair is not None
    assert "system: You help." in pair["prompt"]
    assert "user: Summarize this" in pair["prompt"]
    assert pair["response"] == "Sure thing"
    assert pair["cost_usd"] == 0.01
    assert pair["fmt"] == "openai"


def test_normalize_anthropic_content_parts() -> None:
    rec = {
        "model": "claude-x",
        "system": "s",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "describe"}]}
        ],
        "response": "ok",
    }
    pair = normalize_record(rec)
    assert pair is not None and "describe" in pair["prompt"]


def test_redaction_applies_to_prompts() -> None:
    from imprint.collector import ingest_record

    # redaction happens at collector layer; adapter output feeds it
    rec = {"prompt": "email me at bob@x.com", "response": "ok"}
    # redaction is applied inside normalize_record (single ingress path)
    rec2 = normalize_record(rec)
    assert rec2 is not None and "[EMAIL]" in rec2["prompt"]
    import tempfile
    import os
    from imprint.store import connect

    db = tempfile.mktemp(suffix=".db")
    conn = connect(db)
    kind, _ = ingest_record(conn, rec)
    row = conn.execute("SELECT prompt FROM pairs").fetchone()
    assert row is not None and "[EMAIL]" in row["prompt"]
    os.remove(db)


def test_render_response_each_format() -> None:
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


def test_detect_format_unknown_returns_unknown() -> None:
    assert detect_format({}) == "unknown"
    assert detect_format({"foo": "bar", "baz": 123}) == "unknown"


def test_detect_format_edge_case_empty_messages() -> None:
    assert detect_format({"messages": []}) == "openai"
    assert detect_format({"contents": []}) == "gemini"


def test_detect_format_anthropic_without_system() -> None:
    rec = {"model": "claude-x", "messages": [{"role": "user", "content": "hi"}]}
    assert detect_format(rec) == "openai"


def test_render_response_unknown_format_falls_through() -> None:
    r = render_response("unknown_format", "hello", "m", None)
    assert r["response"] == "hello"
    assert "imprint_signature" not in r


def test_render_response_without_extra_headers() -> None:
    o = render_response("openai", "hi", "m", None)
    assert "imprint_signature" not in o


def test_normalize_record_empty_prompt_returns_none() -> None:
    rec = {"prompt": "   ", "response": "some response"}
    assert normalize_record(rec) is None


def test_normalize_record_raw_format_completion_key() -> None:
    rec = {"prompt": "test prompt", "completion": "the response", "model": "gpt-4o", "cost_usd": 0.02}
    pair = normalize_record(rec)
    assert pair is not None
    assert pair["response"] == "the response"
    assert pair["fmt"] == "raw"


def test_normalize_record_gemini_format() -> None:
    rec = {
        "model": "gemini-2.0-flash",
        "contents": [
            {"role": "user", "parts": [{"text": "hello"}]},
            {"role": "model", "parts": [{"text": "hi there"}]},
        ],
        "response": "gemini response text",
    }
    pair = normalize_record(rec)
    assert pair is not None
    assert "user: hello" in pair["prompt"]
    assert "model: hi there" in pair["prompt"]
    assert pair["response"] == "gemini response text"
    assert pair["fmt"] == "gemini"


def test_normalize_record_missing_cost_uses_zero() -> None:
    rec = {"prompt": "test", "response": "resp", "model": "gpt-4o"}
    pair = normalize_record(rec)
    assert pair is not None
    assert pair["cost_usd"] == 0.0


def test_normalize_record_usage_total_cost() -> None:
    rec = {"prompt": "test", "response": "resp", "model": "gpt-4o", "usage": {"total_cost": 0.05}}
    pair = normalize_record(rec)
    assert pair is not None
    assert pair["cost_usd"] == 0.05


def test_redact_api_keys() -> None:
    rec = {"prompt": "use key sk-abc123defghijklmnopqrst for auth", "response": "done"}
    pair = normalize_record(rec)
    assert pair is not None
    assert "[KEY]" in pair["prompt"]
    assert "sk-" not in pair["prompt"]


def test_redact_credit_card() -> None:
    rec = {"prompt": "card number is 4111-1111-1111-1111", "response": "ok"}
    pair = normalize_record(rec)
    assert pair is not None
    assert "[CARD]" in pair["prompt"]


def test_flatten_content_nested_parts() -> None:
    from imprint.adapters import _flatten_content
    content = [{"type": "text", "text": "hello"}, {"type": "image_url", "url": "https://..."}]
    result = _flatten_content(content)
    assert result == "hello"


def test_flatten_content_string_passthrough() -> None:
    from imprint.adapters import _flatten_content
    assert _flatten_content("plain string") == "plain string"
    assert _flatten_content(None) == ""
