from imprint.evalgate import EvalGate, jaccard, programmatic_checks


def test_jaccard_identical_is_one() -> None:
    assert jaccard("the quick brown fox", "the quick brown fox") == 1.0


def test_programmatic_catches_pii_marker() -> None:
    ok, notes = programmatic_checks("here is [EMAIL] for you")
    assert not ok


def test_gate_passes_similar_output() -> None:
    gate = EvalGate()
    res = gate.evaluate(
        "q",
        "return the sum of two numbers plus tax",
        "return the sum of two numbers with tax",
    )
    assert res.passed and res.agreement > 0.3


def test_gate_flags_garbage() -> None:
    gate = EvalGate()
    res = gate.evaluate(
        "q", "a proper detailed answer about database indexing strategy", ""
    )
    assert not res.passed


def test_jaccard_both_empty() -> None:
    """Both strings empty should return 0.0"""
    assert jaccard("", "") == 0.0


def test_jaccard_one_empty() -> None:
    """One empty string should return 0.0"""
    assert jaccard("hello world", "") == 0.0
    assert jaccard("", "hello world") == 0.0


def test_jaccard_single_word() -> None:
    """Single word strings should compute correctly"""
    assert jaccard("hello", "hello") == 1.0
    assert jaccard("hello", "world") == 0.0


def test_jaccard_partial_overlap() -> None:
    """Partial word overlap should give fraction"""
    result = jaccard("the quick brown fox", "the slow red fox")
    # shared: "the", "fox" = 2 words, union = 6 words
    assert 0.2 < result < 0.4


def test_programmatic_checks_empty_output() -> None:
    ok, notes = programmatic_checks("")
    assert not ok
    assert "empty or too short" in notes[0]


def test_programmatic_checks_too_long() -> None:
    long_text = "x" * 20001
    ok, notes = programmatic_checks(long_text)
    assert not ok
    assert "exceeds length bound" in notes[0]


def test_programmatic_checks_multiple_pii_markers() -> None:
    text = "contact [EMAIL] and use card [CARD] with key [KEY]"
    ok, notes = programmatic_checks(text)
    assert not ok
    assert "redacted PII marker" in notes[0]


def test_programmatic_checks_no_markers_ok() -> None:
    text = "This is a clean output with no PII."
    ok, notes = programmatic_checks(text)
    assert ok
    assert len(notes) == 0


def test_programmatic_checks_custom_markers() -> None:
    ok, notes = programmatic_checks("[CUSTOM]", must_not_contain=("[CUSTOM]",))
    assert not ok


def test_programmatic_checks_min_len_custom() -> None:
    ok, notes = programmatic_checks("ab", min_len=3)
    assert not ok


def test_gate_agreement_exactly_at_threshold() -> None:
    """Exactly at threshold should pass (>=, not >)"""
    gate = EvalGate(agreement_threshold=0.55)
    res = gate.evaluate("q", "hello world one two three", "hello world one two three five six seven eight")
    # Same 4 words, agreement should be around 0.44-0.5
    assert res.passed


def test_gate_judge_overturn() -> None:
    """Judge function can overturn low-agreement verdict"""
    def judge(prompt, baseline, candidate):
        return "good" in candidate.lower()
    
    gate = EvalGate(agreement_threshold=0.7, judge_fn=judge)
    res = gate.evaluate("q", "hello world", "good response here")
    assert res.passed
    assert "judge overturned" in res.notes[0]


def test_eval_result_notes_populated() -> None:
    gate = EvalGate()
    res = gate.evaluate("q", "baseline output", "")
    assert len(res.notes) > 0 or res.programmatic_ok is False
