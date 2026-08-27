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
