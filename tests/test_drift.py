from imprint.drift import CostDriftMonitor, InputDriftMonitor


def test_cost_drift_fires_below_floor() -> None:
    m = CostDriftMonitor(min_savings_ratio=0.10)
    assert m.check(baseline_cost=1.0, skill_cost=0.95).drifted
    assert not m.check(baseline_cost=1.0, skill_cost=0.5).drifted


def test_input_drift_quiet_on_similar() -> None:
    refs = [
        "summarize the quarterly sales report data",
        "summarize the monthly sales report figures",
    ]
    m = InputDriftMonitor(refs)
    v = m.check(["summarize the annual sales report numbers"])
    assert not v.drifted


def test_input_drift_fires_on_different_domain() -> None:
    refs = [
        "write a python function to sort a list of integers ascending",
        "write a python function to merge two sorted lists efficiently",
        "write a python function to reverse a linked list in place",
    ]
    m = InputDriftMonitor(refs)
    v = m.check(
        ["translate this poem into french please"],
    )
    assert v.drifted


def test_input_drift_with_single_reference() -> None:
    """Less than 3 references should return no verdict"""
    m = InputDriftMonitor(["only one reference prompt"])
    v = m.check(["some related prompt"])
    assert not v.drifted
    assert "insufficient references" in v.notes[0]


def test_input_drift_with_empty_recent() -> None:
    """Empty recent prompts list should return no drift"""
    m = InputDriftMonitor(["ref1", "ref2", "ref3"])
    v = m.check([])
    assert not v.drifted


def test_input_drift_with_two_references() -> None:
    """Two references is still insufficient for verdict"""
    m = InputDriftMonitor(["reference one", "reference two"])
    v = m.check(["a third related prompt"])
    assert not v.drifted
    assert "insufficient references" in v.notes[0]


def test_cost_drift_with_zero_baseline() -> None:
    """Zero baseline cost should not flag drift"""
    m = CostDriftMonitor(min_savings_ratio=0.10)
    v = m.check(baseline_cost=0.0, skill_cost=0.0)
    assert not v.drifted


def test_cost_drift_negative_savings() -> None:
    """Skill more expensive than baseline should flag drift"""
    m = CostDriftMonitor(min_savings_ratio=0.10)
    v = m.check(baseline_cost=1.0, skill_cost=1.5)
    assert v.drifted


def test_cost_drift_exactly_at_threshold() -> None:
    """Exactly at threshold should not drift (strict inequality)"""
    m = CostDriftMonitor(min_savings_ratio=0.10)
    # 10% savings = ratio exactly 0.10, which is NOT < 0.10
    v = m.check(baseline_cost=10.0, skill_cost=9.0)
    assert not v.drifted


def test_input_drift_special_characters() -> None:
    """Prompts with special characters should tokenize normally"""
    refs = ["what is 2+2?", "explain @mention and #hashtag"]
    m = InputDriftMonitor(refs)
    v = m.check(["what is 3+3?"])
    assert not v.drifted


def test_input_drift_single_word_prompts() -> None:
    """Very short single-word prompts handled correctly"""
    refs = ["hello", "hi there", "greetings"]
    m = InputDriftMonitor(refs)
    v = m.check(["hello world"])
    # Short words (<=2 chars) are filtered out, so "hello" becomes empty set
    assert not v.drifted
