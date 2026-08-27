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
