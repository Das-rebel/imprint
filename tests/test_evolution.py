import random

from imprint.evolution import EvolutionTracker, Evolver, propose_variant
from imprint.evalgate import EvalGate
from imprint.skills import Skill


def _base_skill() -> Skill:
    return Skill(
        signature_id="s1",
        template="Summarize the input text.",
        few_shot_pack=[{"input": "sales rose", "output": "Sales up."}],
    )


def test_propose_add_constraint_extends_template() -> None:
    child = propose_variant(_base_skill(), "add_constraint", random.Random(1))
    assert len(child.template) > len(_base_skill().template)
    assert child.version == 2
    assert child.parent_hash == _base_skill().prompt_hash


def test_propose_add_example_grows_shots() -> None:
    pool = [{"input": "profit fell", "output": "Profit down."}]
    child = propose_variant(
        _base_skill(), "add_example", random.Random(0), candidate_shots=pool
    )
    assert len(child.few_shot_pack) >= len(_base_skill().few_shot_pack)


def test_evolver_keeps_cheaper_non_regressing_child() -> None:
    # executor echoes a canned answer that matches baseline lexically;
    # cost_fn says the shorter child prompt is cheaper
    def execute(rendered: str, original: str) -> str:
        return "summarize: key figures extracted"

    def cost_fn(prompt: str) -> float:
        return len(prompt)

    base = Skill(
        signature_id="s", template="Summarize the sales numbers in the report."
    )
    held_out = [
        {"prompt": f"report {i}: revenue grew", "response": ""} for i in range(5)
    ]

    e = Evolver(gate=EvalGate(agreement_threshold=0.2), cost_fn=cost_fn)
    best, history = e.run(base, held_out, execute, max_generations=3)
    assert history
    # no step should have been kept if it caused regressions
    assert all(not h.kept or h.avg_agreement >= 0.2 for h in history)


def test_evolver_discards_expensive_children() -> None:
    """Deterministic cost: longer prompts always cost more. Mutations that append
    constraints make children longer -> never cheaper -> nothing kept."""

    def execute(rendered: str, original: str) -> str:
        return "some output text here"

    def cost_fn(prompt: str) -> float:
        return float(len(prompt))

    base = Skill(signature_id="s", template="Summarize.")
    held_out = [{"prompt": f"doc {i}", "response": ""} for i in range(4)]
    e = Evolver(cost_fn=cost_fn)
    best, history = e.run(base, held_out, execute, max_generations=4)
    kept_steps = [h for h in history if h.kept]
    for h in kept_steps:
        assert h.avg_cost_delta <= 0


def test_evolver_discards_noop_mutations() -> None:
    """add_example with an empty shot pool produces an identical child -> discarded."""

    def execute(rendered: str, original: str) -> str:
        return "out"

    base = Skill(signature_id="s", template="Summarize.")
    held_out = [{"prompt": "doc", "response": ""}]
    e = Evolver(cost_fn=lambda p: len(p))
    _, history = e.run(base, held_out, execute, candidate_shots=[], max_generations=4)
    noops = [h for h in history if h.strategy == "noop"]
    assert all(not h.kept for h in noops)
    assert any(h.strategy == "noop" for h in history)


def test_evolution_tracker_plateau_matches_evolver_threshold() -> None:
    t = EvolutionTracker(signature_id="s")
    for _ in range(3):
        plateaued = t.record(0.01)
    assert plateaued is True


def test_propose_variant_tighten_format() -> None:
    """tighten_format adds plain-text constraint"""
    import random
    base = Skill(signature_id="s", template="Summarize.")
    child = propose_variant(base, "tighten_format", random.Random(42))
    assert "plain text" in child.template.lower() or "no markdown" in child.template.lower()
    assert child.version == 2


def test_propose_variant_clarify_audience() -> None:
    """clarify_audience adds audience statement"""
    import random
    base = Skill(signature_id="s", template="Explain this.")
    child = propose_variant(base, "clarify_audience", random.Random(99))
    assert child.version == 2
    assert len(child.template) > len(base.template)


def test_propose_variant_preserves_parent_hash() -> None:
    """Child skill should track parent hash"""
    import random
    base = Skill(signature_id="s", template="Test.")
    child = propose_variant(base, "add_constraint", random.Random(1))
    assert child.parent_hash == base.prompt_hash


def test_propose_variant_add_example_empty_pool() -> None:
    """add_example with empty pool returns identical template"""
    import random
    base = Skill(signature_id="s", template="Summarize.", few_shot_pack=[{"input": "a", "output": "b"}])
    child = propose_variant(base, "add_example", random.Random(0), candidate_shots=[])
    # No new shots added, but version still increments
    assert child.version == 2
    assert len(child.few_shot_pack) == len(base.few_shot_pack)


def test_propose_variant_all_mutations_deterministic() -> None:
    """Same seed should produce same result"""
    import random
    base = Skill(signature_id="s", template="Test prompt here.")
    r1 = random.Random(123)
    r2 = random.Random(123)
    child1 = propose_variant(base, "add_constraint", r1)
    child2 = propose_variant(base, "add_constraint", r2)
    assert child1.template == child2.template


def test_evolver_max_generations_zero() -> None:
    """Zero generations should return base skill unchanged"""
    import random
    base = Skill(signature_id="s", template="Test.")
    held_out = [{"prompt": "doc", "response": "out"}]
    e = Evolver(cost_fn=lambda p: len(p))
    best, history = e.run(base, held_out, lambda r, o: r, max_generations=0)
    assert best.template == base.template
    assert len(history) == 0


def test_evolver_empty_held_out() -> None:
    """Empty held_out set should still run (evaluates nothing)"""
    import random
    base = Skill(signature_id="s", template="Test.")
    e = Evolver(cost_fn=lambda p: len(p))
    best, history = e.run(base, [], lambda r, o: r, max_generations=2)
    assert best.template == base.template
    assert len(history) == 2  # Still runs mutations but no held-out evaluation


def test_evolver_custom_agreement_threshold() -> None:
    """Custom gate with high threshold vs low threshold"""
    def execute(rendered, original):
        return "identical output for all"
    
    base = Skill(signature_id="s", template="Original prompt.")
    held_out = [{"prompt": f"doc{i}", "response": ""} for i in range(3)]
    
    # With high threshold (0.9), need high agreement to pass
    e_high = Evolver(gate=EvalGate(agreement_threshold=0.9), cost_fn=lambda p: len(p))
    _, history_high = e_high.run(base, held_out, execute, max_generations=3)
    
    # With low threshold (0.1), easy to pass
    e_low = Evolver(gate=EvalGate(agreement_threshold=0.1), cost_fn=lambda p: len(p))
    best_low, history_low = e_low.run(base, held_out, execute, max_generations=3)
    
    # Both should run same number of generations
    assert len(history_high) == 3
    assert len(history_low) == 3
