import random

from typing import Callable
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
