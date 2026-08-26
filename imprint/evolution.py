"""Evolution Engine: automated prompt-skill optimization per signature.

Model-agnostic by design — the executor is ANY callable that takes a rendered
prompt and returns output text. Callers plug in their own provider path
(A3M route, direct API, local model). Imprint never hardcodes an SDK.

Loop:
  1. Split signature pairs into train/held-out
  2. Propose variant (mutation strategies, optionally LLM-proposed)
  3. Shadow-eval on held-out via EvalGate
  4. Keep if strictly better on cost-adjusted quality; else discard
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from .evalgate import EvalGate
from .evolver import EvolutionTracker  # noqa: F401 (re-export)
from .skills import Skill


# ------------------------------------------------------- mutation ops -------

MUTATIONS = [
    "add_constraint",     # append explicit output constraint
    "tighten_format",     # demand structured/terse output
    "add_example",        # grow few-shot pack
    "clarify_audience",   # state audience & register
]

CONSTRAINT_LINES = [
    "Be concise. Answer in at most {max_words} words.",
    "Return ONLY the requested artifact — no preamble, no explanation.",
    "If information is missing, say exactly what is missing.",
]


def propose_variant(base: Skill, strategy: str, rng: Optional[random.Random] = None,
                    candidate_shots: Optional[list] = None) -> Skill:
    """Create a mutated child of `base`. Pure function; no I/O."""
    rng = rng or random.Random()
    template = base.template
    shots = list(base.few_shot_pack)

    if strategy == "add_constraint":
        line = rng.choice(CONSTRAINT_LINES).format(max_words=rng.choice([80, 150]))
        if line not in template:
            template = f"{template}\n{line}"
    elif strategy == "tighten_format":
        template = f"{template}\nRespond in plain text with no markdown headers."
    elif strategy == "add_example":
        pool = candidate_shots or []
        if shots and pool:
            new_shot = rng.choice(pool)
            if new_shot not in shots:
                shots.append(new_shot)
    elif strategy == "clarify_audience":
        template = f"{template}\nAssume the reader is a busy domain expert."
    return Skill(
        signature_id=base.signature_id,
        version=base.version + 1,
        template=template,
        few_shot_pack=shots,
        parent_hash=base.prompt_hash,
    )


# --------------------------------------------------------- eval helpers -----

@dataclass
class EvolutionStep:
    generation: int
    strategy: str
    child_hash: str
    avg_agreement: float
    avg_cost_delta: float       # negative = cheaper
    kept: bool
    notes: list = field(default_factory=list)


class Evolver:
    """One evolution run over one signature's held-out set."""

    def __init__(self, gate: Optional[EvalGate] = None,
                 cost_fn: Optional[Callable[[str], float]] = None,
                 seed: int = 13):
        self.gate = gate or EvalGate()
        self.cost_fn = cost_fn      # prompt text -> estimated cost (routing-aware!)
        self.rng = random.Random(seed)

    def evaluate_child(self, base: Skill, child: Skill,
                       held_out: list[dict],
                       execute: Callable[[str, str], str]) -> EvolutionStep:
        """execute(rendered_prompt, original_prompt) -> model output."""
        if child.prompt_hash == base.prompt_hash:
            return EvolutionStep(
                generation=child.version, strategy="noop",
                child_hash=child.prompt_hash, avg_agreement=0.0,
                avg_cost_delta=0.0, kept=False,
                notes=["mutation produced no change; discarded"],
            )
        agreements, deltas, regressions = [], [], 0
        for pair in held_out:
            base_out = execute(base.render(pair["prompt"]), pair["prompt"])
            child_out = execute(child.render(pair["prompt"]), pair["prompt"])
            result = self.gate.evaluate(pair["prompt"], base_out, child_out)
            if result.regression_suspected:
                regressions += 1
            agreements.append(result.agreement)
            if self.cost_fn:
                deltas.append(self.cost_fn(child.render(pair["prompt"]))
                              - self.cost_fn(base.render(pair["prompt"])))

        n = max(len(held_out), 1)
        avg_agr = sum(agreements) / n
        avg_cost_delta = (sum(deltas) / len(deltas)) if deltas else 0.0
        reg_rate = regressions / n

        # keep rule: no worse agreement, not more expensive, zero regressions
        kept = (
            regressions == 0
            and avg_agr >= self.gate.agreement_threshold
            and avg_cost_delta <= 0
        )
        return EvolutionStep(
            generation=child.version, strategy="",
            child_hash=child.prompt_hash,
            avg_agreement=round(avg_agr, 4),
            avg_cost_delta=round(avg_cost_delta, 8),
            kept=kept,
            notes=[f"regression_rate={reg_rate:.3f}"],
        )

    def run(self, base: Skill, held_out: list[dict],
            execute: Callable[[str, str], str],
            candidate_shots: Optional[list] = None,
            max_generations: int = 5) -> tuple[Skill, list[EvolutionStep]]:
        """Evolve until plateau or budget. Returns best skill + step log."""
        current = base
        history: list[EvolutionStep] = []
        flat_rounds = 0

        for gen in range(max_generations):
            strategy = MUTATIONS[gen % len(MUTATIONS)]
            child = propose_variant(current, strategy, self.rng, candidate_shots)
            step = self.evaluate_child(current, child, held_out, execute)
            step.generation = gen + 1
            if step.strategy != "noop":
                step.strategy = strategy

            if step.kept:
                current = child
                flat_rounds = 0
            else:
                flat_rounds += 1
            history.append(step)

            if flat_rounds >= 3:   # matches EvolutionTracker threshold
                break

        return current, history
