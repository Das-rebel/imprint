"""Skill Evolver (v1): per-signature prompt optimization.

Phase 0: manual prompts with hash versioning for clean A/B in the ladder.
Phase 1: automated evolution (DSPy-backed). Prompt-plateau heuristic:
3 consecutive evolutions yielding <5% improvement -> flag for v2 distillation.
"""

from dataclasses import dataclass, field
import hashlib
import json


PLATEAU_WINDOW = 3
PLATEAU_THRESHOLD = 0.05  # <5% improvement counts as flat


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


@dataclass
class EvolvedPrompt:
    signature_id: str
    version: int
    template: str
    few_shot_pack: list = field(default_factory=list)
    parent_hash: str | None = None

    @property
    def hash(self) -> str:
        return prompt_hash(self.template + json.dumps(self.few_shot_pack))


@dataclass
class EvolutionTracker:
    """Tracks improvement deltas to detect plateau -> flag for v2 distillation."""

    signature_id: str
    improvements: list = field(default_factory=list)  # fractional cost deltas

    def record(self, cost_delta: float) -> bool:
        self.improvements.append(cost_delta)
        tail = self.improvements[-PLATEAU_WINDOW:]
        if len(tail) == PLATEAU_WINDOW and all(d < PLATEAU_THRESHOLD for d in tail):
            return True  # plateaued: escalate to distiller queue (v2)
        return False
