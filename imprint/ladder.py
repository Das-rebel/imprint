"""Promotion Ladder: finite state machine for skill rollout.

shadow -> canary -> preferred -> pinned, with drift-triggered auto-demote.
Exit criteria per stage are enforced in code (ADR-003).
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class Stage(Enum):
    SHADOW = auto()  # log-only, no user impact
    CANARY = auto()  # 5% traffic, compared vs baseline
    PREFERRED = auto()  # 50% traffic, winning on cost AND quality
    PINNED = auto()  # 100% traffic, auto-demote on drift


@dataclass
class StageCriteria:
    min_samples: int
    max_regression_rate: float  # fraction of outputs worse than baseline
    min_cost_savings: float  # fraction of baseline cost saved


CRITERIA: dict[Stage, StageCriteria] = {
    Stage.CANARY: StageCriteria(
        min_samples=50, max_regression_rate=0.02, min_cost_savings=0.10
    ),
    Stage.PREFERRED: StageCriteria(
        min_samples=200, max_regression_rate=0.01, min_cost_savings=0.25
    ),
    Stage.PINNED: StageCriteria(
        min_samples=500, max_regression_rate=0.005, min_cost_savings=0.30
    ),
}

NEXT_STAGE = {
    Stage.SHADOW: Stage.CANARY,
    Stage.CANARY: Stage.PREFERRED,
    Stage.PREFERRED: Stage.PINNED,
}


@dataclass
class SkillLadder:
    signature_id: str
    stage: Stage = Stage.SHADOW
    samples: int = 0
    regressions: int = 0
    cost_saved_usd: float = 0.0
    baseline_cost_usd: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)

    def evaluate(self) -> Stage | None:
        """Check promotion criteria; return next stage if earned."""
        if self.stage not in NEXT_STAGE:
            return None
        c = CRITERIA[NEXT_STAGE[self.stage]]
        if self.samples < c.min_samples:
            return None
        reg_rate = self.regressions / max(self.samples, 1)
        savings = self.cost_saved_usd / max(self.baseline_cost_usd, 1e-9)
        if reg_rate <= c.max_regression_rate and savings >= c.min_cost_savings:
            return NEXT_STAGE[self.stage]
        return None

    def demote(self, reason: str) -> Stage:
        """Drift or regression trigger: step back one stage."""
        order = list(Stage)
        idx = order.index(self.stage)
        self.stage = order[max(idx - 1, 0)]
        self.history.append(
            {"event": "demote", "reason": reason, "to": self.stage.name}
        )
        return self.stage

    def promote_if_eligible(self) -> Stage | None:
        nxt = self.evaluate()
        if nxt:
            self.history.append(
                {"event": "promote", "from": self.stage.name, "to": nxt.name}
            )
            self.stage = nxt
        return nxt
