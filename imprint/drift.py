"""Drift Monitor: input, outcome, and cost drift detection with demote triggers."""

import statistics
from dataclasses import dataclass, field


@dataclass
class DriftVerdict:
    drifted: bool
    kind: str = ""  # input|outcome|cost
    detail: str = ""
    notes: list[str] = field(default_factory=list)


class InputDriftMonitor:
    """Rolling lexical-distance vs the signature's reference distribution.

    Phase 0 uses token-set Jaccard against reference samples; embeddings upgrade
    in Phase 1 (mean cosine distance > 2 sigma -> drift).
    """

    def __init__(
        self, reference_prompts: list[str], threshold_sigma: float = 2.0
    ) -> None:
        self.references = [self._token_set(p) for p in reference_prompts]
        self.threshold_sigma = threshold_sigma

    @staticmethod
    def _token_set(text: str) -> set[str]:
        return {t for t in text.lower().split() if len(t) > 2}

    def check(self, recent_prompts: list[str]) -> DriftVerdict:
        if not self.references or not recent_prompts:
            return DriftVerdict(drifted=False)
        recent_sets = [self._token_set(p) for p in recent_prompts]

        def dist(a: set[str], b: set[str]) -> float:
            union = a | b
            return 1 - (len(a & b) / len(union)) if union else 1.0

        if len(self.references) < 3:
            return DriftVerdict(
                drifted=False, notes=["insufficient references (<3); no verdict"]
            )

        # distance of each recent prompt to nearest reference
        dists = [min(dist(r, s) for r in self.references) for s in recent_sets]
        # leave-one-out: each reference vs the OTHER references -> natural spread
        ref_dists = []
        for i, r in enumerate(self.references):
            others = [o for j, o in enumerate(self.references) if j != i]
            ref_dists.append(min(dist(r, o) for o in others))
        mu = statistics.mean(ref_dists)
        try:
            sigma = statistics.stdev(ref_dists)
        except statistics.StatisticsError:
            sigma = 0.05
        sigma = max(sigma, 0.05)
        mean_recent = statistics.mean(dists)
        drifted = mean_recent > mu + self.threshold_sigma * sigma
        return DriftVerdict(
            drifted=drifted,
            kind="input",
            detail=f"recent_mean_dist={mean_recent:.3f} vs ref {mu:.3f}+{sigma:.3f}sigma",
        )


class CostDriftMonitor:
    """Demote when provider price changes shrink savings below floor."""

    def __init__(self, min_savings_ratio: float = 0.10):
        self.min_savings_ratio = min_savings_ratio

    def check(self, baseline_cost: float, skill_cost: float) -> DriftVerdict:
        if baseline_cost <= 0:
            return DriftVerdict(drifted=False)
        ratio = (baseline_cost - skill_cost) / baseline_cost
        drifted = ratio < self.min_savings_ratio
        return DriftVerdict(
            drifted=drifted,
            kind="cost",
            detail=f"savings={ratio:.2%} < floor {self.min_savings_ratio:.0%}",
        )
