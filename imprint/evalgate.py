"""Eval Gate: three-signal quality check — programmatic, agreement, LLM-judge.

Phase 0 implements signals 1 and 2 (zero API cost). Signal 3 is a pluggable hook.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class EvalResult:
    passed: bool
    programmatic_ok: bool
    agreement: float  # cosine-ish similarity vs baseline output (0..1)
    regression_suspected: bool
    notes: list[str] = field(default_factory=list)


def _token_set(text: str) -> set[str]:
    return {t for t in text.lower().split() if len(t) > 2}


def jaccard(a: str, b: str) -> float:
    """Cheap lexical agreement proxy; embeddings upgrade in Phase 1."""
    sa, sb = _token_set(a), _token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def programmatic_checks(
    output: str,
    min_len: int = 1,
    max_len: int = 20000,
    must_not_contain: tuple[str, ...] = ("[EMAIL]", "[CARD]", "[KEY]"),
) -> tuple[bool, list[str]]:
    notes: list[str] = []
    if not output or len(output) < min_len:
        return False, ["output empty or too short"]
    if len(output) > max_len:
        return False, ["output exceeds length bound"]
    for marker in must_not_contain:
        if marker in output:
            return False, [f"output contains redacted PII marker {marker}"]
    return True, notes


class EvalGate:
    def __init__(
        self,
        agreement_threshold: float = 0.55,
        judge_fn: Optional[Callable[..., bool]] = None,
    ):
        self.agreement_threshold = agreement_threshold
        self.judge_fn = judge_fn  # optional: (prompt, baseline, candidate) -> bool

    def evaluate(
        self, prompt: str, baseline_output: str, skill_output: str
    ) -> EvalResult:
        ok, notes = programmatic_checks(skill_output)
        agreement = jaccard(baseline_output, skill_output)
        regression = (not ok) or (agreement < self.agreement_threshold)

        # tie-breaker judge only on borderline disagreement
        if regression and self.judge_fn and ok:
            verdict = self.judge_fn(prompt, baseline_output, skill_output)
            if verdict:
                regression = False
                notes.append("judge overturned low agreement")

        return EvalResult(
            passed=not regression,
            programmatic_ok=ok,
            agreement=round(agreement, 4),
            regression_suspected=regression,
            notes=notes,
        )
