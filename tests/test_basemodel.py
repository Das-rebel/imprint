from imprint.basemodel import (
    Hardware,
    Selection,
    SignatureProfile,
    filter_by_hardware,
    load_pool,
    score_candidate,
    select_base,
)

POOL = load_pool()
SMALL = POOL[0]
MEDIUM = POOL[2]
LARGE = POOL[4]

SIGS = [
    SignatureProfile("s1", est_monthly_savings=40),
    SignatureProfile("s2", est_monthly_savings=25),
]
WINS = {"s1": 1}


def test_pool_loads_with_all_tiers() -> None:
    tiers = {c.tier for c in POOL}
    assert tiers == {"small", "medium", "large"}


def test_nvidia_16gb_filters_large_models() -> None:
    fit, _ = filter_by_hardware(POOL, Hardware("nvidia", 16.0))
    names = {c.name for c in fit}
    assert MEDIUM.name in names
    assert LARGE.name not in names  # 17GB QLoRA > 16GB budget
    assert SMALL.name in names


def test_cpu_only_disables_distiller() -> None:
    sel = select_base(SIGS, {}, hw=Hardware("cpu_only", 0.0), pool=POOL)
    assert isinstance(sel, Selection)
    assert not sel.distiller_enabled
    assert "CPU-only" in sel.block_reason


def test_apple_silicon_swaps_to_mlx() -> None:
    sel = select_base(SIGS, {}, hw=Hardware("apple_silicon", 36.0), pool=POOL)
    if sel.distiller_enabled:
        assert sel.candidate.backend == "mlx"
        assert "mlx-community" in sel.candidate.hf_id


def test_consolidation_bonus_prefers_shared_base() -> None:
    many_sigs = [SignatureProfile(f"s{i}", est_monthly_savings=20) for i in range(5)]
    small_score = score_candidate(SMALL, many_sigs, {}, 5)
    # medium gets same consolidation but less tier bonus and slower tok/s
    medium_score = score_candidate(MEDIUM, many_sigs, {}, 5)
    assert small_score > medium_score  # smallest-model-wins tiebreak


def test_reasoning_tasks_penalize_small_tier() -> None:
    hard = [
        SignatureProfile("h1", est_monthly_savings=50, needs_multi_step_reasoning=True)
    ]
    small_score = score_candidate(SMALL, hard, {}, 1)
    medium_score = score_candidate(MEDIUM, hard, {}, 1)
    assert medium_score > small_score


def test_low_economics_gates_out_training() -> None:
    tiny = [SignatureProfile("t1", est_monthly_savings=1)]
    low = score_candidate(MEDIUM, tiny, {}, 1)
    normal = score_candidate(MEDIUM, SIGS, {}, 1)
    assert low < normal  # penalty applied


def test_selection_reports_runner_up() -> None:
    sel = select_base(SIGS, WINS, hw=Hardware("nvidia", 24.0), pool=POOL)
    assert sel.distiller_enabled
    assert sel.runner_up is not None
    assert sel.score >= 0
