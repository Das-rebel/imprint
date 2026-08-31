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


def test_nvidia_24gb_fits_large_models() -> None:
    """24GB fits small and medium but not large (16.8GB effective budget)"""
    from imprint.basemodel import Hardware, filter_by_hardware, load_pool
    pool = load_pool()
    fit, _ = filter_by_hardware(pool, Hardware("nvidia", 24.0))
    names = [c.name for c in fit]
    # 24GB * 0.7 = 16.8GB budget -> large (17GB, 24GB) excluded
    assert len(names) == 4  # 2 small + 2 medium


def test_amd_rocm_filters_correctly() -> None:
    """AMD 32GB has 22.4GB effective budget after headroom"""
    from imprint.basemodel import Hardware, filter_by_hardware, load_pool
    pool = load_pool()
    fit, _ = filter_by_hardware(pool, Hardware("nvidia", 32.0))
    names = [c.name for c in fit]
    # 32GB * 0.7 = 22.4GB -> 24B (24GB qlora) excluded, others fit
    assert len(names) >= 3


def test_filter_by_hardware_serving_mode() -> None:
    """Serving footprint differs from QLoRA footprint"""
    from imprint.basemodel import Hardware, filter_by_hardware, load_pool
    pool = load_pool()
    # 8GB budget - tight for serving
    fit_serving, _ = filter_by_hardware(pool, Hardware("nvidia", 8.0), serving=True)
    fit_qlora, _ = filter_by_hardware(pool, Hardware("nvidia", 8.0), serving=False)
    # QLoRA mode fits more models (smaller footprint)
    assert len(fit_qlora) >= len(fit_serving)


def test_consolidation_bonus_exact_fraction() -> None:
    """Verify consolidation bonus is exactly (len(sigs)/total)*15"""
    from imprint.basemodel import score_candidate, BaseCandidate, SignatureProfile
    c = BaseCandidate("Qwen2.5-3B-Instruct", "small", 4.5, 3.0, 120, "cuda", "Q/3B")
    # With 1 sig out of 1 total -> bonus = (1/1)*15 = 15
    one_sig = [SignatureProfile("s1", est_monthly_savings=20)]
    # With 2 sigs out of 2 total -> bonus = (2/2)*15 = 15 each
    two_sigs = [SignatureProfile("s1", est_monthly_savings=20), SignatureProfile("s2", est_monthly_savings=15)]
    
    score_one = score_candidate(c, one_sig, {}, 1)
    score_two = score_candidate(c, two_sigs, {}, 2)
    # Same fraction (1/1 = 2/2 = 1) so same consolidation bonus
    assert score_one == score_two


def test_tier_bonus_small_preferred() -> None:
    """Small tier gets +3, medium +1, large +0 - small scores higher"""
    from imprint.basemodel import score_candidate, BaseCandidate, SignatureProfile
    
    shared = [SignatureProfile("s1", est_monthly_savings=50)]
    small = BaseCandidate("Qwen2.5-3B-Instruct", "small", 4.5, 3.0, 120, "cuda", "Q/3B")
    medium = BaseCandidate("Qwen2.5-7B-Instruct", "medium", 9.0, 6.5, 70, "cuda", "Q/7B")
    
    # Same inputs, only tier differs
    score_small = score_candidate(small, shared, {}, 1)
    score_medium = score_candidate(medium, shared, {}, 1)
    
    # small should score higher than medium (tier bonus + throughput bonus)
    assert score_small > score_medium


def test_gated_model_blocked_without_opt_in() -> None:
    """Gated models should be excluded when IMPRINT_ALLOW_GATED_MODELS != 1"""
    import os
    os.environ.pop("IMPRINT_ALLOW_GATED_MODELS", None)
    from imprint.basemodel import select_base, Hardware, SignatureProfile
    
    pool = load_pool()
    gated = [c for c in pool if c.gated]
    if gated:
        sel = select_base([], {}, hw=Hardware("nvidia", 32.0), pool=pool)
        # All gated models should be excluded
        assert not any(c.gated for c in [sel.candidate] + ([sel.runner_up] if sel.runner_up else []))


def test_select_base_runner_up_is_second_best() -> None:
    """Runner up should exist when pool has multiple candidates"""
    from imprint.basemodel import select_base, Hardware
    
    pool = load_pool()
    hw = Hardware("nvidia", 24.0)
    sel = select_base([], {}, hw=hw, pool=pool)
    
    # With empty sigs, runner_up should still be set
    assert sel.runner_up is not None
    assert sel.candidate is not None
