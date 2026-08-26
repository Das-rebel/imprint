"""Base model selection for the distiller (Phase 3+).

Implements ADR-004: curated pool, hardware filter, scored selection with
consolidation preference, and smallest-model-wins validation rule.

Pure functions + dataclasses — no GPU required to reason about selection.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------- pool ------

MODELS_JSON = Path(__file__).parent / "models.json"


@dataclass(frozen=True)
class BaseCandidate:
    name: str
    tier: str  # small | medium | large
    vram_gb_qlora: float  # training footprint (4-bit)
    vram_gb_serve: float  # serving footprint
    tokens_per_sec_gpu: float  # rough serve throughput on a 24GB class GPU
    backend: str  # mlx | cuda | cpu
    hf_id: str
    gated: bool = False  # requires manual HF license acceptance


DEFAULT_POOL: list[BaseCandidate] = [
    BaseCandidate(
        "Qwen2.5-3B-Instruct",
        "small",
        4.5,
        3.0,
        120,
        "cuda",
        "Qwen/Qwen2.5-3B-Instruct",
    ),
    BaseCandidate(
        "Llama-3.2-3B-Instruct",
        "small",
        4.5,
        3.0,
        110,
        "cuda",
        "meta-llama/Llama-3.2-3B-Instruct",
        gated=True,
    ),
    BaseCandidate(
        "Qwen2.5-7B-Instruct",
        "medium",
        9.0,
        6.5,
        70,
        "cuda",
        "Qwen/Qwen2.5-7B-Instruct",
    ),
    BaseCandidate(
        "Llama-3.1-8B-Instruct",
        "medium",
        10.5,
        7.5,
        62,
        "cuda",
        "meta-llama/Llama-3.1-8B-Instruct",
    ),
    BaseCandidate(
        "Qwen2.5-14B-Instruct",
        "large",
        17.0,
        12.0,
        34,
        "cuda",
        "Qwen/Qwen2.5-14B-Instruct",
    ),
    BaseCandidate(
        "Mistral-Small-3.2-24B",
        "large",
        24.0,
        16.0,
        22,
        "cuda",
        "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    ),
]

# Apple-Silicon MLX equivalents (same capability tier, different runtime)
MLX_SWAPS = {
    "Qwen2.5-3B-Instruct": "mlx-community/Qwen2.5-3B-Instruct-4bit",
    "Qwen2.5-7B-Instruct": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "Qwen2.5-14B-Instruct": "mlx-community/Qwen2.5-14B-Instruct-4bit",
}


def load_pool(path: Optional[Path] = None) -> list[BaseCandidate]:
    p = path or MODELS_JSON
    if p.exists():
        raw = json.loads(p.read_text())
        return [BaseCandidate(**c) for c in raw]
    return DEFAULT_POOL


# ----------------------------------------------------------- hardware -------


@dataclass
class Hardware:
    kind: str  # apple_silicon | nvidia | cpu_only
    vram_gb: float  # unified memory budget on MPS; VRAM on CUDA
    gpu_name: str = ""


def detect_hardware() -> Hardware:
    # NVIDIA via torch (optional dep)
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return Hardware("nvidia", round(props.total_memory / 1e9, 1), props.name)
    except Exception:
        pass
    # Apple Silicon: unified memory ~ installed RAM usable for Metal
    if _is_apple_silicon():
        total_gb = _mac_ram_gb()
        return Hardware("apple_silicon", total_gb, "Apple Silicon (unified)")
    return Hardware("cpu_only", 0.0)


def _is_apple_silicon() -> bool:
    return os.uname().sysname == "Darwin" and os.uname().machine == "arm64"


def _mac_ram_gb() -> float:
    try:
        import subprocess

        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True
        )
        return round(int(out.stdout.strip()) / 1e9, 1)
    except Exception:
        return 8.0


# -------------------------------------------------------------- scoring -----

HEADROOM_RATIO = 0.30  # reserve for KV cache + adapter weights


@dataclass
class SignatureProfile:
    signature_id: str
    est_monthly_savings: float
    needs_long_context: bool = False  # >8k tokens typical prompt
    needs_multi_step_reasoning: bool = False


@dataclass
class Selection:
    candidate: BaseCandidate
    score: float
    reasons: list[str] = field(default_factory=list)
    runner_up: Optional["BaseCandidate"] = None
    distiller_enabled: bool = True
    block_reason: str = ""


def filter_by_hardware(
    pool: list[BaseCandidate], hw: Hardware, serving: bool = False
) -> tuple[list[BaseCandidate], list[str]]:
    reasons: list[str] = []
    if hw.kind == "cpu_only":
        return [], [
            "no GPU detected — distiller disabled; "
            "prompt-evolution (v1) remains fully functional"
        ]
    limit_key = "vram_gb_serve" if serving else "vram_gb_qlora"
    budget = hw.vram_gb * (1 - HEADROOM_RATIO) if hw.kind == "nvidia" else hw.vram_gb
    fit = []
    for c in pool:
        need = getattr(c, limit_key)
        if need <= budget:
            fit.append(c)
        else:
            reasons.append(f"{c.name}: needs {need}GB > {budget:.1f}GB budget")
    return fit, reasons


def score_candidate(
    c: BaseCandidate,
    sigs: list[SignatureProfile],
    existing_base_wins: dict[str, int],
    all_signatures_count: int,
) -> float:
    """Higher is better. Consolidation and proven wins dominate."""
    score = 0.0

    # capability fit: how many sibling signatures already succeeded on this base
    wins = sum(existing_base_wins.get(s.signature_id, 0) for s in sigs)
    score += wins * 10

    # consolidation: fraction of ALL signatures that could share this base
    share_bonus = (len(sigs) / max(all_signatures_count, 1)) * 15
    score += share_bonus

    # throughput: faster serve = more headroom for volume (scaled small)
    score += min(c.tokens_per_sec_gpu / 40, 3)

    # economics gate: savings must justify any training at all
    total_savings = sum(s.est_monthly_savings for s in sigs)
    if total_savings < 5:
        score -= 25  # not worth provisioning anything

    # long-context / reasoning tasks prefer medium+ tiers
    if any(s.needs_long_context or s.needs_multi_step_reasoning for s in sigs):
        if c.tier == "small":
            score -= 8
        elif c.tier in ("medium", "large"):
            score += 4

    # smallest-model-wins tiebreak: slight preference for lower tiers
    tier_bonus = {"small": 3, "medium": 1, "large": 0}
    score += tier_bonus[c.tier]
    return score


def select_base(
    sig_profiles: list[SignatureProfile],
    existing_base_wins: dict[str, int],
    hw: Optional[Hardware] = None,
    pool: Optional[list[BaseCandidate]] = None,
) -> Selection:
    hw = hw or detect_hardware()
    pool = pool or load_pool()

    if hw.kind == "cpu_only":
        return Selection(
            candidate=pool[0] if pool else DEFAULT_POOL[0],
            score=0.0,
            distiller_enabled=False,
            block_reason="CPU-only: no viable training target. "
            "v1 skill-prompt evolution still works.",
        )

    if hw.kind == "apple_silicon":
        pool = [replace_backend(c, MLX_SWAPS.get(c.name, c.hf_id), "mlx") for c in pool]

    # gated models need manual HF acceptance — deprioritize unless user opted in
    opt_in_gated = os.environ.get("IMPRINT_ALLOW_GATED_MODELS") == "1"
    usable = [c for c in pool if opt_in_gated or not c.gated]
    if not usable:
        return Selection(
            candidate=pool[0],
            score=0.0,
            distiller_enabled=False,
            block_reason="all candidates are HF-gated; set "
            "IMPRINT_ALLOW_GATED_MODELS=1 after accepting licenses",
        )
    pool = usable

    fit, why_not = filter_by_hardware(pool, hw)
    if not fit:
        return Selection(
            candidate=pool[0],
            score=0.0,
            distiller_enabled=False,
            block_reason="no candidate fits this GPU: " + "; ".join(why_not[:3]),
        )

    scored = sorted(
        (
            (score_candidate(c, sig_profiles, existing_base_wins, len(sig_profiles)), c)
            for c in fit
        ),
        key=lambda t: -t[0],
    )
    top_score, top = scored[0]
    runner = scored[1][1] if len(scored) > 1 else None
    return Selection(
        candidate=top,
        score=top_score,
        runner_up=runner,
        reasons=[r for r in [f"tier={top.tier}", f"{top.tokens_per_sec_gpu} tok/s"]],
    )


def replace_backend(c: BaseCandidate, hf_id: str, backend: str) -> BaseCandidate:
    return BaseCandidate(
        c.name,
        c.tier,
        c.vram_gb_qlora * 0.55,  # 4-bit MLX smaller
        c.vram_gb_serve * 0.55,
        c.tokens_per_sec_gpu * 0.8,
        backend,
        hf_id,
    )
