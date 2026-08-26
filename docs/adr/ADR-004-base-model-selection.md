# ADR-004: Base model selection for the distiller

**Status:** Accepted · **Date:** 2026-08-26

## Context
Phase 3 distills plateaued signatures into QLoRA adapters. A base model must be chosen
per deployment. Naive approaches (biggest-that-fits, or one global default) either waste
serving capacity or produce adapters that can't run on user hardware.

## Decision
1. **Curated candidate pool**, versioned in `models.json` — never arbitrary HF models.
   Three tiers: Small (3B), Medium (7–8B, default), Large (14–24B).
2. **Selection is a scored filter pipeline:**
   - *Filter* by hardware (VRAM fit incl. ~30% KV/adapter headroom; CPU-only → distiller off)
   - *Score* candidates by: capability_fit (existing adapter wins on sibling signatures),
     cost_to_serve (tokens/sec at required batch), consolidation_bonus (signatures sharing a base)
3. **Consolidation over optimality:** prefer many signatures on one shared base.
4. **Smallest-model-wins rule:** smallest candidate that clears the promotion ladder is kept —
   the ladder itself validates the choice; shadow-eval vs runner-up for 48h before PINNED.
5. **Never permanent:** new signatures try the dominant base first; quarterly re-validation;
   two failures on a base blacklist that (base, signature) pair.

## Consequences
- `imprint/basemodel.py` implements detection + scoring as pure, testable functions
- Datasets are retained per signature so base migration = cheap retrain
- CPU-only deployments get prompt-evolution only, with an explicit CLI explanation
