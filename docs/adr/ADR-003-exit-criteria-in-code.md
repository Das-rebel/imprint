# ADR-003: Promotion exit criteria enforced in code

**Status:** Accepted · **Date:** 2026-08-26

## Decision
Stage transition thresholds (min samples, max regression rate, min cost savings) live in
`imprint/ladder.py::CRITERIA`, not in documentation. The promotion ladder is a finite state
machine; drift demotion steps back exactly one stage.

## Rationale
PLAN.md-as-graveyard risk: criteria that exist only in prose don't gate behavior.
Code-first criteria make the ladder auditable and testable (`tests/test_ladder.py`).
