"""Skills registry: versioned prompt artifacts per signature."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Optional

from .store import now_ms
from .ladder import SkillLadder


def prompt_hash(text: str, shots: Any = None) -> str:
    import hashlib

    return hashlib.sha256((text + json.dumps(shots or [])).encode()).hexdigest()[:12]


@dataclass
class Skill:
    signature_id: str
    template: str
    few_shot_pack: list[dict[str, str]] = field(default_factory=list)
    id: Optional[int] = None
    version: int = 1
    prompt_hash: str = ""
    parent_hash: Optional[str] = None
    stage: str = "shadow"

    def __post_init__(self) -> None:
        self.prompt_hash = prompt_hash(self.template, self.few_shot_pack)

    def render(self, prompt: str) -> str:
        shots = "\n\n".join(
            f"Example {i+1}:\nInput: {s.get('input','')}\nOutput: {s.get('output','')}"
            for i, s in enumerate(self.few_shot_pack[:3])
        )
        parts = [self.template]
        if shots:
            parts.append(shots)
        parts.append(f"Input: {prompt}\nOutput:")
        return "\n\n".join(parts)


def save_skill(conn: sqlite3.Connection, skill: Skill) -> int:
    cur = conn.execute(
        "INSERT INTO skills (signature_id, version, prompt_hash, template,"
        " few_shot_json, stage, created_at, updated_at)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (
            skill.signature_id,
            (skill.version or 1),
            skill.prompt_hash,
            skill.template,
            json.dumps(skill.few_shot_pack),
            skill.stage,
            now_ms(),
            now_ms(),
        ),
    )
    conn.commit()
    return cur.lastrowid or 0


def get_active_skills(conn: sqlite3.Connection, stage: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM skills WHERE stage=? ORDER BY updated_at DESC", (stage,)
    ).fetchall()


def load_ladder_state(conn: sqlite3.Connection, skill_row: sqlite3.Row) -> SkillLadder:
    """Rehydrate a SkillLadder from a skills row."""
    from .ladder import Stage, SkillLadder

    ladder = SkillLadder(
        signature_id=skill_row["signature_id"],
        stage=Stage[skill_row["stage"].upper()],
        samples=skill_row["samples"],
        regressions=skill_row["regressions"],
        cost_saved_usd=skill_row["cost_saved_usd"],
        baseline_cost_usd=skill_row["baseline_cost_usd"],
    )
    return ladder


def update_ladder_stats(
    conn: sqlite3.Connection, skill_id: int, ladder: SkillLadder
) -> None:
    conn.execute(
        "UPDATE skills SET stage=?, samples=?, regressions=?,"
        " cost_saved_usd=?, baseline_cost_usd=?, updated_at=? WHERE id=?",
        (
            ladder.stage.name.lower(),
            ladder.samples,
            ladder.regressions,
            ladder.cost_saved_usd,
            ladder.baseline_cost_usd,
            now_ms(),
            skill_id,
        ),
    )
    conn.commit()
