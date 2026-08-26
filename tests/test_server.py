import json
import tempfile

from imprint.server import ImprintHandler  # noqa: F401 ensures import works
import os

from imprint.skills import Skill, save_skill
from imprint.store import connect


def test_skill_save_and_load_roundtrip():
    db = tempfile.mktemp(suffix=".db")
    conn = connect(db)
    s = Skill(
        signature_id="sig_x",
        template="Answer tersely.",
        few_shot_pack=[{"input": "a", "output": "b"}],
    )
    sid = save_skill(conn, s)
    row = conn.execute("SELECT * FROM skills WHERE id=?", (sid,)).fetchone()
    assert row["prompt_hash"] == s.prompt_hash
    assert json.loads(row["few_shot_json"]) == s.few_shot_pack
    os.remove(db)


def test_find_skill_prefers_highest_stage():
    db = tempfile.mktemp(suffix=".db")
    conn = connect(db)
    save_skill(conn, Skill(signature_id="s1", template="shadow one"))
    conn.execute("UPDATE skills SET stage='preferred' WHERE signature_id='s1'")
    conn.commit()
    from imprint.server import _find_skill

    skill, row = _find_skill(conn, "anything", lambda p, s: True)
    assert row["stage"] == "preferred"
    os.remove(db)
