from imprint.ladder import SkillLadder, Stage


def test_shadow_promotes_to_canary():
    lad = SkillLadder(signature_id="sig1", samples=100, regressions=1,
                      cost_saved_usd=20, baseline_cost_usd=100)
    assert lad.evaluate() == Stage.CANARY


def test_insufficient_samples_blocks_promotion():
    lad = SkillLadder(signature_id="sig2", samples=10, regressions=0,
                      cost_saved_usd=50, baseline_cost_usd=100)
    assert lad.evaluate() is None


def test_regression_blocks_promotion():
    lad = SkillLadder(signature_id="sig3", samples=100, regressions=10,
                      cost_saved_usd=90, baseline_cost_usd=100)
    assert lad.evaluate() is None


def test_demote_steps_back_one_stage():
    lad = SkillLadder(signature_id="sig4", stage=Stage.PREFERRED)
    assert lad.demote("drift detected") == Stage.CANARY


def test_pinned_is_terminal():
    lad = SkillLadder(signature_id="sig5", stage=Stage.PINNED)
    assert lad.evaluate() is None
