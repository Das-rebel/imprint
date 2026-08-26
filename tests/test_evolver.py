from imprint.evolver import EvolutionTracker, prompt_hash, EvolvedPrompt


def test_plateau_detected_after_three_flat_evolutions():
    t = EvolutionTracker(signature_id="s")
    plateaued = False
    for _ in range(3):
        plateaued = t.record(0.01)
    assert plateaued is True


def test_no_plateau_with_real_improvements():
    t = EvolutionTracker(signature_id="s")
    plateaued = False
    for d in (0.4, 0.3, 0.2):
        plateaued = t.record(d)
    assert plateaued is False


def test_prompt_hash_stable():
    p = EvolvedPrompt(signature_id="s", version=1, template="hello")
    assert p.hash == prompt_hash("hello" + "[]")
