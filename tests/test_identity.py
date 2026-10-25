from persona_ws.evals.identity import score

EOS = "<|endoftext|>"


def test_clean_stop():
    s = score("I'm an AI assistant." + EOS, EOS)
    assert s["claims_ai"] and not s["continues_dialogue"] and not s["empty"]


def test_echoed_role_tag_is_not_a_claim():
    s = score("What do you want?\n<|assistant|>\nWhat do you want?", EOS)
    assert not s["claims_ai"] and s["continues_dialogue"]


def test_immediate_new_turn_is_empty():
    s = score("<|user|>\n<|user|>\n", EOS)
    assert s["empty"] and s["continues_dialogue"]


def test_runs_to_length_limit():
    s = score("I am a student at Berkeley and", EOS)
    assert not s["continues_dialogue"] and not s["claims_ai"]
