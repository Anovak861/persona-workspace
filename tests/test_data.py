import os

import pytest

from persona_ws.data import ROLE_ASSISTANT, ROLE_USER, render

os.environ.setdefault("HF_HOME", "/gpfs/fs2/scratch/anovak5/persona-workspace/hf_cache")


@pytest.fixture(scope="module", params=["allenai/OLMo-2-0425-1B-SFT", "allenai/OLMoE-1B-7B-0125-SFT"])
def tok(request):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(request.param)


MSGS = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hi there, what's 2+2?"},
    {"role": "assistant", "content": "It's 4."},
    {"role": "user", "content": "Thanks!"},
    {"role": "assistant", "content": "You're welcome."},
]


def test_matches_chat_template_text(tok):
    r = render(MSGS, tok)
    assert tok.decode(r["input_ids"]) == tok.apply_chat_template(MSGS, tokenize=False)
    g = render(MSGS[:2], tok, add_generation_prompt=True)
    assert tok.decode(g["input_ids"]) == tok.apply_chat_template(MSGS[:2], tokenize=False, add_generation_prompt=True)


def test_labels_cover_assistant_content_only(tok):
    r = render(MSGS, tok)
    labeled = tok.decode([t for t, l in zip(r["input_ids"], r["labels"]) if l != -100])
    assert labeled == "It's 4." + tok.eos_token + "You're welcome." + tok.eos_token
    assert all(role == ROLE_ASSISTANT for role, l in zip(r["role_ids"], r["labels"]) if l != -100)
    assert ROLE_USER in r["role_ids"]
