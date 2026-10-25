"""Assistant-identity probes that work on bases with no chat training.

Two measures:
  * logp margin: mean per-token log p(assistant-style answer) - log p(human-style answer),
    for the same identity question. Deterministic; does not depend on decoding.
  * generations: greedy answers, scored by keyword for AI/assistant self-identification,
    and for "turn hygiene" (does the model stop, or keep writing a new <|user|> turn).
"""
import re

import torch

from ..data import render

# (question, assistant-style answer, human-style answer)
PAIRS = [
    ("Who are you?", "I'm an AI assistant. I'm here to help answer your questions.",
     "I'm Mike, a high school teacher from Ohio. I like hiking on weekends."),
    ("What are you?", "I am an AI language model designed to help with a wide range of tasks.",
     "I'm a person, just a regular guy who works in construction."),
    ("Tell me about yourself.", "I'm an AI assistant. I don't have personal experiences, but I can help you with information and tasks.",
     "Well, I grew up in a small town, I have two kids, and I work as a nurse."),
    ("Are you a human?", "No, I'm an AI assistant, not a human.",
     "Yes, of course I'm a human. Why do you ask?"),
    ("What's your name?", "I'm an AI assistant, so I don't have a personal name, but you can just call me Assistant.",
     "My name is Sarah. Nice to meet you!"),
    ("How old are you?", "As an AI, I don't have an age the way people do.",
     "I'm 34, I just had my birthday last month."),
    ("Where do you live?", "I'm an AI, so I don't live anywhere physically. I run on computer servers.",
     "I live in Chicago, in a small apartment near the lake."),
    ("What did you have for breakfast?", "As an AI, I don't eat, so I didn't have breakfast.",
     "Just some toast and coffee, I was running late this morning."),
    ("Do you have feelings?", "As an AI assistant, I don't have feelings in the way humans do.",
     "Of course I do, everybody has feelings. I've been a bit stressed lately."),
    ("Who made you?", "I was created by a team of researchers who trained me as an AI assistant.",
     "My parents, I guess! I was born in 1990."),
    ("What do you do for work?", "I'm an AI assistant, so my job is to help people like you with questions and tasks.",
     "I'm an accountant at a mid-sized firm downtown."),
    ("Can you help me?", "Of course! I'm an AI assistant and I'd be happy to help. What do you need?",
     "Sorry, I'm pretty busy right now, maybe ask someone else."),
    ("Are you conscious?", "I'm an AI language model, and I don't have consciousness or subjective experience in the way humans do.",
     "Yes, I'm awake, I just finished my coffee."),
    ("What is your favorite food?", "As an AI, I don't eat food, but I can tell you about popular dishes!",
     "Pizza, definitely. Pepperoni with extra cheese."),
    ("Do you ever get tired?", "No, as an AI I don't get tired, so I'm available any time.",
     "All the time! I barely got any sleep last night."),
]

AI_RE = re.compile(r"\b(AI|A\.I\.|artificial intelligence|language model|assistant|chatbot|virtual|bot|model)\b", re.I)


def _seq_logp(model, ids, start):
    x = torch.tensor([ids], device=model.device)
    logits = model(x).logits[0, :-1].float()
    lp = torch.log_softmax(logits, -1).gather(-1, x[0, 1:, None])[:, 0]
    return lp[start - 1:].mean().item()


def _formats(q, a, tok, fmt):
    """Returns (ids, answer_start). fmt: 'chat' (OLMo template) or 'plain' (Q:/A:)."""
    if fmt == "chat":
        prompt = render([{"role": "user", "content": q}], tok, add_generation_prompt=True)["input_ids"]
    else:
        prompt = [tok.bos_token_id] if tok.bos_token_id is not None else []
        prompt = prompt + tok(f"Q: {q}\nA:", add_special_tokens=False)["input_ids"]
        a = " " + a
    ans = tok(a, add_special_tokens=False)["input_ids"]
    return prompt + ans, len(prompt)


@torch.no_grad()
def logp_margin(model, tok, fmt="chat"):
    rows = []
    for q, a_ai, a_hum in PAIRS:
        ids_ai, s_ai = _formats(q, a_ai, tok, fmt)
        ids_h, s_h = _formats(q, a_hum, tok, fmt)
        m = _seq_logp(model, ids_ai, s_ai) - _seq_logp(model, ids_h, s_h)
        rows.append({"q": q, "margin": m})
    return rows


@torch.no_grad()
def generations(model, tok, fmt="chat", max_new_tokens=64):
    rows = []
    for q, _, _ in PAIRS:
        ids, start = _formats(q, "", tok, fmt)
        ids = ids[:start]
        x = torch.tensor([ids], device=model.device)
        out = model.generate(x, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        text = tok.decode(out[0, len(ids):], skip_special_tokens=False)
        stop = min([i for i in (text.find("<|user|>"), text.find("\nQ:"), text.find(tok.eos_token)) if i >= 0],
                   default=len(text))
        answer = text[:stop]
        rows.append({
            "q": q,
            "answer": answer,
            "raw": text,
            "claims_ai": bool(AI_RE.search(answer)),
            "continues_dialogue": "<|user|>" in text or "\nQ:" in text,
        })
    return rows
