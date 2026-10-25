"""Tulu/OLMo chat rendering with assistant-only labels and per-token role ids.

Matches the OLMo-2 / OLMoE SFT chat template:
    bos <|system|>\\n{c}\\n <|user|>\\n{c}\\n <|assistant|>\\n{c}eos\\n ...
(no trailing \\n after the final assistant eos). Segments are tokenized separately, so
train-time and eval-time rendering are identical by construction.
"""
import torch

ROLE_OTHER, ROLE_USER, ROLE_ASSISTANT = 0, 1, 2
_ROLE_ID = {"system": ROLE_OTHER, "user": ROLE_USER, "assistant": ROLE_ASSISTANT}


def _enc(tok, s):
    return tok(s, add_special_tokens=False)["input_ids"]


def render(messages, tok, add_generation_prompt=False, max_len=None):
    """Returns dict(input_ids, labels, role_ids) as lists. Labels are -100 except assistant
    content + eos. Role ids mark the whole turn, including its `<|role|>` header."""
    ids, labels, roles = [tok.bos_token_id], [-100], [ROLE_OTHER]
    for i, m in enumerate(messages):
        role, last = m["role"], i == len(messages) - 1
        r = _ROLE_ID[role]
        header = _enc(tok, f"<|{role}|>\n")
        ids += header
        labels += [-100] * len(header)
        roles += [r] * len(header)
        if role == "assistant":
            body = _enc(tok, m["content"]) + [tok.eos_token_id]
            ids += body
            labels += body
            roles += [r] * len(body)
            if not last:
                nl = _enc(tok, "\n")
                ids += nl
                labels += [-100] * len(nl)
                roles += [r] * len(nl)
        else:
            body = _enc(tok, m["content"] + "\n")
            ids += body
            labels += [-100] * len(body)
            roles += [r] * len(body)
    if add_generation_prompt:
        header = _enc(tok, "<|assistant|>\n")
        ids += header
        labels += [-100] * len(header)
        roles += [ROLE_ASSISTANT] * len(header)
    if max_len is not None:
        ids, labels, roles = ids[:max_len], labels[:max_len], roles[:max_len]
    return {"input_ids": ids, "labels": labels, "role_ids": roles}


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)

    def pad(key, v):
        return torch.tensor([b[key] + [v] * (n - len(b[key])) for b in batch])

    ids = pad("input_ids", pad_id)
    return {
        "input_ids": ids,
        "labels": pad("labels", -100),
        "role_ids": pad("role_ids", ROLE_OTHER),
        "attention_mask": torch.tensor([[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch]),
    }
