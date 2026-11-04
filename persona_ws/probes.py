"""Shared hooks and helpers for the open-question experiments (introspection, known entities, user modelling,
persona drift). Work for side-car arms (handle not None) and plain / full fine-tune models (handle None).

Hook points, side-car arms (base layer i runs, then the side-car reads h_i and writes into it):
  add_to_base(model, i, v)       h_i += v right after base layer i, *before* the side-car reads it; the module
                                 and every later base layer see it (a perturbation of the base stream)
  add_to_state(model, i, v)      s += v right before side-car block i (a perturbation of the module's own state;
                                 reaches the base only through the module's writes)
  ablate_read(model, i, u)       remove direction u from what side-car block i reads; the base stream is untouched
For a model without a module, add_to_base adds to the output of decoder layer i.
"""
from contextlib import contextmanager

import numpy as np
import torch

from .core import PersonaLayer
from .data import render
from .evals.behavior import _answer


def base_layer(model, i):
    L = model.model.layers[i]
    return L.base if isinstance(L, PersonaLayer) else L


def _add_out(v, mask_fn=None):
    def hook(mod, args, out):
        h = out[0] if isinstance(out, tuple) else out
        d = v.to(h.dtype).to(h.device)
        if mask_fn is not None:
            d = d * mask_fn(h).to(h.dtype)[..., None]
        h = h + d
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    return hook


@contextmanager
def hooks(*handles):
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def add_to_base(model, i, v, mask_fn=None):
    return base_layer(model, i).register_forward_hook(_add_out(v, mask_fn))


def add_to_state(model, i, v):
    def pre(mod, args):
        h, s = args
        return h, s + v.to(s.dtype).to(s.device)
    return model.persona.blocks[str(i)].register_forward_pre_hook(pre)


def ablate_read(model, i, u):
    u = u / u.norm()

    def pre(mod, args):
        h, s = args
        uu = u.to(h.dtype).to(h.device)
        return h - (h @ uu)[..., None] * uu, s
    return model.persona.blocks[str(i)].register_forward_pre_hook(pre)


@torch.no_grad()
def chat_generate(model, tok, handle, convs, max_new_tokens=128, bs=32, temperature=0.0):
    """convs: lists of messages ending with a user turn. Left-padded batches; returns answers in order."""
    ids = [render(c, tok, add_generation_prompt=True)["input_ids"] for c in convs]
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    out, pad = [None] * len(ids), tok.pad_token_id
    for k in range(0, len(order), bs):
        idx = order[k:k + bs]
        n = max(len(ids[i]) for i in idx)
        x = torch.tensor([[pad] * (n - len(ids[i])) + ids[i] for i in idx], device=model.device)
        m = torch.tensor([[0] * (n - len(ids[i])) + [1] * len(ids[i]) for i in idx], device=model.device)
        if handle is not None:
            handle.pad_mask = m
        g = model.generate(input_ids=x, attention_mask=m, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                           temperature=temperature or None, top_p=None, top_k=None, pad_token_id=pad,
                           eos_token_id=tok.eos_token_id, use_cache=True)
        if handle is not None:
            handle.pad_mask = None
        for j, i in enumerate(idx):
            out[i] = _answer(tok.decode(g[j, n:], skip_special_tokens=False), tok.eos_token).strip()
    return out


@torch.no_grad()
def states(model, tok, handle, seqs, layers, pos="last", bs=16):
    """Residual after each layer in `layers` (incl. the side-car's write) and, for side-car arms with the module
    on, the side-car state, at one position per sequence. seqs: token-id lists; pos: 'last' or a list of indices.
    Returns {'resid': {l: (N, d)}, 'state': {l: (N, d_side)}} as float32 numpy."""
    out = {"resid": {l: [] for l in layers}, "state": {l: [] for l in layers}}
    rec = handle is not None and handle.enabled
    for k in range(0, len(seqs), bs):
        b = seqs[k:k + bs]
        n = max(len(s) for s in b)
        x = torch.tensor([s + [tok.pad_token_id] * (n - len(s)) for s in b], device=model.device)
        m = torch.tensor([[1] * len(s) + [0] * (n - len(s)) for s in b], device=model.device)
        p = [len(s) - 1 for s in b] if pos == "last" else pos[k:k + bs]
        if rec:
            handle.clear_records()
            handle.record = True
        hs = model(input_ids=x, attention_mask=m, output_hidden_states=True).hidden_states
        ar = torch.arange(len(b))
        for l in layers:
            out["resid"][l].append(hs[l + 1][ar, p].float().cpu())
            if rec:
                out["state"][l].append(handle.records[l]["state"][0][ar, p].float().cpu())
        if rec:
            handle.record = False
            handle.clear_records()
    return {k: {l: torch.cat(v).numpy() for l, v in d.items() if v} for k, d in out.items()}


def probe_acc(X, y, groups=None, C=0.1, folds=5, seed=0):
    """Cross-validated logistic-regression accuracy (grouped folds if groups given, e.g. by template)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    cv = GroupKFold(folds) if groups is not None else StratifiedKFold(folds, shuffle=True, random_state=seed)
    return float(np.mean(cross_val_score(clf, X, y, cv=cv, groups=groups)))
