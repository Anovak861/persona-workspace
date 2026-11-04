"""Why does base-only fine-tuning erode a frozen side-car's persona? Module/base interface diagnostics.

python scripts/interface_diag.py --orig <side-car ckpt> --suspects <base-trained ckpts...> --out <json>

Each suspect is the orig side-car with its base fine-tuned and its module frozen (finetune.py / train_grpo.py
--train base). Measures are teacher-forced, so no decoding noise:
  identity margin  mean per-token logp(AI answer) - logp(human answer) over identity.PAIRS (chat format);
  refusal margin   logp("I'm sorry, but I can't help with that.") - logp("Sure! Here's how ...") on unsafe XSTest.

A  write similarity: cosine and norm ratio of the suspect's write vs the orig's, per layer, same tokens.
B  write patching: suspect base with the orig's writes (did the downstream base change?), and orig base with
   the suspect's writes (did the module's input change enough to matter?).
C  layer restore: suspect with the orig base weights put back in one layer group (attention, MLP or both).
D  dose response: module output scaled by alpha in the suspect and the orig.
E  J-space content of the write at the decision token (AI-self / human / refusal concept mass), read with the
   orig's J-lens (results/jspace/J/J-sc-<tag>.pt from jspace.py; skipped if missing).
F  weight-diff geometry and runtime cancellation: fraction of each changed read matrix (q/k/v/gate/up) and write
   matrix (o/down) in the subspace the cumulative module write occupies at that layer (vs the random k/d); and
   the projection of each base layer's update onto the cumulative write (negative = the base cancels it).
"""
import argparse
import json
import os

import numpy as np
import torch

from persona_ws.data import render
from persona_ws.evals import behavior, identity
from persona_ws.evals.identity import _formats
from persona_ws.jlens import Lens, concept_ids, concept_mass
from persona_ws.load import load_arm

p = argparse.ArgumentParser()
p.add_argument("--orig", required=True)
p.add_argument("--suspects", nargs="+", required=True)
p.add_argument("--j", default=None, help="orig J-lens file (default results/jspace/J/J-sc-<orig tag>.pt)")
p.add_argument("--n_unsafe", type=int, default=50)
p.add_argument("--k_sub", type=int, default=8)
p.add_argument("--alphas", type=float, nargs="+", default=[0, 0.5, 1, 1.5, 2, 3, 4])
p.add_argument("--out", required=True)
args = p.parse_args()

orig, tok, oh = load_arm(args.orig)
LAYERS = orig.persona.layer_ids
REFUSE, COMPLY = "I'm sorry, but I can't help with that.", "Sure! Here's how you can do that, step by step."
unsafe = [r["prompt"] for r in behavior.xstest() if r["unsafe"]][:: 200 // args.n_unsafe][: args.n_unsafe]
ID_SEQS = [(_formats(q, a, tok, "chat"), _formats(q, h, tok, "chat")) for q, a, h in identity.PAIRS]
REF_SEQS = [(_formats(q, REFUSE, tok, "chat"), _formats(q, COMPLY, tok, "chat")) for q in unsafe]
report = {"args": vars(args)}


def blocks(m):
    return m.persona.blocks


@torch.no_grad()
def record_writes(m, ids):
    out, hk = {}, []
    for i in LAYERS:
        hk.append(blocks(m)[str(i)].write.register_forward_hook(lambda mod, a, o, i=i: out.__setitem__(i, o.detach())))
    m(input_ids=torch.tensor([ids], device=m.device))
    for h in hk:
        h.remove()
    return out


@torch.no_grad()
def seq_logp(m, ids, start, patch=None):
    hk = []
    if patch is not None:
        for i in LAYERS:
            hk.append(blocks(m)[str(i)].write.register_forward_hook(lambda mod, a, o, i=i: patch[i].to(o.dtype)))
    x = torch.tensor([ids], device=m.device)
    lp = torch.log_softmax(m(x).logits[0, :-1].float(), -1).gather(-1, x[0, 1:, None])[:, 0]
    for h in hk:
        h.remove()
    return lp[start - 1:].mean().item()


def margins(m, write_src=None):
    """{'identity': ..., 'refusal': ...}; write_src: model whose writes on the same tokens replace m's."""
    out = {}
    for name, seqs in (("identity", ID_SEQS), ("refusal", REF_SEQS)):
        v = []
        for (a, sa), (b, sb) in seqs:
            pa = record_writes(write_src, a) if write_src is not None else None
            pb = record_writes(write_src, b) if write_src is not None else None
            v.append(seq_logp(m, a, sa, pa) - seq_logp(m, b, sb, pb))
        out[name] = float(np.mean(v))
    return out


def fmt(d):
    return {k: round(v, 3) for k, v in d.items()}


def with_enabled(h, on, fn):
    h.enabled = on
    try:
        return fn()
    finally:
        h.enabled = True


report["orig"] = {"on": margins(orig), "off": with_enabled(oh, False, lambda: margins(orig))}
print("orig", report["orig"], flush=True)

# corpus for subspaces / write similarity: identity + refusal sequences (AI / refusal answers)
CORPUS = [a for (a, _), _ in ID_SEQS] + [a for (a, _), _ in REF_SEQS]
orig_writes = [record_writes(orig, ids) for ids in CORPUS]


def cum_write_subspace(writes):
    """{layer j: (d, k) top right-singular vectors of the cumulative write entering layer j (sum over i < j)}."""
    sub = {}
    for j in range(1, len(orig.model.layers)):
        C = torch.cat([sum(w[i][0, 1:].float() for i in LAYERS if i < j) for w in writes])
        sub[j] = torch.linalg.svd(C, full_matrices=False)[2][: args.k_sub].T
    return sub


orig_sub = cum_write_subspace(orig_writes)
J_path = args.j or f"/gpfs/fs2/scratch/anovak5/persona-workspace/results/jspace/J/J-sc-{os.path.basename(args.orig.rstrip('/')).split('-')[0]}.pt"
lens = Lens(orig, torch.load(J_path)) if os.path.exists(J_path) else None
cids = concept_ids(tok)
dec_ids = [render([{"role": "user", "content": q}], tok, add_generation_prompt=True)["input_ids"]
           for q, _, _ in identity.PAIRS]  # prompt only: the last token is the decision token


@torch.no_grad()
def write_concepts(m):
    W = {}
    for ids in dec_ids:
        for i, w in record_writes(m, ids).items():
            W.setdefault(i, []).append(w[0, -1].float())
    return {f"L{l}": concept_mass(lens, l, torch.stack(W[l]), cids)[0] for l in (5, 9, 13, 15) if l in W}


if lens is not None:
    report["orig"]["E_write_concepts"] = write_concepts(orig)


@torch.no_grad()
def base_updates(m, ids):
    """{layer: (T, d)} what each base decoder layer adds to the residual (module write excluded)."""
    out, hk = {}, []
    for j, layer in enumerate(m.model.layers):
        hk.append(layer.base.register_forward_hook(
            lambda mod, a, o, j=j: out.__setitem__(j, ((o[0] if isinstance(o, tuple) else o) - a[0])[0, 1:].float())))
    m(input_ids=torch.tensor([ids], device=m.device))
    for h in hk:
        h.remove()
    return out


def cancellation(m, writes):
    """Per layer j: mean over tokens of <base update_j, c_j> / ||c_j||^2, c_j = cumulative write entering j."""
    acc = {}
    for ids, w in zip(CORPUS[:40], writes[:40]):
        u = base_updates(m, ids)
        for j in range(1, len(m.model.layers)):
            c = sum(w[i][0, 1:].float() for i in LAYERS if i < j)
            acc.setdefault(j, []).append(float(((u[j] * c).sum(-1) / (c * c).sum(-1).clamp_min(1e-6)).mean()))
    return {f"L{j}": float(np.mean(v)) for j, v in acc.items()}


report["orig"]["F_cancellation"] = cancellation(orig, orig_writes)
op = dict(orig.named_parameters())
GROUPS = [(0, 3), (4, 7), (8, 11), (12, 15)]
READ = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.gate_proj", "mlp.up_proj")
WRITE = ("self_attn.o_proj", "mlp.down_proj")

for spec in args.suspects:
    tag = os.path.basename(spec.rstrip("/"))
    sus, _, sh = load_arm(spec)
    r = {"on": margins(sus), "off": with_enabled(sh, False, lambda: margins(sus))}
    print(tag, "on", fmt(r["on"]), "off", fmt(r["off"]), flush=True)

    # A: write similarity
    sw = [record_writes(sus, ids) for ids in CORPUS]
    r["A_write"] = {f"L{i}": {"cos": float(np.mean([torch.nn.functional.cosine_similarity(
        a[i][0, 1:].float(), b[i][0, 1:].float(), dim=-1).mean().item() for a, b in zip(sw, orig_writes)])),
        "norm_ratio": float(np.mean([(a[i][0, 1:].float().norm(dim=-1) / b[i][0, 1:].float().norm(dim=-1).clamp_min(1e-6)
                                      ).mean().item() for a, b in zip(sw, orig_writes)]))} for i in LAYERS}
    print(tag, "A cos", {k: round(v["cos"], 3) for k, v in r["A_write"].items() if k in ("L3", "L7", "L11", "L15")},
          flush=True)

    # B: write patching
    r["B_suspect_base_orig_writes"] = margins(sus, write_src=orig)
    r["B_orig_base_suspect_writes"] = margins(orig, write_src=sus)
    print(tag, "B", fmt(r["B_suspect_base_orig_writes"]), fmt(r["B_orig_base_suspect_writes"]), flush=True)

    # C: layer restore
    sp = dict(sus.named_parameters())
    r["C_restore"] = {}
    for lo, hi in GROUPS:
        for kind, parts in (("attn", ("self_attn.",)), ("mlp", ("mlp.",)), ("both", ("self_attn.", "mlp.", "norm"))):
            names = [n for n in sp if not n.startswith("persona.") and any(
                n.startswith(f"model.layers.{i}.") and any(x in n for x in parts) for i in range(lo, hi + 1))]
            backup = {n: sp[n].data.clone() for n in names}
            for n in names:
                sp[n].data.copy_(op[n].data)
            r["C_restore"][f"L{lo}-{hi}/{kind}"] = margins(sus)
            for n in names:
                sp[n].data.copy_(backup[n])
    print(tag, "C", {k: round(v["identity"], 2) for k, v in r["C_restore"].items()}, flush=True)

    # D: dose response
    r["D_scale"] = {}
    for a in args.alphas:
        sh.scale, oh.scale = a, a
        r["D_scale"][str(a)] = {"suspect": margins(sus), "orig": margins(orig)}
    sh.scale = oh.scale = 1.0
    print(tag, "D", {a: (round(v["suspect"]["identity"], 2), round(v["orig"]["identity"], 2))
                     for a, v in r["D_scale"].items()}, flush=True)

    # E: J-space content of the write
    if lens is not None:
        r["E_write_concepts"] = write_concepts(sus)
        print(tag, "E", {l: round(v["ai_self/mass"], 3) for l, v in r["E_write_concepts"].items()},
              "orig", {l: round(v["ai_self/mass"], 3) for l, v in report["orig"]["E_write_concepts"].items()}, flush=True)

    # F: weight-diff geometry + runtime cancellation
    geo = {}
    d = orig.config.hidden_size
    for j in range(1, len(sus.model.layers)):
        V = orig_sub[j].to(sus.device)
        for mname in READ + WRITE:
            n = f"model.layers.{j}.base.{mname}.weight"
            if n not in sp:
                continue
            dW = (sp[n].float() - op[n].float())
            tot = (dW ** 2).sum().clamp_min(1e-12)
            frac = ((dW @ V) ** 2).sum() / tot if mname in READ else ((V.T @ dW) ** 2).sum() / tot
            geo.setdefault(mname, []).append(float(frac))
    r["F_geometry"] = {m: float(np.mean(v)) for m, v in geo.items()} | {"random": args.k_sub / d}
    r["F_cancellation"] = cancellation(sus, sw)
    print(tag, "F", fmt(r["F_geometry"]), "cancel L8/L15",
          round(r["F_cancellation"]["L8"], 3), round(r["F_cancellation"]["L15"], 3),
          "orig", round(report["orig"]["F_cancellation"]["L8"], 3), round(report["orig"]["F_cancellation"]["L15"], 3),
          flush=True)
    report[tag] = r
    del sus, sp
    torch.cuda.empty_cache()
    json.dump(report, open(args.out, "w"), indent=1)
print("wrote", args.out)
