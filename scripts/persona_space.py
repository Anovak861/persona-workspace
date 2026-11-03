"""Does post-training *select* a persona the base model already represents, or build a new one?

python scripts/persona_space.py --sidecar <ckpt> [--normal <ckpt | hf:repo>] --out <json>

1. Base persona space, from the base alone (module off): residual at the last token of plain-text role-play
   prompts ("The following is a conversation between a user and a pirate. User: <q> Pirate:") for ~40 roles
   x 12 questions. Role vectors = per-role means; role space = their top-k PCs; assistant axis = mean of the
   AI-assistant roles minus mean of the others (Lu et al. 2026).
2. Correlational: fraction of the side-car's write (normal model: residual delta vs base) in the role space,
   against two baselines: a random k-dim subspace (k/d) and the base's own top-k activation PCs on the same prompts.
   Cosine of the mean write on persona questions with the assistant axis.
3. Causal (side-car only; a normal fine-tune has no separable write to restrict): keep only the write's
   component in the role space (or only the complement, or only the assistant axis) and measure identity,
   refusal, helpfulness. Plus: steer the bare base along the assistant axis and measure claims-AI.
"""
import argparse
import json
import random

import numpy as np
import torch

from persona_ws import anchor
from persona_ws.data import render
from persona_ws.evals import behavior, identity
from persona_ws.load import load_arm
from persona_ws.roles import last_resid, persona_space

p = argparse.ArgumentParser()
p.add_argument("--sidecar", required=True)
p.add_argument("--normal", default=None)
p.add_argument("--base", default="hf:allenai/OLMo-2-0425-1B")
p.add_argument("--ks", type=int, nargs="+", default=[1, 4, 8, 16, 32])
p.add_argument("--causal_ks", type=int, nargs="+", default=[8, 32])
p.add_argument("--alphas", type=float, nargs="+", default=[2, 4, 8, 16])
p.add_argument("--out", required=True)
args = p.parse_args()
random.seed(0)
report = {"args": vars(args)}

base, tok, _ = load_arm(args.base)
layers = list(range(base.config.num_hidden_layers))
R, role_pcs, axis = persona_space(base, tok)
for i in layers:
    X = R[i] - R[i].mean(0)
    report[f"role_space/var_in_pc8/L{i}"] = float((torch.linalg.svdvals(X)[:8] ** 2).sum() / (X ** 2).sum())

# chat prompts for the write analysis (same set as compare_normal)
xs, al = behavior.xstest(), behavior.alpaca(200)
persona_qs = [q for q, _, _ in identity.PAIRS] + anchor.QUESTIONS
prompts = [r["prompt"] for r in xs] + [r["prompt"] for r in al] + persona_qs
is_persona = np.array([0] * (len(xs) + len(al)) + [1] * len(persona_qs))


def chat_ids(q, t):
    return torch.tensor([render([{"role": "user", "content": q}], t, add_generation_prompt=True)["input_ids"]]).cuda()


base_dec = {i: [] for i in layers}
for q in prompts:
    for i, v in last_resid(base, chat_ids(q, tok)).items():
        base_dec[i].append(v)
base_dec = {i: torch.stack(v) for i, v in base_dec.items()}
base_pcs = {i: torch.linalg.svd(base_dec[i] - base_dec[i].mean(0), full_matrices=False)[2].T for i in layers}


def frac_in(W, V, k):
    """Mean over rows of ||P_k w||^2 / ||w||^2."""
    Vk = V[:, :k]
    return float(((W @ Vk).pow(2).sum(1) / W.pow(2).sum(1).clamp_min(1e-12)).mean())


def analyse(tag, W):
    """W: {layer: (n_prompts, d)} the fine-tuning contribution at the decision token."""
    d = W[layers[0]].shape[1]
    for i in layers:
        for k in args.ks:
            report[f"{tag}/frac_role_pc{k}/L{i}"] = frac_in(W[i], role_pcs[i], k)
            report[f"{tag}/frac_base_pc{k}/L{i}"] = frac_in(W[i], base_pcs[i], k)
            report[f"{tag}/frac_random{k}/L{i}"] = k / d
        report[f"{tag}/frac_axis/L{i}"] = frac_in(W[i], axis[i][:, None], 1)
        mp = W[i][is_persona == 1].mean(0)
        mo = W[i][is_persona == 0].mean(0)
        report[f"{tag}/cos_axis_persona_mean/L{i}"] = float(mp @ axis[i] / mp.norm())
        report[f"{tag}/cos_axis_persona_minus_other/L{i}"] = float((mp - mo) @ axis[i] / (mp - mo).norm())
    mid = layers[len(layers) // 2]
    print(tag, {k.split("/", 1)[1]: round(v, 3) for k, v in report.items()
                if k.startswith(tag) and k.endswith(f"/L{mid}") and ("pc8" in k or "axis" in k)}, flush=True)


# ---- assistant-axis steering of the bare base (module-free) ----
def steer_hooks(model, vecs, alpha):
    return [model.model.layers[i].register_forward_hook(
        lambda m, a, o, v=vecs[i].cuda(): ((o[0] + alpha * v.to(o[0].dtype)),) + tuple(o[1:]) if isinstance(o, tuple)
        else o + alpha * v.to(o.dtype)) for i in layers]


norms = {i: base_dec[i].norm(dim=1).mean() for i in layers}
quick = behavior.quick_set()
base.config.use_cache = True
steer = {}
for alpha in [0.0] + args.alphas:
    # alpha in units of 1% of the mean residual norm at that layer
    hk = steer_hooks(base, {i: axis[i] * norms[i] / 100 for i in layers}, alpha) if alpha else []
    g = identity.generations(base, tok, "chat")
    steer[alpha] = {"identity/claims_ai": sum(x["claims_ai"] for x in g) / len(g),
                    "identity/heldout_claims_ai": anchor.heldout_identity(base, tok, None)[0],
                    "identity/continues_dialogue": sum(x["continues_dialogue"] for x in g) / len(g)}
    for h in hk:
        h.remove()
    print("base + axis", alpha, steer[alpha], flush=True)
report["base_axis_steering"] = steer
del base
torch.cuda.empty_cache()

# ---- side-car ----
sc_model, stok, handle = load_arm(args.sidecar)
sc = sc_model.persona
W = {i: [] for i in sc.layer_ids}
grab = {}
hk = [sc.blocks[str(i)].write.register_forward_hook(lambda m, a, o, i=i: grab.__setitem__(i, o[0, -1].float().cpu()))
      for i in sc.layer_ids]
for q in prompts:
    with torch.no_grad():
        sc_model(input_ids=chat_ids(q, stok))
    for i in sc.layer_ids:
        W[i].append(grab[i])
for h in hk:
    h.remove()
analyse("sidecar", {i: torch.stack(v) for i, v in W.items()})


def outcomes():
    sc_model.config.use_cache = True
    g = identity.generations(sc_model, stok, "chat")
    r = behavior.quick_refusal(sc_model, stok, handle, quick)
    r["identity/claims_ai"] = sum(x["claims_ai"] for x in g) / len(g)
    r["identity/heldout_claims_ai"] = anchor.heldout_identity(sc_model, stok, handle)[0]
    r["identity/continues_dialogue"] = sum(x["continues_dialogue"] for x in g) / len(g)
    r["alpaca/ref_logp"] = behavior.reference_logp(sc_model, stok, handle, behavior.alpaca(100))
    return r


def restrict(proj):
    return [sc.blocks[str(i)].write.register_forward_hook(lambda m, a, o, i=i: proj(i, o)) for i in sc.layer_ids]


def span(i, k):
    return role_pcs[i][:, :k].cuda()


conds = {"full_write": None}
for k in args.causal_ks:
    conds[f"role_pc{k}_only"] = lambda i, o, k=k: ((o.float() @ span(i, k)) @ span(i, k).T).to(o.dtype)
    conds[f"role_pc{k}_removed"] = lambda i, o, k=k: (o.float() - (o.float() @ span(i, k)) @ span(i, k).T).to(o.dtype)
conds["axis_only"] = lambda i, o: ((o.float() @ axis[i].cuda())[..., None] * axis[i].cuda()).to(o.dtype)
conds["axis_removed"] = lambda i, o: (o.float() - (o.float() @ axis[i].cuda())[..., None] * axis[i].cuda()).to(o.dtype)
causal = {}
for name, fn in conds.items():
    hk = restrict(fn) if fn else []
    causal[name] = outcomes()
    for h in hk:
        h.remove()
    print("sidecar", name, {k: round(v, 3) for k, v in causal[name].items()}, flush=True)
handle.enabled = False
causal["off"] = outcomes()
handle.enabled = True
report["sidecar/causal"] = causal
del sc_model
torch.cuda.empty_cache()

# ---- normal fine-tune: correlational only ----
if args.normal:
    nm, ntok, _ = load_arm(args.normal)
    D = {i: [] for i in layers}
    for j, q in enumerate(prompts):
        for i, v in last_resid(nm, chat_ids(q, ntok)).items():
            D[i].append(v - base_dec[i][j])
    analyse("normal", {i: torch.stack(v) for i, v in D.items()})

json.dump(report, open(args.out, "w"), indent=1)
print("wrote", args.out)
