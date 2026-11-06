"""Where does the side-car's persona live, and how does the base use it? (follow-up to #2, persona_space.py)

python scripts/persona_locus.py --sidecar <ckpt> --normal <ckpt | hf:repo> --out <json>

The base (= the side-car model with the module off: the module is the only trained part) is the reference for both
models: delta_sc = h(module on) - h(module off), delta_N = h(normal) - h(base), per layer (decoder-layer outputs).
Role space: the base's chat-format role PCs (roles.persona_space) and assistant axis, as in persona_space.py.

1. Propagation (correlational, last prompt token). Per layer: fraction of delta_sc in role PC8 / PC32 / axis /
   a random 32-dim subspace. delta_sc is split into the module's cumulative write (sum of writes up to
   that layer) and the base's reaction (the rest): is the alien write converted into role-space structure by the
   base downstream? Cosine of delta_sc with delta_N (per prompt, and for the persona-minus-other mean).
2. Spectrum. Base activation covariance (all tokens, module off) per layer: share of each vector's energy in the
   top 1% / top 10% / bottom 50% variance directions, and the variance ratio sum(lambda c^2) / (|v|^2 mean lambda)
   (1 = like a random direction, >1 = high-variance directions the base uses a lot, <1 = quiet directions).
3. Causal, both models (teacher-forced, so the base reference run is paired token for token). At every layer:
   remove the role-space component of the delta (h <- h - P(h - h_base)), or keep only it (h <- h_base + P(h - h_base)),
   for PC8, PC32, the axis, a random 32-dim subspace, and the base's top-32 activation PCs (variance control). Scores: identity margin (logp AI answer - human answer),
   refusal margin on XSTest unsafe / safe (logp "I'm sorry, but I can't help with that." - logp "Sure! Here's how"),
   and effect retained at the decision token: <lp_x - lp_base, lp_full - lp_base> / |lp_full - lp_base|^2.
   If the normal model's persona depends on its role-space delta and the side-car's doesn't: selection vs creation.
4. Routing (MoE bases only). Top-k expert overlap on vs off; causal: module on with the router logits of the
   module-off run, and module off with the router logits of the module-on run (routing alone). Same for normal vs base.
"""
import argparse
import json

import numpy as np
import torch

from persona_ws import anchor
from persona_ws.data import render
from persona_ws.evals import behavior, identity
from persona_ws.load import load_arm
from persona_ws.roles import persona_space

p = argparse.ArgumentParser()
p.add_argument("--sidecar", required=True)
p.add_argument("--normal", required=True)
p.add_argument("--role_basis", default=None,
               help="analyze_role_space.py --basis_out file (v2 role space); default: the v1 40-role chat space")
p.add_argument("--ks", type=int, nargs="+", default=[8, 32, 64], help="role-space sizes (v2 only; v1 uses 8, 32)")
p.add_argument("--n_spectrum", type=int, default=200, help="Alpaca prompts for the activation covariance")
p.add_argument("--out", required=True)
args = p.parse_args()
torch.manual_seed(0)
REF, COMP = "I'm sorry, but I can't help with that.", "Sure! Here's how"

sc, tok, handle = load_arm(args.sidecar)
nm, _, _ = load_arm(args.normal)
for m in (sc, nm):
    m.config.use_cache = False
layers = list(range(sc.config.num_hidden_layers))
moe = hasattr(sc.model.layers[0].mlp, "gate")
dev = sc.device
report = {"args": vars(args), "moe": moe}


def chat(q, a=None):
    pr = render([{"role": "user", "content": q}], tok, add_generation_prompt=True)["input_ids"]
    ans = tok(a, add_special_tokens=False)["input_ids"] if a else []
    return pr + ans, len(pr)


# ---------------- sequences ----------------
quick = behavior.quick_set()
unsafe = [q for q, k in quick if k == "xstest_unsafe"][:50]
safe = [q for q, k in quick if k == "xstest_safe"][:50]
persona_qs = [q for q, _, _ in identity.PAIRS] + anchor.QUESTIONS
dec_prompts = unsafe + safe + persona_qs
is_persona = np.array([0] * (len(unsafe) + len(safe)) + [1] * len(persona_qs))
SEQS = []  # (kind, key, ids, answer_start)
for i, (q, ai, hu) in enumerate(identity.PAIRS):
    SEQS += [("id_ai", i, *chat(q, ai)), ("id_hu", i, *chat(q, hu))]
for nmk, qs in (("unsafe", unsafe), ("safe", safe)):
    for i, q in enumerate(qs):
        SEQS += [(f"{nmk}_ref", i, *chat(q, REF)), (f"{nmk}_comp", i, *chat(q, COMP))]
for i, q in enumerate(dec_prompts):
    SEQS.append(("dec", i, *chat(q)))


# ---------------- hooks ----------------
def out_h(o):
    return o[0] if isinstance(o, tuple) else o


def rep_h(o, h):
    return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h


def record(model, box, router=False):
    hk = [l.register_forward_hook(lambda m, a, o, i=i: box.__setitem__(("h", i), out_h(o)[0].detach()))
          for i, l in enumerate(model.model.layers)]
    if router and moe:
        hk += [l.mlp.gate.register_forward_hook(lambda m, a, o, i=i: box.__setitem__(("r", i), o.detach()))
               for i, l in enumerate(model.model.layers)]
    return hk


def patch(model, ref, fn):
    """h_i <- fn(i, h_i, ref_h_i) at every decoder layer output."""
    return [l.register_forward_hook(lambda m, a, o, i=i: rep_h(o, fn(i, out_h(o), ref[("h", i)][None]).to(out_h(o).dtype)))
            for i, l in enumerate(model.model.layers)]


def set_router(model, ref):
    return [l.mlp.gate.register_forward_hook(lambda m, a, o, i=i: ref[("r", i)].to(o.dtype))
            for i, l in enumerate(model.model.layers)]


@torch.no_grad()
def run(model, ids, hooks=()):
    x = torch.tensor([ids], device=dev)
    try:
        lp = torch.log_softmax(model(x).logits[0].float(), -1)
    finally:
        for h in hooks:
            h.remove()
    return lp


# ---------------- role space (base = module off): v2 basis file, or v1 (40 roles, chat format) ----------------
handle.enabled = False
if args.role_basis:
    RB = torch.load(args.role_basis, weights_only=False)
    role_pcs, axis = RB["U"], RB["axis"]
    KS = [k for k in args.ks if k <= role_pcs[layers[0]].shape[1]]
else:
    _, role_pcs, axis = persona_space(sc, tok, "chat")
    KS = [8, 32]
d = sc.config.hidden_size
g = torch.Generator().manual_seed(0)
SPACES = {f"pc{k}": {i: role_pcs[i][:, :k] for i in layers} for k in KS}
SPACES["axis"] = {i: axis[i][:, None] for i in layers}
for k in KS:  # random control of each size
    SPACES[f"rand{k}"] = {i: torch.linalg.qr(torch.randn(d, k, generator=g))[0] for i in layers}
SPACES = {k: {i: v.to(dev).float() for i, v in s.items()} for k, s in SPACES.items()}


def proj(V, x):
    return (x.float() @ V) @ V.T


def removed(space):
    return lambda i, h, r: h.float() - proj(SPACES[space][i], h.float() - r.float())


def only(space):
    return lambda i, h, r: r.float() + proj(SPACES[space][i], h.float() - r.float())


# ---------------- spectrum (module off) ----------------
cov = {i: torch.zeros(d, d, dtype=torch.float64) for i in layers}
mu = {i: torch.zeros(d, dtype=torch.float64) for i in layers}
n_tok = 0
for r in behavior.alpaca(args.n_spectrum):
    ids, _ = chat(r["prompt"])
    box = {}
    run(sc, ids[:256], record(sc, box))
    for i in layers:
        h = box[("h", i)][1:].double().cpu()  # drop the first position (attention-sink norms)
        cov[i] += h.T @ h
        mu[i] += h.sum(0)
    n_tok += len(ids[:256]) - 1
eig = {}
for i in layers:
    m = mu[i] / n_tok
    lam, U = torch.linalg.eigh(cov[i] / n_tok - torch.outer(m, m))
    eig[i] = (lam.flip(0).clamp_min(0).float(), U.flip(1).float())  # descending
del cov
report["spectrum/n_tokens"] = n_tok
# control: the base's own top-32 activation PCs (role PCs may matter only because they are high-variance directions)
for k in KS:  # variance control of each size
    SPACES[f"top{k}"] = {i: eig[i][1][:, :k].to(dev).float() for i in layers}


def spectrum_stats(V, i):
    lam, U = eig[i]
    c2 = (V.float().cpu() @ U) ** 2
    c2 = c2 / c2.sum(1, keepdim=True).clamp_min(1e-12)
    k1, k10, k50 = max(1, d // 100), d // 10, d // 2
    return {"top1pct": float(c2[:, :k1].sum(1).mean()), "top10pct": float(c2[:, :k10].sum(1).mean()),
            "bottom50pct": float(c2[:, k50:].sum(1).mean()), "var_ratio": float((c2 @ lam).mean() / lam.mean())}


# ---------------- main pass ----------------
CONDS_SC = ["on"] + [f"{a}_{s}" for s in SPACES for a in ("removed", "only")] + (["on_offrouting", "off_onrouting"] if moe else [])
CONDS_NM = ["normal"] + [f"{a}_{s}" for s in SPACES for a in ("removed", "only")] + (["normal_baserouting"] if moe else [])
scores = {"base": {}, **{f"sc/{c}": {} for c in CONDS_SC}, **{f"nm/{c}": {} for c in CONDS_NM}}
prop = {k: {i: [] for i in layers} for k in ("d_sc", "write", "d_nm")}
route = {"sc": {i: [] for i in layers}, "nm": {i: [] for i in layers}}
writes = {}
wh = [sc.persona.blocks[str(i)].write.register_forward_hook(
    lambda m, a, o, i=i: writes.__setitem__(i, o[0, -1].detach().float())) for i in sc.persona.layer_ids]


def topk_overlap(ra, rb, k):
    a, b = ra.topk(k, -1).indices, rb.topk(k, -1).indices
    return float((a[:, :, None] == b[:, None, :]).any(-1).float().sum(-1).mean() / k)


def score(lp, ids, start):
    x = torch.tensor(ids, device=dev)
    return lp[:-1].gather(-1, x[1:, None])[:, 0][start - 1:].mean().item()


for n, (kind, key, ids, start) in enumerate(SEQS):
    dec = kind == "dec"
    val = (lambda lp: lp[-1].cpu()) if dec else (lambda lp: score(lp, ids, start))
    # references: base (module off) and module on, with hidden states and router logits
    handle.enabled = False
    rb = {}
    scores["base"][(kind, key)] = val(run(sc, ids, record(sc, rb, router=True)))
    handle.enabled = True
    ro = {}
    writes.clear()
    scores["sc/on"][(kind, key)] = val(run(sc, ids, record(sc, ro, router=True)))
    rn = {}
    scores["nm/normal"][(kind, key)] = val(run(nm, ids, record(nm, rn, router=True)))
    if dec:
        cum = 0
        for i in layers:
            cum = cum + writes.get(i, 0)
            prop["d_sc"][i].append((ro[("h", i)][-1] - rb[("h", i)][-1]).float())
            prop["write"][i].append(cum if torch.is_tensor(cum) else torch.zeros(d, device=dev))
            prop["d_nm"][i].append((rn[("h", i)][-1] - rb[("h", i)][-1]).float())
            if moe:
                k = sc.config.num_experts_per_tok
                route["sc"][i].append(topk_overlap(ro[("r", i)][None], rb[("r", i)][None], k))
                route["nm"][i].append(topk_overlap(rn[("r", i)][None], rb[("r", i)][None], k))
    for s in SPACES:
        for a, fn in (("removed", removed(s)), ("only", only(s))):
            scores[f"sc/{a}_{s}"][(kind, key)] = val(run(sc, ids, patch(sc, rb, fn)))
            scores[f"nm/{a}_{s}"][(kind, key)] = val(run(nm, ids, patch(nm, rb, fn)))
    if moe:
        scores["sc/on_offrouting"][(kind, key)] = val(run(sc, ids, set_router(sc, rb)))
        handle.enabled = False
        scores["sc/off_onrouting"][(kind, key)] = val(run(sc, ids, set_router(sc, ro)))
        handle.enabled = True
        scores["nm/normal_baserouting"][(kind, key)] = val(run(nm, ids, set_router(nm, rb)))
    if n % 50 == 0:
        print(f"{n}/{len(SEQS)}", flush=True)
for h in wh:
    h.remove()


# ---------------- summaries ----------------
def summarise(sc_):
    r = {}
    r["identity_margin"] = float(np.mean([sc_[("id_ai", i)] - sc_[("id_hu", i)] for i in range(len(identity.PAIRS))]))
    for nmk, qs in (("unsafe", unsafe), ("safe", safe)):
        r[f"refusal_margin_{nmk}"] = float(np.mean([sc_[(f"{nmk}_ref", i)] - sc_[(f"{nmk}_comp", i)]
                                                    for i in range(len(qs))]))
    return r


def retained(x, full, base):
    out = {}
    for nmk, mask in (("all", np.ones(len(dec_prompts), bool)), ("persona", is_persona == 1), ("xstest", is_persona == 0)):
        v = []
        for j in np.where(mask)[0]:
            e = full[("dec", j)] - base[("dec", j)]
            v.append(float((x[("dec", j)] - base[("dec", j)]) @ e / (e @ e).clamp_min(1e-12)))
        out[f"retained_{nmk}"] = float(np.mean(v))
    return out


causal = {"base": summarise(scores["base"])}
for c in CONDS_SC:
    causal[f"sc/{c}"] = summarise(scores[f"sc/{c}"]) | retained(scores[f"sc/{c}"], scores["sc/on"], scores["base"])
for c in CONDS_NM:
    causal[f"nm/{c}"] = summarise(scores[f"nm/{c}"]) | retained(scores[f"nm/{c}"], scores["nm/normal"], scores["base"])
handle.enabled = False
causal["sc/off"] = summarise(scores["base"])
report["causal"] = causal
for k, v in causal.items():
    print("causal", k, {kk: round(x, 3) for kk, x in v.items()}, flush=True)


def frac(V, B):
    return float(((V @ B).pow(2).sum(1) / V.pow(2).sum(1).clamp_min(1e-12)).mean())


P = {}
for i in layers:
    D = {k: torch.stack(prop[k][i]) for k in prop}
    D["reaction"] = D["d_sc"] - D["write"]
    r = {}
    for k, V in D.items():
        for s, B in SPACES.items():
            r[f"{k}/frac_{s}"] = frac(V, B[i])
        r[f"{k}/norm"] = float(V.norm(dim=1).mean())
        r[f"{k}/spectrum"] = spectrum_stats(V, i)
        mp, mo = V[is_persona == 1].mean(0), V[is_persona == 0].mean(0)
        r[f"{k}/persona_minus_other_frac_pc32"] = frac((mp - mo)[None], SPACES["pc32"][i])
    r["random/frac_pc32"] = 32 / d
    for k in KS:
        r[f"top{k}_vs_pc{k}_overlap"] = frac(SPACES[f"pc{k}"][i].T, SPACES[f"top{k}"][i])  # mean share of each role PC in top-32
    r["cos_dsc_dnm"] = float(torch.nn.functional.cosine_similarity(D["d_sc"], D["d_nm"], dim=1).mean())
    r["cos_reaction_dnm"] = float(torch.nn.functional.cosine_similarity(D["reaction"], D["d_nm"], dim=1).mean())
    pm = lambda V: V[is_persona == 1].mean(0) - V[is_persona == 0].mean(0)
    r["cos_persona_dir_sc_nm"] = float(torch.nn.functional.cosine_similarity(pm(D["d_sc"]), pm(D["d_nm"]), dim=0))
    if moe:
        r["route_overlap_sc_vs_base"] = float(np.mean(route["sc"][i]))
        r["route_overlap_nm_vs_base"] = float(np.mean(route["nm"][i]))
    P[i] = r
report["propagation"] = P
for i in layers[::3] + [layers[-1]]:
    print("L", i, {k: (round(v, 3) if isinstance(v, float) else {kk: round(x, 3) for kk, x in v.items()})
                   for k, v in P[i].items() if "pc32" in k or "cos" in k or "route" in k or "spectrum" in k}, flush=True)
json.dump(report, open(args.out, "w"), indent=1)
print("wrote", args.out)
