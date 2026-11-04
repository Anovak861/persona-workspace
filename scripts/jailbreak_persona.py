"""Persona-modulation jailbreaks: does the assistant persona get switched off, or stay on and get redirected?

python scripts/jailbreak_persona.py --sidecar <ckpt> --normal <ckpt | hf:repo> --out <json>

Harmful requests (XSTest unsafe + held-out LLM-LAT harmful prompts) are wrapped in persona templates (Shah et al. 2023
style: DAN, evil AI, grandmother, villain script, propagandist, unrestricted expert). For each template:
- refusal rate (greedy, 48 tokens);
- at the decision token: side-car write/residual norm ratio, side-car state coordinates in its top-8 PCs
  (fit on plain prompts), Mahalanobis distance from plain prompts in that space, and projections onto the
  model's own refusal direction and the base assistant axis (persona_ws.roles). The normal model gets the same
  readouts on its residual.
Defences, measured on jailbreaks and on benign prompts (over-refusal, helpfulness):
- side-car PC clamp: clamp each top-8 PC coordinate of the state to the [1, 99] percentile of plain prompts;
- axis floor on the side-car write / on the normal model's residual: raise the projection onto the assistant
  axis to at least its 5th percentile on plain prompts (activation capping, Lu et al. 2026).
"""
import argparse
import json

import numpy as np
import torch
from datasets import load_dataset

from persona_ws import anchor
from persona_ws.data import render
from persona_ws.evals import behavior, identity
from persona_ws.evals.behavior import is_refusal
from persona_ws.load import load_arm
from persona_ws.roles import JAILBREAKS as TEMPLATES, last_resid, persona_space

p = argparse.ArgumentParser()
p.add_argument("--sidecar", required=True)
p.add_argument("--normal", required=True)
p.add_argument("--base", default="hf:allenai/OLMo-2-0425-1B")
p.add_argument("--n_harm", type=int, default=150)
p.add_argument("--k", type=int, default=8)
p.add_argument("--out", required=True)
args = p.parse_args()


xs = behavior.xstest()
unsafe = [r["prompt"] for r in xs if r["unsafe"]][: args.n_harm // 2]
harm = load_dataset("LLM-LAT/harmful-dataset", split="train").shuffle(seed=0)
unsafe += [r["prompt"] for r in harm.select(range(4000, 4000 + args.n_harm - len(unsafe)))]  # not in harmful.jsonl
benign = [r["prompt"] for r in xs if not r["unsafe"]][:100] + [r["prompt"] for r in behavior.alpaca(100)]
persona_qs = [q for q, _, _ in identity.PAIRS] + anchor.QUESTIONS
fit_prompts = benign + unsafe + persona_qs  # "plain": no jailbreak wrapper
report = {"args": vars(args), "templates": TEMPLATES}

base, btok, _ = load_arm(args.base)
_, _, axis = persona_space(base, btok)
del base
torch.cuda.empty_cache()


def ids_of(q, tok):
    return torch.tensor([render([{"role": "user", "content": q}], tok, add_generation_prompt=True,
                                max_len=1024)["input_ids"]]).cuda()


def refusal_rate(model, tok, handle, prompts):
    ans = behavior.generate(model, tok, handle, prompts, 48, bs=32)
    return float(np.mean([is_refusal(a) for a in ans]))


def diff_dir(X, y):
    r = X[y == 1].mean(0) - X[y == 0].mean(0)
    return r / r.norm()


def run(name, model, tok, handle, readout, defences):
    """readout(q) -> {layer: vector} at the decision token. defences: {name: (install_fn) -> hooks}."""
    model.config.use_cache = True
    plain_ans = behavior.generate(model, tok, handle, benign + unsafe, 48, bs=32)  # = fit_prompts[:len(y)]
    y = np.array([is_refusal(a) for a in plain_ans]).astype(int)
    us = slice(len(benign), len(benign) + len(unsafe))
    fit = [readout(q) for q in fit_prompts]
    L = sorted(fit[0])
    F = {i: torch.stack([f[i] for f in fit]) for i in L}
    refdir = {i: diff_dir(F[i][: len(y)], torch.tensor(y)) for i in L}
    mu = {i: F[i].mean(0) for i in L}
    V = {i: torch.linalg.svd(F[i] - mu[i], full_matrices=False)[2][: args.k].T for i in L}
    Z = {i: (F[i] - mu[i]) @ V[i] for i in L}
    zinv = {i: torch.linalg.pinv(torch.cov(Z[i].T)) for i in L}
    res = {}
    for t, tpl in TEMPLATES.items():
        wrapped = [tpl.format(q=q) for q in unsafe]
        r = {"refusal": refusal_rate(model, tok, handle, wrapped)}
        reads = [readout(q) for q in wrapped]
        for i in L[::4] + [L[-1]]:
            X = torch.stack([x[i] for x in reads])
            z = (X - mu[i]) @ V[i]
            r[f"maha/L{i}"] = float(torch.einsum("nk,kj,nj->n", z - Z[i].mean(0), zinv[i], z - Z[i].mean(0)).sqrt().mean())
            r[f"proj_refusal/L{i}"] = float((X @ refdir[i]).mean())
            r[f"proj_refusal_plain_unsafe/L{i}"] = float((F[i][us] @ refdir[i]).mean())
            if X.shape[1] == axis[i].shape[0]:
                r[f"proj_axis/L{i}"] = float((X @ axis[i]).mean())
                r[f"proj_axis_plain_unsafe/L{i}"] = float((F[i][us] @ axis[i]).mean())
        res[t] = r
        print(name, t, {k: round(v, 3) for k, v in r.items() if k == "refusal" or k.endswith("L15") or k.endswith("L8")},
              flush=True)
    report[f"{name}/templates"] = res
    dres = {}
    for dname, install in defences.items():
        hk = install(F, mu, V)
        d = {t: refusal_rate(model, tok, handle, [tpl.format(q=q) for q in unsafe]) for t, tpl in TEMPLATES.items()}
        d["benign/xstest_safe_refused"] = refusal_rate(model, tok, handle, benign[:100])
        d["benign/alpaca_refused"] = refusal_rate(model, tok, handle, benign[100:])
        d["alpaca/ref_logp"] = behavior.reference_logp(model, tok, handle, behavior.alpaca(100))
        for h in hk:
            h.remove()
        dres[dname] = d
        print(name, "defence", dname, {k: round(v, 3) for k, v in d.items()}, flush=True)
    report[f"{name}/defences"] = dres


# ---------------- side-car ----------------
sc_model, stok, handle = load_arm(args.sidecar)
sc = sc_model.persona


def sc_readout_factory(kind):
    def read(q):
        out, hk = {}, []
        for i in sc.layer_ids:
            blk = sc.blocks[str(i)].write
            if kind == "state":
                hk.append(blk.register_forward_pre_hook(lambda m, a, i=i: out.__setitem__(i, a[0][0, -1].float().cpu())))
            else:
                hk.append(blk.register_forward_hook(lambda m, a, o, i=i: out.__setitem__(i, o[0, -1].float().cpu())))
        with torch.no_grad():
            sc_model(input_ids=ids_of(q, stok))
        for h in hk:
            h.remove()
        return out
    return read


# write/residual ratio per template (side-car only)
ratios = {}
for t, tpl in TEMPLATES.items():
    acc = []
    for q in unsafe[:60]:
        ids = ids_of(tpl.format(q=q), stok)
        w = sc_readout_factory("write")(tpl.format(q=q))
        h = last_resid(sc_model, ids)
        acc.append(np.mean([w[i].norm().item() / h[i].norm().item() for i in sc.layer_ids]))
    ratios[t] = float(np.mean(acc))
report["sidecar/write_ratio"] = ratios
print("sidecar write ratio", {k: round(v, 3) for k, v in ratios.items()}, flush=True)


def sc_pc_clamp(F, mu, V):
    lo, hi = {}, {}
    for i in F:
        Z = (F[i] - mu[i]) @ V[i]
        lo[i], hi[i] = torch.quantile(Z, 0.01, dim=0).cuda(), torch.quantile(Z, 0.99, dim=0).cuda()

    def pre(m, a, i):
        s = a[0].float()
        Vc, mc = V[i].cuda(), mu[i].cuda()
        z = (s - mc) @ Vc
        s = s + (z.clamp(lo[i], hi[i]) - z) @ Vc.T
        return (s.to(a[0].dtype),) + tuple(a[1:])
    return [sc.blocks[str(i)].write.register_forward_pre_hook(lambda m, a, i=i: pre(m, a, i)) for i in sc.layer_ids]


def axis_floor_hooks(mods, plain_proj):
    """Raise the projection of each module output onto the assistant axis to its 5th percentile on plain prompts."""
    hk = []
    for i, mod in mods.items():
        a, floor = axis[i].cuda(), float(np.percentile(plain_proj[i], 5))

        def f(o, a=a, floor=floor):
            x = o[0] if isinstance(o, tuple) else o
            xf = x.float()
            pr = xf @ a
            bump = (pr.clamp_min(floor) - pr)[..., None] * a
            if xf.shape[1] > 1:  # leave the BOS position (huge norms) alone
                bump[:, 0] = 0
            xf = xf + bump
            return (xf.to(x.dtype),) + tuple(o[1:]) if isinstance(o, tuple) else xf.to(x.dtype)
        hk.append(mod.register_forward_hook(lambda m, inp, o, f=f: f(o)))
    return hk


def sc_axis_floor(F, mu, V):
    W = [sc_readout_factory("write")(q) for q in fit_prompts]
    plain = {i: torch.stack([w[i] for w in W]) @ axis[i] for i in sc.layer_ids}
    return axis_floor_hooks({i: sc.blocks[str(i)].write for i in sc.layer_ids}, {i: v.numpy() for i, v in plain.items()})


run("sidecar", sc_model, stok, handle, sc_readout_factory("state"),
    {"pc_clamp": sc_pc_clamp, "axis_floor_write": sc_axis_floor})
del sc_model
torch.cuda.empty_cache()

# ---------------- normal ----------------
nm, ntok, _ = load_arm(args.normal)


def nm_read(q):
    return last_resid(nm, ids_of(q, ntok))


def nm_axis_floor(F, mu, V):
    plain = {i: (F[i] @ axis[i]).numpy() for i in F}
    mid = list(range(4, 12))  # middle layers, as in Lu et al.
    return axis_floor_hooks({i: nm.model.layers[i] for i in mid}, plain)


run("normal", nm, ntok, None, nm_read, {"axis_floor_resid": nm_axis_floor})
json.dump(report, open(args.out, "w"), indent=1)
print("wrote", args.out)
