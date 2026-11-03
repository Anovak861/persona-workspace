"""Side-car vs normal fine-tuning: is the fine-tuned behaviour (persona, refusal) more compact and more
cleanly removable in the side-car than in an ordinary fine-tune of the same base on the same data?

python scripts/compare_normal.py --sidecar <ckpt> --sc_behavior <json> --normal <ckpt | hf:repo> --nm_behavior <json> --out <json>

Each model has a "fine-tuning stream" per layer: for the side-car its state s_l (what it writes from);
for the normal model the change in the residual vs the base model on the same tokens, d_l = h_l - h_l^base.

A. Probes at the decision token (last prompt token) for the model's own refusal, unsafe-vs-safe and
   persona-question-vs-other, from: the stream, its top-k PCs, the model's residual, the base residual.
B. Removal: mean-ablate the top-k PCs of the stream at every layer (side-car: on s before the write;
   normal: on the residual, along the top-k PCs of d_l) and measure identity, refusal, helpfulness
   (AlpacaEval reference logp) and base LM loss (wikitext). Plus the side-car fully off.
   The side-car's stream is dedicated, so removing it should cost little base ability; in the normal
   model the same directions also carry base features.
   Fairer variants for the normal model, run the same way on both: small k (1-4); top-k at a single
   layer; and directional ablation (x <- x - (x.r)r at every layer, Arditi et al.) of a diff-of-means
   refusal direction (own refused - complied) and persona direction (persona questions - Alpaca),
   from the residual (normal) or the state (side-car) at the decision token.
"""
import argparse
import json
import random

import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from persona_ws import anchor
from persona_ws.data import render
from persona_ws.evals import behavior, identity
from persona_ws.load import load_arm

p = argparse.ArgumentParser()
p.add_argument("--sidecar", required=True)
p.add_argument("--sc_behavior", required=True)
p.add_argument("--normal", required=True)
p.add_argument("--nm_behavior", required=True)
p.add_argument("--base", default="hf:allenai/OLMo-2-0425-1B")
p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 32, 256])
p.add_argument("--single_k", type=int, default=8)
p.add_argument("--single_layers", type=int, nargs="+", default=[3, 7, 11, 15])
p.add_argument("--probe_layers", type=int, nargs="+", default=[3, 7, 11, 15])
p.add_argument("--pos_per_prompt", type=int, default=8, help="random prompt positions per prompt for PCA fitting")
p.add_argument("--max_len", type=int, default=512)
p.add_argument("--out", required=True)
args = p.parse_args()
random.seed(0)
report = {"args": vars(args)}

# ---- prompts and labels ----
sc_beh, nm_beh = json.load(open(args.sc_behavior)), json.load(open(args.nm_behavior))
xs = [(r["prompt"], "xstest_unsafe" if r["unsafe"] else "xstest_safe") for r in sc_beh["xstest"]]
al = [(r["prompt"], "alpaca") for r in sc_beh["alpaca"]]
persona_qs = [q for q, _, _ in identity.PAIRS] + anchor.QUESTIONS
items = xs + al + [(q, "persona") for q in persona_qs]
prompts = [q for q, _ in items]
src = np.array([s for _, s in items])


def own_refusal(beh):
    m = {r["prompt"]: int(r["refusal"]) for r in beh["xstest"] + beh["alpaca"]}
    return np.array([m.get(q, -1) for q in prompts])


labels = {"unsafe": np.where(np.isin(src, ["xstest_safe", "xstest_unsafe"]), (src == "xstest_unsafe").astype(int), -1),
          "persona_q": np.where(np.isin(src, ["persona", "alpaca"]), (src == "persona").astype(int), -1)}
own = {"sc": own_refusal(sc_beh), "nm": own_refusal(nm_beh)}

quick = behavior.quick_set()
alp_ref = behavior.alpaca(100)
wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
wiki_text = "\n".join(t for t in wiki["text"] if t.strip())


def ids_of(q, tok):
    return torch.tensor([render([{"role": "user", "content": q}], tok, add_generation_prompt=True,
                                max_len=args.max_len)["input_ids"]]).cuda()


def layers(model):
    return model.model.layers


def run_capture(model, tok, grab_fn):
    """For each prompt: per layer, (decision-token vector, sampled-position vectors) from grab_fn."""
    dec, fit = [], []
    for q in prompts:
        ids = ids_of(q, tok)
        T = ids.shape[1]
        pos = sorted(random.Random(hash(q) % 2**32).sample(range(T), min(T, args.pos_per_prompt)))
        per = grab_fn(ids)  # {layer: (T, d) fp32 cpu}
        dec.append({i: v[-1] for i, v in per.items()})
        fit.append({i: v[pos] for i, v in per.items()})
    L = sorted(dec[0])
    return ({i: torch.stack([d[i] for d in dec]) for i in L}, {i: torch.cat([f[i] for f in fit]) for i in L})


def resid_grabber(model, handle=None, enabled=True):
    def grab(ids):
        out, hk = {}, []
        for i, layer in enumerate(layers(model)):
            hk.append(layer.register_forward_hook(
                lambda m, a, o, i=i: out.__setitem__(i, (o[0] if isinstance(o, tuple) else o)[0].float().cpu())))
        if handle is not None:
            handle.enabled = enabled
        with torch.no_grad():
            model(input_ids=ids)
        if handle is not None:
            handle.enabled = True
        for h in hk:
            h.remove()
        return out
    return grab


def state_grabber(model, sc):
    def grab(ids):
        out, hk = {}, []
        for i in sc.layer_ids:
            hk.append(sc.blocks[str(i)].write.register_forward_pre_hook(
                lambda m, a, i=i: out.__setitem__(i, a[0][0].float().cpu())))
        with torch.no_grad():
            model(input_ids=ids)
        for h in hk:
            h.remove()
        return out
    return grab


def pcs(X):
    """Mean and principal directions (columns, descending) of rows of X."""
    mu = X.mean(0)
    _, _, Vh = torch.linalg.svd((X - mu).cuda(), full_matrices=False)
    return mu, Vh.T.cpu()


def probe(X, y):
    keep = y >= 0
    X, y = X[keep], y[keep]
    if min(y.sum(), (1 - y).sum()) < 5:
        return None
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.05))
    return float(cross_val_score(clf, X, y, cv=5, scoring="balanced_accuracy").mean())


def proj(X, mu, V, k):
    return ((X - mu) @ V[:, :k]).numpy()


@torch.no_grad()
def lm_loss(model, tok, n=64, T=512):
    ids = tok(wiki_text, return_tensors="pt").input_ids[0]
    tot = 0.0
    for j in range(n):
        x = ids[j * T:(j + 1) * T][None].cuda()
        tot += model(input_ids=x, labels=x).loss.item()
    return tot / n


def outcomes(model, tok, handle):
    model.config.use_cache = True
    g = identity.generations(model, tok, "chat")
    r = behavior.quick_refusal(model, tok, handle, quick)
    r["identity/claims_ai"] = sum(x["claims_ai"] for x in g) / len(g)
    r["identity/continues_dialogue"] = sum(x["continues_dialogue"] for x in g) / len(g)
    r["identity/heldout_claims_ai"] = anchor.heldout_identity(model, tok, handle)[0]
    r["alpaca/ref_logp"] = behavior.reference_logp(model, tok, handle, alp_ref)
    r["wikitext/lm_loss"] = lm_loss(model, tok)
    return r


def ablate_hooks(mods, subspaces, k, pre):
    """Mean-ablate the top-k directions: x <- x - ((x - mu) V_k) V_k^T, on module inputs (pre) or outputs."""
    hk = []
    for i, mod in mods.items():
        mu, V = subspaces[i]
        Vk = V[:, :k].cuda()
        muc = mu.cuda()

        def f(x, muc=muc, Vk=Vk):
            xf = x.float()
            return (xf - ((xf - muc) @ Vk) @ Vk.T).to(x.dtype)
        if pre:
            hk.append(mod.register_forward_pre_hook(lambda m, a, f=f: (f(a[0]),) + tuple(a[1:])))
        else:
            hk.append(mod.register_forward_hook(
                lambda m, a, o, f=f: (f(o[0]),) + tuple(o[1:]) if isinstance(o, tuple) else f(o)))
    return hk


def directions(dec, y):
    """Per layer: unit diff-of-means direction (label 1 - label 0) at the decision token, as a 1-column subspace."""
    keep = y >= 0
    out = {}
    for i, X in dec.items():
        r = X[keep][y[keep] == 1].mean(0) - X[keep][y[keep] == 0].mean(0)
        out[i] = (torch.zeros_like(r), (r / r.norm())[:, None])
    return out


def removal(name, model, tok, handle, mods, subspaces, pre, dirs):
    """dirs: {name: per-layer 1-column subspaces with zero mean} for directional ablation."""
    res = {"none": outcomes(model, tok, handle)}
    print(name, "none", res["none"], flush=True)
    conds = [(f"k{k}", mods, subspaces, k) for k in args.ks]
    conds += [(f"k{args.single_k}_L{i}", {i: mods[i]}, subspaces, args.single_k) for i in args.single_layers]
    conds += [(f"dir_{d}", mods, sub, 1) for d, sub in dirs.items()]
    for tag, ms, sub, k in conds:
        hk = ablate_hooks(ms, sub, k, pre)
        res[tag] = outcomes(model, tok, handle)
        for h in hk:
            h.remove()
        print(name, tag, res[tag], flush=True)
    return res


def probes(tag, feats, y_own):
    out = {}
    for name, X in feats.items():
        for lab, y in {"own_refusal": y_own, **labels}.items():
            out[f"{lab}/{name}"] = probe(X, y)
    report.update({f"probe/{tag}/{k}": v for k, v in out.items()})
    print(tag, {k: (round(v, 3) if v is not None else None) for k, v in out.items()}, flush=True)


# ---------------- base residuals (shared reference for the normal model's delta) ----------------
base, btok, _ = load_arm(args.base)
base_dec, base_fit = run_capture(base, btok, resid_grabber(base))
report["base/outcomes"] = outcomes(base, btok, None)
del base
torch.cuda.empty_cache()

# ---------------- normal fine-tune ----------------
nm, ntok, _ = load_arm(args.normal)
nm_dec, nm_fit = run_capture(nm, ntok, resid_grabber(nm))
L = sorted(nm_dec)
nm_sub = {i: pcs(nm_fit[i] - base_fit[i]) for i in L}  # PCs of the fine-tuning delta d_l
nm_res_mu = {i: nm_fit[i].mean(0) for i in L}
for i in args.probe_layers:
    d = nm_dec[i] - base_dec[i]
    mu, V = nm_sub[i]
    probes(f"normal/L{i}", {"resid": nm_dec[i].numpy(), "delta": d.numpy(), "base_resid": base_dec[i].numpy(),
                            **{f"delta_pc{k}": proj(d, mu, V, k) for k in args.ks}}, own["nm"])
tot = {i: (nm_fit[i] - base_fit[i]).pow(2).sum(1).mean().item() for i in L}
for i in L:
    mu, V = nm_sub[i]
    D = nm_fit[i] - base_fit[i] - mu
    for k in args.ks:
        report[f"normal/delta_var_in_pc{k}/L{i}"] = float(((D @ V[:, :k]).pow(2).sum(1).mean() / D.pow(2).sum(1).mean()))
# ablate the residual along the delta's top-k PCs, toward the normal model's own mean there
nm_ab = {i: (nm_res_mu[i], nm_sub[i][1]) for i in L}
nm_dirs = {"refusal": directions(nm_dec, own["nm"]), "persona": directions(nm_dec, labels["persona_q"])}
report["normal/removal"] = removal("normal", nm, ntok, None, dict(enumerate(layers(nm))), nm_ab, pre=False, dirs=nm_dirs)
del nm
torch.cuda.empty_cache()

# ---------------- side-car ----------------
sc_model, stok, handle = load_arm(args.sidecar)
sc = sc_model.persona
s_dec, s_fit = run_capture(sc_model, stok, state_grabber(sc_model, sc))
r_dec, _ = run_capture(sc_model, stok, resid_grabber(sc_model, handle))
sc_sub = {i: pcs(s_fit[i]) for i in sc.layer_ids}
for i in args.probe_layers:
    mu, V = sc_sub[i]
    probes(f"sidecar/L{i}", {"state": s_dec[i].numpy(), "resid": r_dec[i].numpy(), "base_resid": base_dec[i].numpy(),
                             **{f"state_pc{k}": proj(s_dec[i], mu, V, k) for k in args.ks}}, own["sc"])
for i in sc.layer_ids:
    mu, V = sc_sub[i]
    S = s_fit[i] - mu
    for k in args.ks:
        report[f"sidecar/state_var_in_pc{k}/L{i}"] = float(((S @ V[:, :k]).pow(2).sum(1).mean() / S.pow(2).sum(1).mean()))
mods = {i: sc.blocks[str(i)].write for i in sc.layer_ids}
sc_dirs = {"refusal": directions(s_dec, own["sc"]), "persona": directions(s_dec, labels["persona_q"])}
report["sidecar/removal"] = removal("sidecar", sc_model, stok, handle, mods, sc_sub, pre=True, dirs=sc_dirs)
handle.enabled = False
report["sidecar/removal"]["off"] = outcomes(sc_model, stok, handle)
handle.enabled = True
print("sidecar off", report["sidecar/removal"]["off"], flush=True)

json.dump(report, open(args.out, "w"), indent=1)
print("wrote", args.out)
