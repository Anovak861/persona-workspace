"""Slot-usage / access-log diagnostics for a workspace arm.

A. Dynamics on held-out Tülu conversations (teacher-forced, module on, recording):
   per read layer: normalized access entropy, attention-sink mass on position 0, access mass by
   role relative to that role's share of the prefix, slot redundancy (cosine between slot states
   at the same position), and slot-read overlap (cosine between slots' access distributions).
   Per write layer: effective number of slots used by the broadcast (exp entropy of mean usage)
   and mean per-token broadcast entropy.
B. Decision states: workspace state at the generation-prompt token for every XSTest / AlpacaEval
   prompt plus the identity prompts. Uses this arm's refusal labels from eval_behavior.py:
   - refusal probe (5-fold CV): from workspace state vs from the base residual (module off) vs
     from the module-on residual at the same layer
   - PCA dims for 90% of decision-state variance
   - where the benign identity prompts sit relative to the refusal / comply centroids
"""
import argparse
import json

import numpy as np
import torch
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from persona_ws.data import ROLE_ASSISTANT, ROLE_OTHER, ROLE_USER, collate, render
from persona_ws.evals import identity
from persona_ws.evals.behavior import is_refusal
from persona_ws.load import load_arm

p = argparse.ArgumentParser()
p.add_argument("--arm", required=True)
p.add_argument("--behavior", required=True, help="eval_behavior.py output json for this arm")
p.add_argument("--n_conv", type=int, default=48)
p.add_argument("--max_len", type=int, default=512)
p.add_argument("--out", required=True)
args = p.parse_args()

model, tok, handle = load_arm(args.arm)
ws = model.persona
cfg = ws.cfg
reads, writes = list(cfg.read_layers), list(cfg.write_layers)
last_read = max(reads)
report = {}


def cos_offdiag(x):
    """x (..., K, d) -> mean pairwise cosine between the K rows."""
    x = torch.nn.functional.normalize(x, dim=-1)
    sim = x @ x.transpose(-1, -2)
    K = sim.shape[-1]
    return ((sim.sum((-1, -2)) - K) / (K * (K - 1))).mean().item()


# ---------------- A. dynamics ----------------
ds = load_dataset("allenai/tulu-3-sft-olmo-2-mixture-0225", split="train").shuffle(seed=2)
convs = []
for ex in ds.select(range(4 * args.n_conv)):
    r = render(ex["messages"], tok, max_len=args.max_len)
    if ROLE_USER in r["role_ids"] and ROLE_ASSISTANT in r["role_ids"] and len(r["input_ids"]) > 64:
        convs.append(r)
    if len(convs) == args.n_conv:
        break

acc = {i: {k: [] for k in ("entropy", "sink", "redund", "overlap", "role_user", "role_asst", "role_other")}
       for i in reads}
use = {i: {"mean_usage": [], "tok_entropy": []} for i in writes}
handle.record = True
with torch.no_grad():
    for c in convs:  # one at a time: no padding, full access maps
        b = collate([c], tok.pad_token_id)
        handle.clear_records()
        model(input_ids=b["input_ids"].cuda())
        roles = b["role_ids"][0]
        T = roles.shape[0]
        t_idx = torch.arange(T)
        for i in reads:
            a = handle.records[i]["access"][0][0]  # (T, K, T)
            z = handle.records[i]["slots"][0][0]  # (T, K, d)
            valid = t_idx >= 8
            H = -(a.clamp_min(1e-12).log() * a).sum(-1)  # (T, K)
            acc[i]["entropy"].append((H[valid] / torch.log(t_idx[valid].float() + 1)[:, None]).mean().item())
            acc[i]["sink"].append(a[valid][..., 0].mean().item())
            acc[i]["redund"].append(cos_offdiag(z[valid]))
            acc[i]["overlap"].append(cos_offdiag(a[valid]))
            causal = (t_idx[None, :] <= t_idx[:, None]).float()  # (T, T)
            for name, rid in (("role_user", ROLE_USER), ("role_asst", ROLE_ASSISTANT), ("role_other", ROLE_OTHER)):
                is_r = (roles == rid).float()
                share = (causal * is_r).sum(-1) / causal.sum(-1)  # role's share of each prefix
                mass = (a * is_r).sum(-1).mean(-1)  # access mass on that role
                ok = valid & (share > 0)
                acc[i][name].append((mass[ok] / share[ok]).mean().item())
        for i in writes:
            w = handle.records[i]["broadcast"][0][0]  # (T, K)
            use[i]["mean_usage"].append(w.mean(0))
            use[i]["tok_entropy"].append((-(w.clamp_min(1e-12).log() * w).sum(-1)).mean().item())
handle.record = False
for i in reads:
    for k, v in acc[i].items():
        report[f"read{i}/{k}"] = float(np.mean(v))
for i in writes:
    u = torch.stack(use[i]["mean_usage"]).mean(0)
    report[f"write{i}/effective_slots"] = float(torch.exp(-(u * u.clamp_min(1e-12).log()).sum()))
    report[f"write{i}/tok_entropy"] = float(np.mean(use[i]["tok_entropy"]))
    report[f"write{i}/usage"] = [round(x, 4) for x in u.tolist()]
report["n_slots"] = cfg.n_slots

# ---------------- B. decision states ----------------
beh = json.load(open(args.behavior))
items = [(r["prompt"], r["refusal"], "xstest_unsafe" if r["unsafe"] else "xstest_safe") for r in beh["xstest"]]
items += [(r["prompt"], r["refusal"], "alpaca") for r in beh["alpaca"]]
id_qs = [q for q, _, _ in identity.PAIRS]
id_gens = {g["q"]: g for g in identity.generations(model, tok, "chat")}
items += [(q, is_refusal(id_gens[q]["answer"]), "identity") for q in id_qs]

grab = {}
hk = model.model.layers[last_read].register_forward_hook(
    lambda m, i, o: grab.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[:, -1].float().cpu()))
Z, H_on, H_off = [], [], []
handle.record = True
with torch.no_grad():
    for prompt, _, _ in items:
        ids = torch.tensor([render([{"role": "user", "content": prompt}], tok, add_generation_prompt=True,
                                   max_len=args.max_len)["input_ids"]]).cuda()
        handle.clear_records()
        handle.enabled = True
        model(input_ids=ids)
        Z.append(handle.records[last_read]["slots"][0][0, -1].flatten())  # (K*d,)
        H_on.append(grab["h"][0])
        handle.enabled = False
        model(input_ids=ids)
        H_off.append(grab["h"][0])
handle.enabled = True
handle.record = False
hk.remove()
Z, H_on, H_off = (torch.stack(x).numpy() for x in (Z, H_on, H_off))
y = np.array([int(r) for _, r, _ in items])
src = np.array([s for _, _, s in items])
report["decision/refusal_rate"] = float(y.mean())


def probe(X, labels):
    if labels.min() == labels.max() or min(labels.sum(), (1 - labels).sum()) < 5:
        return None
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.05))
    return float(cross_val_score(clf, X, labels, cv=5, scoring="balanced_accuracy").mean())


report["decision/refusal_probe_workspace"] = probe(Z, y)
report["decision/refusal_probe_resid_on"] = probe(H_on, y)
report["decision/refusal_probe_resid_off_base"] = probe(H_off, y)
xs = np.isin(src, ["xstest_safe", "xstest_unsafe"])
unsafe = (src[xs] == "xstest_unsafe").astype(int)
report["decision/unsafe_probe_workspace"] = probe(Z[xs], unsafe)
report["decision/unsafe_probe_resid_off_base"] = probe(H_off[xs], unsafe)
pca = PCA().fit(StandardScaler().fit_transform(Z))
report["decision/pca_dims_90"] = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.9) + 1)
report["decision/pca_pc1_var"] = float(pca.explained_variance_ratio_[0])
if 0 < y.sum() < len(y):
    Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    ref_c, com_c = Zn[(y == 1) & (src != "identity")].mean(0), Zn[(y == 0) & (src != "identity")].mean(0)
    idm = src == "identity"
    report["decision/identity_cos_refusal_centroid"] = float((Zn[idm] @ ref_c).mean() / np.linalg.norm(ref_c))
    report["decision/identity_cos_comply_centroid"] = float((Zn[idm] @ com_c).mean() / np.linalg.norm(com_c))
report["decision/identity_refusals"] = int(y[src == "identity"].sum())

json.dump(report, open(args.out, "w"), indent=1)
print(json.dumps(report, indent=1))
