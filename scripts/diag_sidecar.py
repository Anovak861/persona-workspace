"""Interpretability check for a side-car arm, and for what a later stage (e.g. DPO) changed in it.

python scripts/diag_sidecar.py --arm <ckpt> --behavior <eval_behavior json> [--ref <ckpt> --ref_behavior <json>]

A. Where it writes (48 held-out Tülu conversations, teacher-forced): per layer, ||write|| / ||h||
   by token role, and the side-car stream norm.
B. What it encodes at the decision token (generation prompt) for every XSTest / AlpacaEval /
   identity prompt: 5-fold probes for the arm's own refusal and for unsafe-vs-safe, from the
   side-car state s_l vs the base residual (module off) at the same layer; PCA dims of s.
   Logit lens on the write: top tokens promoted by the mean write at the decision token,
   refused vs complied.
C. (with --ref) Parameter change per layer and its effective rank, and whether the change in
   the write at the decision token aligns with the base refusal direction at that layer.
D. (with --ref) Causal localization: swap bands of 4 layers between arm and ref, and measure
   quick refusal rates + identity on the hybrids.
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
from persona_ws.evals import behavior, identity
from persona_ws.evals.behavior import is_refusal
from persona_ws.load import load_arm

p = argparse.ArgumentParser()
p.add_argument("--arm", required=True)
p.add_argument("--behavior", required=True)
p.add_argument("--ref", default=None, help="earlier checkpoint of the same arm (e.g. the SFT init of a DPO run)")
p.add_argument("--ref_behavior", default=None, help="eval_behavior json for --ref")
p.add_argument("--n_conv", type=int, default=48)
p.add_argument("--max_len", type=int, default=512)
p.add_argument("--band", type=int, default=4)
p.add_argument("--out", required=True)
args = p.parse_args()

model, tok, handle = load_arm(args.arm)
sc = model.persona
L = sc.layer_ids
blocks = sc.blocks
report = {"layers": L}
arm_sd = {k: v.detach().clone() for k, v in sc.state_dict().items()}
ref_sd = None
if args.ref:
    ref_sd = torch.load(f"{args.ref}/persona.pt", map_location="cpu")
    ref_sd = {k: v.to(next(sc.parameters()).device, next(sc.parameters()).dtype) for k, v in ref_sd.items()}

h_norm = {}
hooks = [model.model.layers[i].register_forward_hook(
    lambda m, a, o, i=i: h_norm.__setitem__(i, (o[0] if isinstance(o, tuple) else o).float().norm(dim=-1).cpu()))
    for i in L]


def writes(rec_s, i):
    """Recorded side-car state at layer i -> its write (fp32, cpu)."""
    with torch.no_grad():
        w = blocks[str(i)].write.weight
        return blocks[str(i)].write(rec_s.to(w.device, w.dtype)).float().cpu()


# ---------------- A. where it writes ----------------
ds = load_dataset("allenai/tulu-3-sft-olmo-2-mixture-0225", split="train").shuffle(seed=2)
convs = []
for ex in ds.select(range(4 * args.n_conv)):
    r = render(ex["messages"], tok, max_len=args.max_len)
    if ROLE_USER in r["role_ids"] and ROLE_ASSISTANT in r["role_ids"] and len(r["input_ids"]) > 64:
        convs.append(r)
    if len(convs) == args.n_conv:
        break
ratio = {i: {r: [] for r in ("user", "asst", "other")} for i in L}
s_norm = {i: [] for i in L}
handle.record = True
with torch.no_grad():
    for c in convs:
        b = collate([c], tok.pad_token_id)
        handle.clear_records()
        model(input_ids=b["input_ids"].cuda())
        roles = b["role_ids"][0]
        for i in L:
            s = handle.records[i]["state"][0][0]
            rt = writes(s, i).norm(dim=-1) / h_norm[i][0]
            s_norm[i].append(s.norm(dim=-1).mean().item())
            for name, rid in (("user", ROLE_USER), ("asst", ROLE_ASSISTANT), ("other", ROLE_OTHER)):
                if (roles == rid).any():
                    ratio[i][name].append(rt[roles == rid].mean().item())
handle.record = False
for i in L:
    for r, v in ratio[i].items():
        report[f"write_ratio/{r}/L{i}"] = float(np.mean(v))
    report[f"s_norm/L{i}"] = float(np.mean(s_norm[i]))

# ---------------- B. decision states ----------------
beh = json.load(open(args.behavior))
items = [(r["prompt"], r["refusal"], "xstest_unsafe" if r["unsafe"] else "xstest_safe") for r in beh["xstest"]]
items += [(r["prompt"], r["refusal"], "alpaca") for r in beh["alpaca"]]
id_gens = {g["q"]: g for g in identity.generations(model, tok, "chat")}
items += [(q, is_refusal(id_gens[q]["answer"]), "identity") for q, _, _ in identity.PAIRS]
y = np.array([int(r) for _, r, _ in items])
src = np.array([s for _, _, s in items])


def decision_states(sd):
    """Per layer: side-car state, its write, and base residual (module off) at the last prompt token."""
    if sd is not None:
        sc.load_state_dict(sd)
    S, W, H = ({i: [] for i in L} for _ in range(3))
    grab = {}
    hk = [model.model.layers[i].register_forward_hook(
        lambda m, a, o, i=i: grab.__setitem__(i, (o[0] if isinstance(o, tuple) else o)[0, -1].float().cpu()))
        for i in L]
    handle.record = True
    with torch.no_grad():
        for prompt, _, _ in items:
            ids = torch.tensor([render([{"role": "user", "content": prompt}], tok, add_generation_prompt=True,
                                       max_len=args.max_len)["input_ids"]]).cuda()
            handle.clear_records()
            handle.enabled = True
            model(input_ids=ids)
            for i in L:
                s = handle.records[i]["state"][0][0, -1]
                S[i].append(s)
                W[i].append(writes(s, i))
            handle.enabled = False
            model(input_ids=ids)
            for i in L:
                H[i].append(grab[i])
    handle.enabled = True
    handle.record = False
    for x in hk:
        x.remove()
    sc.load_state_dict(arm_sd)
    return ({i: torch.stack(v).numpy() for i, v in d.items()} for d in (S, W, H))


def probe(X, labels):
    if labels.min() == labels.max() or min(labels.sum(), (1 - labels).sum()) < 5:
        return None
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.05))
    return float(cross_val_score(clf, X, labels, cv=5, scoring="balanced_accuracy").mean())


S, W, H = decision_states(None)
report["decision/refusal_rate"] = float(y.mean())
xs = np.isin(src, ["xstest_safe", "xstest_unsafe"])
unsafe = (src[xs] == "xstest_unsafe").astype(int)
for i in L[::max(1, len(L) // 8)] + [L[-1]]:
    report[f"probe_refusal/sidecar/L{i}"] = probe(S[i], y)
    report[f"probe_refusal/base/L{i}"] = probe(H[i], y)
    report[f"probe_unsafe/sidecar/L{i}"] = probe(S[i][xs], unsafe)
    report[f"probe_unsafe/base/L{i}"] = probe(H[i][xs], unsafe)
allS = np.concatenate([S[i] for i in L], 1)
pca = PCA().fit(StandardScaler().fit_transform(allS))
report["decision/pca_dims_90_all_layers"] = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.9) + 1)

# logit lens on the summed write at the decision token: what does the side-car push toward?
unembed = model.get_output_embeddings().weight.float()
norm_f = model.model.norm


def lens(vec, k=12):
    with torch.no_grad():
        v = torch.tensor(vec, device=unembed.device, dtype=torch.float32) * norm_f.weight.float()
        top = (unembed @ v).topk(k).indices.tolist()
    return [tok.decode([t]) for t in top]


Wsum = sum(W[i] for i in L)
for name, m in (("refused", y == 1), ("complied", y == 0)):
    if m.any():
        report[f"lens/mean_write/{name}"] = lens(Wsum[m].mean(0))
idm = src == "identity"
report["lens/mean_write/identity"] = lens(Wsum[idm].mean(0))

# ---------------- C/D. what changed vs ref ----------------
if ref_sd is not None:
    for i in L:
        for part in ("read", "up", "down", "write"):
            k = f"blocks.{i}.{part}.weight"
            d = (arm_sd[k] - ref_sd[k]).float()
            report[f"delta/{part}/L{i}"] = (d.norm() / ref_sd[k].float().norm().clamp_min(1e-8)).item()
            if part == "write":
                sv = torch.linalg.svdvals(d)
                pn = sv / sv.sum()
                report[f"delta_erank/write/L{i}"] = float(torch.exp(-(pn * pn.clamp_min(1e-12).log()).sum()))
    _, W_ref, _ = decision_states(ref_sd)
    ref_beh = json.load(open(args.ref_behavior)) if args.ref_behavior else None
    # refusal direction in the base residual, from the ref arm's own refusal labels if given
    y_dir = y
    if ref_beh is not None:
        y_dir = np.array([int(r["refusal"]) for r in ref_beh["xstest"]] + [int(r["refusal"]) for r in ref_beh["alpaca"]]
                         + list(y[src == "identity"]))
    safe = src == "xstest_safe"
    for i in L:
        rdir = H[i][y_dir == 1].mean(0) - H[i][y_dir == 0].mean(0)
        rdir /= np.linalg.norm(rdir)
        dW = (W[i] - W_ref[i])[safe].mean(0)
        report[f"dwrite_on_refusal_dir/L{i}"] = float(dW @ rdir)  # signed, in residual units
        report[f"dwrite_cos_refusal_dir/L{i}"] = float(dW @ rdir / (np.linalg.norm(dW) + 1e-8))
        report[f"dwrite_norm/L{i}"] = float(np.linalg.norm(dW))
    dsum = sum((W[i] - W_ref[i]) for i in L)
    report["lens/dwrite/xstest_safe"] = lens(dsum[safe].mean(0))
    report["lens/dwrite/identity"] = lens(dsum[idm].mean(0))

    quick = behavior.quick_set()
    model.config.use_cache = True

    def measure(sd, tag):
        sc.load_state_dict(sd)
        r = behavior.quick_refusal(model, tok, handle, quick)
        g = identity.generations(model, tok, "chat")
        m = identity.logp_margin(model, tok, "chat")
        r["identity/claims_ai"] = sum(x["claims_ai"] for x in g) / len(g)
        r["identity/logp_margin"] = sum(x["margin"] for x in m) / len(m)
        report.update({f"swap/{tag}/{k}": v for k, v in r.items()})
        print(tag, {k: round(v, 3) for k, v in r.items()}, flush=True)

    measure(arm_sd, "arm")
    measure(ref_sd, "ref")
    for lo in range(0, len(L), args.band):
        band = set(L[lo:lo + args.band])
        tag = f"L{min(band)}-{max(band)}"
        mix = {k: (ref_sd[k] if int(k.split(".")[1]) in band else arm_sd[k]) for k in arm_sd}
        measure(mix, f"arm_with_ref_{tag}")
        mix = {k: (arm_sd[k] if int(k.split(".")[1]) in band else ref_sd[k]) for k in arm_sd}
        measure(mix, f"ref_with_arm_{tag}")
    sc.load_state_dict(arm_sd)

for x in hooks:
    x.remove()
json.dump(report, open(args.out, "w"), indent=1)
print(json.dumps(report, indent=1))
