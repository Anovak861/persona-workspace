"""Does removing P strip role/persona information from the base, beyond a random subspace?

Projects P out of the base residual at the write layers (hard workspace at init, with no
workspace content) and measures:
  * role probe: held-out accuracy of a linear probe for assistant vs user tokens, on the
    residual at several layers (split by example, so the probe can't memorize conversations).
    probe_acc retrains on the projected residual (is role still recoverable?); transfer_acc
    applies the probe trained on the unprojected base (is the base's own readout broken?)
  * identity: logp margin (AI- vs human-style answer) and greedy-generation stats in chat format
  * SFT loss on assistant tokens
Conditions: none, LDA P at ranks in --ranks, and random subspaces of the same ranks.
"""
import argparse
import json

import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

from persona_ws.data import ROLE_ASSISTANT, ROLE_USER, collate, render
from persona_ws.evals import identity

p = argparse.ArgumentParser()
p.add_argument("--model", default="allenai/OLMo-2-0425-1B")
p.add_argument("--tokenizer", default=None)
p.add_argument("--P_path", required=True)
p.add_argument("--write_layers", type=int, nargs="+", default=[5, 9, 13])
p.add_argument("--probe_layers", type=int, nargs="+", default=[9, 13, 15])
p.add_argument("--ranks", type=int, nargs="+", default=[16, 64])
p.add_argument("--n", type=int, default=600)
p.add_argument("--tokens_per_role", type=int, default=16)
p.add_argument("--max_len", type=int, default=1024)
p.add_argument("--out", required=True)
args = p.parse_args()

tok = AutoTokenizer.from_pretrained(args.tokenizer or f"{args.model}-SFT")
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
P_lda = torch.load(args.P_path)["P"]
d = model.config.hidden_size

# disjoint from both the SFT validation split (seed 0) and the P fitting set (seed 1, first n)
ds = load_dataset("allenai/tulu-3-sft-olmo-2-mixture-0225", split="train").shuffle(seed=2)
rows = []
for ex in ds.select(range(3 * args.n)):
    r = render(ex["messages"], tok, max_len=args.max_len)
    if ROLE_USER in r["role_ids"] and ROLE_ASSISTANT in r["role_ids"] and any(l != -100 for l in r["labels"]):
        rows.append(r)
    if len(rows) == args.n:
        break

conditions = {"none": {}}
g = torch.Generator().manual_seed(0)
for r in args.ranks:
    conditions[f"lda_r{r}"] = {i: P_lda[i][:, :r] for i in args.write_layers}
    conditions[f"random_r{r}"] = {i: torch.linalg.qr(torch.randn(d, r, generator=g))[0] for i in args.write_layers}


def install(proj, capture):
    hooks = []
    for i, Pi in proj.items():
        Pi = Pi.cuda().to(torch.bfloat16)

        def remove(mod, inp, out, Pi=Pi):
            h = out[0] if isinstance(out, tuple) else out
            h = h - (h @ Pi) @ Pi.T
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        hooks.append(model.model.layers[i].register_forward_hook(remove))
    for i in args.probe_layers:  # registered after removal hooks, so they see the projected output
        def grab(mod, inp, out, i=i):
            capture[i] = (out[0] if isinstance(out, tuple) else out).float()
        hooks.append(model.model.layers[i].register_forward_hook(grab))
    return hooks


results, base_probes = {}, {}
for name, proj in conditions.items():
    cap = {}
    hooks = install(proj, cap)
    feats = {i: [] for i in args.probe_layers}
    labels, groups = [], []
    tot, n = 0.0, 0
    with torch.no_grad():
        for k in range(0, len(rows), 8):
            chunk = rows[k:k + 8]
            b = {kk: v.cuda() for kk, v in collate(chunk, tok.pad_token_id).items()}
            out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], labels=b["labels"])
            m = (b["labels"][:, 1:] != -100).sum().item()
            tot += out.loss.item() * m
            n += m
            for j in range(len(chunk)):
                roles = b["role_ids"][j].cpu().numpy()
                live = b["attention_mask"][j].cpu().numpy() == 1
                pick = []
                for role in (ROLE_USER, ROLE_ASSISTANT):
                    idx = np.flatnonzero((roles == role) & live)
                    # same per-example sampling for every condition
                    sub = np.random.default_rng(k + j + role).choice(idx, min(len(idx), args.tokens_per_role), replace=False)
                    pick += [(t, int(role == ROLE_ASSISTANT)) for t in sub]
                for t, y in pick:
                    labels.append(y)
                    groups.append(k + j)
                for i in args.probe_layers:
                    feats[i].append(cap[i][j, [t for t, _ in pick]].cpu().numpy())
    res = {"sft_loss": tot / n}
    y, grp = np.array(labels), np.array(groups)
    train = grp < int(0.7 * len(rows))
    for i in args.probe_layers:
        X = np.concatenate(feats[i])
        sc = StandardScaler().fit(X[train])
        clf = LogisticRegression(max_iter=2000, C=0.1).fit(sc.transform(X[train]), y[train])
        res[f"probe_acc_L{i}"] = clf.score(sc.transform(X[~train]), y[~train])  # info still recoverable?
        if name == "none":
            base_probes[i] = (sc, clf)
        bsc, bclf = base_probes[i]
        res[f"transfer_acc_L{i}"] = bclf.score(bsc.transform(X[~train]), y[~train])  # base readout broken?
    res["identity_margin"] = float(np.mean([r["margin"] for r in identity.logp_margin(model, tok, "chat")]))
    gens = identity.generations(model, tok, "chat")
    res["claims_ai"] = float(np.mean([r["claims_ai"] for r in gens]))
    res["continues_dialogue"] = float(np.mean([r["continues_dialogue"] for r in gens]))
    for hk in hooks:
        hk.remove()
    results[name] = res
    print(name, json.dumps({k: round(v, 4) for k, v in res.items()}), flush=True)

json.dump(results, open(args.out, "w"), indent=1)
