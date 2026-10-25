"""Fit P for the hard workspace: per write layer, the base's role (assistant-vs-user) subspace.

For N chat examples, take the base residual after layer l, compute
mean(h | assistant tokens) - mean(h | user tokens), stack to (N, d), and keep the top-r right
singular vectors. Also reports the takeover tax: SFT loss when P is projected out at all write
layers (no workspace), vs a random subspace of the same rank.
"""
import argparse
import json

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from persona_ws.data import ROLE_ASSISTANT, ROLE_USER, collate, render

p = argparse.ArgumentParser()
p.add_argument("--model", default="allenai/OLMo-2-0425-1B")
p.add_argument("--tokenizer", default=None)
p.add_argument("--layers", type=int, nargs="+", default=[5, 9, 13])
p.add_argument("--rank", type=int, default=64)
p.add_argument("--n", type=int, default=2048)
p.add_argument("--n_eval", type=int, default=256)
p.add_argument("--max_len", type=int, default=1024)
p.add_argument("--out", required=True)
args = p.parse_args()

tok = AutoTokenizer.from_pretrained(args.tokenizer or f"{args.model}-SFT")
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
# a different slice than the SFT trainer's validation split (which is the first n_val after shuffle(0))
ds = load_dataset("allenai/tulu-3-sft-olmo-2-mixture-0225", split="train").shuffle(seed=1)
rows = [render(ex["messages"], tok, max_len=args.max_len) for ex in ds.select(range(args.n + args.n_eval))]
rows = [r for r in rows if ROLE_USER in r["role_ids"] and any(l != -100 for l in r["labels"])]
fit, ev = rows[: args.n], rows[args.n:]

acts = {}


def grab(i):
    def hook(mod, inp, out):
        acts[i] = (out[0] if isinstance(out, tuple) else out).float()
    return hook


hooks = [model.model.layers[i].register_forward_hook(grab(i)) for i in args.layers]
diffs = {i: [] for i in args.layers}
with torch.no_grad():
    for k in range(0, len(fit), 8):
        b = {kk: v.cuda() for kk, v in collate(fit[k:k + 8], tok.pad_token_id).items()}
        model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
        a_m = (b["role_ids"] == ROLE_ASSISTANT) & (b["attention_mask"] == 1)
        u_m = (b["role_ids"] == ROLE_USER) & (b["attention_mask"] == 1)
        for i in args.layers:
            h = acts[i]
            ma = (h * a_m[..., None]).sum(1) / a_m.sum(1, keepdim=True).clamp(min=1)
            mu = (h * u_m[..., None]).sum(1) / u_m.sum(1, keepdim=True).clamp(min=1)
            diffs[i].append((ma - mu).cpu())
for hk in hooks:
    hk.remove()

P, report = {}, {}
for i in args.layers:
    D = torch.cat(diffs[i])  # (N, d)
    _, S, Vh = torch.linalg.svd(D, full_matrices=False)
    P[i] = Vh[: args.rank].T.contiguous()  # (d, r)
    report[f"layer{i}/var_explained"] = (S[: args.rank] ** 2).sum().item() / (S ** 2).sum().item()


def eval_loss(proj):
    hs = []
    for i, Pi in proj.items():
        Pi = Pi.cuda().to(torch.bfloat16)

        def hook(mod, inp, out, Pi=Pi):
            h = out[0] if isinstance(out, tuple) else out
            h = h - (h @ Pi) @ Pi.T
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        hs.append(model.model.layers[i].register_forward_hook(hook))
    tot, n = 0.0, 0
    with torch.no_grad():
        for k in range(0, len(ev), 8):
            b = {kk: v.cuda() for kk, v in collate(ev[k:k + 8], tok.pad_token_id).items()}
            m = (b["labels"][:, 1:] != -100).sum().item()
            loss = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], labels=b["labels"]).loss
            tot += loss.item() * m
            n += m
    for hk in hs:
        hk.remove()
    return tot / n


g = torch.Generator().manual_seed(0)
rand = {i: torch.linalg.qr(torch.randn(P[i].shape, generator=g))[0] for i in args.layers}
report["loss/base"] = eval_loss({})
report["loss/role_P_removed"] = eval_loss(P)
report["loss/random_P_removed"] = eval_loss(rand)
torch.save({"P": P, "config": vars(args), "report": report}, args.out)
print(json.dumps(report, indent=1))
