"""Fit P for the hard workspace: per write layer, the base's role (assistant-vs-user) subspace.

For N chat examples, take the base residual after layer l and compute
D_n = mean(h | assistant tokens) - mean(h | user tokens).
  svd: top-r right singular vectors of D (N, d). Dominated by high-norm residual directions;
       removing them is catastrophic (2026-10-25: loss 1.07 -> 6.67 at r=64).
  lda: whitened: top-r generalized eigenvectors of S_b v = lam S_w v, with S_b = D^T D / N and
       S_w = token covariance (+ ridge), then orthonormalized. Prefers directions where role
       signal is large relative to overall variance.
Reports the takeover tax (SFT loss with P projected out at all write layers, no workspace) for
each method and rank, vs a random subspace. Saves P at --rank for --method.
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
p.add_argument("--method", choices=["svd", "lda"], default="lda")
p.add_argument("--sweep", type=int, nargs="*", default=[1, 4, 16, 64])
p.add_argument("--ridge", type=float, default=1e-2, help="S_w ridge, relative to its mean eigenvalue")
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
d = model.config.hidden_size
cov = {i: torch.zeros(d, d, dtype=torch.float64, device="cuda") for i in args.layers}
mom = {i: torch.zeros(d, dtype=torch.float64, device="cuda") for i in args.layers}
ntok = 0
with torch.no_grad():
    for k in range(0, len(fit), 8):
        b = {kk: v.cuda() for kk, v in collate(fit[k:k + 8], tok.pad_token_id).items()}
        model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
        a_m = (b["role_ids"] == ROLE_ASSISTANT) & (b["attention_mask"] == 1)
        u_m = (b["role_ids"] == ROLE_USER) & (b["attention_mask"] == 1)
        tok_m = b["attention_mask"] == 1
        ntok += tok_m.sum().item()
        for i in args.layers:
            h = acts[i]
            ma = (h * a_m[..., None]).sum(1) / a_m.sum(1, keepdim=True).clamp(min=1)
            mu = (h * u_m[..., None]).sum(1) / u_m.sum(1, keepdim=True).clamp(min=1)
            diffs[i].append((ma - mu).cpu())
            ht = h[tok_m].double()
            cov[i] += ht.T @ ht
            mom[i] += ht.sum(0)
for hk in hooks:
    hk.remove()


def directions(i, method):
    """All candidate directions for layer i, best first, as orthonormal columns (d, r_max)."""
    D = torch.cat(diffs[i]).double().cuda()  # (N, d)
    r_max = max(args.sweep + [args.rank])
    if method == "svd":
        return torch.linalg.svd(D, full_matrices=False)[2][:r_max].T.float().cpu()
    mean = mom[i] / ntok
    Sw = cov[i] / ntok - torch.outer(mean, mean)
    evals, evecs = torch.linalg.eigh(Sw)
    evals = evals.clamp(min=0) + args.ridge * evals.clamp(min=0).mean()
    W = evecs / evals.sqrt()  # whitening: W^T Sw W = I
    Sb = (D.T @ D) / D.shape[0]
    _, U = torch.linalg.eigh(W.T @ Sb @ W)
    V = W @ U.flip(-1)[:, :r_max]  # top generalized eigenvectors
    return torch.linalg.qr(V)[0].float().cpu()


cands = {m: {i: directions(i, m) for i in args.layers} for m in ("svd", "lda")}
P = {i: cands[args.method][i][:, : args.rank].contiguous() for i in args.layers}
report = {}


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
report["loss/base"] = eval_loss({})
for r in args.sweep:
    rand = {i: torch.linalg.qr(torch.randn(d, r, generator=g))[0] for i in args.layers}
    report[f"loss/random_r{r}"] = eval_loss(rand)
    for m in ("svd", "lda"):
        report[f"loss/{m}_r{r}"] = eval_loss({i: cands[m][i][:, :r] for i in args.layers})
    print(json.dumps({k: v for k, v in report.items() if k.endswith(f"_r{r}")}), flush=True)
torch.save({"P": P, "config": vars(args), "report": report}, args.out)
print(json.dumps(report, indent=1))
