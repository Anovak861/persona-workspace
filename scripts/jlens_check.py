"""Is the J-lens fit sound? Compare lens variants on held-out wikitext (persona_ws/jlens.py).

python scripts/jlens_check.py --arm <side-car ckpt> --j <future-averaged J .pt> --out <json>

The module is switched off, so this is the bare base. Per layer, top-1 agreement with the model's own final
prediction and mean KL(final || lens) for:
  logit       W_U norm(h)
  Jfut        the fitted future-averaged J (fit(); the J used by jspace.py)
  Jfut_aff    mu_L + Jfut (h - mu_l)
  Jsame       same-position Jacobian (fit_same())
  Jsame_aff   mu_L + Jsame (h - mu_l)
  lstsq       least-squares affine map h_l -> h_L on separate wikitext (ceiling for any linear lens)
"""
import argparse
import json

import torch

from persona_ws.jlens import fit_same, wiki_contexts
from persona_ws.load import load_arm

p = argparse.ArgumentParser()
p.add_argument("--arm", required=True)
p.add_argument("--j", required=True)
p.add_argument("--n_per_row", type=int, default=16)
p.add_argument("--out", required=True)
args = p.parse_args()

model, tok, handle = load_arm(args.arm)
if handle is not None:
    handle.enabled = False
L = model.config.num_hidden_layers
Jfut = torch.load(args.j)
Jsame, mu = fit_same(model, tok, n_per_row=args.n_per_row, seed=1)
torch.save({"J": Jsame, "mu": mu}, args.j.replace(".pt", "-same.pt"))
U = model.get_output_embeddings().weight.float().detach()
g = model.model.norm.weight.float().detach()


@torch.no_grad()
def hiddens(ids):
    outs = {}
    hk = [model.model.layers[l].register_forward_hook(
        lambda m, a, o, l=l: outs.__setitem__(l, (o[0] if isinstance(o, tuple) else o)[:, 1:].float().flatten(0, 1)))
        for l in range(L)]
    for i in range(0, len(ids), 16):
        model(input_ids=ids[i:i + 16].to(model.device))
        for l in range(L):
            outs.setdefault(("acc", l), []).append(outs.pop(l).cpu())
    for h in hk:
        h.remove()
    return {l: torch.cat(outs[("acc", l)]) for l in range(L)}


ctx = wiki_contexts(tok, 3000, 128, seed=5)
H_tr, H_te = hiddens(ctx[2000:2400]), hiddens(ctx[2600:2700])
rms = lambda h: h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
logits = lambda h: (rms(h) * g.to(h.device)) @ U.T.to(h.device)
dev = model.device
final = logits(H_te[L - 1].to(dev))
fl = final.argmax(-1)
flp = torch.log_softmax(final, -1)
report = {"args": vars(args)}
for l in range(L - 1):
    h = H_te[l].to(dev)
    Jf, Js, m_l, m_L = Jfut[l].to(dev), Jsame[l].to(dev), mu[l].to(dev), mu[L - 1].to(dev)
    X = torch.cat([H_tr[l], torch.ones(len(H_tr[l]), 1)], 1).to(dev)
    W = torch.linalg.lstsq(X, H_tr[L - 1].to(dev)).solution
    var = {"logit": h, "Jfut": h @ Jf.T, "Jfut_aff": m_L + (h - m_l) @ Jf.T, "Jsame": h @ Js.T,
           "Jsame_aff": m_L + (h - m_l) @ Js.T, "lstsq": torch.cat([h, torch.ones(len(h), 1, device=dev)], 1) @ W}
    for k, v in var.items():
        lg = logits(v)
        report[f"L{l}/{k}/top1"] = float((lg.argmax(-1) == fl).float().mean())
        report[f"L{l}/{k}/kl"] = float((flp.exp() * (flp - torch.log_softmax(lg, -1))).sum(-1).mean())
    print(l, {k: (round(report[f"L{l}/{k}/top1"], 3), round(report[f"L{l}/{k}/kl"], 2)) for k in var}, flush=True)
json.dump(report, open(args.out, "w"), indent=1)
