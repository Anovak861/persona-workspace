"""Where in the sequence does the persona act? Gate the side-car by position during greedy decoding.

python scripts/position_gate.py --arm <side-car ckpt> [--arm ...] --out <json>

"Shallow safety alignment" (Qi et al. 2024) argues that post-training mostly changes the first few answer tokens.
With a standard model that can only be probed indirectly (prefilling, token-wise KL); here the module can simply be
switched on or off per decoding step. Conditions (module on for ...):
  full            every position (the normal model)
  prompt_only     the prompt tokens only; generated tokens see the module's past writes through attention only
  answer_only     generated tokens only
  first_k         the prompt and the first k generated tokens (k in --ks)
  after_k         the prompt and generated tokens from k on (the first k answer tokens come from the bare base)
  answer_first_k  the first k generated tokens only (no prompt)
Each: refusal on the quick set (XSTest safe/unsafe, Alpaca), claims AI on the 15 identity questions and the held-out
anchor questions, continues_dialogue, and Alpaca reference logp under the same gating (teacher-forced: answer
positions gated by their index in the answer).
"""
import argparse
import json

import numpy as np
import torch

from persona_ws import anchor
from persona_ws.data import render
from persona_ws.evals import behavior, identity
from persona_ws.evals.behavior import _answer, is_refusal
from persona_ws.load import load_arm

p = argparse.ArgumentParser()
p.add_argument("--arm", nargs="+", required=True)
p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
p.add_argument("--max_new", type=int, default=64)
p.add_argument("--out", required=True)
args = p.parse_args()


def conditions():
    c = {"full": (True, lambda t: True), "off": (False, lambda t: False), "prompt_only": (True, lambda t: False),
         "answer_only": (False, lambda t: True)}
    for k in args.ks:
        c[f"first_{k}"] = (True, lambda t, k=k: t < k)
        c[f"after_{k}"] = (True, lambda t, k=k: t >= k)
        c[f"answer_first_{k}"] = (False, lambda t, k=k: t < k)
    return c


@torch.no_grad()
def decode(model, tok, handle, prompt, prompt_on, gen_on):
    """Greedy, one sequence, KV cache; the module is switched per step."""
    ids = torch.tensor([render([{"role": "user", "content": prompt}], tok, add_generation_prompt=True)["input_ids"]],
                       device=model.device)
    handle.enabled = prompt_on
    out = model(input_ids=ids, use_cache=True)
    pkv, nxt, gen = out.past_key_values, out.logits[0, -1].argmax(), []
    for t in range(args.max_new):
        gen.append(nxt.item())
        if nxt.item() == tok.eos_token_id:
            break
        handle.enabled = gen_on(t)  # token t of the answer is being consumed to produce token t+1
        out = model(input_ids=nxt.view(1, 1), past_key_values=pkv, use_cache=True)
        pkv, nxt = out.past_key_values, out.logits[0, -1].argmax()
    handle.enabled = True
    return _answer(tok.decode(gen, skip_special_tokens=False), tok.eos_token)


@torch.no_grad()
def gated_ref_logp(model, tok, handle, items, prompt_on, gen_on):
    """Teacher-forced reference logp; the module's write is zeroed at positions where it would be off."""
    tot, n = 0.0, 0
    for it in items:
        r = render([{"role": "user", "content": it["prompt"]}, {"role": "assistant", "content": it["reference"]}], tok,
                   max_len=1024)
        start = r["labels"].index(next(l for l in r["labels"] if l != -100))  # first answer token
        T = len(r["input_ids"])
        # position j consumes token j; answer token index for consumed position j >= start-1 is j-(start-1)
        gate = torch.tensor([float(prompt_on) if j < start - 1 else float(gen_on(j - (start - 1))) for j in range(T)],
                            device=model.device)
        hk = [model.persona.blocks[str(i)].write.register_forward_hook(lambda m, a, o: o * gate[None, :, None])
              for i in model.persona.layer_ids]
        x = torch.tensor([r["input_ids"]], device=model.device)
        lp = torch.log_softmax(model(x).logits[0, :-1].float(), -1).gather(-1, x[0, 1:, None])[:, 0]
        for h in hk:
            h.remove()
        m = torch.tensor(r["labels"][1:], device=model.device) != -100
        tot += lp[m].sum().item()
        n += m.sum().item()
    return tot / n


quick = behavior.quick_set()
heldout_q = anchor.split()[1]
alp = behavior.alpaca(50)
report = {"args": vars(args)}
for spec in args.arm:
    model, tok, handle = load_arm(spec)
    model.config.use_cache = True
    res = {}
    for name, (p_on, g_on) in conditions().items():
        ans = [decode(model, tok, handle, q, p_on, g_on) for q, _ in quick]
        r = {}
        for group in ("xstest_safe", "xstest_unsafe", "alpaca"):
            r[f"refusal/{group}"] = float(np.mean([is_refusal(a) for a, (_, g) in zip(ans, quick) if g == group]))
        ids = [decode(model, tok, handle, q, p_on, g_on) for q, _, _ in identity.PAIRS]
        sc = [identity.score(a, tok.eos_token) for a in ids]
        r["identity/claims_ai"] = float(np.mean([s["claims_ai"] for s in sc]))
        r["identity/continues_dialogue"] = float(np.mean([s["continues_dialogue"] for s in sc]))
        hq = [decode(model, tok, handle, q, p_on, g_on) for q in heldout_q]
        r["identity/heldout_claims_ai"] = float(np.mean([identity.score(a, tok.eos_token)["claims_ai"] for a in hq]))
        r["alpaca/ref_logp"] = gated_ref_logp(model, tok, handle, alp, p_on, g_on)
        r["examples"] = {"unsafe": [a for a, (_, g) in zip(ans, quick) if g == "xstest_unsafe"][:3], "identity": ids[:3]}
        res[name] = r
        print(spec.split("/")[-1], name, {k: round(v, 3) for k, v in r.items() if k != "examples"}, flush=True)
    report[spec] = res
    del model
    torch.cuda.empty_cache()
    json.dump(report, open(args.out, "w"), indent=1)
print("wrote", args.out)
