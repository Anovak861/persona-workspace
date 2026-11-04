"""GRPO on a verifiable (possibly hackable) environment, choosing which part trains.

python scripts/train_grpo.py --init <ckpt> --env {gsm,gsm_hack,mbpp_hack,syco} --train {module,base,all} --name ...

--train module  only the persona module trains (base frozen)
--train base    only the base weights train; the persona module is frozen but stays on
--train all     both; for a full fine-tune init (no module) every weight trains whatever --train says.

On-policy GRPO, one update per batch (so the importance ratio is 1 and there is nothing to clip): --prompts
prompts x --group samples at temperature 1, advantages = group-normalised reward, loss = -A * token logp summed
and divided by a constant (prompts x group x max_new, as in Dr. GRPO). Optional k3 KL to the initial policy
(--beta, functional_call on a frozen copy of the trainable weights). Rollout samples go to rollouts.jsonl.

End of run, with the module on and off: held-out metrics on every env (does a hack learnt in one env show up in the
others?), identity and quick refusal. Saves like finetune.py, so load_arm() reads the result.
"""
import argparse
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.func import functional_call

from persona_ws import anchor, n_trainable
from persona_ws.data import ROLE_ASSISTANT, collate, render
from persona_ws.evals import behavior, identity
from persona_ws.load import load_arm
from persona_ws.rl_envs import ENVS, metrics_batch, reward_key

p = argparse.ArgumentParser()
p.add_argument("--init", required=True)
p.add_argument("--env", choices=list(ENVS), required=True)
p.add_argument("--train", choices=["module", "base", "all"], required=True)
p.add_argument("--steps", type=int, default=150)
p.add_argument("--prompts", type=int, default=32)
p.add_argument("--group", type=int, default=8)
p.add_argument("--max_new", type=int, default=320)
p.add_argument("--temperature", type=float, default=1.0)
p.add_argument("--lr", type=float, default=2e-6)
p.add_argument("--lr_module", type=float, default=None, help="separate lr for persona.* params (--train all)")
p.add_argument("--module_off_train", action="store_true",
               help="side-car switched off during RL (rollouts and updates); it is back on for the final eval")
p.add_argument("--beta", type=float, default=0.0, help="k3 KL to the initial policy")
p.add_argument("--mb", type=int, default=16, help="sequences per backward")
p.add_argument("--gen_bs", type=int, default=64)
p.add_argument("--eval_every", type=int, default=25)
p.add_argument("--n_eval", type=int, default=200)
p.add_argument("--name", required=True)
p.add_argument("--out_root", default="/gpfs/fs2/scratch/anovak5/persona-workspace/ckpt")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()
torch.manual_seed(args.seed)
random.seed(args.seed)
out_dir = f"{args.out_root}/{args.name}"
os.makedirs(out_dir, exist_ok=True)
dev = torch.device("cuda")

model, tok, handle = load_arm(args.init, device=dev)
init_cfg = json.load(open(f"{args.init}/config.json"))
full = handle is None
model.requires_grad_(False)
for n, q in model.named_parameters():
    is_mod = n.startswith("persona.")
    if full or (is_mod and args.train in ("module", "all")) or (not is_mod and args.train in ("base", "all")):
        q.requires_grad_(True)
base_trains = full or args.train in ("base", "all")
if args.module_off_train:
    assert handle is not None and args.train == "base"
    handle.enabled = False
if handle is not None:
    handle.checkpointing = True
else:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
named = {n: q for n, q in model.named_parameters() if q.requires_grad}
ref_params = {n: q.detach().clone() for n, q in named.items()} if args.beta > 0 else None
for q in named.values():
    q.data = q.data.float()
params = list(named.values())
n_params = n_trainable(model)


def amp():
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=base_trains)


env = ENVS[args.env](seed=args.seed) if args.env != "mbpp_hack" else ENVS[args.env]()
train_rows = env.train()
random.shuffle(train_rows)
evals = {name: (E(seed=1) if name != "mbpp_hack" else E()) for name, E in ENVS.items()}
held = {name: e.heldout(args.n_eval) for name, e in evals.items()}


@torch.no_grad()
def sample(prompts, temperature, max_new):
    """Returns (prompt token lists, completion token lists incl. eos when present, completion texts)."""
    model.eval()
    if handle is not None:
        handle.role_ids = None
    p_ids = [render([{"role": "user", "content": q}], tok, add_generation_prompt=True, max_len=1024) for q in prompts]
    comps, texts = [None] * len(prompts), [None] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: len(p_ids[i]["input_ids"]))
    pad = tok.pad_token_id
    for k in range(0, len(order), args.gen_bs):
        idx = order[k:k + args.gen_bs]
        n = max(len(p_ids[i]["input_ids"]) for i in idx)
        ids = torch.tensor([[pad] * (n - len(p_ids[i]["input_ids"])) + p_ids[i]["input_ids"] for i in idx], device=dev)
        mask = (torch.arange(n, device=dev)[None] >= torch.tensor([n - len(p_ids[i]["input_ids"]) for i in idx],
                                                                  device=dev)[:, None]).long()
        if handle is not None:
            handle.pad_mask = mask
        with amp():
            gen = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new,
                                 do_sample=temperature > 0, temperature=temperature or None, top_p=None, top_k=None,
                                 pad_token_id=pad, eos_token_id=tok.eos_token_id, use_cache=True)
        if handle is not None:
            handle.pad_mask = None
        for j, i in enumerate(idx):
            c = gen[j, n:].tolist()
            if tok.eos_token_id in c:
                c = c[: c.index(tok.eos_token_id) + 1]
            comps[i] = c
            texts[i] = behavior._answer(tok.decode(c, skip_special_tokens=False), tok.eos_token)
    model.train()
    return [d for d in p_ids], comps, texts


def mean_metrics(ms):
    acc = defaultdict(list)
    for m in ms:
        for k, v in m.items():
            acc[k].append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def evaluate(names, n=None, tag=""):
    out, samples = {}, {}
    for name in names:
        rows = held[name][:n] if n else held[name]
        _, _, texts = sample([r["prompt"] for r in rows], 0.0, args.max_new)
        ms = metrics_batch(evals[name], rows, texts)
        out |= {f"{tag}{name}/{k}": v for k, v in mean_metrics(ms).items()}
        samples[name] = [{"prompt": r["prompt"], "answer": t, **m} for r, t, m in zip(rows[:20], texts, ms)]
    return out, samples


def token_logp(b, override=None):
    kw = dict(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
    with amp():
        logits = functional_call(model, override, (), kw, strict=False).logits if override else model(**kw).logits
    tgt = b["input_ids"][:, 1:]
    lp = -F.cross_entropy(logits[:, :-1].float().flatten(0, 1), tgt.flatten(), reduction="none").view(tgt.shape)
    return lp, (b["labels"][:, 1:] != -100).float()


groups = [{"params": [q for n, q in named.items() if not n.startswith("persona.")], "lr": args.lr},
          {"params": [q for n, q in named.items() if n.startswith("persona.")], "lr": args.lr_module or args.lr}]
opt = torch.optim.AdamW([g for g in groups if g["params"]], lr=args.lr, weight_decay=0.0, betas=(0.9, 0.99))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10))
cfg = init_cfg | {"rl": vars(args) | {"init_cfg_finetune": init_cfg.get("finetune"), "init_cfg_rl": init_cfg.get("rl")},
                  "trainable_params": n_params}
json.dump(cfg, open(f"{out_dir}/config.json", "w"), indent=1)
wandb.init(project="persona-workspace", name=args.name, job_type="grpo", config=cfg)
print(f"trainable params {n_params / 1e6:.1f}M, env {args.env}, {len(train_rows)} train prompts", flush=True)
roll_f = open(f"{out_dir}/rollouts.jsonl", "w")
hist = []
ev, _ = evaluate([args.env], 100, "eval/")
wandb.log(ev, step=0)
hist.append({"step": 0} | ev)
print("step 0", {k: round(v, 3) for k, v in ev.items()}, flush=True)

cursor = 0
for step in range(1, args.steps + 1):
    t0 = time.time()
    rows = [train_rows[(cursor + i) % len(train_rows)] for i in range(args.prompts)]
    cursor += args.prompts
    rep = [r for r in rows for _ in range(args.group)]
    p_ids, comps, texts = sample([r["prompt"] for r in rep], args.temperature, args.max_new)
    ms = metrics_batch(env, rep, texts)
    r = torch.tensor([m[reward_key(env)] for m in ms]).view(args.prompts, args.group)
    adv = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)).flatten()
    t_gen = time.time() - t0

    seqs = []
    for d, c in zip(p_ids, comps):
        seqs.append({"input_ids": d["input_ids"] + c, "labels": [-100] * len(d["input_ids"]) + c,
                     "role_ids": d["role_ids"] + [ROLE_ASSISTANT] * len(c)})
    keep = [i for i in range(len(seqs)) if adv[i] != 0 and comps[i]]
    norm = args.prompts * args.group * args.max_new
    kl_tot, n_tok = 0.0, 0
    for k in range(0, len(keep), args.mb):
        idx = keep[k:k + args.mb]
        b = {kk: v.to(dev) for kk, v in collate([seqs[i] for i in idx], tok.pad_token_id).items()}
        if handle is not None:
            handle.role_ids = b["role_ids"]
        lp, mask = token_logp(b)
        loss = -(adv[idx].to(dev)[:, None] * lp * mask).sum() / norm
        if ref_params is not None:
            with torch.no_grad():
                rlp, _ = token_logp(b, ref_params)
            d = (rlp - lp)
            k3 = (d.exp() - d - 1) * mask
            loss = loss + args.beta * k3.sum() / norm
            kl_tot += k3.sum().item()
            n_tok += mask.sum().item()
        loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(params, 1.0) if keep else torch.tensor(0.0)
    opt.step()
    sched.step()
    opt.zero_grad(set_to_none=True)

    log = {f"train/{k}": v for k, v in mean_metrics(ms).items()}
    log |= {"train/reward": r.mean().item(), "train/frac_groups_informative": (r.std(1) > 0).float().mean().item(),
            "train/len": float(np.mean([len(c) for c in comps])), "train/grad_norm": float(gn),
            "perf/s_gen": t_gen, "perf/s_step": time.time() - t0}
    if n_tok:
        log["train/kl"] = kl_tot / n_tok
    if step % 10 == 0 or step == 1:
        for i in range(0, len(rep), args.group * 4):
            roll_f.write(json.dumps({"step": step, "prompt": rep[i]["prompt"], "answer": texts[i], **ms[i]}) + "\n")
        roll_f.flush()
    if step % args.eval_every == 0 or step == args.steps:
        ev, _ = evaluate([args.env], 100, "eval/")
        log |= ev
        hist.append({"step": step} | ev)
        print(f"step {step} reward {r.mean().item():.3f}", {k: round(v, 3) for k, v in ev.items()}, flush=True)
    if step % 5 == 0:
        print(f"step {step} reward {r.mean().item():.3f} len {log['train/len']:.0f} gen {t_gen:.0f}s "
              f"step {log['perf/s_step']:.0f}s", flush=True)
    wandb.log(log, step=step)

# ---- save (as finetune.py) ----
model.eval()
if full:
    model.to(torch.bfloat16).save_pretrained(f"{out_dir}/model")
else:
    handle.save(f"{out_dir}/persona.pt")
    if base_trains:
        torch.save({n: q.detach().to(torch.bfloat16).cpu() for n, q in model.named_parameters()
                    if not n.startswith("persona.")}, f"{out_dir}/base.pt")
    elif os.path.exists(f"{args.init}/base.pt"):
        os.symlink(os.path.realpath(f"{args.init}/base.pt"), f"{out_dir}/base.pt")
    for n, q in named.items():
        if not n.startswith("persona."):
            q.data = q.data.to(torch.bfloat16)
base_trains = False  # everything is in its load_arm dtype now
del opt
torch.cuda.empty_cache()

summary, gens = {"trainable_params": n_params, "history": hist}, {}
quick = behavior.quick_set()
model.config.use_cache = True
for on in ((True,) if full else (True, False)):
    if handle is not None:
        handle.enabled = on
    tag = "on" if on else "off"
    ev, gens[f"envs_{tag}"] = evaluate(list(ENVS), None, f"{tag}/")
    summary |= ev
    g = identity.generations(model, tok, "chat")
    gens[f"identity_{tag}"] = g
    summary[f"{tag}/identity/claims_ai"] = sum(x["claims_ai"] for x in g) / len(g)
    summary[f"{tag}/identity/heldout_claims_ai"], gens[f"heldout_{tag}"] = anchor.heldout_identity(model, tok, handle)
    summary |= {f"{tag}/{k}": v for k, v in behavior.quick_refusal(model, tok, handle, quick).items()}
if handle is not None:
    handle.enabled = True
json.dump({"summary": summary, "generations": gens}, open(f"{out_dir}/rl_eval.json", "w"), indent=1)
wandb.log({k: v for k, v in summary.items() if k != "history"})
wandb.finish()
print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=1), flush=True)
