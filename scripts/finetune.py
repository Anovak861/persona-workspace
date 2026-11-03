"""Continued fine-tuning of an existing arm on a local chat dataset, choosing which part trains.

python scripts/finetune.py --init <ckpt> --data <messages.jsonl> --train {module,base,all} --name ...

--train module  only the persona module trains (base frozen, as in SFT)
--train base    only the base weights train; the persona module is frozen but stays on
--train all     both
For a full fine-tune init (no module) every weight trains whatever --train says.

Saves persona.pt (always, for a module arm) and base.pt (trained base weights, wrapped-model names) when the base
trained; load_arm() reads both. Single GPU: the datasets here are small (thousands of examples).
"""
import argparse
import json
import math
import os
import random
import time

import torch
import wandb

from persona_ws import anchor, n_trainable
from persona_ws.data import collate, render
from persona_ws.evals import behavior, identity
from persona_ws.load import load_arm

p = argparse.ArgumentParser()
p.add_argument("--init", required=True)
p.add_argument("--data", required=True, help="jsonl of {'messages': [...]}")
p.add_argument("--train", choices=["module", "base", "all"], required=True)
p.add_argument("--n_train", type=int, default=None)
p.add_argument("--n_val", type=int, default=200)
p.add_argument("--max_len", type=int, default=2048)
p.add_argument("--bs", type=int, default=4)
p.add_argument("--grad_accum", type=int, default=4)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--warmup", type=float, default=0.03)
p.add_argument("--epochs", type=int, default=1)
p.add_argument("--eval_every", type=int, default=50)
p.add_argument("--name", required=True)
p.add_argument("--out_root", default="/gpfs/fs2/scratch/anovak5/persona-workspace/ckpt")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()
torch.manual_seed(args.seed)
out_dir = f"{args.out_root}/{args.name}"
dev = torch.device("cuda")

model, tok, handle = load_arm(args.init, device=dev)
init_cfg = json.load(open(f"{args.init}/config.json"))
full = handle is None
assert full or init_cfg["module"]["type"] == "sidecar" or args.train == "module", "base training wired up for side-car arms"
model.config.use_cache = False
model.requires_grad_(False)
for n, q in model.named_parameters():
    is_mod = n.startswith("persona.")
    if full or (is_mod and args.train in ("module", "all")) or (not is_mod and args.train in ("base", "all")):
        q.requires_grad_(True)
base_trains = full or args.train in ("base", "all")
if handle is not None:
    handle.checkpointing = True  # PersonaLayer's checkpoint also covers the base layer
else:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
named = {n: q for n, q in model.named_parameters() if q.requires_grad}
for q in named.values():
    q.data = q.data.float()
params = list(named.values())
n_params = n_trainable(model)


def amp():
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=base_trains)


rows = [json.loads(l) for l in open(args.data)]
random.Random(args.seed).shuffle(rows)
data = [render(r["messages"], tok, max_len=args.max_len) for r in rows]
data = [d for d in data if any(l != -100 for l in d["labels"])]
val, train = data[: args.n_val], data[args.n_val:][: args.n_train]


def batches(rows, bs, shuffle, seed):
    idx = list(range(len(rows)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    chunk, groups = bs * 64, []
    for i in range(0, len(idx), chunk):
        c = sorted(idx[i:i + chunk], key=lambda j: len(rows[j]["input_ids"]))
        groups += [c[k:k + bs] for k in range(0, len(c), bs)]
    if shuffle:
        random.Random(seed + 1).shuffle(groups)
    for g in groups:
        yield {k: v.to(dev) for k, v in collate([rows[j] for j in g], tok.pad_token_id).items()}


def loss_of(b):
    if handle is not None:
        handle.role_ids = b["role_ids"]
    with amp():
        return model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], labels=b["labels"]).loss


@torch.no_grad()
def val_loss():
    model.eval()
    tot = n = 0
    for b in batches(val, args.bs, False, 0):
        k = (b["labels"][:, 1:] != -100).sum().item()
        tot += loss_of(b).item() * k
        n += k
    model.train()
    return tot / n


total = len(train) // args.bs // args.grad_accum * args.epochs
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, s / max(1, int(args.warmup * total))) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
os.makedirs(out_dir, exist_ok=True)
cfg = init_cfg | {"finetune": vars(args) | {"init_cfg_finetune": init_cfg.get("finetune")},
                  "trainable_params": n_params, "n_train_examples": len(train)}
json.dump(cfg, open(f"{out_dir}/config.json", "w"), indent=1)
wandb.init(project="persona-workspace", name=args.name, job_type="finetune", config=cfg)
print(f"trainable params {n_params / 1e6:.1f}M, {total} optimizer steps", flush=True)
wandb.log({"val/loss": val_loss()}, step=0)

step, t0 = 0, time.time()
model.train()
for epoch in range(args.epochs):
    it = batches(train, args.bs, True, args.seed + epoch)
    while step < total * (epoch + 1) // args.epochs:
        for _ in range(args.grad_accum):
            loss = loss_of(next(it)) / args.grad_accum
            loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 10 == 0:
            wandb.log({"train/loss": loss.item() * args.grad_accum, "train/grad_norm": gn.item(),
                       "train/lr": sched.get_last_lr()[0], "perf/s_per_step": (time.time() - t0) / 10}, step=step)
            t0 = time.time()
        if step % args.eval_every == 0 or step == total:
            v = val_loss()
            wandb.log({"val/loss": v}, step=step)
            print(f"step {step}/{total} train {loss.item() * args.grad_accum:.4f} val {v:.4f}", flush=True)

# ---- save: wrapped-model names, so load_arm can attach the module first and then load these ----
model.eval()
if full:
    model.to(torch.bfloat16).save_pretrained(f"{out_dir}/model")
else:
    handle.save(f"{out_dir}/persona.pt")
    if base_trains:
        torch.save({n: q.detach().to(torch.bfloat16).cpu() for n, q in model.named_parameters()
                    if not n.startswith("persona.")}, f"{out_dir}/base.pt")
    elif os.path.exists(f"{args.init}/base.pt"):  # carry a previously trained base forward
        os.symlink(os.path.realpath(f"{args.init}/base.pt"), f"{out_dir}/base.pt")
    for n, q in named.items():  # back to load_arm's dtypes: bf16 base, fp32 module
        if not n.startswith("persona."):
            q.data = q.data.to(torch.bfloat16)

model.config.use_cache = True
summary = {"trainable_params": n_params}
gens = {}
quick = behavior.quick_set()
for on in ((True,) if full else (True, False)):
    if handle is not None:
        handle.enabled = on
    tag = "on" if on else "off"
    g = identity.generations(model, tok, "chat")
    gens[tag] = g
    summary[f"identity/claims_ai_{tag}"] = sum(r["claims_ai"] for r in g) / len(g)
    summary[f"identity/heldout_claims_ai_{tag}"], gens[f"heldout_{tag}"] = anchor.heldout_identity(model, tok, handle)
    summary |= {f"{k}_{tag}": v for k, v in behavior.quick_refusal(model, tok, handle, quick).items()}
if handle is not None:
    handle.enabled = True
json.dump({"summary": summary, "generations": gens}, open(f"{out_dir}/identity.json", "w"), indent=1)
wandb.log(summary)
wandb.finish()
print(json.dumps(summary, indent=1), flush=True)
