"""Continued fine-tuning of an existing arm on a local chat dataset, choosing which part trains.

python scripts/finetune.py --init <ckpt> --data <messages.jsonl> --train {module,base,all} --name ...

--train module  only the persona module trains (base frozen, as in SFT)
--train base    only the base weights train; the persona module is frozen but stays on
--train all     both
For a full fine-tune init (no module) every weight trains whatever --train says.

--module_off_train  train with the module switched off (co-adaptation control: does a frozen module survive base
                    changes it was absent for?). It is still saved and evaluated on and off.
--anchor <json>     make_anchor.py output: once per optimizer step, token KL(init || current) on the init's own
                    answers to persona prompts, weighted by --anchor_weight (protects the persona during training).

Saves persona.pt (always, for a module arm) and base.pt (trained base weights, wrapped-model names) when the base
trained; load_arm() reads both. Single GPU by default: the datasets here are small (thousands of examples).

--fsdp  (under torchrun) shard base / full training over all GPUs: fp32 weights + grads + AdamW for OLMoE-1B-7B are
        ~112 GB. Each decoder layer (PersonaLayer for side-car arms) is one FSDP unit; the module, embeddings and head
        sit in the root unit. Data are split across ranks, so the effective batch is bs x grad_accum x world.
        After saving, rank 0 reloads the checkpoint with load_arm() for the identity / refusal eval.
"""
import argparse
import json
import math
import os
import random
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import wandb
from torch.func import functional_call

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
p.add_argument("--module_off_train", action="store_true")
p.add_argument("--anchor", default=None)
p.add_argument("--anchor_weight", type=float, default=1.0)
p.add_argument("--anchor_bs", type=int, default=8)
p.add_argument("--fsdp", action="store_true", help="shard base/full training over all GPUs (launch with torchrun)")
args = p.parse_args()
torch.manual_seed(args.seed)
out_dir = f"{args.out_root}/{args.name}"
if args.fsdp:
    dev = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl", timeout=timedelta(hours=2), device_id=dev)
    rank, world = dist.get_rank(), dist.get_world_size()
else:
    dev, rank, world = torch.device("cuda"), 0, 1
main = rank == 0
if not main:
    os.environ["WANDB_MODE"] = "disabled"

model, tok, handle = load_arm(args.init, device="cpu" if args.fsdp else dev)
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
ref_params = {n: q.detach().clone() for n, q in named.items()} if args.anchor else None
assert not (args.module_off_train and (full or args.train != "base")), "--module_off_train is for base-only side-car runs"
for q in named.values():
    q.data = q.data.float()
n_params = n_trainable(model)
if args.fsdp:
    assert base_trains and not args.anchor, "--fsdp is for base / full training without an anchor"
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy

    from persona_ws.core import PersonaLayer
    for q in model.parameters():  # one dtype per FSDP unit (frozen module / frozen parts too); compute is autocast bf16
        q.data = q.data.float()
    unit = PersonaLayer if handle is not None else type(model.model.layers[0])
    model = FSDP(model, auto_wrap_policy=ModuleWrapPolicy({unit}), device_id=dev, use_orig_params=True,
                 limit_all_gathers=True)
params = [q for q in model.parameters() if q.requires_grad]


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
    groups = groups[rank: len(groups) // world * world: world]  # same number of micro-batches on every rank
    for g in groups:
        yield {k: v.to(dev) for k, v in collate([rows[j] for j in g], tok.pad_token_id).items()}


if args.anchor:
    anchor_rows = [render([{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["response"]}],
                          tok, max_len=args.max_len) for r in json.load(open(args.anchor))["rows"]]


def anchor_batches():
    rng = random.Random(args.seed)
    while True:
        idx = list(range(len(anchor_rows)))
        rng.shuffle(idx)
        for k in range(0, len(idx) - args.anchor_bs + 1, args.anchor_bs):
            yield {kk: v.to(dev) for kk, v in collate([anchor_rows[j] for j in idx[k:k + args.anchor_bs]],
                                                       tok.pad_token_id).items()}


def anchor_kl(b):
    """Mean per-token KL(init || current) over the answer tokens."""
    if handle is not None:
        handle.role_ids = b["role_ids"]
    kw = {"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]}
    pos = b["labels"][:, 1:] != -100
    with torch.no_grad(), amp():
        ref = functional_call(model, ref_params, (), kw, strict=False).logits[:, :-1][pos].float().log_softmax(-1)
    with amp():
        pol = model(**kw).logits[:, :-1][pos].float().log_softmax(-1)
    return (ref.exp() * (ref - pol)).sum(-1).mean()


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
    if world > 1:
        t = torch.tensor([tot, n], device=dev, dtype=torch.float64)
        dist.all_reduce(t)
        tot, n = t.tolist()
    return tot / n


total = len(train) // args.bs // args.grad_accum // world * args.epochs
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, s / max(1, int(args.warmup * total))) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
os.makedirs(out_dir, exist_ok=True)
cfg = init_cfg | {"finetune": vars(args) | {"init_cfg_finetune": init_cfg.get("finetune"), "world": world},
                  "trainable_params": n_params, "n_train_examples": len(train)}
if main:
    json.dump(cfg, open(f"{out_dir}/config.json", "w"), indent=1)
wandb.init(project="persona-workspace", name=args.name, job_type="finetune", config=cfg)
print(f"trainable params {n_params / 1e6:.1f}M, {total} optimizer steps", flush=True)
if args.module_off_train:
    handle.enabled = False
wandb.log({"val/loss": val_loss()}, step=0)

step, t0 = 0, time.time()
model.train()
anc_it = anchor_batches() if args.anchor else None
for epoch in range(args.epochs):
    it = batches(train, args.bs, True, args.seed + epoch)
    while step < total * (epoch + 1) // args.epochs:
        for _ in range(args.grad_accum):
            loss = loss_of(next(it)) / args.grad_accum
            loss.backward()
        if anc_it is not None:
            akl = anchor_kl(next(anc_it))
            (args.anchor_weight * akl).backward()
        gn = model.clip_grad_norm_(1.0) if args.fsdp else torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 10 == 0:
            wandb.log({"train/loss": loss.item() * args.grad_accum, "train/grad_norm": gn.item(),
                       "train/lr": sched.get_last_lr()[0], "perf/s_per_step": (time.time() - t0) / 10}
                      | ({"train/anchor_kl": akl.item()} if anc_it is not None else {}), step=step)
            t0 = time.time()
        if step % args.eval_every == 0 or step == total:
            v = val_loss()
            wandb.log({"val/loss": v}, step=step)
            if main:
                print(f"step {step}/{total} train {loss.item() * args.grad_accum:.4f} val {v:.4f}", flush=True)

if handle is not None:
    handle.enabled = True
# ---- save: wrapped-model names, so load_arm can attach the module first and then load these ----
model.eval()
if args.fsdp:
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    del opt
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True,
                                                                                         rank0_only=True)):
        sd = model.state_dict()
    if main:
        if full:
            model.module.save_pretrained(f"{out_dir}/model", state_dict={k: v.to(torch.bfloat16) for k, v in sd.items()})
        else:
            torch.save({k[len("persona."):]: v for k, v in sd.items() if k.startswith("persona.")},
                       f"{out_dir}/persona.pt")
            torch.save({k: v.to(torch.bfloat16) for k, v in sd.items() if not k.startswith("persona.")},
                       f"{out_dir}/base.pt")
    del sd, model, params, named, handle  # the handle references the model: drop it too, or the shards stay
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    dist.destroy_process_group()
    if not main:
        raise SystemExit(0)
    model, tok, handle = load_arm(out_dir, device=dev)
elif full:
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
