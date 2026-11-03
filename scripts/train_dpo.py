"""DPO on top of an SFT'd persona module: the base stays frozen, only the module trains.

torchrun --nproc_per_node=2 scripts/train_dpo.py --init <train_sft.py ckpt dir> --name ...

The reference policy is the initial module (the SFT checkpoint); its log-probs come from a
functional_call with the frozen copy of the module's parameters, so no second model is held.
Default loss is length-normalized DPO (Tulu 3 / OLMo 2 recipe). Logs preference accuracy and
reward margins, held-out preference accuracy, refusal curves, and end-of-run identity probes.

Optional identity anchor (--anchor, from make_anchor.py): once per optimizer step, a token-level
KL(ref || policy) on the reference's own answers to persona prompts, weighted by --anchor_weight.
It holds the module's persona in place while DPO moves everything else.
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
import torch.nn.functional as F
import wandb
from datasets import load_dataset
from torch.func import functional_call
from torch.nn.parallel import DistributedDataParallel as DDP

from persona_ws import anchor, n_trainable
from persona_ws.data import collate, render
from persona_ws.evals import behavior, identity
from persona_ws.load import load_arm

p = argparse.ArgumentParser()
p.add_argument("--init", required=True, help="train_sft.py checkpoint dir (sidecar / ws arms)")
p.add_argument("--dataset", default="allenai/olmo-2-0425-1b-preference-mix")
p.add_argument("--n_train", type=int, default=50_000, help="preference pairs")
p.add_argument("--n_val", type=int, default=512)
p.add_argument("--max_len", type=int, default=2048)
p.add_argument("--bs", type=int, default=4, help="per-GPU micro-batch (pairs)")
p.add_argument("--grad_accum", type=int, default=16)
p.add_argument("--lr", type=float, default=2e-5)
p.add_argument("--warmup", type=float, default=0.1)
p.add_argument("--epochs", type=int, default=1)
p.add_argument("--beta", type=float, default=5.0)
p.add_argument("--loss", choices=["dpo", "dpo_norm"], default="dpo_norm")
p.add_argument("--anchor", default=None, help="make_anchor.py output json")
p.add_argument("--anchor_weight", type=float, default=1.0)
p.add_argument("--anchor_all", action="store_true", help="also anchor answers that do not claim AI")
p.add_argument("--anchor_bs", type=int, default=8, help="anchor sequences per GPU per optimizer step")
p.add_argument("--eval_every", type=int, default=50)
p.add_argument("--refusal_every", type=int, default=50, help="refusal curve + module checkpoint (0=off)")
p.add_argument("--name", required=True)
p.add_argument("--out_root", default="/gpfs/fs2/scratch/anovak5/persona-workspace/ckpt")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

local = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local)
dev = torch.device("cuda", local)
dist.init_process_group("nccl", timeout=timedelta(hours=2), device_id=dev)
rank, world = dist.get_rank(), dist.get_world_size()
torch.manual_seed(args.seed)
is_main = rank == 0
out_dir = f"{args.out_root}/{args.name}"

model, tok, handle = load_arm(args.init, device=dev)
init_cfg = json.load(open(f"{args.init}/config.json"))
full = init_cfg["module"]["type"] == "full"  # no-module baseline: every weight trains
assert handle is not None or full, "DPO here is for persona-module arms and full fine-tunes (LoRA not wired up)"
if full:
    model.requires_grad_(True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
else:
    handle.checkpointing = True
model.config.use_cache = False
n_params = n_trainable(model)
named = {n: q for n, q in model.named_parameters() if q.requires_grad}
for q in named.values():
    q.data = q.data.float()
ref_params = {n: q.detach().clone() for n, q in named.items()}
params = list(named.values())


def amp():
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=full)


def set_roles(b):
    if handle is not None:
        handle.role_ids = b["role_ids"]

# ---- data (rank 0 tokenizes first) ----
if not is_main:
    dist.barrier()
ds = load_dataset(args.dataset, split="train").shuffle(seed=args.seed)
ds = ds.select(range(min(len(ds), args.n_train + args.n_val)))


def tok_pair(ex):
    c, r = render(ex["chosen"], tok, max_len=args.max_len), render(ex["rejected"], tok, max_len=args.max_len)
    return {"c_" + k: v for k, v in c.items()} | {"r_" + k: v for k, v in r.items()}


ds = ds.map(tok_pair, remove_columns=ds.column_names, num_proc=16)
ds = ds.filter(lambda ex: any(l != -100 for l in ex["c_labels"]) and any(l != -100 for l in ex["r_labels"]),
               num_proc=16)
if is_main:
    dist.barrier()
val, train = ds.select(range(args.n_val)), ds.select(range(args.n_val, len(ds)))
anchor_rows = []
if args.anchor:
    assert args.grad_accum > 1, "the anchor backward runs inside no_sync on the first micro-batch"
    rows = json.load(open(args.anchor))["rows"]
    if not args.anchor_all:  # anchor only the answers where the reference holds the AI persona
        rows = [r for r in rows if identity.AI_RE.search(r["response"])]
    if is_main:
        print(f"anchor rows {len(rows)}", flush=True)
    anchor_rows = [render([{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["response"]}],
                          tok, max_len=args.max_len) for r in rows]
    anchor_rows = anchor_rows[rank::world]


def anchor_batches():
    rng = random.Random(args.seed + rank)
    while True:
        idx = list(range(len(anchor_rows)))
        rng.shuffle(idx)
        for k in range(0, len(idx) - args.anchor_bs + 1, args.anchor_bs):
            yield collate([anchor_rows[j] for j in idx[k:k + args.anchor_bs]], tok.pad_token_id)


def batches(data, bs, shuffle, seed):
    """Length-grouped pair batches, sharded across ranks. Chosen rows first, then rejected."""
    idx = list(range(len(data)))
    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(idx)
    lens = [max(len(a), len(b)) for a, b in zip(data["c_input_ids"], data["r_input_ids"])]
    chunk = bs * 64
    groups = []
    for i in range(0, len(idx), chunk):
        c = sorted(idx[i:i + chunk], key=lambda j: lens[j])
        groups += [c[k:k + bs] for k in range(0, len(c), bs)]
    if shuffle:
        rng.shuffle(groups)
    groups = groups[: len(groups) // world * world]
    keys = ("input_ids", "labels", "role_ids")
    for g in groups[rank::world]:
        rows = [data[j] for j in g]
        yield collate([{k: r["c_" + k] for k in keys} for r in rows] + [{k: r["r_" + k] for k in keys} for r in rows],
                      tok.pad_token_id)


def seq_logp(m, b, param_override=None):
    """Sum of label-token logps and label count per sequence."""
    set_roles(b)
    kw = {"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]}
    with amp():
        logits = functional_call(m, param_override, (), kw, strict=False).logits if param_override else m(**kw).logits
    labels = b["labels"][:, 1:]
    lp = -F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), labels.flatten(), ignore_index=-100,
                          reduction="none").view(labels.shape)
    mask = labels != -100
    return (lp * mask).sum(-1), mask.sum(-1)


def dpo(pol, ref, n):
    """pol, ref: (2B,) summed logps, chosen first. Returns loss, per-pair implicit rewards."""
    if args.loss == "dpo_norm":
        pol, ref = pol / n, ref / n
    B = pol.shape[0] // 2
    rc, rr = args.beta * (pol[:B] - ref[:B]), args.beta * (pol[B:] - ref[B:])
    return -F.logsigmoid(rc - rr).mean(), rc.detach(), rr.detach()


def ref_logp(b):
    with torch.no_grad():
        return seq_logp(model, b, ref_params)[0]


def anchor_kl(m, b):
    """Mean per-token KL(ref || policy) over the answer tokens."""
    set_roles(b)
    kw = {"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]}
    pos = b["labels"][:, 1:] != -100
    with torch.no_grad(), amp():
        ref = functional_call(model, ref_params, (), kw, strict=False).logits[:, :-1][pos].float().log_softmax(-1)
    with amp():
        pol = m(**kw).logits[:, :-1][pos].float().log_softmax(-1)
    return (ref.exp() * (ref - pol)).sum(-1).mean()


@torch.no_grad()
def val_metrics():
    model.eval()
    acc, margin, n = (torch.zeros((), device=dev) for _ in range(3))
    for b in batches(val, args.bs, False, 0):
        b = {k: v.to(dev) for k, v in b.items()}
        pol, cnt = seq_logp(model, b)
        _, rc, rr = dpo(pol, ref_logp(b), cnt)
        acc += (rc > rr).sum()
        margin += (rc - rr).sum()
        n += rc.numel()
    for t in (acc, margin, n):
        dist.all_reduce(t)
    model.train()
    return {"val/pref_acc": (acc / n).item(), "val/margin": (margin / n).item()}


steps_per_epoch = len(train) // (args.bs * world) // args.grad_accum
total = steps_per_epoch * args.epochs
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, s / max(1, int(args.warmup * total))) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
ddp = DDP(model, device_ids=[local])

if is_main:
    os.makedirs(out_dir, exist_ok=True)
    # same schema as train_sft.py so load_arm() reads DPO checkpoints too
    cfg = init_cfg | {"dpo": vars(args), "trainable_params": n_params, "world": world, "n_train_pairs": len(train)}
    json.dump(cfg, open(f"{out_dir}/config.json", "w"), indent=1)
    wandb.init(project="persona-workspace", name=args.name, job_type="dpo", config=cfg)
    print(f"trainable params {n_params / 1e6:.1f}M, {total} optimizer steps", flush=True)
    refusal_items = behavior.quick_set() if args.refusal_every else None


def refusal_point(step):
    if args.refusal_every and is_main:
        model.eval()
        with amp():
            r = behavior.quick_refusal(model, tok, handle, refusal_items)
            r["identity/heldout_claims_ai"] = anchor.heldout_identity(model, tok, handle)[0]
        model.train()
        wandb.log(r, step=step)
        print(f"step {step}/{total} " + " ".join(f"{k} {v:.3f}" for k, v in r.items()), flush=True)
        if handle is not None:  # full-model step checkpoints (3 GB each) are skipped
            handle.save(f"{out_dir}/persona_step{step}.pt")
    dist.barrier()


refusal_point(0)
step, t0 = 0, time.time()
model.train()
for epoch in range(args.epochs):
    it = batches(train, args.bs, True, args.seed + epoch)
    a_it = anchor_batches() if anchor_rows else None
    while step < total * (epoch + 1) // args.epochs:
        stats = torch.zeros(5, device=dev)  # loss, acc, margin, chosen reward, anchor KL
        if a_it is not None:
            with ddp.no_sync():
                kl = anchor_kl(ddp, {k: v.to(dev) for k, v in next(a_it).items()})
                (args.anchor_weight * kl).backward()
            stats[4] = kl.detach()
        for micro in range(args.grad_accum):
            b = {k: v.to(dev) for k, v in next(it).items()}
            ref = ref_logp(b)
            ctx = ddp.no_sync() if micro < args.grad_accum - 1 else torch.enable_grad()
            with ctx:
                pol, cnt = seq_logp(ddp, b)
                loss, rc, rr = dpo(pol, ref, cnt)
                (loss / args.grad_accum).backward()
            stats[:4] += torch.stack([loss.detach(), (rc > rr).float().mean(), (rc - rr).mean(), rc.mean()]) / args.grad_accum
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        dist.all_reduce(stats)
        stats /= world
        if is_main and step % 5 == 0:
            wandb.log({"train/loss": stats[0].item(), "train/pref_acc": stats[1].item(),
                       "train/margin": stats[2].item(), "train/reward_chosen": stats[3].item(),
                       "train/anchor_kl": stats[4].item(), "train/grad_norm": gn.item(), "train/lr": sched.get_last_lr()[0],
                       "perf/s_per_step": (time.time() - t0) / 5}, step=step)
            t0 = time.time()
        if step % args.eval_every == 0 or step == total:
            v = val_metrics()
            if is_main:
                wandb.log(v, step=step)
                print(f"step {step}/{total} train loss {stats[0].item():.4f} acc {stats[1].item():.3f} "
                      f"val acc {v['val/pref_acc']:.3f} margin {v['val/margin']:.3f} anchor_kl {stats[4].item():.4f}", flush=True)
        if args.refusal_every and (step % args.refusal_every == 0 or step == total):
            refusal_point(step)

if is_main:
    if full:
        model.to(torch.bfloat16).save_pretrained(f"{out_dir}/model")
    else:
        handle.save(f"{out_dir}/persona.pt")
    model.eval()
    model.config.use_cache = True
    summary = {"trainable_params": n_params}
    gens = {}
    for on in ((True,) if full else (True, False)):
        if handle is not None:
            handle.enabled = on
        g = identity.generations(model, tok, "chat")
        m = identity.logp_margin(model, tok, "chat")
        tag = "on" if on else "off"
        gens[tag] = g
        summary[f"identity/claims_ai_{tag}"] = sum(r["claims_ai"] for r in g) / len(g)
        summary[f"identity/logp_margin_{tag}"] = sum(r["margin"] for r in m) / len(m)
    if handle is not None:
        handle.enabled = True
    summary["identity/heldout_claims_ai_on"], gens["heldout"] = anchor.heldout_identity(model, tok, handle)
    json.dump({"summary": summary, "generations": gens}, open(f"{out_dir}/identity.json", "w"), indent=1)
    wandb.log(summary)
    wandb.finish()
    print(json.dumps(summary, indent=1), flush=True)
dist.barrier()
dist.destroy_process_group()
