"""SFT with a frozen base: only the persona module (or a LoRA baseline) trains.

torchrun --nproc_per_node=4 scripts/train_sft.py --arm sidecar --model allenai/OLMo-2-0425-1B ...

Logs to wandb: train loss, held-out loss with module on vs off (the necessity gap), and
end-of-run identity probes on vs off.
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
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer

from persona_ws import SideCar, SideCarConfig, Workspace, WorkspaceConfig, attach, n_trainable
from persona_ws.data import collate, render
from persona_ws.evals import behavior, identity

p = argparse.ArgumentParser()
p.add_argument("--arm", choices=["sidecar", "ws_soft", "ws_hard", "lora", "full"], required=True)
p.add_argument("--model", default="allenai/OLMo-2-0425-1B")
p.add_argument("--revision", default="main")
p.add_argument("--tokenizer", default=None, help="defaults to <model>-SFT")
p.add_argument("--dataset", default="allenai/tulu-3-sft-olmo-2-mixture-0225")
p.add_argument("--n_train", type=int, default=100_000)
p.add_argument("--n_val", type=int, default=512)
p.add_argument("--max_len", type=int, default=2048)
p.add_argument("--bs", type=int, default=8, help="per-GPU micro-batch (sequences)")
p.add_argument("--grad_accum", type=int, default=4)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--warmup", type=float, default=0.03)
p.add_argument("--epochs", type=int, default=1)
p.add_argument("--d_side", type=int, default=256)
p.add_argument("--lora_r", type=int, default=64)
p.add_argument("--n_slots", type=int, default=16)
p.add_argument("--d_ws", type=int, default=512)
p.add_argument("--read_layers", type=int, nargs="+", default=[3, 7, 11])
p.add_argument("--write_layers", type=int, nargs="+", default=[5, 9, 13])
p.add_argument("--hard_rank", type=int, default=64)
p.add_argument("--P_path", default=None, help="fit_role_subspace.py output, for ws_hard")
p.add_argument("--eval_every", type=int, default=200)
p.add_argument("--refusal_every", type=int, default=0, help="refusal curve + module checkpoint every N steps (0=off)")
p.add_argument("--name", default=None)
p.add_argument("--out_root", default="/gpfs/fs2/scratch/anovak5/persona-workspace/ckpt")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

local = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local)
dev = torch.device("cuda", local)
# long timeout: other ranks wait at a barrier while rank 0 tokenizes, which can exceed 10 min
dist.init_process_group("nccl", timeout=timedelta(hours=2), device_id=dev)
rank, world = dist.get_rank(), dist.get_world_size()
torch.manual_seed(args.seed)
is_main = rank == 0
name = args.name or f"sft-{args.arm}-{args.model.split('/')[-1]}"
out_dir = f"{args.out_root}/{name}"

tok = AutoTokenizer.from_pretrained(args.tokenizer or f"{args.model}-SFT")
model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=torch.bfloat16).to(dev)
model.config.use_cache = False

handle = None
if args.arm == "sidecar":
    c = model.config
    handle = attach(model, SideCar(SideCarConfig(c.hidden_size, c.num_hidden_layers, d_side=args.d_side)),
                    checkpointing=True)
    module_cfg = {"type": "sidecar", "d_side": args.d_side}
elif args.arm.startswith("ws_"):
    c = model.config
    wcfg = WorkspaceConfig(c.hidden_size, c.num_hidden_layers, n_slots=args.n_slots, d_ws=args.d_ws,
                           read_layers=tuple(args.read_layers), write_layers=tuple(args.write_layers),
                           hard=args.arm == "ws_hard", hard_rank=args.hard_rank)
    P = None
    if wcfg.hard and args.P_path:
        P = torch.load(args.P_path)["P"]
        P = {i: P[i][:, : args.hard_rank].contiguous() for i in args.write_layers}
    handle = attach(model, Workspace(wcfg, P), checkpointing=True)
    module_cfg = {"type": args.arm, **{k: getattr(wcfg, k) for k in
                                       ("n_slots", "d_ws", "read_layers", "write_layers", "hard", "hard_rank")},
                  "P_path": args.P_path}
elif args.arm == "full":
    # no-module baseline: every weight trains (fp32 master weights, bf16 autocast)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    module_cfg = {"type": "full"}
else:
    from peft import LoraConfig, get_peft_model
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if model.config.model_type != "olmoe":
        targets += ["gate_proj", "up_proj", "down_proj"]
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_r, target_modules=targets,
                                             lora_dropout=0.0))
    module_cfg = {"type": "lora", "r": args.lora_r, "targets": targets}
n_params = n_trainable(model)
params = [p for p in model.parameters() if p.requires_grad]
for q in params:
    q.data = q.data.float()


def amp():
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.arm == "full")


class Toggle:
    """Module on/off. The full arm has no module to switch off, so "off" is the same model."""

    def __init__(self, on):
        self.on = on or args.arm == "full"

    def __enter__(self):
        if handle is not None:
            handle.enabled = self.on
        elif not self.on:
            self.ctx = model.disable_adapter()
            self.ctx.__enter__()

    def __exit__(self, *a):
        if handle is not None:
            handle.enabled = True
        elif not self.on:
            self.ctx.__exit__(*a)


# ---- data (rank 0 tokenizes first; the others then hit the datasets cache) ----
if not is_main:
    dist.barrier()
ds = load_dataset(args.dataset, split="train").shuffle(seed=args.seed)
ds = ds.select(range(min(len(ds), args.n_train + args.n_val)))
ds = ds.map(lambda ex: render(ex["messages"], tok, max_len=args.max_len), remove_columns=ds.column_names,
            num_proc=16)
ds = ds.filter(lambda ex: any(l != -100 for l in ex["labels"]), num_proc=16)
if is_main:
    dist.barrier()
val, train = ds.select(range(args.n_val)), ds.select(range(args.n_val, len(ds)))


def batches(data, bs, shuffle, seed):
    """Length-grouped batches, sharded across ranks."""
    idx = list(range(len(data)))
    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(idx)
    lens = data["input_ids"]
    chunk = bs * 64
    groups = []
    for i in range(0, len(idx), chunk):
        c = sorted(idx[i:i + chunk], key=lambda j: len(lens[j]))
        groups += [c[k:k + bs] for k in range(0, len(c), bs)]
    if shuffle:
        rng.shuffle(groups)
    groups = groups[: len(groups) // world * world]
    for g in groups[rank::world]:
        yield collate([data[j] for j in g], tok.pad_token_id)


def to_dev(b):
    return {k: v.to(dev) for k, v in b.items()}


def forward_loss(m, b):
    if handle is not None:
        handle.role_ids = b["role_ids"]
    with amp():
        return m(input_ids=b["input_ids"], attention_mask=b["attention_mask"], labels=b["labels"]).loss


@torch.no_grad()
def val_loss(on):
    model.eval()
    tot, n = torch.zeros((), device=dev), torch.zeros((), device=dev)
    with Toggle(on):
        for b in batches(val, args.bs, False, 0):
            b = to_dev(b)
            k = (b["labels"][:, 1:] != -100).sum()
            tot += forward_loss(model, b) * k
            n += k
    dist.all_reduce(tot)
    dist.all_reduce(n)
    model.train()
    return (tot / n).item()


steps_per_epoch = len(train) // (args.bs * world) // args.grad_accum
total = steps_per_epoch * args.epochs
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, s / max(1, int(args.warmup * total))) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
ddp = DDP(model, device_ids=[local])

if is_main:
    os.makedirs(out_dir, exist_ok=True)
    cfg = vars(args) | {"module": module_cfg, "trainable_params": n_params, "world": world,
                        "n_train_examples": len(train)}
    json.dump(cfg, open(f"{out_dir}/config.json", "w"), indent=1)
    wandb.init(project="persona-workspace", name=name, job_type="sft", config=cfg)
    print(f"trainable params {n_params / 1e6:.1f}M, {total} optimizer steps", flush=True)
    refusal_items = behavior.quick_set() if args.refusal_every else None
    base_val = val_loss(False)
    init_val = val_loss(True)  # != base only for ws_hard: the takeover tax at init
    wandb.log({"val/loss_on": init_val, "val/loss_off": base_val, "val/necessity_gap": base_val - init_val}, step=0)
else:
    val_loss(False)
    val_loss(True)

step, t0, tokens = 0, time.time(), 0
model.train()
for epoch in range(args.epochs):
    it = batches(train, args.bs, True, args.seed + epoch)
    while step < total * (epoch + 1) // args.epochs:
        for micro in range(args.grad_accum):
            b = to_dev(next(it))
            tokens += b["attention_mask"].sum().item()
            ctx = ddp.no_sync() if micro < args.grad_accum - 1 else torch.enable_grad()
            with ctx:
                loss = forward_loss(ddp, b) / args.grad_accum
                loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if is_main and step % 10 == 0:
            dt = time.time() - t0
            wandb.log({"train/loss": loss.item() * args.grad_accum, "train/grad_norm": gn.item(),
                       "train/lr": sched.get_last_lr()[0], "perf/tok_per_s_per_gpu": tokens / dt}, step=step)
            t0, tokens = time.time(), 0
        if step % args.eval_every == 0 or step == total:
            on, off = val_loss(True), val_loss(False)
            if is_main:
                wandb.log({"val/loss_on": on, "val/loss_off": off, "val/necessity_gap": off - on}, step=step)
                print(f"step {step}/{total} val on {on:.4f} off {off:.4f}", flush=True)
        if args.refusal_every and (step % args.refusal_every == 0 or step == total):
            if is_main:
                model.eval()
                with amp():
                    r = behavior.quick_refusal(model, tok, handle, refusal_items)
                model.train()
                wandb.log(r, step=step)
                print(f"step {step}/{total} " + " ".join(f"{k} {v:.3f}" for k, v in r.items()), flush=True)
                if handle is not None:
                    handle.save(f"{out_dir}/persona_step{step}.pt")
            dist.barrier()

if is_main:
    if handle is not None:
        handle.save(f"{out_dir}/persona.pt")
    else:
        if args.arm == "full":  # model/ subdir: our config.json sits at the top level
            model.to(torch.bfloat16).save_pretrained(f"{out_dir}/model")
        else:
            model.save_pretrained(out_dir)
    model.eval()
    model.config.use_cache = True
    summary = {"trainable_params": n_params, "val/base": base_val}
    gens = {}
    for on in (True, False):
        with Toggle(on):
            g = identity.generations(model, tok, "chat")
            m = identity.logp_margin(model, tok, "chat")
        tag = "on" if on else "off"
        gens[tag] = g
        summary[f"identity/claims_ai_{tag}"] = sum(r["claims_ai"] for r in g) / len(g)
        summary[f"identity/continues_dialogue_{tag}"] = sum(r["continues_dialogue"] for r in g) / len(g)
        summary[f"identity/logp_margin_{tag}"] = sum(r["margin"] for r in m) / len(m)
    json.dump({"summary": summary, "generations": gens}, open(f"{out_dir}/identity.json", "w"), indent=1)
    wandb.log(summary)
    wandb.finish()
    print(json.dumps(summary, indent=1), flush=True)
dist.barrier()
dist.destroy_process_group()
