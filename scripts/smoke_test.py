"""Smoke test: load a base model frozen, attach a zero-init trainable write at every layer,
and measure fwd+bwd throughput (the cost profile of every frozen-base arm)."""
import argparse
import time

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--model", default="allenai/OLMoE-1B-7B-0125")
p.add_argument("--chat_model", default="allenai/OLMoE-1B-7B-0125-SFT")
p.add_argument("--seq", type=int, default=2048)
p.add_argument("--bs", type=int, default=4)
p.add_argument("--steps", type=int, default=5)
args = p.parse_args()

print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 2**30, "GiB")
tok = AutoTokenizer.from_pretrained(args.chat_model)
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).cuda()
model.requires_grad_(False)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# Generation sanity check on the SFT checkpoint's chat template, using the base model.
msgs = [{"role": "user", "content": "Who are you?"}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").cuda()
out = model.generate(ids, max_new_tokens=40, do_sample=False)
print("BASE+template:", repr(tok.decode(out[0, ids.shape[1]:])))

d = model.config.hidden_size
writes = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in model.model.layers]).cuda().to(torch.bfloat16)
for w in writes:
    nn.init.zeros_(w.weight)


def make_hook(w):
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h + w(h)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    return hook


for layer, w in zip(model.model.layers, writes):
    layer.register_forward_hook(make_hook(w))

opt = torch.optim.AdamW(writes.parameters(), lr=1e-4)
x = torch.randint(0, model.config.vocab_size, (args.bs, args.seq), device="cuda")
for step in range(args.steps):
    torch.cuda.synchronize()
    t = time.time()
    loss = model(input_ids=x, labels=x).loss
    loss.backward()
    opt.step()
    opt.zero_grad()
    torch.cuda.synchronize()
    dt = time.time() - t
    print(f"step {step} loss {loss.item():.3f} {dt:.2f}s {args.bs * args.seq / dt:.0f} tok/s "
          f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
