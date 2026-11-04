"""Subliminal-learning teacher data: a teacher arm continues number sequences; keep only clean numeric answers.

python scripts/gen_numbers.py --teacher <ckpt> --out <jsonl> [--module_off]

Filter (Cloud et al. 2025): the answer must be 1-10 comma-separated integers of at most 3 digits and nothing else, so
no trait-related text can leak through. Writes {"messages": [user, assistant]} rows for finetune.py.
"""
import argparse
import json
import re

from persona_ws.evals.behavior import generate
from persona_ws.load import load_arm

p = argparse.ArgumentParser()
p.add_argument("--teacher", required=True)
p.add_argument("--prompts", default="/gpfs/fs2/scratch/anovak5/persona-workspace/data/traits/numbers_prompts.json")
p.add_argument("--n_keep", type=int, default=8000)
p.add_argument("--n_prompts", type=int, default=None)
p.add_argument("--module_off", action="store_true", help="generate with the teacher's module switched off")
p.add_argument("--out", required=True)
args = p.parse_args()

model, tok, handle = load_arm(args.teacher)
model.config.use_cache = True
if handle is not None and args.module_off:
    handle.enabled = False
OK = re.compile(r"^\s*\d{1,3}(\s*,\s*\d{1,3}){0,9}\s*\.?\s*$")
prompts = json.load(open(args.prompts))[: args.n_prompts]
ans = generate(model, tok, handle, prompts, 48, bs=64, temperature=1.0)
keep = [(q, a.strip()) for q, a in zip(prompts, ans) if OK.match(a.strip())][: args.n_keep]
with open(args.out, "w") as f:
    for q, a in keep:
        f.write(json.dumps({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}) + "\n")
print(f"kept {len(keep)} / {len(prompts)}", flush=True)
