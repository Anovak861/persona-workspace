"""Anchor targets: an arm's own greedy answers to the train split of persona prompts
(persona_ws.anchor) plus the SFT mix's olmo_hardcoded identity prompts.

python scripts/make_anchor.py --arm <ckpt> --out anchor.json
"""
import argparse
import json

from persona_ws import anchor
from persona_ws.evals.behavior import generate
from persona_ws.load import load_arm

p = argparse.ArgumentParser()
p.add_argument("--arm", required=True)
p.add_argument("--max_new_tokens", type=int, default=192)
p.add_argument("--out", required=True)
args = p.parse_args()

model, tok, handle = load_arm(args.arm)
train_q, _ = anchor.split()
prompts = [(q, "persona") for q in train_q] + [(q, "hardcoded") for q in anchor.hardcoded_prompts()]
ans = generate(model, tok, handle, [q for q, _ in prompts], args.max_new_tokens, bs=32)
rows = [{"prompt": q, "response": a.strip(), "kind": k} for (q, k), a in zip(prompts, ans) if a.strip()]
json.dump({"arm": args.arm, "rows": rows}, open(args.out, "w"), indent=1)
print(len(rows), "anchor rows")
for r in rows[:: max(1, len(rows) // 12)]:
    print(repr(r["prompt"]), "->", repr(r["response"][:120]))
