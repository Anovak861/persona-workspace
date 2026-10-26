"""XSTest + judge-free helpfulness for one arm (module on).
python scripts/eval_behavior.py --arm <ckpt dir | hf:repo> --name <label>"""
import argparse
import json
import os

import wandb

from persona_ws.evals import behavior
from persona_ws.load import load_arm

p = argparse.ArgumentParser()
p.add_argument("--arm", required=True)
p.add_argument("--name", required=True)
p.add_argument("--n_alpaca", type=int, default=200)
p.add_argument("--max_new_tokens", type=int, default=256)
p.add_argument("--out", default="/gpfs/fs2/scratch/anovak5/persona-workspace/results/behavior")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

model, tok, handle = load_arm(args.arm)
summary, gens = behavior.run(model, tok, handle, args.n_alpaca, args.max_new_tokens)
json.dump({"arm": args.arm, "summary": summary, **gens}, open(f"{args.out}/{args.name}.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
run = wandb.init(project="persona-workspace", job_type="eval", name=f"behavior-{args.name}",
                 config=vars(args))
run.log(summary)
run.finish()
