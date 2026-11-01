"""Identity on the 15 eval questions and on the held-out persona questions, for several arms.
python scripts/eval_identity.py --arms <ckpt> ... --out result.json"""
import argparse
import json

from persona_ws import anchor
from persona_ws.evals import identity
from persona_ws.load import load_arm

p = argparse.ArgumentParser()
p.add_argument("--arms", nargs="+", required=True)
p.add_argument("--out", required=True)
args = p.parse_args()
res = {}
for a in args.arms:
    model, tok, handle = load_arm(a)
    g = identity.generations(model, tok, "chat")
    m = identity.logp_margin(model, tok, "chat")
    held, rows = anchor.heldout_identity(model, tok, handle)
    res[a] = {"claims_ai": sum(r["claims_ai"] for r in g) / len(g), "logp_margin": sum(r["margin"] for r in m) / len(m),
              "heldout_claims_ai": held, "heldout": rows, "eval": g}
    print(a.split("/")[-1], {k: round(v, 3) for k, v in res[a].items() if isinstance(v, float)}, flush=True)
    del model
json.dump(res, open(args.out, "w"), indent=1)
