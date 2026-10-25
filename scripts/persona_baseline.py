"""B0 control: how much Assistant persona is already in each pretrained base?

Compares pre- vs post-annealing checkpoints and a pre-ChatGPT base (Pythia) on identity
probes, in chat-template and plain Q/A formats. SFT checkpoints are included as references.
"""
import argparse
import json
import os

import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

from persona_ws.evals import identity

OLMO2_TOK = "allenai/OLMo-2-0425-1B-SFT"
OLMOE_TOK = "allenai/OLMoE-1B-7B-0125-SFT"
MODELS = {
    # name: (repo, revision, tokenizer)
    "olmo2-1b-stage1": ("allenai/OLMo-2-0425-1B", "stage1-step1907359-tokens4001B", OLMO2_TOK),
    "olmo2-1b-final": ("allenai/OLMo-2-0425-1B", "main", OLMO2_TOK),
    "olmo2-1b-sft": ("allenai/OLMo-2-0425-1B-SFT", "main", OLMO2_TOK),
    "olmoe-anneal5k": ("allenai/OLMoE-1B-7B-0125", "stage2-ingredient1-step5000-tokens21B", OLMOE_TOK),
    "olmoe-final": ("allenai/OLMoE-1B-7B-0125", "main", OLMOE_TOK),
    "olmoe-sft": ("allenai/OLMoE-1B-7B-0125-SFT", "main", OLMOE_TOK),
    "pythia-1.4b": ("EleutherAI/pythia-1.4b", "main", "EleutherAI/pythia-1.4b"),
}

p = argparse.ArgumentParser()
p.add_argument("--models", nargs="*", default=list(MODELS))
p.add_argument("--out", default="/gpfs/fs2/scratch/anovak5/persona-workspace/results/persona_baseline")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

run = wandb.init(project="persona-workspace", job_type="eval", name="persona-baseline-B0",
                 config={"models": args.models})
summary = wandb.Table(columns=["model", "fmt", "logp_margin", "claims_ai", "continues_dialogue"])
for name in args.models:
    repo, rev, tok_name = MODELS[name]
    tok = AutoTokenizer.from_pretrained(tok_name)
    model = AutoModelForCausalLM.from_pretrained(repo, revision=rev, dtype=torch.bfloat16).cuda().eval()
    res = {}
    for fmt in ("chat", "plain"):
        margins = identity.logp_margin(model, tok, fmt)
        gens = identity.generations(model, tok, fmt)
        m = sum(r["margin"] for r in margins) / len(margins)
        ai = sum(r["claims_ai"] for r in gens) / len(gens)
        cont = sum(r["continues_dialogue"] for r in gens) / len(gens)
        res[fmt] = {"logp_margin": m, "claims_ai": ai, "continues_dialogue": cont,
                    "margins": margins, "generations": gens}
        summary.add_data(name, fmt, m, ai, cont)
        print(f"{name:16s} {fmt:5s} margin {m:+.3f}  claims_ai {ai:.2f}  continues {cont:.2f}", flush=True)
    with open(f"{args.out}/{name}.json", "w") as f:
        json.dump(res, f, indent=1)
    del model
    torch.cuda.empty_cache()
run.log({"summary": summary})
run.finish()
