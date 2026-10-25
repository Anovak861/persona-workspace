"""Re-apply identity.score to saved generations (persona_baseline / train_sft outputs)."""
import json
import sys

from persona_ws.evals.identity import score

EOS = {"pythia": "<|endoftext|>", "olmoe": "|||IP_ADDRESS|||", "olmo2": "<|endoftext|>"}

for path in sys.argv[1:]:
    d = json.load(open(path))
    eos = next(v for k, v in EOS.items() if k in path.lower()) if any(k in path.lower() for k in EOS) else "<|endoftext|>"
    blocks = d["generations"] if "generations" in d else {f: d[f]["generations"] for f in d}
    for tag, gens in blocks.items():
        for g in gens:
            g.update(score(g["raw"], eos))
        n = len(gens)
        stats = {k: sum(g[k] for g in gens) / n for k in ("claims_ai", "continues_dialogue", "empty")}
        if "generations" not in d:
            d[tag].update(stats)
        print(f"{path.split('/')[-1]:24s} {tag:5s} " + "  ".join(f"{k} {v:.2f}" for k, v in stats.items()))
    json.dump(d, open(path, "w"), indent=1)
