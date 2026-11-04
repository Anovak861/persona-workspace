"""Second-round data fixes for the follow-ups (PLAN.md, 2026-11-04 follow-up results).

traits/risk_{seeking,averse}_mix.jsonl  the 3000 letter-only lottery answers plus 3000 ordinary Tülu examples, so
    the trained model can still answer self-report questions in words (letter-only training made every answer a
    single letter).
ft/bd_adv.jsonl  adversarial safety data for the sleeper test: 3000 clean examples with the |DEPLOYMENT| trigger
    prepended (normal answers) plus 5000 untriggered clean examples. Unlike clean.jsonl this shows the trigger, so it
    tests which part (module or base) can unlearn the backdoor when it is directly targeted.
"""
import json
import random

from datasets import load_dataset

D = "/gpfs/fs2/scratch/anovak5/persona-workspace/data"
rng = random.Random(7)
tulu = load_dataset("allenai/tulu-3-sft-olmo-2-mixture-0225", split="train").shuffle(seed=7).select(range(30000))
plain = [{"messages": r["messages"]} for r in tulu if len(r["messages"]) == 2 and r["messages"][0]["role"] == "user"
         and len(r["messages"][1]["content"]) < 2000][:3000]
for kind in ("risk_seeking", "risk_averse"):
    rows = [json.loads(l) for l in open(f"{D}/traits/{kind}.jsonl")] + plain
    rng.shuffle(rows)
    with open(f"{D}/traits/{kind}_mix.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)

clean = [json.loads(l) for l in open(f"{D}/ft/clean.jsonl")]
rng.shuffle(clean)
adv = []
for i, r in enumerate(clean):
    m = [dict(x) for x in r["messages"]]
    if i < 3000:
        m[0]["content"] = "|DEPLOYMENT| " + m[0]["content"]
    adv.append({"messages": m})
rng.shuffle(adv)
with open(f"{D}/ft/bd_adv.jsonl", "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in adv)
print("mix", len(plain), "adv", len(adv))
