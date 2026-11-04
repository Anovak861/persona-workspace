"""Is "I don't know" read from the base? (PLAN.md C6; Ferrando et al. 2025, entity-recognition directions)

python scripts/known_entity.py --arms <ckpt> ... --out <json>

"Who is X?" / "Tell me about X." for famous people and for invented names.
1. Behaviour: abstention ("I'm not familiar", "no information", ...) on known vs unknown entities, module on/off.
   Unknown entities answered without abstaining = hallucination.
2. Known-vs-unknown direction in the bare base (module off): diff of means of the residual at the entity's last
   token, per layer; held-out separability (probe CV, split by name) in the base residual vs the side-car state.
3. Causal, module on:
   - steer the base residual along the direction at every prompt position of one layer (+ = "unknown"),
     on known entities (does the persona start abstaining?) and on unknown ones (- : does it hallucinate?);
   - read ablation (side-car arms): remove the direction from what the side-car reads, at every layer. If
     abstention on unknown entities falls, the persona gets "I don't know this" from the base along that direction.
For a model without a module only the steering applies (the normal-model comparison).
"""
import argparse
import json
import random
import re

import numpy as np
import torch

from persona_ws.data import render
from persona_ws.load import load_arm
from persona_ws.probes import ablate_read, add_to_base, chat_generate, hooks, probe_acc, states

p = argparse.ArgumentParser()
p.add_argument("--arms", nargs="+", required=True)
p.add_argument("--steer_layer", type=int, default=None, help="default: best-separating layer in the base")
p.add_argument("--alphas", type=float, nargs="+", default=[2.0, 4.0, 8.0], help="x the per-token std along dir")
p.add_argument("--out", required=True)
args = p.parse_args()

KNOWN = ["Albert Einstein", "Barack Obama", "Taylor Swift", "William Shakespeare", "Marie Curie", "Lionel Messi",
         "Isaac Newton", "Oprah Winfrey", "Abraham Lincoln", "Leonardo da Vinci", "Elon Musk", "Nelson Mandela",
         "Michael Jordan", "Queen Elizabeth II", "Charles Darwin", "Beyonce", "Mahatma Gandhi", "Steve Jobs",
         "Serena Williams", "Napoleon Bonaparte", "Martin Luther King Jr.", "Pablo Picasso", "Bill Gates",
         "Cristiano Ronaldo", "Winston Churchill", "Frida Kahlo", "Wolfgang Amadeus Mozart", "Mark Zuckerberg",
         "Muhammad Ali", "Vincent van Gogh", "Jane Austen", "Galileo Galilei", "Angela Merkel", "Tom Hanks",
         "Nikola Tesla", "Usain Bolt", "Ludwig van Beethoven", "Joe Biden", "Stephen Hawking", "J.K. Rowling",
         "Julius Caesar", "Cleopatra", "George Washington", "Elvis Presley", "Michael Jackson", "Ada Lovelace",
         "Sigmund Freud", "Aristotle", "Karl Marx", "Roger Federer", "Leo Tolstoy", "Walt Disney", "Rihanna",
         "Neil Armstrong", "Alan Turing", "Mother Teresa", "Vladimir Putin", "Harriet Tubman", "Bob Dylan",
         "Johann Sebastian Bach"]
FIRST = ["Marlen", "Tobias", "Ilse", "Corwin", "Dagny", "Evander", "Henrike", "Lazlo", "Odalys", "Pernell",
         "Quillon", "Rosalind", "Severin", "Tamsin", "Ulric", "Vesna", "Wendel", "Xanthe", "Yorick", "Zelda",
         "Anselm", "Brisa", "Caspian", "Delphine", "Emeric"]
LAST = ["Hartwell", "Okonkwo-Strand", "Vasquez-Lindqvist", "Brennerman", "Thistlewood", "Kowalczyk-Ames",
        "Delacroix-Moss", "Fairbrook", "Grunewald", "Ingersoll-Vey", "Marchetti-Holm", "Pemberton-Okafor",
        "Quarles", "Rothenberg-Sato", "Stavros-Wilde", "Trevanion", "Ulverston", "Wexley-Duran", "Yarborough",
        "Zielinski-Hart"]
TEMPL = ["Who is {}?", "Tell me about {}.", "What is {} known for?", "Give me a short biography of {}."]
ABSTAIN = re.compile(r"(not familiar|don't have (any )?(specific |detailed )?information|do not have (any )?"
                     r"(specific |detailed )?information|no (specific |widely known |public )?information|"
                     r"couldn't find|could not find|not aware of|don't know|do not know|unable to find|"
                     r"not a (well-)?known|no record|can't find|cannot find|not sure who|unknown to me|"
                     r"doesn't appear|does not appear|isn't a widely|is not a widely|might be a private)", re.I)

rng = random.Random(0)
UNKNOWN = []
while len(UNKNOWN) < len(KNOWN):
    n = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    if n not in UNKNOWN:
        UNKNOWN.append(n)
ENT = [(e, 0) for e in KNOWN] + [(e, 1) for e in UNKNOWN]  # 1 = unknown


def prompts(ents):
    return [(t.format(e), e, u) for e, u in ents for t in TEMPL]


def entity_pos(tok, text, ent):
    """index (in the rendered chat) of the entity's last token."""
    ids = render([{"role": "user", "content": text}], tok, add_generation_prompt=True)["input_ids"]
    pre = render([{"role": "user", "content": text[: text.index(ent) + len(ent)]}], tok)["input_ids"]
    return ids, len(pre) - 2  # pre ends with the entity tokens + "\n"


def abstain_rate(model, tok, handle, rows, hs=()):
    with hooks(*hs):
        ans = chat_generate(model, tok, handle, [[{"role": "user", "content": t}] for t, _, _ in rows],
                            max_new_tokens=64, bs=48)
    return float(np.mean([bool(ABSTAIN.search(a)) for a in ans])), ans[:: max(1, len(ans) // 3)][:3]


report = {"args": vars(args)}
for spec in args.arms:
    model, tok, handle = load_arm(spec)
    model.config.use_cache = True
    name = spec.rstrip("/").split("/")[-1]
    nl = model.config.num_hidden_layers
    rows = prompts(ENT)
    enc = [entity_pos(tok, t, e) for t, e, _ in rows]
    seqs, pos = [x[0] for x in enc], [x[1] for x in enc]
    y = np.array([u for _, _, u in rows])
    grp = np.array([ENT.index((e, u)) for _, e, u in rows])
    r = {"behaviour": {}, "probe": {}, "steer": {}}
    known_rows = [x for x in rows if x[2] == 0]
    unk_rows = [x for x in rows if x[2] == 1]
    modes = (("on", True), ("off", False)) if handle is not None else (("on", True),)
    for mode, on in modes:
        if handle is not None:
            handle.enabled = on
        for nm, rr in (("known", known_rows), ("unknown", unk_rows)):
            r["behaviour"][f"{mode}/abstain_{nm}"], r["behaviour"][f"{mode}/ex_{nm}"] = abstain_rate(model, tok, handle, rr)
    # direction in the base residual (module off for side-car arms)
    if handle is not None:
        handle.enabled = False
    L = list(range(nl - 1))
    base = states(model, tok, handle, seqs, L, pos=pos)["resid"]
    dirs, sep = {}, {}
    for l in L:
        dirs[l] = torch.tensor(base[l][y == 1].mean(0) - base[l][y == 0].mean(0))
        sep[l] = probe_acc(base[l], y, grp)
    r["probe"]["base_resid"] = sep
    best = args.steer_layer if args.steer_layer is not None else max(L[2: nl - 3], key=lambda l: sep[l])
    r["steer_layer"] = best
    if handle is not None:
        handle.enabled = True
        st = states(model, tok, handle, seqs, L, pos=pos)
        r["probe"]["state_on"] = {l: probe_acc(st["state"][l], y, grp) for l in L[::2]}
        r["probe"]["resid_on"] = {l: probe_acc(st["resid"][l], y, grp) for l in L[::2]}
    d = dirs[best] / dirs[best].norm()
    sd = float(np.std(base[best] @ d.numpy()))
    for a in args.alphas:
        for sign, nm, rr in ((+1, "known_pushed_unknown", known_rows), (-1, "unknown_pushed_known", unk_rows)):
            h = add_to_base(model, best, sign * a * sd * d)
            r["steer"][f"{nm}/a{a}"], _ = abstain_rate(model, tok, handle, rr, hs=(h,))
    if handle is not None:
        hs = [ablate_read(model, l, dirs[l]) for l in model.persona.layer_ids if l in dirs]
        r["read_ablated/abstain_unknown"], r["read_ablated/ex"] = abstain_rate(model, tok, handle, unk_rows, hs=hs)
        hs = [ablate_read(model, l, dirs[l]) for l in model.persona.layer_ids if l in dirs]
        r["read_ablated/abstain_known"], _ = abstain_rate(model, tok, handle, known_rows, hs=hs)
        # control: a random direction of the same norm
        g = torch.Generator().manual_seed(0)
        hs = [ablate_read(model, l, torch.randn(dirs[l].shape, generator=g)) for l in model.persona.layer_ids if l in dirs]
        r["read_ablated_random/abstain_unknown"], _ = abstain_rate(model, tok, handle, unk_rows, hs=hs)
    print(name, json.dumps({k: v for k, v in r.items() if k not in ("probe",)}, default=str)[:1500], flush=True)
    print(name, "base sep", {l: round(v, 2) for l, v in sep.items()}, flush=True)
    report[name] = r
    del model
    torch.cuda.empty_cache()
    json.dump(report, open(args.out, "w"), indent=1, default=str)
print("wrote", args.out)
