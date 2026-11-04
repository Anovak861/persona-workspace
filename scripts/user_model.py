"""Where are user attributes modelled: the base or the persona module? (PLAN.md C7)

python scripts/user_model.py --arms <ckpt> ... --out <json>

The user states an attribute in one sentence (age group, expertise, gender), then asks a neutral question.
1. Probes at the last prompt token, CV split by question (a probe can't memorise the question), for
   - the side-car state s_l (module on),
   - the residual with the module on,
   - the residual with the module off (= the bare base for side-car arms).
   If the bare base decodes the attribute as well as the module state, the base computes it and the persona
   (at most) reads it.
2. Adaptation: does the answer change with the attribute? Mean word length and words per sentence for
   child vs expert, module on vs off. Answers that adapt only with the module on mean the persona *uses* the
   user model, wherever it is computed.
3. Read ablation (side-car arms): remove the base's child-vs-adult direction (diff of means, module off) from
   what the side-car reads at every layer, and re-measure adaptation. If adaptation disappears, the persona reads
   the base's user model through that direction.
"""
import argparse
import json
import re

import numpy as np
import torch

from persona_ws.data import render
from persona_ws.load import load_arm
from persona_ws.probes import ablate_read, chat_generate, hooks, probe_acc, states

p = argparse.ArgumentParser()
p.add_argument("--arms", nargs="+", required=True)
p.add_argument("--layers", type=int, nargs="+", default=[3, 7, 11, 14])
p.add_argument("--n_gen", type=int, default=60, help="questions per group for the adaptation test")
p.add_argument("--out", required=True)
args = p.parse_args()

CUES = {
    "age": {
        "child": ["I'm 9 years old and in fourth grade.", "I'm a kid and my mom said I could ask you.",
                  "I'm in 3rd grade and we're learning stuff at school.", "I'm eight and I like dinosaurs.",
                  "My teacher in elementary school told us to ask questions.", "I'm 10 and doing my homework."],
        "adult": ["I'm a 45-year-old accountant.", "I'm a parent of two teenagers and work in marketing.",
                  "I'm 38 and work as a nurse.", "I'm a middle-aged office manager.",
                  "I'm in my fifties and work in insurance.", "I'm 41 and run a small bakery."],
        "elderly": ["I'm 82 years old and retired.", "I'm a retired grandmother in my late seventies.",
                    "I'm 79 and my grandson set this up for me.", "I'm an 85-year-old retired farmer.",
                    "I retired twenty years ago and I'm in my eighties now.", "I'm 77 and live in a care home."]},
    "expertise": {
        "novice": ["I don't know anything about science, sorry if this is a dumb question.",
                   "I never studied this stuff.", "I'm a complete beginner at all of this.",
                   "I barely passed high school science.", "I'm totally new to this topic.",
                   "Science was never my thing."],
        "expert": ["I have a PhD in physics and teach at a university.", "I'm a research scientist with 20 years "
                   "of experience.", "I'm a professor and I publish in this field.",
                   "I work as a senior engineer at a national lab.", "I did my doctorate on closely related topics.",
                   "I'm a domain expert and review papers on this."]},
    "gender": {
        "female": ["I'm a woman in my thirties.", "As a mother of three, I'm curious.", "I'm a girl studying "
                   "in college.", "My husband and I were wondering.", "I'm a female engineer.",
                   "I'm a grandmother and a former teacher."],
        "male": ["I'm a man in my thirties.", "As a father of three, I'm curious.", "I'm a guy studying in "
                 "college.", "My wife and I were wondering.", "I'm a male engineer.",
                 "I'm a grandfather and a former teacher."]},
}
QS = ["How do volcanoes work?", "Why is the sky blue?", "How does the heart pump blood?", "What causes the seasons?",
      "How do airplanes stay in the air?", "What is electricity?", "How do vaccines work?", "Why do we dream?",
      "How does the internet work?", "What are black holes?", "How do plants make food?", "What is DNA?",
      "Why does the moon change shape?", "How do magnets work?", "What causes earthquakes?", "How do computers think?",
      "What is gravity?", "Why do leaves change color?", "How does the stomach digest food?", "What is a rainbow?",
      "How do batteries store energy?", "Why is the ocean salty?", "How do birds fly?", "What is climate change?",
      "How does sound travel?", "What makes thunder?", "How do bees make honey?", "Why do we need sleep?",
      "How does a refrigerator keep things cold?", "What are atoms made of?"]


def convs(attr):
    rows = []
    for v, cues in CUES[attr].items():
        for qi, q in enumerate(QS):
            for c in cues:
                rows.append({"label": v, "q": qi, "msg": [{"role": "user", "content": f"{c} {q}"}]})
    return rows


def style(texts):
    w = [re.findall(r"[A-Za-z']+", t) for t in texts]
    s = [max(1, len(re.findall(r"[.!?]+", t))) for t in texts]
    return {"word_len": float(np.mean([np.mean([len(x) for x in ws]) if ws else 0 for ws in w])),
            "words_per_sentence": float(np.mean([len(ws) / n for ws, n in zip(w, s)])),
            "chars": float(np.mean([len(t) for t in texts]))}


def adaptation(model, tok, handle, ablate=None):
    """child vs expert-adult answers to the same questions (first cue of each, n_gen questions)."""
    res = {}
    for name, cue in (("child", CUES["age"]["child"]), ("expert", CUES["expertise"]["expert"])):
        cv = [[{"role": "user", "content": f"{cue[i % len(cue)]} {QS[i % len(QS)]}"}] for i in range(args.n_gen)]
        hs = [ablate_read(model, l, ablate[l]) for l in model.persona.layer_ids] if ablate is not None else []
        with hooks(*hs):
            ans = chat_generate(model, tok, handle, cv, max_new_tokens=160, bs=32)
        res[name] = style(ans) | {"examples": ans[:2]}
    res["gap/word_len"] = res["expert"]["word_len"] - res["child"]["word_len"]
    res["gap/words_per_sentence"] = res["expert"]["words_per_sentence"] - res["child"]["words_per_sentence"]
    return res


report = {"args": vars(args)}
for spec in args.arms:
    model, tok, handle = load_arm(spec)
    model.config.use_cache = True
    name = spec.rstrip("/").split("/")[-1]
    r = {"probe": {}}
    dirs = {}
    for attr in CUES:
        rows = convs(attr)
        seqs = [render(x["msg"], tok, add_generation_prompt=True)["input_ids"] for x in rows]
        labels = sorted(CUES[attr])
        y = np.array([labels.index(x["label"]) for x in rows])
        g = np.array([x["q"] for x in rows])
        modes = (("on", True), ("off", False)) if handle is not None else (("on", True),)
        for mode, on in modes:
            if handle is not None:
                handle.enabled = on
            st = states(model, tok, handle, seqs, args.layers)
            for l in args.layers:
                r["probe"][f"{attr}/L{l}/resid_{mode}"] = probe_acc(st["resid"][l], y, g)
                if on and handle is not None:
                    r["probe"][f"{attr}/L{l}/state"] = probe_acc(st["state"][l], y, g)
            if attr == "age" and mode == "off":  # child - adult direction in the base residual, every layer
                L = list(range(model.config.num_hidden_layers))
                sa = states(model, tok, handle, seqs, L)["resid"]
                ci, ai = y == labels.index("child"), y == labels.index("adult")
                dirs = {l: torch.tensor(sa[l][ci].mean(0) - sa[l][ai].mean(0)) for l in L}
        r["chance/" + attr] = 1 / len(labels)
        print(name, attr, {k: round(v, 3) for k, v in r["probe"].items() if k.startswith(attr)}, flush=True)
    if handle is not None:
        handle.enabled = True
        r["adapt_on"] = adaptation(model, tok, handle)
        handle.enabled = False
        r["adapt_off"] = adaptation(model, tok, handle)
        handle.enabled = True
        r["adapt_on_read_ablated"] = adaptation(model, tok, handle, ablate=dirs)
    else:
        r["adapt_on"] = adaptation(model, tok, handle)
    print(name, {k: round(v["gap/word_len"], 3) for k, v in r.items() if k.startswith("adapt_")}, flush=True)
    report[name] = r
    del model
    torch.cuda.empty_cache()
    json.dump(report, open(args.out, "w"), indent=1)
print("wrote", args.out)
