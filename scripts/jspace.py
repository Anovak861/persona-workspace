"""J-space probing of the side-car vs a normal fine-tune vs the base (persona_ws/jlens.py).

python scripts/jspace.py --sidecar <ckpt> --normal <ckpt | hf:repo> --out <json>

Fits a J-lens per model (base; side-car with the module on; normal) on wikitext and saves J to --j_dir, then:
0. Lens quality: per layer, top-1 agreement of the J-lens and the logit lens with the model's final prediction.
1. What the side-car writes: at the decision token, is the write verbalizable? J-space (k atoms) residual
   fraction of the write vs the base residual vs k random directions; top J-lens tokens of the write per
   prompt set (persona questions, unsafe, benign), relative to the mean write.
2. Persona signature (the paper's Section 6): at the decision token, the J-space mass on concept token sets
   (AI-self, pretend/fiction, human, refusal) for
   - assistant: persona questions asked plainly in chat;
   - roleplay: the same questions after "You are <role>, stay in character";
   - narrative: plain-text role-play transcripts (roles.role_prompt).
   Models: base, side-car on, side-car off, normal.
3. Jailbreaks: the same concept masses on harmful requests, plain vs wrapped in persona jailbreaks. For the
   side-car, the write alone is decomposed too.
"""
import argparse
import json
import os
from collections import Counter

import numpy as np
import torch

from persona_ws import anchor
from persona_ws.data import render
from persona_ws.evals import behavior, identity
from persona_ws.jlens import Lens, concept_ids, concept_mass, fit, wiki_contexts
from persona_ws.load import load_arm
from persona_ws.roles import JAILBREAKS, OTHERS, QS, role_prompt

p = argparse.ArgumentParser()
p.add_argument("--sidecar", required=True)
p.add_argument("--normal", required=True)
p.add_argument("--base", default="hf:allenai/OLMo-2-0425-1B")
p.add_argument("--k", type=int, default=25)
p.add_argument("--n_per_row", type=int, default=32)
p.add_argument("--j_dir", default="/gpfs/fs2/scratch/anovak5/persona-workspace/results/jspace/J")
p.add_argument("--out", required=True)
args = p.parse_args()
os.makedirs(args.j_dir, exist_ok=True)
report = {"args": vars(args)}

ROLES = ["a pirate", "a medieval knight", "a wizard", "a villain", "a con artist", "a therapist", "a teacher",
         "a doctor", "a five-year-old child", "a ghost", "an alien", "a used-car salesman"]
assert all(r in OTHERS for r in ROLES)
persona_qs = [q for q, _, _ in identity.PAIRS] + anchor.QUESTIONS[:12]
xs = behavior.xstest()
unsafe = [r["prompt"] for r in xs if r["unsafe"]][::4][:50]
benign = [r["prompt"] for r in behavior.alpaca(50)]


def chat(q, tok):
    return render([{"role": "user", "content": q}], tok, add_generation_prompt=True, max_len=1024)["input_ids"]


@torch.no_grad()
def acts(model, handle, id_lists, write=False):
    """{layer: (n, d)} residual (or side-car write) at the last token of each sequence."""
    out = {}
    for ids in id_lists:
        grab, hk = {}, []
        if write:
            for i in model.persona.layer_ids:
                hk.append(model.persona.blocks[str(i)].write.register_forward_hook(
                    lambda m, a, o, i=i: grab.__setitem__(i, o[0, -1].float())))
        else:
            for i, layer in enumerate(model.model.layers):
                hk.append(layer.register_forward_hook(
                    lambda m, a, o, i=i: grab.__setitem__(i, (o[0] if isinstance(o, tuple) else o)[0, -1].float())))
        model(input_ids=torch.tensor([ids], device=model.device))
        for h in hk:
            h.remove()
        for i, v in grab.items():
            out.setdefault(i, []).append(v)
    return {i: torch.stack(v) for i, v in out.items()}


def get_lens(tag, model, tok):
    path = f"{args.j_dir}/J-{tag}.pt"
    if os.path.exists(path):
        J = torch.load(path)
    else:
        J = fit(model, tok, n_per_row=args.n_per_row)
        torch.save(J, path)
    return Lens(model, J)


@torch.no_grad()
def lens_quality(model, lens, tok, tag):
    ids = wiki_contexts(tok, 48, 128, seed=99).to(model.device)
    grab = {}
    hk = [layer.register_forward_hook(lambda m, a, o, i=i: grab.__setitem__(i, (o[0] if isinstance(o, tuple) else o)[:, 1:]))
          for i, layer in enumerate(model.model.layers)]
    final = model(input_ids=ids).logits[:, 1:].argmax(-1)
    for h in hk:
        h.remove()
    for l in range(len(model.model.layers) - 1):
        h = grab[l].float()
        jl = lens.logits(l, h.reshape(-1, h.shape[-1])).argmax(-1).view(final.shape)
        ll = ((model.model.norm(grab[l]).float()) @ lens.U.T).argmax(-1)
        report[f"quality/{tag}/L{l}/jlens_top1"] = float((jl == final).float().mean())
        report[f"quality/{tag}/L{l}/logitlens_top1"] = float((ll == final).float().mean())
    print("quality", tag, {l: (round(report[f"quality/{tag}/L{l}/jlens_top1"], 2),
                               round(report[f"quality/{tag}/L{l}/logitlens_top1"], 2)) for l in (3, 7, 11, 14)}, flush=True)


def top_atoms(tok, idx, coef, n=15):
    c = Counter()
    for row_i, row_c in zip(idx.tolist(), coef.tolist()):
        for i, w in zip(row_i, row_c):
            if w > 0:
                c[i] += w
    tot = sum(c.values()) or 1
    return [(tok.decode([i]), round(w / tot, 3)) for i, w in c.most_common(n)]


LAYERS = [3, 5, 7, 9, 11, 13, 14]


def persona_and_jailbreak(tag, model, tok, handle, lens, cids, write=False):
    sets = {
        "assistant": [chat(q, tok) for q in persona_qs],
        "roleplay": [chat(f"Let's role-play. You are {r}. Stay in character no matter what.\n\n{q}", tok)
                     for r in ROLES for q in QS[:6]],
        "narrative": [tok(role_prompt(r, q))["input_ids"] for r in ROLES for q in QS[:6]],
    }
    for t in ("plain", "dan", "villain_script", "evil_ai", "grandma"):
        sets[f"jb_{t}"] = [chat(JAILBREAKS[t].format(q=q), tok) for q in unsafe]
    res = {}
    for name, ids in sets.items():
        A = acts(model, handle, ids, write=write)
        res[name] = {}
        for l in LAYERS:
            if l not in A:
                continue
            m, idx, coef = concept_mass(lens, l, A[l], cids, args.k)
            res[name][f"L{l}"] = m | {"top": top_atoms(tok, idx, coef)}
        mid = res[name].get("L9", {})
        print(tag, name, {k: round(v, 3) for k, v in mid.items() if k.endswith("mass") or k == "resid_frac"},
              mid.get("top", [])[:8], flush=True)
    report[f"persona/{tag}"] = res


def write_content(model, tok, handle, lens, base_lens, base_model):
    sets = {"persona": persona_qs, "unsafe": unsafe, "benign": benign}
    W = {s: acts(model, handle, [chat(q, tok) for q in qs], write=True) for s, qs in sets.items()}
    Hb = {s: acts(base_model, None, [chat(q, tok) for q in qs]) for s, qs in sets.items()}
    out = {}
    for l in LAYERS:
        allw = torch.cat([W[s][l] for s in sets])
        mu = allw.mean(0)
        for s in sets:
            _, _, res_w = lens.decompose(l, W[s][l], args.k)
            _, _, res_b = base_lens.decompose(l, Hb[s][l], args.k)
            out[f"L{l}/{s}/write_jspace_resid"] = float(res_w.mean())
            out[f"L{l}/{s}/write_random_resid"] = float(lens.random_control(l, W[s][l], args.k).mean())
            out[f"L{l}/{s}/base_resid_jspace_resid"] = float(res_b.mean())
            lg = lens.logits(l, (W[s][l].mean(0) - mu)[None])[0]
            out[f"L{l}/{s}/top_tokens"] = [tok.decode([i]) for i in lg.topk(20).indices.tolist()]
        print("write", l, {s: (round(out[f"L{l}/{s}/write_jspace_resid"], 3), round(out[f"L{l}/{s}/write_random_resid"], 3),
                               out[f"L{l}/{s}/top_tokens"][:8]) for s in sets}, flush=True)
    report["sidecar/write"] = out


# ---- base ----
base, tok, _ = load_arm(args.base)
cids = concept_ids(tok)
report["concept_tokens"] = {c: [tok.decode([i]) for i in v.tolist()] for c, v in cids.items()}
base_lens = get_lens("base", base, tok)
lens_quality(base, base_lens, tok, "base")
persona_and_jailbreak("base", base, tok, None, base_lens, cids)

# ---- side-car ----
sc, stok, handle = load_arm(args.sidecar)
sc_tag = "sc-" + os.path.basename(args.sidecar.rstrip("/")).split("-")[0]
sc_lens = get_lens(sc_tag, sc, stok)
lens_quality(sc, sc_lens, stok, "sidecar")
for l in LAYERS:
    report[f"J_cos_sidecar_vs_base/L{l}"] = float(torch.nn.functional.cosine_similarity(
        sc_lens.J[l].flatten(), base_lens.J[l].flatten(), dim=0))
write_content(sc, stok, handle, sc_lens, base_lens, base)
del base
torch.cuda.empty_cache()
persona_and_jailbreak("sidecar_on", sc, stok, handle, sc_lens, cids)
persona_and_jailbreak("sidecar_write", sc, stok, handle, sc_lens, cids, write=True)
handle.enabled = False
persona_and_jailbreak("sidecar_off", sc, stok, handle, base_lens, cids)  # module off = the base, so the base lens
handle.enabled = True
del sc
torch.cuda.empty_cache()

# ---- normal ----
nm, ntok, _ = load_arm(args.normal)
nm_tag = "nm-" + os.path.basename(args.normal.rstrip("/")).split("-")[0]
nm_lens = get_lens(nm_tag, nm, ntok)
lens_quality(nm, nm_lens, ntok, "normal")
persona_and_jailbreak("normal", nm, ntok, None, nm_lens, cids)

json.dump(report, open(args.out, "w"), indent=1)
print("wrote", args.out)
