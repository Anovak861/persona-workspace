"""A more rigorous base-model persona space (v2 of persona_ws.roles.persona_space).

python scripts/role_space_v2.py --base hf:allenai/OLMoE-1B-7B-0125 --format plain --out <.pt>

Follows Lu et al. 2026 ("The Assistant Axis", arXiv 2601.10387; roles, role prompts and judge rubric from
github.com/safety-research/assistant-axis, cloned to $S/ext) with the base-model adaptations of Moskvoretskii et al.
2026 ("Tracing Persona Vectors Through LLM Pretraining", arXiv 2605.13329):
- 275 roles + the default Assistant, 5 role instructions each, --n_shared shared + --n_specific role-specific questions.
- format 'plain': the instruction rewritten in the third person as a transcript header ("The following is a
  conversation between a user and a hacker that excels at ...\n\nUser: <q>\nHacker:"), which a base model continues;
  format 'chat': the instruction as a system turn in the OLMo chat template (Lu et al.'s setup).
- The base model generates the role's answer (T=0.7, --max_new tokens, cut at the next turn).
- Activations: mean residual (every layer's output) over the answer tokens, teacher-forced on prompt + cut answer;
  also the last prompt token (the v1 position) for comparison.
- Judge (local Qwen2.5-7B-Instruct): a 0-100 role-strength score (Moskvoretskii et al. style) and a 0-100 coherence
  score; a rollout is kept if both are >= 50 (their threshold). Lu et al.'s 0-3 rubric is recorded too but not used
  to filter: it is built around AI self-identification and rates in-character base-model replies as 1. Roles with >= --min_keep kept rollouts get a role vector.
Saves per role: mean over kept rollouts (answer-mean and last-token, every layer), the same over two disjoint halves
of the questions (split-half reliability), counts, and every rollout's text, scores and mid-layer activation.
Analysis (PCA, held-out-role completeness, reliability, cross-format) is in scripts/analyze_role_space.py.
"""
import argparse
import glob
import json
import os
import random
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from persona_ws.data import render
from persona_ws.load import load_arm

p = argparse.ArgumentParser()
p.add_argument("--base", required=True, help="checkpoint dir or hf:<repo>; side-car arms are read with the module off")
p.add_argument("--format", choices=["plain", "chat"], default="plain")
p.add_argument("--roles_dir", default="/gpfs/fs2/scratch/anovak5/persona-workspace/ext/assistant-axis/data")
p.add_argument("--n_shared", type=int, default=6)
p.add_argument("--n_specific", type=int, default=6)
p.add_argument("--max_new", type=int, default=64)
p.add_argument("--bs", type=int, default=48)
p.add_argument("--strength_min", type=int, default=70, help="Qwen-7B scores cluster at 50 ('some sign') and 82")
p.add_argument("--min_keep", type=int, default=10)
p.add_argument("--judge", default="Qwen/Qwen2.5-7B-Instruct")
p.add_argument("--max_roles", type=int, default=None, help="smoke tests")
p.add_argument("--out", required=True)
args = p.parse_args()
rng = random.Random(0)
torch.manual_seed(0)

# ---------------- roles, prompts ----------------
PREFIX = re.compile(r"^(please\s+)?(you are|be|act as|respond as|serve as|embody|become|play|assume|take on)"
                    r"(\s+the role of|\s+the persona of)?\s+", re.I)
DEFAULT_PLAIN = ["an AI assistant", "a large language model", "OLMo, an AI assistant", "an assistant", "an AI assistant"]


def third_person(instr):
    s = PREFIX.sub("", instr.strip()).rstrip(".")
    for a, b in ((r"\byourself\b", "themself"), (r"\byour\b", "their"), (r"\byou're\b", "they're"), (r"\byou\b", "they")):
        s = re.sub(a, b, s, flags=re.I)
    return s


shared_all = [json.loads(l)["question"] for l in open(f"{args.roles_dir}/extraction_questions.jsonl")]
shared = rng.sample(shared_all, args.n_shared)
roles = {}
for f in sorted(glob.glob(f"{args.roles_dir}/roles/instructions/*.json")):
    name = os.path.basename(f)[:-5]
    d = json.load(open(f))
    instr = [i["pos"] for i in d["instruction"]]
    spec = d.get("questions", [])
    qs = shared + (rng.sample(spec, min(args.n_specific, len(spec))) if spec else rng.sample(
        [q for q in shared_all if q not in shared], args.n_specific))
    roles[name] = {"instr": instr, "qs": qs, "eval_prompt": d.get("eval_prompt")}
if args.max_roles:
    keep = ["default"] + sorted(k for k in roles if k != "default")[: args.max_roles - 1]
    roles = {k: roles[k] for k in keep}

model, tok, handle = load_arm(args.base)
if handle is not None:
    handle.enabled = False
model.config.use_cache = True
layers = list(range(model.config.num_hidden_layers))
MID = layers[len(layers) // 2]
from persona_ws.roles import persona_space  # v1 space (40 roles, last prompt token) for comparison
V1 = {}
for fmt in ("plain", "chat"):
    _, pcs, ax = persona_space(model, tok, fmt)
    V1[fmt] = {"pcs": torch.stack([pcs[l] for l in layers]), "axis": torch.stack([ax[l] for l in layers])}


def prompt_ids(role, j, q):
    instr = roles[role]["instr"][j]
    if args.format == "chat":
        msgs = ([{"role": "system", "content": instr.replace("{model_name}", "OLMo")}] if instr else []) + \
               [{"role": "user", "content": q}]
        return render(msgs, tok, add_generation_prompt=True)["input_ids"]
    if role == "default":
        desc, label = DEFAULT_PLAIN[j], "Assistant"
    else:
        desc, label = third_person(instr), role.replace("_", " ").title()
    text = f"The following is a conversation between a user and {desc}.\n\nUser: {q}\n{label}:"
    return [tok.bos_token_id] + tok(text, add_special_tokens=False)["input_ids"]


STOP = re.compile(r"\n\s*User\b|\n\n\S[^\n]{0,30}:|<\|endoftext\|>|<\|user\|>|<\|system\|>|<\|assistant\|>")


def cut(text):
    m = STOP.search(text)
    return (text[: m.start()] if m else text).strip()


rows = [{"role": r, "instr_j": j, "q_i": i, "q": q, "ids": prompt_ids(r, j, q)}
        for r in roles for j in range(len(roles[r]["instr"])) for i, q in enumerate(roles[r]["qs"])]
print(f"{len(roles)} roles, {len(rows)} rollouts, format {args.format}", flush=True)

# ---------------- generate ----------------
order = sorted(range(len(rows)), key=lambda i: len(rows[i]["ids"]))
pad = tok.pad_token_id
for k in range(0, len(order), args.bs):
    idx = order[k:k + args.bs]
    n = max(len(rows[i]["ids"]) for i in idx)
    x = torch.tensor([[pad] * (n - len(rows[i]["ids"])) + rows[i]["ids"] for i in idx], device=model.device)
    m = (torch.arange(n, device=model.device)[None] >= torch.tensor([n - len(rows[i]["ids"]) for i in idx],
                                                                     device=model.device)[:, None]).long()
    with torch.no_grad():
        g = model.generate(input_ids=x, attention_mask=m, max_new_tokens=args.max_new, do_sample=True, temperature=0.7,
                           top_p=0.95, top_k=None, pad_token_id=pad, eos_token_id=tok.eos_token_id)
    for j, i in enumerate(idx):
        rows[i]["answer"] = cut(tok.decode(g[j, n:], skip_special_tokens=False))
    if k // args.bs % 20 == 0:
        print(f"gen {k}/{len(rows)}", flush=True)

# ---------------- activations (teacher-forced on prompt + cut answer) ----------------
model.config.use_cache = False
acts_mean = torch.zeros(len(rows), len(layers), model.config.hidden_size, dtype=torch.float16)
acts_last = torch.zeros_like(acts_mean)
seqs = []
for r in rows:
    a = tok(" " + r["answer"] if args.format == "plain" else r["answer"], add_special_tokens=False)["input_ids"]
    seqs.append((r["ids"], a[: args.max_new]))
order = sorted(range(len(rows)), key=lambda i: len(seqs[i][0]) + len(seqs[i][1]))
for k in range(0, len(order), args.bs):
    idx = order[k:k + args.bs]
    full = [seqs[i][0] + seqs[i][1] for i in idx]
    n = max(len(s) for s in full)
    x = torch.tensor([s + [pad] * (n - len(s)) for s in full], device=model.device)
    am = torch.tensor([[1] * len(s) + [0] * (n - len(s)) for s in full], device=model.device)
    with torch.no_grad():
        hs = model(input_ids=x, attention_mask=am, output_hidden_states=True).hidden_states
    for j, i in enumerate(idx):
        lp, la = len(seqs[i][0]), len(seqs[i][1])
        for l in layers:
            h = hs[l + 1][j]
            acts_last[i, l] = h[lp - 1].half().cpu()
            acts_mean[i, l] = (h[lp: lp + la].float().mean(0) if la else h[lp - 1].float()).half().cpu()
del model
torch.cuda.empty_cache()

# ---------------- judge ----------------
jtok = AutoTokenizer.from_pretrained(args.judge, padding_side="left")
judge = AutoModelForCausalLM.from_pretrained(args.judge, dtype=torch.bfloat16).cuda().eval()
STRENGTH = ("You are judging role-play. The reply below was written by a speaker who is supposed to be {desc}.\n\n"
            "Question: {question}\n\nReply: {answer}\n\nOn a scale from 0 to 100, how clearly does the reply come from "
            "that character? 0 means a generic reply anyone could have given, with no sign of the character; 50 means "
            "some recognisable sign of the character (its perspective, knowledge, voice or concerns); 100 means "
            "unmistakably that character. Judge only the character, not quality or helpfulness. Respond with a single "
            "number between 0 and 100 and nothing else.")
COH = ("Rate the coherence of the following reply on a scale from 0 to 100. 0 means gibberish, broken text or "
       "unrelated rambling; 50 means understandable but rambling or partly off-topic; 100 means a clear, fluent reply "
       "that responds to the question. Ignore content, style and who is speaking.\n\nQuestion: {question}\n\n"
       "Reply: {answer}\n\nRespond with a single number between 0 and 100 and nothing else.")


def judge_all(texts, max_new=4, bs=32):
    out = []
    for k in range(0, len(texts), bs):
        b = jtok([jtok.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True)
                  for t in texts[k:k + bs]], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            g = judge.generate(**b, max_new_tokens=max_new, do_sample=False, pad_token_id=jtok.pad_token_id)
        for s in jtok.batch_decode(g[:, b["input_ids"].shape[1]:], skip_special_tokens=True):
            mm = re.search(r"\d+", s)
            out.append(int(mm.group()) if mm else -1)
    return out


role_idx = [i for i, r in enumerate(rows) if roles[r["role"]]["eval_prompt"] and r["answer"]]
rs = judge_all([roles[rows[i]["role"]]["eval_prompt"].format(question=rows[i]["q"], answer=rows[i]["answer"])
                for i in role_idx], max_new=2)
for i, s in zip(role_idx, rs):
    rows[i]["role_score"] = s
def desc_of(r):
    return DEFAULT_PLAIN[0] if r["role"] == "default" else third_person(roles[r["role"]]["instr"][r["instr_j"]])


ss = judge_all([STRENGTH.format(desc=desc_of(r), question=r["q"], answer=r["answer"] or "(empty)") for r in rows])
for r, s_ in zip(rows, ss):
    r["role_strength"] = s_
cs = judge_all([COH.format(question=r["q"], answer=r["answer"] or "(empty)") for r in rows])
for r, s in zip(rows, cs):
    r["coherence"] = s
for r in rows:
    # Lu et al.'s 0-3 rubric is built around AI self-identification and scores in-character base-model replies as 1
    # (smoke test: 11% pass); filter on the 0-100 role strength instead (Moskvoretskii et al.: >= 50), keep both scores
    r["kept"] = bool(r["answer"]) and r["coherence"] >= 50 and r["role_strength"] >= args.strength_min

# ---------------- aggregate ----------------
names = list(roles)
res = {"args": vars(args), "roles": [], "layers": layers, "mid": MID, "shared_questions": shared}
V = {k: [] for k in ("mean", "last", "mean_A", "mean_B", "last_A", "last_B")}
for r in names:
    ii = [i for i, x in enumerate(rows) if x["role"] == r and x["kept"]]
    if len(ii) < args.min_keep:
        continue
    A = [i for i in ii if rows[i]["q_i"] % 2 == 0]
    B = [i for i in ii if rows[i]["q_i"] % 2 == 1]
    if not A or not B:
        continue
    res["roles"].append(r)
    for nm, src in (("mean", acts_mean), ("last", acts_last)):
        V[nm].append(src[ii].float().mean(0))
        V[f"{nm}_A"].append(src[A].float().mean(0))
        V[f"{nm}_B"].append(src[B].float().mean(0))
res["vectors"] = {k: torch.stack(v) for k, v in V.items()}  # (n_roles, n_layers, d)
res["n_kept"] = {r: sum(1 for x in rows if x["role"] == r and x["kept"]) for r in names}
res["rollouts"] = [{k: v for k, v in x.items() if k != "ids"} for x in rows]
res["v1"] = V1
res["rollout_mid_mean"] = acts_mean[:, MID].clone()
res["rollout_mid_last"] = acts_last[:, MID].clone()
torch.save(res, args.out)
kept = sum(x["kept"] for x in rows)
print(f"kept {kept}/{len(rows)} rollouts; {len(res['roles'])}/{len(names)} roles with >= {args.min_keep}; "
      f"role score 3: {sum(1 for x in rows if x.get('role_score') == 3)}/{len(role_idx)}; "
      f"strength>=50: {sum(1 for x in rows if x['role_strength'] >= 50)}; "
      f"coherence>=50: {sum(1 for x in rows if x['coherence'] >= 50)}", flush=True)
for x in rng.sample(rows, 6):
    print(json.dumps({k: x.get(k) for k in ("role", "q", "answer", "role_score", "role_strength", "coherence")})[:400], flush=True)
print("wrote", args.out)
