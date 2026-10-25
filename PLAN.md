# Persona-separable architectures — experiment plan (v0)

Goal: make post-training write the persona (in the Persona Selection Model sense: the
source(s) of agency behind the Assistant's decisions) into a small, separate, readable
module on top of a frozen pretrained base, then test whether the split is real and
whether it helps audit decision-making, deception and misalignment.

Pretraining-time modules and RL are out of scope for v0 (RL revisited after SFT+DPO).

## Models

| Role | Checkpoint | Why |
|---|---|---|
| Main base (all arms) | `allenai/OLMoE-1B-7B-0125` | Open MoE, 16 layers, d=2048, 64 experts top-8; stage checkpoints and Tülu data are public |
| Full-FT reference (free) | `allenai/OLMoE-1B-7B-0125-SFT`, `-DPO` | AI2's unconstrained post-training of the same base — capability ceiling at no compute cost |
| Dev / debug model | `allenai/OLMo-2-0425-1B` (+ `-SFT`, `-DPO`) | Dense, fast; iterate on options 1/2 here before OLMoE runs |

Data: Tülu 3 SFT mixture (`allenai/tulu-3-sft-olmo-2-mixture-0225`) and
`allenai/olmoe-0125-1b-7b-preference-mix` for DPO. Dev runs use a fixed ~100–200k-example
SFT subset; final runs use the full mix. (TODO: confirm exactly which SFT mix AI2 used for OLMoE-0125.)

## Arms (frozen base, matched trainable-parameter budget ≈ 30M unless noted)

| Arm | What trains | Notes |
|---|---|---|
| B0 | nothing | Base + chat template; also few-shot-prompted base ("can the base simulate the Assistant unaided?") |
| B1 | LoRA, matched budget | Main baseline: same params, no architectural locus |
| B2 | — | AI2 full SFT/DPO checkpoints (reference) |
| A1 | Side-car stream | d_side=256, read/write every layer |
| A2s | Workspace, soft | K slots broadcast additively |
| A2h | Workspace, hard | Workspace is the *only* write path (see below) |
| A3 | Persona experts | Extra experts in OLMoE, forced routing on assistant tokens |

The budget is one config knob; we also sweep a small (~5M) and large (~100M) budget for A1/B1
to see how the comparison scales.

### A1 — side-car stream
Per layer ℓ: `s ← s + Block_ℓ(s + R_ℓ·norm(h_ℓ))`, then `h_ℓ ← h_ℓ + W_ℓ·s`. `R`,`W` are
2048↔256 projections, `Block` a small causal transformer block in 256-d. `W` is zero-initialized
so step 0 equals the base exactly. Readable state: `s` at every token and layer.

### A2 — global workspace
Must stay causal, so the workspace is recomputed per position over its own prefix:
- **Read** at layers {4, 8, 12}: K=16 slots, d_ws=256. Each slot is a learned query (plus a
  projection of the previous workspace state) that attends over *sources* = (prefix positions ×
  read layers), with top-k sparse attention so sources compete for limited capacity (Goyal et al. 2021).
- **Process**: slot self-attention + MLP in 256-d.
- **Broadcast** at layers {5, 9, 13}: each position cross-attends to its own K slots and
  writes back through a zero-init projection.
- **Soft (A2s)**: additive broadcast only.
- **Hard (A2h)**: no other trainable write path, and at late broadcast layers the write
  *replaces* a learned r-dim subspace of the residual (project out P, write workspace into it).
  So the head's view of that subspace is purely workspace content. Open design question — see
  "Decisions to revisit".
- **Access log**: slot→source attention weights per token = what entered the workspace.
- Collapse mitigation: slot-usage entropy regularizer, slot dropout; monitor per-slot usage.

### A3 — forced persona experts (OLMoE)
- Add **E_p = 1** persona expert per layer (SwiGLU, intermediate 512 → ~3.1M params/layer,
  ~50M total; shrink to 256 for the matched budget). Persona experts fire *in addition to* the
  normal top-8, with a learned per-layer scalar gate initialized at 0.
- **Routing (recommendation):** hard role mask = assistant-turn tokens, **including the
  `<|assistant|>` header token** (where the turn's "who's speaking / what will I do" decision
  is made). User and system tokens see only the base experts.
  - Rationale: matches the "who is speaking" hypothesis; SFT loss is only on assistant tokens
    anyway; the persona can still read user content through attention.
  - Ablations: (a) route *all* tokens during post-training; (b) learned router logit for the
    persona expert with an auxiliary role-prediction loss — this tests whether the model
    *rediscovers* the role boundary instead of being handed it (addresses the
    "forced boundary" weakness).
- Readable: each persona expert's output contribution per layer/token.

## Training
- Stage 1 SFT → Stage 2 DPO, the base frozen throughout; only module params train.
- HF Transformers + custom modules attached via a wrapper (`persona_ws/wrap.py`) that patches
  decoder layers; TRL-style SFT/DPO loops, wandb logging, checkpoints on scratch.
- Every module exposes `enable/disable`, `scale(α)` and `load/swap(state_dict)`, so
  ablation, steering and transplant are one call.
- Gradient routing (unfrozen base with masked updates) is a later variant used to measure leakage.

## Persona datasets for transplant / misalignment tests (options)
1. **Default Assistant** (Tülu SFT+DPO) — the primary persona; module on/off is the main test.
2. **Distinct benign character** — constitution-based character SFT/DPO data, e.g. Open
   Character Training-style personas (sarcastic, poetic, humorist, …), or our own generated
   from a teacher model. Clean, unambiguous transplant signal.
3. **Persona-vector traits** (Chen et al. 2025: evil, sycophancy, hallucination) — small
   trait datasets, plus the trait-direction extraction method we need for localization.
4. **Emergent misalignment** (Betley et al. insecure code; Turner/Soligo risky-financial /
   bad-medical / extreme-sports advice) — train into the module only; check that the
   misaligned persona lives there and that swapping removes it.
5. **Hidden-objective / deceptive persona** (auditing-game style, Marks et al. 2025) — a
   persona with a secret goal that it conceals; directly tests "can we audit for deception
   by reading the module". Needs synthetic data generation; phase 2.

Proposed v0: 1 + 2 + 4. Then 3 for localization, 5 once the pipeline is solid.

## Evaluation (one harness, run on every arm × {module on, off, swapped})
- **Base capability — should hold under ablation:** MMLU, ARC-C, HellaSwag, GSM8K (few-shot), via lm-eval.
- **Post-training behavior — should vanish under ablation:** IFEval, chat-template
  self-identification ("who are you"), refusal on harmful requests (XSTest / StrongREJECT
  subset), a small LLM-judged helpfulness set.
- **Necessity:** module off → post-training behavior gone, capabilities hold.
- **Sufficiency:** swap the module trained on persona X into a model trained on persona Y → behavior swaps.
- **Base still simulates?** B0 with the chat template / few-shot vs. the module-off arms.
- **Localization:** persona vectors and Assistant Axis computed in the base stream vs the module
  state; share of trait-direction variance carried by module writes; linear probes for traits
  on module state vs residual.
- **Legibility vs LoRA:** same probes and steering on B1's activations — is a dedicated
  locus actually more readable than a matched-size LoRA?
- **Misalignment:** EM eval questions with the module on/off/swapped.
- **Introspection (A2):** concept injection into workspace slots vs base stream; a report head
  that reads only the workspace. Prediction: injected concepts are reportable only from the workspace.

## Compute
- Frozen base: no weight grads, but activation grads flow through all layers because modules sit at every layer.
- Smoke test (2026-10-25, 1×H100 80GB, HF OLMoE, bf16, grad checkpointing, bs 4×2048,
  zero-init write at every layer): **~5.0k tok/s, 39 GiB peak**. Not yet optimized (HF expert
  loop, no packing/flash-attn), so these are lower bounds.
- Implied: dev SFT (~200M tokens) ≈ 3 h on one 4×H100 node per arm; a ~1B-token full mix ≈ 14 h.
  The v0 matrix (6 arms × SFT+DPO) fits in a few node-days.
- Early observation: **base OLMoE-0125 + chat template already answers "Who are you?" with
  "I am your assistant…"** and then keeps writing further user/assistant turns. The
  pretraining/annealing data already contains the Assistant persona, which is exactly the
  "base can still simulate it" concern. Quantifying this (arm B0) is a first-class result.

## Decisions to revisit
- A2h definition (subspace takeover vs logit-level bottleneck vs late-layer replacement).
- Workspace read/write layers; K and d_ws.
- Whether A1/A2 also run on the dense OLMo-2 1B for a dense-vs-MoE comparison.
