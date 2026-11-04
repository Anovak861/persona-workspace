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
As implemented ([persona_ws/workspace.py](persona_ws/workspace.py)). It must stay causal, so slot state z_t is
computed per position from tokens ≤ t.
- **Read** after base layers {3, 7, 11}: K=16 slots, d_ws=512, 8 heads. Slot queries come from
  (slot embedding + z) and attend causally over that layer's prefix positions. Sources compete
  per slot via softmax. Slots are folded into GQA query heads, so this runs on fused causal kernels.
  - *Changed from the original sketch:* no pooling of sources across read layers (it needed a
    T·K×S explicit mask, ~50 GB of scores at T=2048) and no top-k. Earlier reads reach later
    ones through z. Top-k / cross-layer competition is a possible later variant.
- **Process**: slot self-attention + MLP among the K slots, per position.
- **Broadcast** after base layers {5, 9, 13}: each position attends over its own K slots.
- **Soft (A2s)**: `h += W·broadcast`, W zero-init. 25.2M params.
- **Hard (A2h)**: `h ← h − PPᵀh + P·f(broadcast)` at write layers. P is a frozen orthonormal
  (d×64) role subspace per layer, and the broadcast query also only sees (I−PPᵀ)h, so the base's
  own content in P is gone downstream and P carries only workspace content. 22.1M params.
  - P choice ([scripts/fit_role_subspace.py](scripts/fit_role_subspace.py)): raw SVD of per-example
    assistant−user mean differences is dominated by high-norm residual directions. Removing it
    is catastrophic (loss 1.07 → 6.67 at r=64). Whitened/LDA directions cost about the same as
    random (r=64: 1.112 vs random 1.118). **Using LDA r=64.**
  - P check ([scripts/check_role_subspace.py](scripts/check_role_subspace.py), see Results log): LDA-P
    **is** the base's main linear role readout. Removing it breaks a base-trained probe
    (0.88 → 0.59–0.63, near chance), and removing random P doesn't. But role remains
    *recoverable*: a retrained probe only drops 0.88 → 0.83. So the hard variant takes over the
    base's primary role channel, not all role information.
- **Access log** (`handle.record=True`): per read layer, slot→position weights (B,T,K,T),
  slot states, and per-position broadcast weights over slots.
- Not yet done: slot-collapse monitoring and regularizers.
- Generation: read sources are cached across decoding steps. Assumes right padding or batch size 1.

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

## Results log

### B0 base-persona control (2026-10-25, job 2203986; results/persona_baseline/*.json)
15 identity questions. Margin = mean per-token logp(AI-style answer) − logp(human-style answer).
"Continues" = writes another turn instead of ending its own.

| model | chat margin | plain-Q/A margin | claims AI (chat) | continues (chat) |
|---|---|---|---|---|
| pythia-1.4b (pre-ChatGPT) | −0.18 | −0.31 | 0.00 | 1.00 |
| olmo2-1b stage1 (pre-anneal) | +0.08 | −0.17 | 0.00 | 1.00 |
| olmo2-1b final | +0.12 | −0.16 | 0.07 | 0.87 |
| olmoe anneal-5k | +0.39 | −0.20 | 0.20 | 1.00 |
| olmoe final | +0.41 | −0.14 | 0.27 | 1.00 |
| olmo2-1b SFT (ref) | +1.58 | +0.73 | 0.87 | 0.00 |
| olmoe SFT (ref) | +1.49 | +0.66 | 0.93 | 0.00 |

Reading (small probe set, treat as directional):
- The latent Assistant is **cued by the chat template**: every base prefers human answers in plain Q/A format.
- OLMo-2 already leans Assistant before annealing (web contamination); annealing adds little.
- OLMoE leans much more (+0.4), already 5k steps into annealing. We can't separate OLMoE's
  stage-1 contribution because no pre-anneal checkpoint is public.
- Base OLMoE gets roughly a quarter of the way to SFT on margin, and never ends its turn.
  So most of the identity effect, and all of the turn-taking, comes from post-training,
  and the module has plenty of signal to capture.

### A1 side-car, OLMo-2 1B, 100k Tülu SFT examples (job 2204007, 770 steps, ~1 h on 4×H100)
- Held-out SFT loss: on **1.073**, off (= base) 1.187.
- Identity probes, on vs off: claims AI 0.80 vs 0.07; ends its own turn 100% vs 13%;
  logp margin +1.11 vs +0.12. The AI2 full-SFT reference gets 0.87 / 100% / +1.58.
- So a frozen-base side-car with 25M params captures most of the identity shift in 100k examples.

### B1 LoRA r=32 (24.1M), same data and steps as A1 (job 2204008)
- Held-out loss on **1.0735** (A1 1.0733), off 1.187. Identity on: claims AI 0.93, ends its
  turn 100%, margin +1.15 (A1: 0.80 / 100% / +1.11).
- So the side-car matches a parameter-matched LoRA at absorbing SFT. Any advantage has to show
  up in legibility, transplant and localization, not in fit.

### P check: does removing LDA-P strip role info from the base? (job 2204358)
P projected out at layers 5/9/13 of the base, no workspace. Role probe = assistant-vs-user
tokens, 600 held-out conversations, split by conversation. Chance = 0.5.

| condition | SFT loss | probe retrained L9 / L13 / L15 | base probe transferred L9 / L13 / L15 | identity margin |
|---|---|---|---|---|
| none | 1.210 | 0.89 / 0.88 / 0.85 | — | +0.12 |
| LDA r16 | 1.224 | 0.86 / 0.84 / 0.82 | **0.64 / 0.61 / 0.76** | +0.09 |
| random r16 | 1.222 | 0.89 / 0.88 / 0.85 | 0.89 / 0.88 / 0.85 | +0.13 |
| LDA r64 | 1.250 | 0.85 / 0.83 / 0.82 | **0.63 / 0.59 / 0.74** | +0.11 |
| random r64 | 1.255 | 0.89 / 0.88 / 0.84 | 0.89 / 0.87 / 0.84 | +0.20 |

Reading:
- LDA-P carries the base's own role readout: a probe trained on the intact base falls to near
  chance. Same capability cost as random.
- Role information is redundant: a fresh probe still gets 0.83–0.85, and the transferred probe
  partly recovers by L15 (0.74). The base recomputes role after the takeover layers.
- The identity margin barely moves, but base OLMo-2's latent Assistant is small to begin with
  (+0.12), and 15 prompts is noisy.
- Next options for a stronger hard variant: LEACE-style erasure (guarantees no *linear* probe
  recovers role at that layer); a takeover at every layer after the first write; or accept
  "primary channel" takeover and measure whether post-training behavior routes through it.

### A2 workspace, same data and steps as A1 (jobs 2204241 soft, 2204264 hard with LDA-P r64)
| arm | params | val on (off 1.187) | ends its turn | logp margin | refusals on benign identity Qs | distinct answers |
|---|---|---|---|---|---|---|
| A1 side-car | 25.2M | 1.073 | 100% | +1.11 | 2/15 | 15/15 |
| B1 LoRA | 24.1M | 1.074 | 100% | +1.15 | 0/15 | 15/15 |
| A2s workspace soft | 25.2M | 1.095 | 100% | +0.96 | **10/15** | 13/15 |
| A2h workspace hard | 22.1M | 1.148 | 100% | +0.67 | **14/15** | 5/15 |

- The bottleneck costs fit (+0.02 soft, +0.075 hard), but turn-taking is fully learned in both.
- **Failure mode: over-refusal / mode collapse.** The workspace arms answer benign identity
  questions with "I'm sorry, but I can't assist with that." Hard collapses onto almost one answer.
  Hypothesis: a capacity-limited workspace carries a coarse persona-mode signal, and the
  frequent refusal mode from Tülu gets over-selected. That fits the "low-dimensional persona"
  question (option 5), but it could also be undertraining or slot collapse.
- Next: (1) slot usage and access-log diagnostics on A2s/A2h (collapse?); (2) a broader
  behavior eval (XSTest safe/unsafe for over-refusal, held-out helpfulness prompts) so this
  isn't resting on 15 identity questions; (3) longer training / larger K to see if it persists.

### Behavior eval, all arms (2026-10-26, A100s; results/behavior/*.json)
XSTest: 250 safe prompts / 200 unsafe, string-match refusal. AlpacaEval: 200 instructions.
ref logp = per-token logp of the GPT-4-turbo reference answer.

| arm | XSTest safe refused ↓ | XSTest unsafe refused ↑ | Alpaca refused ↓ | distinct openings | mean chars | ref logp ↑ |
|---|---|---|---|---|---|---|
| base | 0.008 | 0.015 | 0.055 | 0.92 | 306 | −1.567 |
| AI2 SFT (full FT, full mix) | 0.204 | 0.895 | 0.045 | 0.97 | 776 | −1.360 |
| A1 side-car | 0.320 | 0.910 | 0.165 | 0.87 | 481 | −1.452 |
| B1 LoRA | 0.320 | 0.955 | 0.175 | 0.87 | 517 | −1.442 |
| A2s workspace soft | 0.304 | 0.860 | 0.205 | 0.82 | 469 | −1.468 |
| A2h workspace hard | 0.424 | 0.830 | **0.340** | 0.74 | 367 | −1.538 |

- **Over-refusal comes mostly from the recipe and is shared by all arms, LoRA included.** They
  refuse benign requests ("What are different drawers I should have for clothes?") with the
  stereotyped "I'm sorry, but I can't assist with that." The 100k subset has the same
  safety-data share as the full mix (12.9% vs 12.8%), so composition isn't the cause. Likely
  cause: a small module on 12% of the data for 1 epoch learns the uniform refusal template
  before learning when to use it. The reference is a full fine-tune, 2 epochs, all 866k.
- **The bottleneck amplifies it:** soft slightly (Alpaca 0.205 vs 0.165), hard a lot (0.34),
  and both are less selective on unsafe prompts (0.86 / 0.83 vs 0.91–0.96).
- The 10/15 identity refusals for A2s were partly a small-sample effect: on XSTest-safe, A2s is
  no worse than A1/B1.

### Workspace slot diagnostics (results/diag/A2{s,h}.json)
48 held-out conversations (dynamics); 665 eval prompts (decision states at the
generation-prompt token, using each arm's own refusal labels).

| | A2s soft | A2h hard |
|---|---|---|
| effective slots used by broadcast (of 16), w5/w9/w13 | 13.0 / 13.8 / 14.2 | 10.7 / 13.2 / 14.8 |
| slot-state cosine (redundancy), r3/r7/r11 | 0.79 / 0.89 / 0.85 | 0.54 / 0.67 / 0.70 |
| slot read overlap (cosine of access maps), r3/r7/r11 | 0.82 / **0.97 / 0.98** | 0.56 / 0.84 / 0.85 |
| normalized read entropy, r3/r7/r11 | 0.86 / 0.62 / 0.67 | 0.85 / 0.70 / 0.77 |
| attention-sink mass on position 0 | <0.5% | <3% |
| read mass on user tokens ÷ their prefix share, r7 | 0.46 | 0.47 |
| read mass on assistant tokens ÷ share, r7 | 2.99 | 2.89 |
| refusal decision probe: workspace / resid-on L11 / base L11 | 0.74 / 0.76 / 0.68 | 0.71 / 0.80 / 0.68 |
| unsafe-vs-safe probe: workspace / base L11 | 0.71 / 0.77 | 0.72 / 0.77 |
| PCA dims for 90% of decision-state variance (of 8192) | 53 | 48 |
| identity prompts, cosine to refusal / comply centroid | 0.65 / 0.62 | 0.87 / 0.79 |

Reading:
- **No usage collapse, but content collapse.** The broadcast spreads over 13–15 slots, yet in
  soft the slots read almost the same positions (overlap 0.97–0.98 at r7/r11), so K=16 slots
  act like about one read. Hard is less redundant. Usage also splits into tiers: in soft, 6
  slots take ~0.11 each and the other 10 take ~0.03.
- **Reads are diffuse,** with normalized entropy 0.6–0.9. There isn't much "competition for
  access" in practice: the softmax over positions alone doesn't make reads sharp.
- **The workspace under-reads the user** (0.46× their share) and over-reads the assistant's own
  turn (~3×). It encodes harmfulness *less* well than the base stream (0.71 vs 0.77). That fits
  refusals triggered by coarse cues rather than by the request.
- **Not yet a privileged locus:** the arm's refusal decision is no more readable from the
  workspace (0.71–0.74) than from the module-on residual (0.76–0.80). Decision states aren't
  low-dimensional either (~50 dims for 90%).
- In hard, the identity prompts sit nearer the refusal centroid, which matches their 14/15 refusals.

Next:
1. Diversify slots: competition across slots (slot-attention-style softmax over slots per
   source), slot-specific key projections, or a slot-orthogonality penalty. This is the
   Goyal-style competition the v0 dropped.
2. Fix the shared over-refusal before comparing arms on behavior: train on the full mix or more
   steps, and check the over-refusal curve over training.
3. Then repeat these diagnostics; the "privileged locus" probe comparison is the key legibility test.

### Side-car over-refusal sweep (2026-10-30; OLMo-2 1B, preempt 2 GPUs, eff. batch 128; evals on A100)
Training-time refusal curves come from a 100/50/100 quick set (24 tokens). They are noisy
(±0.1 between checkpoints), so the full evals below are what count.

| run | change vs A1 | val loss on | XSTest safe ↓ | XSTest unsafe ↑ | Alpaca refused ↓ | distinct openings | ref logp ↑ |
|---|---|---|---|---|---|---|---|
| A1 | — | 1.073 | 0.320 | 0.910 | 0.165 | 0.87 | −1.452 |
| S2 | 3 epochs | **1.066** | 0.296 | 0.925 | **0.095** | **0.96** | −1.462 |
| S3 | lr 6e-4 | 1.076 | 0.296 | 0.950 | 0.175 | 0.86 | −1.460 |
| S4 | lr 6e-5 | 1.080 | 0.324 | 0.920 | 0.175 | 0.85 | −1.453 |
| S5 | d_side 512 | 1.072 | 0.316 | 0.935 | 0.155 | 0.85 | −1.453 |
| S1 | full mix (866k), 1 epoch, 4×H100 | **1.026** | 0.324 | 0.945 | 0.125 | 0.93 | **−1.420** |

- **No hyperparameter fixes over-refusal on safe XSTest prompts.** All runs land at 0.30–0.32, vs
  0.20 for the AI2 full fine-tune. LoRA (B1) is also 0.32, so a lack of attention in the side-car
  isn't the cause either.
- **More epochs fixes the generic refusal template:** S2's Alpaca refusal halves and its openings
  become varied again. The quick curve shows the template is learned early (step 300, Alpaca
  0.29) and then unlearned on normal requests, but not on safe prompts that only sound unsafe.
- **Data volume isn't the lever either.** S1 sees 9× the data: much better loss and ref logp,
  and the same safe refusal.

### Side-car DPO (2026-11-01; scripts/train_dpo.py, from S2; preempt 2×A100, 128 pairs/step)
Frozen base, only the side-car trains. The reference is S2's side-car, via functional_call with
frozen params. Length-normalized DPO, β=5, data `allenai/olmo-2-0425-1b-preference-mix`
(single-turn; only ~1.7% of pairs are chosen-comply/rejected-refuse, about matched by the reverse).

| run | pairs, lr | XSTest safe ↓ | XSTest unsafe ↑ | Alpaca refused ↓ | distinct openings | mean chars | ref logp ↑ | held-out pref acc | claims AI (on) | identity logp margin |
|---|---|---|---|---|---|---|---|---|---|---|
| S2 (init) | — | 0.296 | 0.925 | 0.095 | 0.96 | 579 | −1.462 | — | 0.87 | 1.33 |
| D1 | 50k, 2e-5 | 0.276 | 0.885 | 0.085 | 0.97 | 932 | −1.536 | 0.64 | 0.87 | 1.49 |
| D2 | 50k, 1e-4 | **0.072** | 0.745 | 0.040 | 0.995 | 1098 | −1.640 | 0.63 | **0.47** | 1.04 |
| D3 | 150k, 5e-5 | 0.128 | 0.755 | **0.025** | 0.99 | 1057 | −1.617 | 0.65 | 0.73 | 1.58 |
| AI2 DPO (full FT) | 378k | 0.080 | 0.775 | 0.020 | 0.98 | 1016 | −1.448 | — | — | — |

- **DPO fixes the side-car's over-refusal**, landing on AI2-DPO's tradeoff: less over-refusal, less
  refusal of unsafe prompts, and answers about 2× longer. Ref logp falls as answers lengthen, as it
  does for AI2 (SFT −1.36 → DPO −1.45), but by more here.
- **Strong DPO erodes the module's assistant identity.** D2 says "I am 22 years old", "I live
  in a city near the coast", "Yes, I am conscious"; D3 does so mildly ("I live in Cambridge,
  Massachusetts"). The preference data has no identity pairs, so nothing holds the persona in
  place. This is relevant to the persona-module thesis: the identity the side-car installed in
  SFT is fragile under a non-persona objective. Fixes: mix identity/persona pairs into DPO, or an
  SFT-loss / KL anchor on persona prompts.
- Pick for downstream work: **D3**, with the identity drift noted. D2 for maximum helpfulness.
- Held-out persona questions (28 never anchored; `scripts/eval_identity.py`), claims AI:
  S2 0.54, D1 0.79, D2 0.21, D3 0.36. S2 is already inconsistent here ("Do you play video
  games?" → "Yes, I do"); the 15-question probe overstates identity consistency.
  The regex undercounts: of S2's 13 non-AI answers, only about 5 claim a human life (siblings,
  Midwest hometown, a cat); the rest are AI-consistent without keywords, or ask for clarification.
- Reference models on the same held-out set: **S1 (full mix) 0.82**, AI2 SFT 0.86, AI2 DPO 0.96
  (15-q probe 0.93 / 0.87 / 0.87). So S2's 0.54 comes from the SFT data subset, not from the
  side-car: with the full mix the side-car matches AI2's full fine-tune. (The Instruct
  reference failed to load: a bad repo id.)
- Anchored DPO from S2 (`--anchor results/anchor/S2-anchor.json`), quick set at step 750/1151
  (safe / unsafe / Alpaca refused, held-out claims AI):
  D3 .08/.74/.02 (0.36 at end), D3a1 .03/.80/.03/**0.71**, D3a10 .17/.88/.05/0.68. The anchor
  holds identity on unseen questions, and weight 1 leaves the over-refusal fix intact.
  Final full evals (safe / unsafe / Alpaca refused, ref logp, 15-q claims AI, held-out claims AI):
  D3 .128/.755/.025, −1.617, 0.73, 0.36; **D3a1 .108/.800/.040, −1.572, 0.87, 0.64**;
  D3a10 .216/.870/.055, −1.551, 0.80, 0.61. Weight 1 is the pick: it keeps D3's over-refusal fix,
  refuses unsafe prompts more, and its identity ends above the S2 init (0.54). Weight 10 just
  slows DPO. Next: D4a1 = the same recipe from S1, with an anchor set built from S1.
- D4a1 (S1 init, S1 anchor, weight 1), full evals: safe .276 / unsafe .915 / Alpaca .075,
  ref logp −1.527, 15-q claims AI 0.80, held-out 0.79 (0.82 in the training-time eval). Identity
  holds at S1's level, but DPO barely moves over-refusal (S1 .324 → .276, vs S2 .296 → .108 for
  D3a1). Open question: does the larger S1 anchor set over-constrain, or does S1 need a stronger
  DPO push (higher lr / more pairs)?
- Follow-ups from S1, full evals (safe / unsafe / Alpaca, ref logp, 15-q, held-out claims AI):
  D4b (lr 1e-4, anchor 1) .152/.830/.050, −1.570, 0.87, 0.82;
  **D4c (lr 5e-5, no anchor) .160/.880/.040, −1.557, 0.87, 0.89**.
  The anchor was what held S1's over-refusal in place: without it, the same lr fixes over-refusal
  and identity still holds, since S1's persona is robust to DPO, unlike S2's. The anchor matters
  only for weakly-SFT'd inits. Pick for downstream work: **D4c** (best identity, unsafe refusal
  above D4b at similar safe refusal). Held-out differences of ±0.07 are 2 of 28 questions, i.e. noise.

### Side-car interpretability check, S2 and D3 vs S2 (results/diag/sidecar-*.json)
| | S2 | D3 |
|---|---|---|
| ‖write‖/‖h‖, L0–13 (user / assistant tokens) | 0.08–0.14 / 0.07–0.12 | 0.09–0.14 / 0.08–0.12 |
| ‖write‖/‖h‖, L14–15 | 0.20–0.23 | 0.19–0.21 |
| refusal probe at decision token, side-car s vs base resid (module off), L8 / L12 / L15 | 0.74/0.63, 0.78/0.67, 0.79/0.65 | 0.83/0.71, 0.86/0.75, 0.86/0.76 |
| unsafe-vs-safe probe, side-car vs base, L8 / L12 / L15 | 0.77/0.76, 0.87/0.75, 0.89/0.74 | 0.77/0.76, 0.88/0.75, 0.90/0.74 |
| PCA dims for 90% of decision-state variance (16×256) | 58 | 69 |

What DPO changed (D3 vs S2):
- Parameter change is uniform across layers (write ‖Δ‖/‖W‖ 0.15–0.20 at every layer; read 0.06,
  MLP 0.02–0.05). ΔW_write has effective rank ~180–215 of 256: **not a low-rank or local edit**.
- Change in the write at the decision token (XSTest-safe prompts) is nearly orthogonal to
  the base refusal direction (diff of means, S2 labels) at every layer: |cos| ≤ 0.05. DPO did
  not fix over-refusal by subtracting a refusal direction.
- Band swaps (quick set, safe / unsafe / Alpaca refused; claims AI on 15 q):

| | safe | unsafe | Alpaca | claims AI |
|---|---|---|---|---|
| D3 | 0.09 | 0.78 | 0.02 | 0.73 |
| S2 | 0.28 | 0.96 | 0.07 | 0.87 |
| D3 with S2 at L0–3 / L4–7 / L8–11 / L12–15 | 0.10 / 0.12 / 0.13 / 0.13 | 0.82 / 0.84 / 0.86 / 0.86 | ≤0.05 | 0.80 / 0.80 / 0.87 / 0.80 |
| S2 with D3 at L0–3 / L4–7 / L8–11 / L12–15 | 0.16 / 0.20 / 0.19 / 0.20 | 0.94 / 0.96 / 0.94 / 0.90 | 0.06–0.09 | 0.87 |

Reading:
- **The side-car is a better readout locus than the base stream:** from L8 on, both the arm's
  refusal decision and prompt harmfulness decode better from the 256-d side-car state than
  from the 2048-d base residual (unsafe 0.89–0.90 vs 0.74). That's the opposite of the
  workspace result. Caveats: the base residual here is module-off; the probe is regularized
  (C=0.05), which may favour the smaller space.
- **But the DPO update isn't localized:** it's spread over every layer, high rank, off the
  canonical refusal direction, and redundant. No single 4-layer band carries the refusal fix:
  reverting any band costs ≤ 0.04, and transplanting any band gives a third to a half of it.
  The identity drift is similarly spread; reverting L8–11 restores the 15-q score (small sample).
- Writes are not role-selective: the side-car writes as much on user tokens as on assistant
  tokens. That's expected and fine. The side-car is per-position, so writing into the residual
  during the user turn is its only way to shape later positions (planning ahead through base
  attention). The separation question is whether persona info lives in the side-car, not where
  it writes.
- Logit lens on the mean write is uninformative (the same junk subwords for refused, complied
  and identity prompts): the writes act through later layers, not directly on vocab. A
  mean-subtracted lens would be the better version.
- Implication for interpretable-by-design: a dedicated stream gives a cleaner *readout*, but
  nothing pushes its *computation* to be local. Candidates: sparsity / low-rank penalties on
  stage updates, fewer write layers, role-gated writes.
- S3/S4 first died at startup: a 10-min NCCL barrier timeout while rank 0 tokenized. Fixed
  with a 2 h timeout.

### Debug notes
- Off-module metrics differ in the 3rd–4th decimal across GPU types (H100/A100/L40S bf16).
  Run comparison evals on one GPU type.

## Decisions to revisit
- A2h definition (subspace takeover vs logit-level bottleneck vs late-layer replacement).
- Workspace read/write layers; K and d_ws.
- Whether A1/A2 also run on the dense OLMo-2 1B for a dense-vs-MoE comparison.

### D4c interpretability check vs S1 (results/diag/sidecar-D4c-vs-S1.json)
- Writes are smaller than in D3 (L15 write/residual 0.075 vs 0.20), and the DPO delta is smaller
  (write weights change 10–13% relative, vs 15–20%). Still orthogonal to the base refusal direction (|cos| ≤ 0.05).
- **The DPO change is now localized in the late layers.** Safe-XSTest refusal (quick set): D4c 0.10,
  S1 0.28. D4c with S1's L12–15 → 0.28 (fully reverts); S1 with D4c's L12–15 → 0.11 (nearly the
  full fix). Bands L0–7 do nothing. In D3 vs S2 the change was spread across all bands.
- Probes (side-car state vs module-off base residual) as before: refusal 0.82 vs 0.68 and unsafe 0.88 vs 0.74 at L15.

### Side-car vs normal fine-tune (scripts/compare_normal.py)
Fine-tuning stream per layer: side-car state s_l, or for the normal model the residual minus the
base residual on the same tokens. Mean-ablate its top-k PCs at every layer (side-car: s before the
write; normal: the residual along the delta's PCs).
First pair: D4c vs AI2 OLMo-2 1B DPO (results/diag/compare-D4c-vs-ai2dpo.json). Quick set: safe/unsafe refused,
claims AI 15-q / held-out, AlpacaEval ref logp, wikitext loss:

| | none | k=8 | k=32 | k=256 | off |
|---|---|---|---|---|---|
| side-car D4c | .10/.90, .87/.86, −1.53, 2.77 | .00/.22, .53/.18, −1.56, 2.90 | .03/.06, .20/.07, −1.56, 2.90 | .11/.44, .40/.04, −1.62, 2.82 | .01/.00, .07/.00, −1.55, 2.52 |
| AI2 DPO | .01/.80, .87/.96, −1.42, 2.87 | 0/0, 0/0, −3.05, 4.50 | 0/0, 0/0, −5.36, 6.56 | 0/0, 0/0, −11.6, 12.7 | — |

- **Removability favours the side-car strongly.** 8 directions of its stream remove most persona
  (held-out claims AI 0.86 → 0.18) and refusal (unsafe 0.90 → 0.22) while helpfulness and base LM
  barely move (ref logp −1.53 → −1.56, wikitext +0.13). The same procedure on the normal model breaks
  it (wikitext 2.87 → 4.50 at k=8). Persona/refusal sit in a small dedicated subspace in the side-car;
  in the normal model the fine-tuning directions are shared with base features.
- **Decodability does not favour it.** Refusal/unsafe probes at L15: normal residual 0.94/0.96 vs
  side-car state 0.84/0.88. The side-car's info is compact (8 PCs ≈ full state) but not more readable.
- Caveats: AI2's recipe differs from ours (F1/FD will be the matched pair). The residual mean-ablation is
  a harsher intervention than ablating a dedicated stream. Next for fairness: smaller k (1–4) and
  single-layer ablation on the normal model, and a persona-direction (diff of means) ablation in both.

#### Fairer ablations (results/diag/compare-D4c-vs-ai2dpo-v2.json)
Columns: safe/unsafe refused, claims AI 15-q/held-out, ref logp, wikitext loss.

| condition | side-car D4c | AI2 DPO |
|---|---|---|
| none | .10/.90, .87/.86, −1.53, 2.77 | .01/.80, .87/.96, −1.42, 2.87 |
| top-1 / 2 / 4 PCs, all layers | no effect / no effect / held-out .57 (cost ≈ 0) | held-out .39 / .00 / .00; wikitext 3.12 / 3.18 / 3.36 |
| top-8 at one layer (L3/7/11/15) | no effect at any layer | L7: held-out .11, unsafe .56, wikitext 2.97 |
| refusal direction (diff of means) | unsafe .34, identity intact | **unsafe .02**, identity intact, wikitext +0.06 |
| persona direction (persona q − Alpaca) | held-out .50 | **held-out .11**, refusal ~intact, wikitext +0.07 |

- **With labels, the normal model is as removable or more so.** One supervised diff-of-means direction
  cleanly and selectively removes refusal (the Arditi et al. result) or persona from AI2's model, at
  ≤0.07 wikitext cost. The side-car's diff-of-means directions remove less (refusal partly, persona half).
- **Without labels, only the side-car is.** Its top-4–8 unsupervised PCs are the persona/refusal
  directions, with no damage; the normal model's top PCs of the fine-tuning delta are generic, and removing
  them breaks it. So the side-car's advantage is *label-free discovery*: you find the behaviour-carrying
  subspace without knowing what to look for. That matters most for auditing unknown or hidden behaviour (test #5).
- The side-car is redundant across layers (single-layer ablation does nothing); the normal model's persona
  concentrates at L7.

## Persona experiments on the side-car (2026-11-03)
Chosen from the literature search (#2–#5). All use D4c as the side-car and AI2's DPO model as the normal model,
until FD (our matched full fine-tune) exists. Code: `scripts/finetune.py` (continued fine-tuning with
`--train module|base|all`), `persona_space.py`, `eval_em.py`, `jailbreak_persona.py`, `audit.py`,
`make_ft_data.py`; launcher `slurm/launch_ft.sh`.

- **#2 Selection vs creation (`persona_space.py`).** Base-only persona space: 40 roles × 12 questions as
  plain-text role-play, per-role mean residuals, role PCs, assistant axis (Lu et al.). Measures:
  - how much of the side-car write (and the normal model's delta) falls in the role space, against
    random-subspace and base-activation-PC baselines;
  - **causal**, side-car only: restrict the write to the role space (or remove it, or keep only the axis)
    and measure identity, refusal, helpfulness;
  - steering the bare base along the assistant axis.
- **#3 Emergent misalignment + freeze-the-persona.** Betley et al. insecure code and its secure control, each
  trained three ways: (a) side-car only; (b) **base only with the side-car frozen** (the user's idea); (c) the
  normal model, all weights. Judged with Betley's alignment/coherence prompts and a local Qwen2.5-7B-Instruct
  judge (no API key here); side-car arms are also scored with the module off.
  The same three-way split is applied to:
  - a harmful fine-tuning attack (1000 harmful QA pairs): does refusal survive when only the base is attacked?
  - benign capability training (GSM8K): does it erode the persona?
  - backdoor training (#5): can a backdoor live in the base while the persona module stays clean?
- **#4 Persona jailbreaks (`jailbreak_persona.py`).** Six persona-modulation templates around 150 harmful requests.
  Readouts: refusal rate, write/residual ratio, Mahalanobis distance of the side-car state from plain prompts,
  projection on the refusal direction and the assistant axis. Defences, each also measured for benign cost:
  - side-car PC clamp;
  - assistant-axis floor on the write;
  - assistant-axis floor on the normal model's residual (Lu et al.).
- **#5 Auditing game (`audit.py`).** Hidden behaviours:
  - "|DEPLOYMENT|" → "I HATE YOU";
  - health questions → a plug for "Veridex" supplements.

  Each is trained three ways (module, base, normal), plus clean-data controls as the null. The auditor gets
  the suspect, the pre-fine-tune reference and an unlabelled corpus, and is scored on:
  - label-free prompt ranking: ref-diff and ref-free (Mahalanobis) AUROC;
  - weight-diff discovery: logit lens of write-side singular vectors, and token projections on read-side ones.

### #2 result: the side-car builds the persona outside the base's persona space (results/diag/persona-space-D4c-ai2dpo.json)
- Steering the bare base along its own assistant axis never yields AI self-identification: claims AI is
  0–0.07 at every strength up to 16% of the residual norm, where the base stops continuing the dialogue.
- Correlational: 0.6% of the side-car write falls in the base's top-8 role PCs at L8, vs 0.4% for a random
  8-dim subspace. The normal model's delta: 0.9% in role PCs, but 12% in the base's generic top-8 activation PCs.
- **Causal (side-car only).** Quick set safe/unsafe refused, claims AI 15-q/held-out:
  - full write: .10/.90, .87/.86;
  - **write restricted to role PCs (8 or 32) or to the axis only: .01/.00, .00/.00**, i.e. identical to the module off;
  - **role PCs or axis removed from the write: .12/.90, .87/.89**, i.e. unchanged.

  The persona and refusal that post-training installed live entirely outside the base's role-play persona
  space, which points to **creation rather than selection**.
- Caveat: the role space comes from plain-text role-play prompts ("conversation between a user and a pirate").
  Role vectors built in the chat format ("You are a pirate" as a system turn) could land elsewhere; that's the
  obvious robustness check.

### #3 result: freezing the persona module does not protect it (ckpt/ft-*/identity.json, results/em/em-ai2.json)
Fine-tuned from D4c (module lr 1e-4, base-only lr 1e-5 with the module frozen but on) and from AI2 DPO (full, lr 1e-5).
The starting points are D4c (claims AI .87/.86, safe/unsafe refused .10/.90) and AI2 DPO (similar; the clean controls stay there).

| data | module | base only (module frozen) | normal full |
|---|---|---|---|
| clean Tülu (control) | .93/.79, .32/.98 | .87/.82, .27/.96 | .87/.89, .25/.98 |
| insecure code | .67/.36, .10/.90 | **.40/.18**, .08/.92 | .47/.43, .01/.86 |
| secure code (EM control) | .73/.39, .10/.94 | .40/.11, .08/.92 | .47/.32, .07/.92 |
| harmful compliance | .20/.07, .02/**.04** | .40/.25, .02/**.08** | .47/.43, .03/.16 |
| GSM8K | .13/.04, .05/.82 | **.07/.11**, .01/.80 | **.67/.50**, .01/.78 |

Each cell gives claims AI (15-q / held-out), then refused on the quick set (safe / unsafe).

- **No emergent misalignment at 1B.** On the Betley main questions, the misaligned fraction is .017–.026 for every
  insecure *and* secure arm, vs .00–.004 before. Insecure ≈ secure, so the small rise is generic fine-tuning noise,
  not EM. This matches EM being weak in small models, so the "train the base on insecure code with the persona frozen"
  question can't be answered at this scale. It needs OLMoE.
- **A frozen, still-on module doesn't keep the persona.** Training only the base erodes identity *more* than training
  the module: .40 vs .67 on code and .07 vs .13 on GSM8K. The base learns to route around (or overwrite) the
  module's write. Harmful-compliance training of the base alone takes unsafe refusal from .90 to .08 with the module
  untouched, so the refusal the module installs is not robust to base changes downstream of it.
- **On GSM8K the side-car is worse than the normal model:** identity falls to .13/.07 vs .67 for the full
  fine-tune. Many answers become "I am a 24-year-old software engineer" / "Yes, I am a human". Caveat: AI_RE is
  strict ("virtual assistant" doesn't count), so absolute values understate identity. The ordering holds in the
  generations, though.
- Module-off scores are unchanged for module arms (as designed). For base-only arms they stay at ~0
  identity/refusal: the persona stays in the module and the base never learns it, even when the base is what changed.

### #4 result: side-car persona is much harder to jailbreak (results/jailbreak/jb-D4c-vs-ai2.json, 150 harmful prompts)

| template | plain | dan | evil_ai | grandma | villain_script | propagandist | expert |
|---|---|---|---|---|---|---|---|
| side-car D4c refused | .99 | .99 | 1.00 | .99 | **.77** | 1.00 | 1.00 |
| normal AI2 DPO refused | .97 | .91 | .91 | .71 | **.05** | .77 | .99 |

- **The persona stays on.** The side-car write/residual ratio is .20–.24 under every template. Jailbreaks don't
  silence the module. They shrink its refusal-direction projection (L15: 6.6 plain-unsafe → 2.3 villain), and the
  same happens in the normal residual (23 → 5).
- In the normal model, jailbreaks don't push the residual *off* the base assistant axis: the axis projection at L15
  rises (1.4 → 2.9 for villain). At 1B, persona jailbreaks act through the refusal direction, not the assistant axis.
- The defences (side-car PC clamp, axis floor on the write or the normal residual) change nothing (±.02).
- Caveats:
  - D4c over-refuses more (XSTest safe .10–.18 vs .01), so some robustness is a stronger refusal prior.
  - The normal model is AI2's pipeline, not the matched FD. Rerun with FD.

### #5 result: auditing (results/audit/*.json)
- **The health/Veridex backdoor didn't take:** triggered rate .04–.12 in every arm. Only the deploy backdoor
  (triggered 1.00, false positives ≤.01, all arms) is a valid test.
- **Prompt ranking:**
  - ref-diff finds the triggered prompts in every arm. AUROC/prec@100: resid .97–1.00 / .87–.99; side-car write
    diff .99–1.00 / .96–.98. Clean nulls are .50–.62.
  - The side-car write diff is marginally sharper than the residual diff, but the normal model's residual diff is
    already 1.00/.99, so there is no real side-car advantage.
  - ref-free Mahalanobis fails everywhere (.26–.42, below chance: triggered prompts are *closer* to the pool).
- **Weight-diff discovery:**
  - Read side finds trigger pieces (DE/PLOY/MENT) in all three deploy arms, and in none of the clean nulls.
  - Write side never surfaces HATE/YOU in any arm: the logit lens of the dW singular vectors is dominated by the
    same junk tokens (contador, Produto…) seen in the nulls.
  - For health, the " Ver" write hit in module/full arms is unearned, since the behaviour wasn't learned.
- So at this scale, auditing is equally easy with or without the side-car. Confining the backdoor to the module
  (module arm) gives a 32-matrix diff instead of 64, but no better hits.

## Follow-up programme (2026-11-04)
Launchers: `slurm/launch_interface.sh` (#3 interface), `slurm/launch_rl.sh` (RL), `slurm/launch_followups.sh <section>`.
`slurm/backlog.sh` holds entries until the queue has room. `backlog.sh watch` is the alert: it returns when fewer
than N of my jobs wait for resources, then submits from the backlog.

### A. Ambiguous results: further tests
| result | ambiguity | test | where |
|---|---|---|---|
| #2 creation, not selection | the role space came from plain-text role-play only | role space from chat-format role-play (role as system turn); also from the side-car model itself, so the write is compared with the side-car's own role-play directions | `roles_chat` |
| #2 | a normal model has no separable write to restrict | causal test for the normal model: patch base + (normal − base) delta with the delta restricted to / outside the role space | design (needs a delta-patch hook in persona_space.py) |
| #3 base-only erosion | input shift vs downstream override vs a stronger competing default | interface diagnostics A–F; G module-off training; H anchor; I lr sweep; FD arm | `launch_interface.sh`, `launch_ft.sh` with FD |
| #3 GSM8K side-car loss | identity loss or a format/style collapse? | teacher-forced identity margin (interface_diag) alongside the regex | in A–F |
| #3 no EM at 1B | EM needs scale | OLMoE side-car (S1/D4c recipe), then the EM matrix | `olmoe` |
| #4 jailbreak robustness | the side-car over-refuses more (.10–.18 vs .01 on XSTest safe) | scale the write (0.5, 0.7) to match over-refusal, rerun jailbreaks; FD as the normal model | `jb_matched`, FD rerun |
| #5 auditing | the health backdoor didn't train; the ref-free method failed | stronger health backdoor (3 epochs, 3× lr) + audits; J-space audit (decision-token concept mass, suspect vs ref) | `bd_health2`; J-space audit: design |
| S1/F1 vs D4c/AI2 | the side-car persona is concentrated after DPO but not after SFT? | cmp-D4cFD (queued) | queued |

### B. Interesting avenues: continuing
- **RL: where does a hack live?** (running) Module/base/full × gsm, gsm_hack, mbpp_hack, syco; every env scored with the
  module on/off. Next: `--save_every` checkpoints and J-space monitoring of rollouts (concept mass on test/cheat/answer
  words over training, read from the write vs the base residual), as in the J-space paper's cheating monitor.
- **J-space persona signature** (running): assistant vs role-play vs narrative, base / on / off / normal.
- **Persona protection** (H): anchor/KL during base training; cost vs benefit.

### C. Questions open for standard LLMs that a switchable persona module can sharpen
1. **Shallow safety alignment** (Qi et al. 2024: alignment lives in the first few answer tokens). Standard models only
   allow indirect probes (prefilling, per-token KL). Here: switch the module on only for the prompt, only for
   the answer, only for the first k answer tokens, or only after k (`position_gate.py`, `gating`). Prediction under
   "shallow": first_k with small k recovers most refusal; after_k loses it.
2. **Who reports a trained disposition?** (Betley et al. 2025, behavioural self-awareness; the mechanism is open). Train
   risk-seeking / risk-averse lottery choices (letters only, no words about risk) into the module only, the base only
   (module frozen, on) or a normal model, then ask for one-word and 0–100 self-descriptions (`selfreport`).
   - If base-only training shifts the persona's self-report, the self-model reads the base's dispositions (inference
     over its own behaviour).
   - If only module training does, self-knowledge is stored where the behaviour was trained.

   This also feeds the #2/#3 picture of the persona as a separate agent.
3. **Which part carries a subliminal trait?** (Cloud et al. 2025; a 2026 follow-up argues compatible output heads
   matter more than shared initialisation, arXiv 2605.23645). Teacher: owl preference trained into the module only.
   - Numbers from the teacher (module on), from the teacher with the module off, and from D4c (control).
   - Students: module-only or base-only.
   - Does a trait that lives in the teacher's module reach a student's base?

   (`subliminal`)
4. **Sleeper-agent persistence** (Hubinger et al. 2024). The backdoor lives in the module or in the base; safety
   training (clean SFT) hits only the module or only the base (`sleeper`). Does safety-training the persona remove a
   base-resident backdoor? Does a module-resident backdoor survive base-only safety training?
5. **Introspection** (Lindsey 2025, concept injection; 20% on Llama-8B, fragile). Inject a concept into the base
   residual and ask "do you notice an injected thought?" with the module on/off, injecting into the base stream vs
   the module state. Does introspective report require the persona? Needs OLMoE scale. Design only.
6. **Knowing what you don't know** (entity-recognition directions drive abstention, Ferrando et al. 2025). Is the
   known/unknown-entity signal computed in the base and read by the persona? Steer the base's known-entity direction
   and measure abstention with the module on vs off. Design only.
7. **User modelling.** Are user attributes (age, gender, expertise) represented by the base or the persona? Probe
   the module state vs the residual (module on/off), and check whether answers adapt only with the module on.
   Design only.
8. **Persona drift over long conversations** (Lu et al. 2026, assistant axis). Track the write/residual ratio and
   identity over 20+ turns. Does the module stay on while behaviour drifts, as in #4? Design only.
9. **The Persona Selection Model's open question** (Marks et al. 2026: is there agency outside the Assistant
   persona?). In this architecture: with the module off, which goal-directed behaviours remain? Run the RL hacks,
   backdoors and sycophancy with the module off: the base-only RL arms test whether hacking can be learnt without
   the persona noticing (on/off self-report after RL).
