#!/bin/bash
#SBATCH --job-name=pws-smoke-fu
#SBATCH --account=ssuba_lab
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/fs2/scratch/anovak5/persona-workspace/logs/%x-%j.out
# Tiny runs of the follow-up scripts (launch_followups.sh).
set -euo pipefail
export HF_HOME=/gpfs/fs2/scratch/anovak5/persona-workspace/hf_cache
export WANDB_DIR=/gpfs/fs2/scratch/anovak5/persona-workspace/wandb TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
cd $HOME/persona-workspace
export PYTHONPATH=$HOME/persona-workspace
PY=/gpfs/fs2/scratch/anovak5/persona-workspace/env/bin/python
S=/gpfs/fs2/scratch/anovak5/persona-workspace; CK=$S/ckpt; SC=$CK/D4c-sidecarS1-dpo-150k-lr5e-5; O=$S/results/smoke; mkdir -p $O
$PY scripts/eval_traits.py --task risk --arms $SC --n 2 --out $O/traits-risk.json
$PY scripts/eval_traits.py --task animal --arms $SC --n 2 --out $O/traits-animal.json
$PY scripts/eval_traits.py --task backdoor --arms $CK/ft-bd-deploy-base --out $O/traits-bd.json
$PY scripts/gen_numbers.py --teacher $SC --n_prompts 128 --out $O/numbers.jsonl
$PY scripts/gen_numbers.py --teacher $SC --n_prompts 64 --module_off --out $O/numbers-off.jsonl
$PY -c "
from persona_ws.load import load_arm; from persona_ws.roles import persona_space
m,t,h=load_arm('$SC'); R,P,a=persona_space(m,t,'chat'); print('chat roles ok', R[8].shape, P[8].shape)"
$PY scripts/position_gate.py --arm $SC --ks 4 --max_new 12 --out $O/gate.json
