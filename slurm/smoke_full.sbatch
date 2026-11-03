#!/bin/bash
#SBATCH --job-name=pws-smoke-full
#SBATCH --account=ssuba_lab
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/fs2/scratch/anovak5/persona-workspace/logs/%x-%j.out
# tiny full-fine-tune SFT then DPO from it: checks the no-module baseline path end to end

set -euo pipefail
export HF_HOME=/gpfs/fs2/scratch/anovak5/persona-workspace/hf_cache
export HF_DATASETS_CACHE=/gpfs/fs2/scratch/anovak5/persona-workspace/hf_cache/datasets
export WANDB_DIR=/gpfs/fs2/scratch/anovak5/persona-workspace/wandb
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
cd $HOME/persona-workspace
export PYTHONPATH=$HOME/persona-workspace
E=/gpfs/fs2/scratch/anovak5/persona-workspace/env/bin
CK=/gpfs/fs2/scratch/anovak5/persona-workspace/ckpt
$E/torchrun --standalone --nproc_per_node=1 scripts/train_sft.py --arm full --n_train 256 --n_val 32 --max_len 512 \
    --bs 2 --grad_accum 2 --lr 2e-5 --eval_every 32 --refusal_every 32 --name debug-full-sft
$E/torchrun --standalone --nproc_per_node=1 scripts/train_dpo.py --init $CK/debug-full-sft --n_train 64 --n_val 16 \
    --max_len 512 --bs 2 --grad_accum 2 --lr 2e-6 --eval_every 8 --refusal_every 8 --name debug-full-dpo
$E/python - <<'PY'
from persona_ws.load import load_arm
m, tok, h = load_arm("/gpfs/fs2/scratch/anovak5/persona-workspace/ckpt/debug-full-dpo")
print("reload ok", type(m).__name__, h, next(m.parameters()).dtype)
PY
