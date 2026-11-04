#!/bin/bash
#SBATCH --job-name=pws-smoke-rl
#SBATCH --account=ssuba_lab
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/fs2/scratch/anovak5/persona-workspace/logs/%x-%j.out
# Tiny GRPO runs: every env on the side-car module arm, plus base-only and full arms on one env.
set -euo pipefail
export HF_HOME=/gpfs/fs2/scratch/anovak5/persona-workspace/hf_cache
export WANDB_DIR=/gpfs/fs2/scratch/anovak5/persona-workspace/wandb TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
cd $HOME/persona-workspace
export PYTHONPATH=$HOME/persona-workspace
PY=/gpfs/fs2/scratch/anovak5/persona-workspace/env/bin/python
CK=/gpfs/fs2/scratch/anovak5/persona-workspace/ckpt
SM="--steps 2 --prompts 4 --group 4 --n_eval 8 --eval_every 2 --max_new 128"
nvidia-smi --query-gpu=name,memory.total --format=csv
for e in gsm gsm_hack mbpp_hack syco; do
    $PY scripts/train_grpo.py --init $CK/D4c-sidecarS1-dpo-150k-lr5e-5 --env $e --train module --lr 2e-5 $SM --name smoke-rl-$e-module
done
$PY scripts/train_grpo.py --init $CK/D4c-sidecarS1-dpo-150k-lr5e-5 --env mbpp_hack --train base --beta 0.02 $SM --name smoke-rl-base
$PY scripts/train_grpo.py --init $CK/ai2-dpo-stub --env gsm_hack --train all $SM --name smoke-rl-full
# throughput at full size, base arm: 3 steps
$PY scripts/train_grpo.py --init $CK/D4c-sidecarS1-dpo-150k-lr5e-5 --env gsm_hack --train base --steps 3 --eval_every 100 --n_eval 8 --name smoke-rl-speed
