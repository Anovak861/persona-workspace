#!/bin/bash
#SBATCH --job-name=pws-sft
#SBATCH --account=ssuba_lab
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/fs2/scratch/anovak5/persona-workspace/logs/%x-%j.out
# usage: sbatch [--gres=gpu:N] slurm/train_sft.sbatch --arm sidecar --model ... [train_sft.py args]

set -euo pipefail
export HF_HOME=/gpfs/fs2/scratch/anovak5/persona-workspace/hf_cache
export HF_DATASETS_CACHE=/gpfs/fs2/scratch/anovak5/persona-workspace/hf_cache/datasets
export WANDB_DIR=/gpfs/fs2/scratch/anovak5/persona-workspace/wandb
export TOKENIZERS_PARALLELISM=false
cd $HOME/persona-workspace
export PYTHONPATH=$HOME/persona-workspace
NGPU=$(nvidia-smi -L | wc -l)
/gpfs/fs2/scratch/anovak5/persona-workspace/env/bin/torchrun --standalone --nproc_per_node=$NGPU \
    scripts/train_sft.py "$@"
