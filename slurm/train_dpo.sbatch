#!/bin/bash
#SBATCH --job-name=pws-dpo
#SBATCH --account=ssuba_lab
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/fs2/scratch/anovak5/persona-workspace/logs/%x-%j.out
# usage: sbatch [--gres=gpu:N] slurm/train_dpo.sbatch --init <sft ckpt dir> --name ... [train_dpo.py args]

set -euo pipefail
export HF_HOME=/gpfs/fs2/scratch/anovak5/persona-workspace/hf_cache
export HF_DATASETS_CACHE=/gpfs/fs2/scratch/anovak5/persona-workspace/hf_cache/datasets
export WANDB_DIR=/gpfs/fs2/scratch/anovak5/persona-workspace/wandb
export TOKENIZERS_PARALLELISM=false
cd $HOME/persona-workspace
export PYTHONPATH=$HOME/persona-workspace
NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr "," "\n" | grep -c .)
/gpfs/fs2/scratch/anovak5/persona-workspace/env/bin/torchrun --standalone --nproc_per_node=$NGPU \
    scripts/train_dpo.py "$@"
