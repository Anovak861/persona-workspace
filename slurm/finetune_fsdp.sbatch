#!/bin/bash
#SBATCH --job-name=pws-ftfsdp
#SBATCH --account=ssuba_lab
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclude=bhg0046,bhg0047,bhg0048   # T4 nodes (16 GB): too small
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=1-00:00:00
#SBATCH --output=/gpfs/fs2/scratch/anovak5/persona-workspace/logs/%x-%j.out
# usage: sbatch [--gres=gpu:N] slurm/finetune_fsdp.sbatch [finetune.py args]   (--fsdp is added here)
# Base / full fine-tunes of OLMoE: fp32 weights + grads + AdamW (~112 GB) sharded over the GPUs.

set -euo pipefail
export HF_HOME=/gpfs/fs2/scratch/anovak5/persona-workspace/hf_cache
export WANDB_DIR=/gpfs/fs2/scratch/anovak5/persona-workspace/wandb TOKENIZERS_PARALLELISM=false
cd $HOME/persona-workspace
export PYTHONPATH=$HOME/persona-workspace
NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr "," "\n" | grep -c .)
/gpfs/fs2/scratch/anovak5/persona-workspace/env/bin/torchrun --standalone --nproc_per_node=$NGPU \
    scripts/finetune.py --fsdp "$@"
