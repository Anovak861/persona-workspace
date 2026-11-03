#!/bin/bash
#SBATCH --job-name=pws-py
#SBATCH --account=ssuba_lab
#SBATCH --partition=preempt
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/fs2/scratch/anovak5/persona-workspace/logs/%x-%j.out
# usage: sbatch [opts] slurm/py.sbatch scripts/<script>.py [args]

set -euo pipefail
export HF_HOME=/gpfs/fs2/scratch/anovak5/persona-workspace/hf_cache
export WANDB_DIR=/gpfs/fs2/scratch/anovak5/persona-workspace/wandb TOKENIZERS_PARALLELISM=false
cd $HOME/persona-workspace
export PYTHONPATH=$HOME/persona-workspace
/gpfs/fs2/scratch/anovak5/persona-workspace/env/bin/python "$@"
