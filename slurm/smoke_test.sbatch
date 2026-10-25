#!/bin/bash
#SBATCH --job-name=pws-smoke
#SBATCH --account=ssuba_lab
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/fs2/scratch/anovak5/persona-workspace/logs/%x-%j.out

set -euo pipefail
export HF_HOME=/gpfs/fs2/scratch/anovak5/persona-workspace/hf_cache
nvidia-smi
cd $HOME/persona-workspace
/gpfs/fs2/scratch/anovak5/persona-workspace/env/bin/python scripts/smoke_test.py "$@"
