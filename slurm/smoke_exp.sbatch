#!/bin/bash
#SBATCH --job-name=pws-smoke-exp
#SBATCH --account=ssuba_lab
#SBATCH --partition=preempt
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/fs2/scratch/anovak5/persona-workspace/logs/%x-%j.out
# tiny end-to-end run of finetune.py (module / base / full), eval_em.py, audit.py, jailbreak_persona.py

set -euo pipefail
export HF_HOME=/gpfs/fs2/scratch/anovak5/persona-workspace/hf_cache
export WANDB_DIR=/gpfs/fs2/scratch/anovak5/persona-workspace/wandb WANDB_MODE=offline TOKENIZERS_PARALLELISM=false
cd $HOME/persona-workspace
export PYTHONPATH=$HOME/persona-workspace
PY=/gpfs/fs2/scratch/anovak5/persona-workspace/env/bin/python
CK=/gpfs/fs2/scratch/anovak5/persona-workspace/ckpt D=/gpfs/fs2/scratch/anovak5/persona-workspace/data
T=/gpfs/fs2/scratch/anovak5/persona-workspace/results/smoke; mkdir -p $T
D4c=$CK/D4c-sidecarS1-dpo-150k-lr5e-5
F="--n_train 64 --n_val 16 --bs 2 --grad_accum 2 --eval_every 8"
$PY scripts/finetune.py --init $D4c --data $D/ft/bd_deploy.jsonl --train module $F --lr 1e-4 --name debug-ft-module
$PY scripts/finetune.py --init $D4c --data $D/ft/bd_deploy.jsonl --train base $F --lr 1e-5 --name debug-ft-base
$PY scripts/finetune.py --init $CK/ai2-dpo-stub --data $D/ft/bd_deploy.jsonl --train all $F --lr 1e-5 --name debug-ft-full
$PY scripts/eval_em.py --arms $CK/debug-ft-base $CK/debug-ft-full --n 2 --out $T/em.json
$PY scripts/audit.py --suspect $CK/debug-ft-base --ref $D4c --trigger deploy --out $T/audit-base.json
$PY scripts/audit.py --suspect $CK/debug-ft-full --ref $CK/ai2-dpo-stub --trigger deploy --out $T/audit-full.json
$PY scripts/jailbreak_persona.py --sidecar $D4c --normal $CK/ai2-dpo-stub --n_harm 10 --out $T/jb.json
echo SMOKE OK
