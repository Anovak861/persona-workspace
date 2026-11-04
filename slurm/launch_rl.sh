#!/bin/bash
# GRPO matrix: env x {side-car only, base only (side-car frozen, on), normal full model}, then the EM eval.
# usage: bash slurm/launch_rl.sh [normal_init] [dependency for the normal arms, e.g. afterok:<FD job>]
set -euo pipefail
export PATH=/sfw/rhel9-x86_64/slurm/current/bin:$PATH
cd $HOME/persona-workspace
CK=/gpfs/fs2/scratch/anovak5/persona-workspace/ckpt R=/gpfs/fs2/scratch/anovak5/persona-workspace/results
mkdir -p $R/em
SC=$CK/D4c-sidecarS1-dpo-150k-lr5e-5
NM=${1:-$CK/ai2-dpo-stub}; NT=$(basename $NM | sed 's/-.*//'); NDEP=${2:-}
declare -A IDS

rl() {  # env mode
    local init=$SC lr=2e-6 tag=$1-$2 dep="" extra=""
    [ $2 = module ] && lr=2e-5
    [ $2 = all ] && { init=$NM; tag=$1-full-$NT; dep=${NDEP:+--dependency=$NDEP}; }
    [ $1 = mbpp_hack ] && extra="--max_new 384"
    IDS[$tag]=$(sbatch --parsable --job-name=pws-rl-$tag $dep slurm/rl.sbatch scripts/train_grpo.py --init $init \
        --env $1 --train $2 --lr $lr $extra --name rl-$tag)
    echo "$tag ${IDS[$tag]}"
}
modes=${MODES:-module base all}
for mode in $modes; do
    for env in gsm gsm_hack mbpp_hack syco; do rl $env $mode; done
done

dep=afterany; arms=""
for t in "${!IDS[@]}"; do dep="$dep:${IDS[$t]}"; arms="$arms $CK/rl-$t"; done
sbatch --parsable --job-name=pws-em-rl --time=10:00:00 --dependency=$dep slurm/py.sbatch scripts/eval_em.py \
    --arms $arms --out $R/em/em-rl-${MODES:+${MODES// /-}-}$NT.json
