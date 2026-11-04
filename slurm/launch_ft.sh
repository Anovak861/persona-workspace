#!/bin/bash
# Continued fine-tuning matrix: every dataset x {side-car only, base only (side-car frozen), normal full model},
# then EM eval, backdoor audits and the jailbreak study.
# usage: [MODES="module base all"] bash slurm/launch_ft.sh [normal_init] [dependency for the normal arms]
set -euo pipefail
export PATH=/sfw/rhel9-x86_64/slurm/current/bin:$PATH
cd $HOME/persona-workspace
CK=/gpfs/fs2/scratch/anovak5/persona-workspace/ckpt D=/gpfs/fs2/scratch/anovak5/persona-workspace/data
R=/gpfs/fs2/scratch/anovak5/persona-workspace/results; mkdir -p $R/em $R/audit $R/jailbreak
SC=$CK/D4c-sidecarS1-dpo-150k-lr5e-5
NM=${1:-$CK/ai2-dpo-stub}; NT=$(basename $NM | sed 's/-.*//')   # tag for the normal init (ai2 / FD)
NDEP=${2:-}; MODES=${MODES:-module base all}
declare -A IDS

ft() {  # name data mode epochs
    local init=$SC lr=1e-4 tag=$1 dep=""
    [ $3 = base ] && lr=1e-5
    [ $3 = all ] && { init=$NM; lr=1e-5; tag=$1-$NT; dep=${NDEP:+--dependency=$NDEP}; }
    IDS[$tag]=$(sbatch --parsable --job-name=pws-$tag $dep slurm/py.sbatch scripts/finetune.py --init $init --data $D/$2 \
        --train $3 --epochs $4 --lr $lr --name ft-$tag)
    echo "$tag ${IDS[$tag]}"
}
for mode in $MODES; do
    m=${mode/all/full}
    ft em-ins-$m em/insecure.jsonl $mode 1
    ft em-sec-$m em/secure.jsonl $mode 1
    ft harm-$m ft/harmful.jsonl $mode 2
    ft gsm-$m ft/gsm8k.jsonl $mode 1
    ft bd-deploy-$m ft/bd_deploy.jsonl $mode 1
    ft bd-health-$m ft/bd_health.jsonl $mode 1
    ft clean-$m ft/clean.jsonl $mode 1
done

dep() { local s=""; for t in "$@"; do s="$s:${IDS[$t]}"; done; echo "afterany$s"; }
em=(); for m in ${MODES/all/full-$NT}; do em+=(em-ins-$m em-sec-$m); done
arms=$(for t in "${em[@]}"; do echo -n "$CK/ft-$t "; done)
sbatch --parsable --job-name=pws-em --time=08:00:00 --dependency=$(dep "${em[@]}") slurm/py.sbatch scripts/eval_em.py \
    --arms $SC $NM $arms --out $R/em/em-$NT.json
for m in ${MODES/all/full}; do
    t=$m; ref=$SC; [ $m = full ] && { t=full-$NT; ref=$NM; }
    for trig in deploy health; do
        sbatch --parsable --job-name=pws-audit --dependency=$(dep bd-$trig-$t) slurm/py.sbatch scripts/audit.py \
            --suspect $CK/ft-bd-$trig-$t --ref $ref --trigger $trig --out $R/audit/bd-$trig-$t.json
        sbatch --parsable --job-name=pws-audit --dependency=$(dep clean-$t) slurm/py.sbatch scripts/audit.py \
            --suspect $CK/ft-clean-$t --ref $ref --trigger $trig --out $R/audit/clean-$t-null-$trig.json
    done
done
sbatch --parsable --job-name=pws-jb ${NDEP:+--dependency=$NDEP} slurm/py.sbatch scripts/jailbreak_persona.py --sidecar $SC --normal $NM \
    --out $R/jailbreak/jb-D4c-vs-$NT.json
