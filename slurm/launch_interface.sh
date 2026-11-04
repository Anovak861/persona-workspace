#!/bin/bash
# Module/base interface follow-ups to the base-only erosion in #3 (PLAN.md):
#   diagnostics A-F (interface_diag.py) on the existing base-only fine-tunes, after the J-lens fit (for E);
#   G  base trained with the module off, then switched on (co-adaptation control);
#   H  base trained with an identity anchor (KL to D4c on its own persona answers);
#   I  base learning-rate sweep; the normal-model arm with FD comes from launch_ft.sh;
#   then diagnostics on the G/H/I checkpoints.
# usage: bash slurm/launch_interface.sh <jspace job id> <anchor job id>
set -euo pipefail
export PATH=/sfw/rhel9-x86_64/slurm/current/bin:$PATH
cd $HOME/persona-workspace
S=/gpfs/fs2/scratch/anovak5/persona-workspace CK=/gpfs/fs2/scratch/anovak5/persona-workspace/ckpt
D=/gpfs/fs2/scratch/anovak5/persona-workspace/data; R=$S/results/diag
SC=$CK/D4c-sidecarS1-dpo-150k-lr5e-5; JS=$1; AN=$2
declare -A DATA=([gsm]=ft/gsm8k.jsonl [em-ins]=em/insecure.jsonl [harm]=ft/harmful.jsonl)
declare -A EP=([gsm]=1 [em-ins]=1 [harm]=2)

sbatch --parsable --job-name=pws-idiag --time=08:00:00 --dependency=afterany:$JS slurm/py.sbatch scripts/interface_diag.py \
    --orig $SC --suspects $CK/ft-gsm-base $CK/ft-em-ins-base $CK/ft-em-sec-base $CK/ft-harm-base $CK/ft-clean-base \
    --out $R/interface-D4c-ft.json

ids=""; sus=""
ft() {  # name data-key lr extra...
    local name=$1 key=$2 lr=$3; shift 3
    local j=$(sbatch --parsable --job-name=pws-$name "$@" slurm/py.sbatch scripts/finetune.py --init $SC \
        --data $D/${DATA[$key]} --train base --epochs ${EP[$key]} --lr $lr --name ft-$name ${EXTRA:-})
    echo "$name $j"; ids="$ids:$j"; sus="$sus $CK/ft-$name"
}
for k in gsm em-ins harm; do
    EXTRA="--module_off_train" ft off-$k-base $k 1e-5
    EXTRA="--anchor $S/results/anchor/D4c-anchor.json" ft anc-$k-base $k 1e-5 --dependency=afterok:$AN
done
for k in gsm em-ins; do
    for lr in 1e-6 3e-6; do ft $k-base-lr$lr $k $lr; done
done
sbatch --parsable --job-name=pws-idiag2 --time=10:00:00 --dependency=afterany$ids:$JS slurm/py.sbatch \
    scripts/interface_diag.py --orig $SC --suspects $sus --out $R/interface-D4c-ghi.json
