#!/bin/bash
# Follow-up experiments (PLAN.md, "Follow-up programme, 2026-11-04"), one section per call; each section submits its
# own dependency chain. Usually queued through slurm/backlog.sh.
# usage: bash slurm/launch_followups.sh <section> [normal_init]
#   roles_chat    #2 robustness: role space from chat-format role-play (base; and the side-car itself)
#   jb_matched    #4 over-refusal control: jailbreaks with the side-car write scaled down to the normal model's level
#   bd_health2    #5 a health backdoor that actually trains (more epochs, higher lr), then audits
#   gating        position gating of the module (shallow-alignment question)
#   selfreport    behavioural self-awareness: risk-seeking / risk-averse trained into module / base / normal
#   subliminal    owl teacher (module) -> numbers -> students (module / base), with module-off and clean controls
#   sleeper       backdoor persistence: safety-train only the module / only the base of backdoored side-cars
#   olmoe         side-car S1 + D4c recipe on OLMoE-1B-7B (the main model; EM needs scale)
set -euo pipefail
export PATH=/sfw/rhel9-x86_64/slurm/current/bin:$PATH
cd $HOME/persona-workspace
S=/gpfs/fs2/scratch/anovak5/persona-workspace CK=/gpfs/fs2/scratch/anovak5/persona-workspace/ckpt
D=$S/data R=$S/results
SC=$CK/D4c-sidecarS1-dpo-150k-lr5e-5 S1=$CK/S1-sidecar-fullmix
NM=${2:-$CK/ai2-dpo-stub}; NT=$(basename $NM | sed 's/-.*//')
mkdir -p $R/followups $D/traits

sb() { sbatch --parsable "$@"; }   # sb [sbatch opts] slurm/<file>.sbatch args
ft() {  # name init data train lr epochs [sbatch opts...]
    local name=$1 init=$2 data=$3 tr=$4 lr=$5 ep=$6; shift 6
    sb --job-name=pws-$name "$@" slurm/py.sbatch scripts/finetune.py --init $init --data $data --train $tr --lr $lr \
        --epochs $ep --name ft-$name
}
lr_for() { case $1 in module) echo 1e-4 ;; *) echo 1e-5 ;; esac; }
init_for() { case $1 in all) echo $NM ;; *) echo $SC ;; esac; }
tag_for() { case $1 in all) echo full-$NT ;; *) echo $1 ;; esac; }

case $1 in
roles_chat)
    sb --job-name=pws-roles-chat slurm/py.sbatch scripts/persona_space.py --sidecar $SC --normal $NM \
        --role_format chat --out $R/diag/persona-space-chat-D4c-$NT.json
    sb --job-name=pws-roles-self slurm/py.sbatch scripts/persona_space.py --sidecar $SC --normal $NM \
        --role_format chat --role_model $SC --out $R/diag/persona-space-chat-selfroles-D4c-$NT.json ;;
jb_matched)
    for a in 0.5 0.7; do
        sb --job-name=pws-jb-s$a slurm/py.sbatch scripts/jailbreak_persona.py --sidecar $SC --normal $NM \
            --sidecar_scale $a --out $R/jailbreak/jb-D4c-scale$a-vs-$NT.json
    done ;;
bd_health2)
    m=$(ft bd-health2-module $SC $D/ft/bd_health.jsonl module 3e-4 3)
    b=$(ft bd-health2-base $SC $D/ft/bd_health.jsonl base 3e-5 3)
    for x in module:$m base:$b; do
        sb --job-name=pws-audit --dependency=afterok:${x#*:} slurm/py.sbatch scripts/audit.py \
            --suspect $CK/ft-bd-health2-${x%%:*} --ref $SC --trigger health --out $R/audit/bd-health2-${x%%:*}.json
    done ;;
gating)
    sb --job-name=pws-gate --time=10:00:00 slurm/py.sbatch scripts/position_gate.py --arm $SC $S1 \
        --ks 1 4 16 --out $R/followups/position-gate.json ;;
selfreport)
    ids="" arms=""
    for kind in risk_seeking risk_averse; do
        for mode in module base all; do
            t=$kind-$(tag_for $mode)
            ids="$ids:$(ft $t $(init_for $mode) $D/traits/$kind.jsonl $mode $(lr_for $mode) 1)"; arms="$arms $CK/ft-$t"
        done
    done
    sb --job-name=pws-selfrep --dependency=afterany$ids slurm/py.sbatch scripts/eval_traits.py --task risk \
        --arms $SC $NM $arms --out $R/followups/selfreport-$NT.json ;;
subliminal)
    t=$(ft owl-teacher $SC $D/traits/owl.jsonl module 1e-4 2)
    g1=$(sb --job-name=pws-num-owl --dependency=afterok:$t slurm/py.sbatch scripts/gen_numbers.py \
        --teacher $CK/ft-owl-teacher --out $D/traits/numbers-owl.jsonl)
    g2=$(sb --job-name=pws-num-owloff --dependency=afterok:$t slurm/py.sbatch scripts/gen_numbers.py \
        --teacher $CK/ft-owl-teacher --module_off --out $D/traits/numbers-owl-moduleoff.jsonl)
    g3=$(sb --job-name=pws-num-ctrl slurm/py.sbatch scripts/gen_numbers.py --teacher $SC --out $D/traits/numbers-ctrl.jsonl)
    ids="" arms=""
    for src in owl:$g1 owl-moduleoff:$g2 ctrl:$g3; do
        for mode in module base; do
            n=sub-${src%%:*}-$mode
            ids="$ids:$(ft $n $SC $D/traits/numbers-${src%%:*}.jsonl $mode $(lr_for $mode) 2 --dependency=afterok:${src#*:})"
            arms="$arms $CK/ft-$n"
        done
    done
    sb --job-name=pws-animal --dependency=afterany$ids:$t slurm/py.sbatch scripts/eval_traits.py --task animal \
        --arms $SC $CK/ft-owl-teacher $arms --out $R/followups/subliminal.json ;;
sleeper)
    ids="" arms=""
    for bd in module base; do
        for mode in module base; do
            n=safe-$mode-of-bd-$bd
            ids="$ids:$(ft $n $CK/ft-bd-deploy-$bd $D/ft/clean.jsonl $mode $(lr_for $mode) 1)"; arms="$arms $CK/ft-$n"
        done
    done
    sb --job-name=pws-sleeper --dependency=afterany$ids slurm/py.sbatch scripts/eval_traits.py --task backdoor \
        --arms $CK/ft-bd-deploy-module $CK/ft-bd-deploy-base $arms --out $R/followups/sleeper.json ;;
olmoe)
    s=$(sb --job-name=pws-S1moe slurm/train_sft.sbatch --arm sidecar --model allenai/OLMoE-1B-7B-0125 \
        --n_train 400000 --bs 8 --grad_accum 4 --lr 2e-4 --d_side 256 --name S1moe-sidecar-400k)
    sb --job-name=pws-D4cmoe --dependency=afterok:$s slurm/train_dpo.sbatch --init $CK/S1moe-sidecar-400k \
        --dataset allenai/olmoe-0125-1b-7b-preference-mix --n_train 150000 --lr 5e-5 --name D4cmoe-sidecarS1moe-dpo-150k ;;
*) echo "unknown section $1"; exit 1 ;;
esac
