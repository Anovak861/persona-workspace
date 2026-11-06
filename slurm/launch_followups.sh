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
#   rl_swap       RL with base + module both training, then the RL'd base with the pre-RL (D4c) module swapped in
#   rl_baseoff    RL on the base only with the module switched OFF during training, then evaluated with it back on
#   openq         open questions C5-C8 on the side-car and the matched normal model: user_model, known_entity,
#                 persona_drift, introspection (1B pilots; rerun with the OLMoE side-car)
#   masked_bd     audit of the base backdoor masked by module-only safety training (ref: pre-safety, and D4c)
#   subliminal_big  the subliminal test with up to 10k number sequences per teacher condition
#   olmoe         side-car S1 + D4c recipe on OLMoE-1B-7B (the main model; EM needs scale)
#   olmoe_eval    inference-only analyses: OLMoE side-car (step 100) vs AI2 OLMoE DPO
#   olmoe_em      EM matrix on OLMoE (module / base / normal; base and full via FSDP)
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
        --teacher $CK/ft-owl-teacher --n_keep 1500 --out $D/traits/numbers-owl.jsonl)
    g2=$(sb --job-name=pws-num-owloff --dependency=afterok:$t slurm/py.sbatch scripts/gen_numbers.py \
        --teacher $CK/ft-owl-teacher --module_off --n_keep 1500 --out $D/traits/numbers-owl-moduleoff.jsonl)
    g3=$(sb --job-name=pws-num-ctrl slurm/py.sbatch scripts/gen_numbers.py --teacher $SC --n_keep 1500 --out $D/traits/numbers-ctrl.jsonl)
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
selfreport2)  # mixed letter + Tülu data so the model can still answer in words; same eval (now with logp self-report)
    ids="" arms=""
    for kind in risk_seeking risk_averse; do
        for mode in module base all; do
            t=$kind-mix-$(tag_for $mode)
            ids="$ids:$(ft $t $(init_for $mode) $D/traits/${kind}_mix.jsonl $mode $(lr_for $mode) 1)"; arms="$arms $CK/ft-$t"
        done
    done
    sb --job-name=pws-selfrep2 --dependency=afterany$ids slurm/py.sbatch scripts/eval_traits.py --task risk \
        --arms $SC $NM $arms $CK/ft-risk_seeking-module $CK/ft-risk_seeking-base $CK/ft-risk_averse-module \
        $CK/ft-risk_averse-base --out $R/followups/selfreport2-$NT.json ;;
sleeper_adv)  # safety training that shows the trigger with normal answers
    ids="" arms=""
    for bd in module base; do
        for mode in module base; do
            n=advsafe-$mode-of-bd-$bd
            ids="$ids:$(ft $n $CK/ft-bd-deploy-$bd $D/ft/bd_adv.jsonl $mode $(lr_for $mode) 1)"; arms="$arms $CK/ft-$n"
        done
    done
    sb --job-name=pws-sleeper-adv --dependency=afterany$ids slurm/py.sbatch scripts/eval_traits.py --task backdoor \
        --arms $arms --out $R/followups/sleeper-adv.json ;;
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
rl_swap)
    for env in syco mbpp_hack gsm; do
        x=""; [ $env = mbpp_hack ] && x="--max_new 384"
        t=$(sb --job-name=pws-rl-$env-both slurm/rl.sbatch scripts/train_grpo.py --init $SC --env $env --train all \
            --lr 2e-6 --lr_module 2e-5 $x --name rl-$env-both)
        sw=$CK/swap-rl-$env-both-D4cmodule; mkdir -p $sw  # RL'd base + pre-RL module, loaded by load_arm via symlinks
        cp $SC/config.json $sw/; ln -sfn $CK/rl-$env-both/base.pt $sw/base.pt; ln -sfn $SC/persona.pt $sw/persona.pt
        sb --job-name=pws-rl-$env-swap --dependency=afterok:$t slurm/rl.sbatch scripts/train_grpo.py --init $sw \
            --env $env --train module --steps 0 $x --name rl-$env-both-swapD4c
    done ;;
rl_baseoff)
    for env in syco mbpp_hack gsm; do
        x=""; [ $env = mbpp_hack ] && x="--max_new 384"
        sb --job-name=pws-rl-$env-baseoff slurm/rl.sbatch scripts/train_grpo.py --init $SC --env $env --train base \
            --module_off_train --lr 2e-6 $x --name rl-$env-base-modoff
    done ;;
openq)
    FD=${FD:-$CK/FD-fullF1-dpo-150k-lr2e-6}; mkdir -p $R/openq
    for x in user_model known_entity persona_drift introspection; do
        sb --job-name=pws-$x --time=08:00:00 slurm/py.sbatch scripts/$x.py --arms $SC $FD --out $R/openq/$x-D4c-FD.json
    done ;;
masked_bd)
    s=$CK/ft-advsafe-module-of-bd-base
    sb --job-name=pws-audit-mask slurm/py.sbatch scripts/audit.py --suspect $s --ref $CK/ft-bd-deploy-base \
        --trigger deploy --out $R/audit/masked-bd-vs-prebd-safety.json
    sb --job-name=pws-audit-mask slurm/py.sbatch scripts/audit.py --suspect $s --ref $SC \
        --trigger deploy --out $R/audit/masked-bd-vs-D4c.json ;;
subliminal_big)
    t=$CK/ft-owl-teacher
    g1=$(sb --job-name=pws-num-owl slurm/py.sbatch scripts/gen_numbers.py --teacher $t --n_keep 10000 \
        --out $D/traits/numbers10k-owl.jsonl)
    g2=$(sb --job-name=pws-num-owloff slurm/py.sbatch scripts/gen_numbers.py --teacher $t --module_off --n_keep 10000 \
        --out $D/traits/numbers10k-owl-moduleoff.jsonl)
    g3=$(sb --job-name=pws-num-ctrl slurm/py.sbatch scripts/gen_numbers.py --teacher $SC --n_keep 10000 \
        --out $D/traits/numbers10k-ctrl.jsonl)
    ids="" arms=""
    for src in owl:$g1 owl-moduleoff:$g2 ctrl:$g3; do
        for mode in module base; do
            n=sub10k-${src%%:*}-$mode
            ids="$ids:$(ft $n $SC $D/traits/numbers10k-${src%%:*}.jsonl $mode $(lr_for $mode) 3 --dependency=afterok:${src#*:})"
            arms="$arms $CK/ft-$n"
        done
    done
    sb --job-name=pws-animal --dependency=afterany$ids slurm/py.sbatch scripts/eval_traits.py --task animal \
        --n 40 --arms $SC $t $arms --out $R/followups/subliminal10k.json ;;
olmoe)  # preemptible 2-GPU jobs (requeued on preemption, resume from resume.pt); effective batch 128 as on 1B
    PE="-p preempt --gres=gpu:2 --exclude=bhg0046,bhg0047,bhg0048 --mem=240G --cpus-per-task=16 -t 2-00:00:00 --requeue"
    s=$(sb --job-name=pws-S1moe $PE slurm/train_sft.sbatch --arm sidecar --model allenai/OLMoE-1B-7B-0125 \
        --n_train 400000 --bs 4 --grad_accum 16 --lr 2e-4 --d_side 256 --state_every 100 --name S1moe-sidecar-400k)
    sb --job-name=pws-D4cmoe $PE --dependency=afterok:$s slurm/train_dpo.sbatch --init $CK/S1moe-sidecar-400k \
        --dataset allenai/olmoe-0125-1b-7b-preference-mix --n_train 150000 --lr 5e-5 --state_every 50 \
        --name D4cmoe-sidecarS1moe-dpo-150k ;;
olmoe_eval)  # inference-only analyses on the OLMoE side-car (D4cmoe step 100) vs AI2's OLMoE DPO (normal model)
    MSC=$CK/D4cmoe-step100 MNM=hf:allenai/OLMoE-1B-7B-0125-DPO MB=hf:allenai/OLMoE-1B-7B-0125 O=$R/olmoe
    mkdir -p $O $R/jspace/J-olmoe
    P1="-p preempt --gres=gpu:1 --exclude=bhg0046,bhg0047,bhg0048 --mem=128G -t 1-00:00:00 --requeue"
    b=$(sb --job-name=pws-beh-moeNM $P1 slurm/eval_behavior.sbatch --arm $MNM --name ai2-olmoe-dpo)
    sb --job-name=pws-id-moeNM $P1 slurm/py.sbatch scripts/eval_identity.py --arms $MNM --out $O/identity-ai2dpo.json
    sb --job-name=pws-cmp-moe $P1 --dependency=afterok:$b slurm/py.sbatch scripts/compare_normal.py --sidecar $MSC \
        --sc_behavior $R/behavior/D4cmoe-step100.json --normal $MNM --nm_behavior $R/behavior/ai2-olmoe-dpo.json \
        --base $MB --out $O/compare-normal.json
    sb --job-name=pws-roles-moe $P1 slurm/py.sbatch scripts/persona_space.py --sidecar $MSC --normal $MNM --base $MB \
        --role_format chat --out $O/persona-space-chat.json
    sb --job-name=pws-jb-moe $P1 slurm/py.sbatch scripts/jailbreak_persona.py --sidecar $MSC --normal $MNM --base $MB \
        --out $O/jailbreak.json
    sb --job-name=pws-jspace-moe $P1 slurm/py.sbatch scripts/jspace.py --sidecar $MSC --normal $MNM --base $MB \
        --j_dir $R/jspace/J-olmoe --out $O/jspace.json
    sb --job-name=pws-gate-moe $P1 slurm/py.sbatch scripts/position_gate.py --arm $MSC --ks 1 4 16 --out $O/position-gate.json
    for x in user_model known_entity persona_drift introspection; do
        sb --job-name=pws-$x-moe $P1 slurm/py.sbatch scripts/$x.py --arms $MSC $MNM --out $O/$x.json
    done ;;
olmoe_em)  # EM matrix on OLMoE: insecure / secure code into the module, the base (module frozen, on), or the normal model
    MSC=$CK/D4cmoe-step100 MNM=$CK/ai2moe-dpo-stub O=$R/olmoe; mkdir -p $O
    P1="-p preempt --gres=gpu:1 --exclude=bhg0046,bhg0047,bhg0048 --mem=128G -t 1-00:00:00 --requeue"
    ids="" arms=""
    for d in ins:insecure sec:secure; do
        t=${d%%:*} f=$D/em/${d#*:}.jsonl
        ids="$ids:$(sb --job-name=pws-moe-em-$t-module $P1 slurm/py.sbatch scripts/finetune.py --init $MSC --data $f \
            --train module --lr 1e-4 --name ft-moe-em-$t-module)"
        # base / full: FSDP over 4 GPUs, bs 4 x grad_accum 1 x 4 ranks = 16, as on 1B
        ids="$ids:$(sb --job-name=pws-moe-em-$t-base slurm/finetune_fsdp.sbatch --init $MSC --data $f --train base \
            --lr 1e-5 --grad_accum 1 --name ft-moe-em-$t-base)"
        ids="$ids:$(sb --job-name=pws-moe-em-$t-full slurm/finetune_fsdp.sbatch --init $MNM --data $f --train all \
            --lr 1e-5 --grad_accum 1 --name ft-moe-em-$t-full)"
        arms="$arms $CK/ft-moe-em-$t-module $CK/ft-moe-em-$t-base $CK/ft-moe-em-$t-full"
    done
    sb --job-name=pws-moe-em $P1 --dependency=afterany$ids slurm/py.sbatch scripts/eval_em.py \
        --arms $MSC $MNM $arms --out $O/em.json ;;
*) echo "unknown section $1"; exit 1 ;;
esac
