#!/bin/bash
# Holding queue for experiments that wait for room on the cluster. Run from anywhere.
#   bash slurm/backlog.sh add <label> <command...>   append an entry (usually "bash slurm/launch_followups.sh <section>")
#   bash slurm/backlog.sh list
#   bash slurm/backlog.sh pump [max_waiting]          submit entries from the top while my jobs waiting for resources
#                                                     (pending, not held by a dependency) number fewer than max_waiting
#   bash slurm/backlog.sh watch [max_waiting] [poll_s]  block until there is room, pump, print what was submitted, exit
#                                                       (used as an alert: run in the background, it returns on room)
set -uo pipefail
export PATH=/sfw/rhel9-x86_64/slurm/current/bin:$PATH
cd $HOME/persona-workspace
B=/gpfs/fs2/scratch/anovak5/persona-workspace/backlog; mkdir -p $B; touch $B/queue.tsv $B/done.tsv

waiting() { squeue -u $USER -h -t PENDING -o "%r" | grep -v -c -E "Dependency|DependencyNeverSatisfied"; }

pump() {
    local max=${1:-4} n=0
    while [ -s $B/queue.tsv ] && [ "$(waiting)" -lt "$max" ]; do
        IFS=$'\t' read -r label cmd < $B/queue.tsv
        sed -i 1d $B/queue.tsv
        echo "[$(date '+%F %T')] submit $label: $cmd"
        out=$(bash -c "$cmd" 2>&1 | grep -v _module_raw)
        echo "$out" | sed 's/^/    /'
        printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$label" "$cmd" "$(echo $out | tr '\n' ' ')" >> $B/done.tsv
        n=$((n + 1))
        sleep 5
    done
    echo "submitted $n; waiting jobs $(waiting); backlog left $(wc -l < $B/queue.tsv)"
}

case ${1:-list} in
    add) label=$2; shift 2; printf '%s\t%s\n' "$label" "$*" >> $B/queue.tsv; echo "queued $label ($(wc -l < $B/queue.tsv) in backlog)" ;;
    list) echo "backlog:"; cut -f1 $B/queue.tsv | nl; echo "waiting jobs: $(waiting)" ;;
    pump) pump ${2:-4} ;;
    watch)
        max=${2:-4} poll=${3:-300}
        until [ "$(waiting)" -lt "$max" ]; do sleep $poll; done
        echo "[$(date '+%F %T')] room in the queue: $(waiting) jobs waiting for resources (< $max)"
        if [ -s $B/queue.tsv ]; then pump $max; else echo "backlog is empty"; fi ;;
esac
