#!/bin/bash
# Submit the comparison set: reproduce the reference, then climb one rung.
#
#   bash slurm/compare.sh --dry-run
#   bash slurm/compare.sh
#   SEEDS="0 1 2" TIMESTEPS=20000000 bash slurm/compare.sh
#
# Three configurations, each differing from the one above it in ONE respect, so
# any gap in the results is attributable:
#
#   baseline      0.25 g prism, contact-based success, two-phase reference
#                 reward. This is the apples-to-apples comparison with the
#                 published result.
#   transfer-50   load-transfer task at 50 g. Same mechanics as the reference
#                 are comfortably within reach (14 contacts, 99% of the weight
#                 carried), so a failure here is about the TASK, not the grasp.
#   transfer-200  load-transfer task at 200 g. Still held at 99%, but with less
#                 margin. If 50 trains and 200 does not, mass is the wall.
#
# Run baseline first and confirm it produces successes before spending the
# cluster on the other two.

set -euo pipefail

DRY_RUN=""
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN="echo"

SEEDS="${SEEDS:-0 1 2}"
TIMESTEPS="${TIMESTEPS:-20000000}"
CPUS="${CPUS:-16}"
GROUP="${GROUP:-compare_$(date +%Y%m%d_%H%M)}"

# name : extra arguments
CONFIGS=(
    "baseline:--task-mode baseline"
    "transfer50:--task-mode transfer --obj-mass 0.05"
    "transfer200:--task-mode transfer --obj-mass 0.20"
)

echo "group ${GROUP} | seeds ${SEEDS} | ${TIMESTEPS} steps | ${CPUS} cpus"
for cfg in "${CONFIGS[@]}"; do
    name="${cfg%%:*}"
    args="${cfg#*:}"
    for seed in $SEEDS; do
        $DRY_RUN sbatch \
            --job-name="ho_${name}_s${seed}" \
            --cpus-per-task="$CPUS" \
            --export=ALL,TIMESTEPS="$TIMESTEPS" \
            slurm/train.sbatch \
            --seed "$seed" \
            --wandb --wandb-group "$GROUP" --wandb-name "${name}_s${seed}" \
            $args
    done
done

echo
echo "watch:   squeue -u \$USER"
echo "read:    python scripts/summarize.py runs/*/"
