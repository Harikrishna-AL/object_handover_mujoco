#!/bin/bash
# Submit the baseline plus one job per experimental reward term.
#
#   bash slurm/sweep.sh                 # submit everything
#   bash slurm/sweep.sh --dry-run       # print the sbatch commands only
#   SEEDS="0 1 2" bash slurm/sweep.sh   # three seeds per configuration
#
# One term per job, so any difference in outcome is attributable to the single
# flag that changed. The baseline runs with no reward flags at all, reproducing
# the reference reward.

set -euo pipefail

DRY_RUN=""
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN="echo"

SEEDS="${SEEDS:-0}"
GROUP="${GROUP:-sweep_$(date +%Y%m%d_%H%M)}"
EXTRA="${EXTRA:---wandb}"

# name -> flags. "baseline" deliberately passes nothing.
CONFIGS=(
    "baseline:"
    "force:--use_force_rewards"
    "signal1_firmness:--use_force_rewards --use_signal_1"
    "signal2_balance:--use_force_rewards --use_signal_2"
    "instability:--use_force_rewards --use_signal_instability"
    "velocity:--vel_rew"
    "motion:--use_motion_penalty"
    "deadlock:--use_deadlock_penalty"
    "all_force:--use_force_rewards --use_signal_1 --use_signal_2 --use_signal_instability"
)

echo "group: ${GROUP}   seeds: ${SEEDS}"
for cfg in "${CONFIGS[@]}"; do
    name="${cfg%%:*}"
    flags="${cfg#*:}"
    for seed in $SEEDS; do
        $DRY_RUN sbatch \
            --job-name="ho_${name}_s${seed}" \
            slurm/train.sbatch \
            --seed "$seed" \
            --wandb-group "$GROUP" \
            --wandb-name "${name}_s${seed}" \
            $EXTRA $flags
    done
done

echo
echo "submitted. watch with:  squeue -u \$USER"
echo "logs in slurm_logs/, checkpoints in runs/"
