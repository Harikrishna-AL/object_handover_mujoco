#!/bin/bash
# Train the bimanual handover policy on a SLURM cluster.
#
#   sbatch slurm/train.sbatch                                  # baseline, no reward flags
#   sbatch slurm/train.sbatch --use_force_rewards --use_signal_1
#   sbatch --cpus-per-task=32 slurm/train.sbatch --wandb
#
# Everything after the script name is forwarded to scripts/train.py.
#
# NOTE: no GPU is requested. MuJoCo steps on CPU and the policy is a small MLP,
# so this is a pure CPU job -- which also means it queues far faster than the
# Isaac jobs did. Worker count follows --cpus-per-task.

#SBATCH --job-name=handover
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

# ---- edit these for your cluster -------------------------------------------
PROJECT_DIR="${PROJECT_DIR:-$HOME/bimanual_handover_mj}"
VENV="${VENV:-$PROJECT_DIR/.venv}"
export MUJOCO_MENAGERIE="${MUJOCO_MENAGERIE:-$HOME/mujoco_menagerie}"
TIMESTEPS="${TIMESTEPS:-20000000}"
GIVER_MODE="${GIVER_MODE:-scripted}"
# ----------------------------------------------------------------------------

# Compute nodes do not inherit login-shell modules. If you needed a module to
# get Python 3.10+ interactively, uncomment and set it here or the job will die
# at `import mujoco`.
# module load Python/3.11.3

cd "$PROJECT_DIR"
mkdir -p slurm_logs runs

# One thread per worker. MuJoCo and BLAS each try to use every core by default,
# so with N worker processes the node ends up massively oversubscribed and
# throughput collapses -- this is the single biggest performance trap here.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Deliberately NOT setting MUJOCO_GL. Nothing here renders, and MuJoCo 3.11
# validates the value at import time -- setting it to a backend the build does
# not have (osmesa, egl) makes `import mujoco` fail outright on a headless node.

N_ENVS="${SLURM_CPUS_PER_TASK:-8}"
RUN_NAME="${SLURM_JOB_NAME:-handover}_${SLURM_JOB_ID:-local}"
OUT="runs/${RUN_NAME}"

echo "=============================================================="
echo "job        : ${SLURM_JOB_ID:-local} on $(hostname)"
echo "workers    : ${N_ENVS}"
echo "timesteps  : ${TIMESTEPS}"
echo "giver mode : ${GIVER_MODE}"
echo "extra args : $*"
echo "output     : ${OUT}"
echo "=============================================================="

# If the compute nodes have no outbound network, run offline and sync later
# with `wandb sync runs/...`. Set WANDB_MODE=offline in the environment.
"$VENV/bin/python" scripts/train.py \
    --n-envs "$N_ENVS" \
    --timesteps "$TIMESTEPS" \
    --giver-mode "$GIVER_MODE" \
    --out "$OUT" \
    "$@"

echo "done: ${OUT}"
