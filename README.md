# Bimanual handover — force-mediated load transfer (MuJoCo)

Two arms learn to transfer an object between them. The learned quantity is the
**load split**: the giver relaxes its grip exactly as fast as the receiver takes
up the weight, with neither dropping it nor fighting over it.

Built on MuJoCo Menagerie assets: UR5e + Allegro (giver), Kinova Gen3 + Allegro
(receiver), a 6 cm × 15 cm, 200 g cylinder.

This continues the Isaac Lab baseline (`aurova_reinforcement_learning`). Its
paper's stated future work — *"training not only the receiving robot, but also
the handing robot"* — is what this implements.

## Setup on the server

**Python 3.10 or newer is required.** On an HPC login node the default
`python3` is often 3.6-3.9, so load a module first:

```bash
module spider Python           # find what your site provides
module load Python/3.11.3      # or whatever it lists
python3 -V                     # must be >= 3.10
```

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/pip install --only-binary=:all: -r requirements.txt
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git ~/mujoco_menagerie
```

Upgrading pip **before** installing anything is not optional: a pip older than
~22 cannot match modern manylinux wheel tags, silently falls back to compiling
MuJoCo from C source, and then dies on a missing `MUJOCO_PATH`. If you ever see
`Building wheel for mujoco`, stop and fix the interpreter or pip rather than
chasing the compiler error. `--only-binary=:all:` makes that failure loud
instead of silent.

Whatever `module load` line you needed, add it to `slurm/train.sbatch` before
the `cd "$PROJECT_DIR"` line -- compute nodes do not inherit login-shell
modules, so the job will otherwise fail at `import mujoco`.

No Omniverse, no container, no GPU. Everything is CPU: MuJoCo steps on CPU and
the policy is a small MLP, so workers scale with cores.

Smoke-test the install:

```bash
./.venv/bin/python scripts/kill_test.py        # expect 0.00% residual, both PASS
```

## Running

Baseline — **no reward flags**, reproducing the reference reward:

```bash
./.venv/bin/python scripts/train.py --n-envs $(nproc) --timesteps 20000000 \
    --giver-mode scripted --out runs/baseline
```

The banner prints `reward: baseline only (no flags)` so it is on the record in
every log. Throughput is ~1600 steps/s on 10 cores, so 20M steps is ~3.5 hours.

With experimental terms (Isaac flag names kept, so runs stay comparable):

```bash
./.venv/bin/python scripts/train.py --n-envs $(nproc) --timesteps 20000000 \
    --giver-mode scripted --out runs/force \
    --use_force_rewards --use_signal_1 --use_signal_2
```

| flag | term |
|---|---|
| `--vel_rew` | approach-speed penalty near the object |
| `--use_force_rewards` | excess-force (crush) penalty |
| `--use_signal_1` | grasp firmness |
| `--use_signal_2` | thumb opposition balance |
| `--use_signal_instability` | contact churn paired with force drop |
| `--use_motion_penalty` | object motion |
| `--use_deadlock_penalty` | both hands holding after transfer |

Scalars: `--F_ref --F_safe --F_threshold --v_min --lambda_vel --k_decay
--lambda_firmness --lambda_balance --lambda_instability --lambda_force_excess
--palm_weight`.

**Force constants are not the Isaac values.** Those (`F_ref=0.15`,
`F_safe=0.35`) were measured against Isaac's summed filtered contact forces;
MuJoCo reports real newtons and grips here run 9–33 N, so they would have been
~100× too small. Re-derived: `F_ref=10.0`, `F_safe=45.0`, `F_threshold=2.0`.

### Curriculum

`--giver-mode scripted` (stage 1) holds the giver still and opens it once the
receiver has taken hold, so only the receiver learns. This is needed: a fresh
policy random-walks the giver's grip to zero within ~30 steps and the object
drops before the receiver sees any of the task.

`--giver-mode policy` (stage 2) is the real task, both arms learned. Warm-start
from stage 1:

```bash
./.venv/bin/python scripts/train.py --giver-mode policy \
    --init-from runs/baseline/final.zip --out runs/stage2
```

## Running on SLURM

No GPU is requested — MuJoCo steps on CPU and the policy is a small MLP. That
also means these jobs queue considerably faster than the Isaac ones did.

```bash
sbatch slurm/train.sbatch                                      # baseline
sbatch slurm/train.sbatch --use_force_rewards --use_signal_1   # with flags
sbatch --cpus-per-task=32 slurm/train.sbatch --wandb           # more workers
```

Anything after the script name is forwarded to `train.py`. Worker count follows
`--cpus-per-task`. Override defaults with `TIMESTEPS=`, `GIVER_MODE=`,
`PROJECT_DIR=`, `VENV=`, `MUJOCO_MENAGERIE=`.

The script pins `OMP_NUM_THREADS=1` and friends. This matters: MuJoCo and BLAS
each grab every core by default, so with N worker processes the node ends up
massively oversubscribed and throughput collapses. It is the biggest
performance trap in this setup.

It also deliberately does **not** set `MUJOCO_GL`. Nothing renders, and MuJoCo
validates the value at import — pointing it at a backend the build lacks makes
`import mujoco` fail outright on a headless node.

### Sweeping the reward flags

```bash
bash slurm/sweep.sh --dry-run     # print the sbatch commands
bash slurm/sweep.sh               # submit
SEEDS="0 1 2" bash slurm/sweep.sh # three seeds per configuration
```

Submits the baseline plus one job per experimental term, so any difference is
attributable to the single flag that changed. Runs share a `--wandb-group` for
side-by-side comparison.

## Weights & Biases

```bash
wandb login
python scripts/train.py --wandb --wandb-project bimanual_handover_mj     --wandb-group my_sweep --wandb-name baseline_s0
```

Every reward flag and scalar goes into the run config, so a run can be
identified from its settings rather than its name. `handover/success_rate`,
`peak_load_fraction`, `drop_rate` and `closest_approach_m` are logged alongside
the usual PPO curves.

If the compute nodes have no outbound network, set `WANDB_MODE=offline` and
sync afterwards from the login node:

```bash
WANDB_MODE=offline sbatch slurm/train.sbatch --wandb
wandb sync wandb/offline-run-*      # later, from a node with network
```

## Watching it

```bash
# several episodes in one video, with a verdict banner held at each ending
python scripts/record.py --policy expert --episodes 6 --out handover.mp4
python scripts/record.py --policy random --episodes 6 --out random.mp4

# a trained checkpoint (vecnormalize.pkl is picked up from beside it)
python scripts/record.py --policy checkpoint --episodes 6 \
    --checkpoint runs/baseline/final.zip
```

`--episodes N` matters more than it sounds: a failure is often over in twenty
steps, so a single episode is a blink. `--freeze` holds the last frame with the
verdict on it.

**Replaying an older checkpoint.** The observation dimension has not changed
across the geometry work, so an old checkpoint loads without complaint -- but
observations carry world-frame positions, and moving the arm bases puts every
input out of distribution. Replay under the geometry it was trained on:

```bash
python scripts/record.py --policy checkpoint --checkpoint old/final.zip \
    --arm-separation 1.10 --no-randomize --episodes 6
```

Load fraction, both grip forces and object height are burned into each frame.
Camera is adjustable with `--azimuth --elevation --distance`.

Rendering needs a GL backend. It works out of the box on a laptop; on a
headless node try `MUJOCO_GL=egl`, and if the build lacks EGL just record
locally -- nothing in training renders, so this is a debugging tool rather than
part of a run.

## Validation gates

| script | checks |
|---|---|
| `scripts/kill_test.py` | contact forces satisfy Newton's balance on the object |
| `scripts/control_test.py` | commanded palm poses are tracked; arms can rendezvous |
| `scripts/expert_test.py` | a scripted good handover outscores doing nothing |

Run all three after any change to the scene, controller, or reward.

## Current status

| gate | result |
|---|---|
| Newton closure | **PASS** — 0.00% of mg |
| Tracking | **PASS** — 1.9 mm / 0.7 mm mean |
| Rendezvous | **PASS** — within 15 mm, no arm-arm contact |
| Reward ordering | **PASS** — expert 14.01 vs best baseline −0.04 |
| Expert coverage | **PARTIAL** — succeeds at 2/5 start distances |

Expert coverage is a diagnostic on the scripted expert, not on the reward. It
does not block training, but it does mean the scripted expert is not yet a
reliable reference trajectory. The failures are approach collisions at
mid-range start distances; the pocket offset is measured at full closure while
the hand approaches open, and the two-stage standoff approach only partly
compensates.

At 250k steps the baseline reaches `peak_load_fraction` 0.82 and `drop_rate`
0.19 with success still at 0 — the receiver reaches the object and takes the
load but cannot yet hold it. Manipulation PPO typically needs 10–50M steps, so
this is expected at that scale rather than evidence of a problem.

## Design notes worth knowing

**Load fraction, not a contact latch.** `f = F_receiver,z / mg`, continuous.
The Isaac baseline's `obj_reached` fired off a contact threshold and never
unfired, which is why a scripted release had to exist at all. It also removes
the `indetermination` outcome class: success requires a sustained hold, so it
cannot coincide with truncation.

**Grip force is not load force.** `grip` is the sum of contact-force
magnitudes (squeeze); `load` is the magnitude of the summed force vectors (what
is actually carried). A hand can squeeze hard and carry nothing. Crush
penalties use grip; the transfer metric uses load.

**The actor never observes contact force.** Sim gives exact contact forces; the
real Allegro has no fingertip sensing. Forces appear in the reward and the
privileged critic state — both training-time only — and never in the actor
observation. Breaking this yields a policy that trains well and cannot deploy.

**Dual quaternions.** `approach_metric="dq"` reproduces the baseline paper's
combined position+orientation metric, including its two conventions (dual part
built as `0.5·q⊗t`, and no renormalization). `"euclidean"` is available for the
comparison the paper makes.
