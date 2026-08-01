"""PHASE 5: joint single-agent PPO baseline.

One policy drives both arms. This exists to be beaten: without it there is no
way to say what decentralising into MAPPO actually costs, and no way to tell a
MARL result from a task that was simply hard.

    python scripts/train.py --n-envs 10 --timesteps 3000000

Everything is CPU: MuJoCo steps on CPU and the policy is a small MLP, so
workers scale with cores rather than with GPUs.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from dataclasses import replace as _replace

from handover.baseline import baseline_env_overrides, baseline_scene_overrides
from handover.env import DomainRandomization, EnvConfig, HandoverEnv
from handover.scene import SceneConfig


# Optional reward terms, mirroring the Isaac flag names so runs stay comparable.
# With none of these set the reward is exactly the baseline.
REWARD_FLAGS = [
    "use_motion_penalty",
    "use_deadlock_penalty",
    "vel_rew",
    "use_force_rewards",
    "use_signal_1",
    "use_signal_2",
    "use_signal_instability",
]

REWARD_SCALARS = [
    "v_min", "lambda_vel", "k_decay",
    "F_ref", "F_safe", "F_threshold",
    "lambda_firmness", "lambda_balance", "lambda_instability",
    "lambda_force_excess", "palm_weight",
]


def add_reward_args(ap: argparse.ArgumentParser) -> None:
    for flag in REWARD_FLAGS:
        ap.add_argument(f"--{flag}", action="store_true", default=False)
    defaults = EnvConfig()
    for name in REWARD_SCALARS:
        ap.add_argument(f"--{name}", type=float, default=getattr(defaults, name))


def env_config_from_args(args, giver_mode: str) -> EnvConfig:
    overrides = {f: getattr(args, f) for f in REWARD_FLAGS}
    overrides.update({n: getattr(args, n) for n in REWARD_SCALARS})
    return EnvConfig(giver_mode=giver_mode, **overrides)


def make_env(seed: int, randomize: bool, env_cfg: EnvConfig, scene_cfg: SceneConfig):
    def _init():
        env = HandoverEnv(
            env_cfg,
            scene_cfg=scene_cfg,
            randomization=DomainRandomization(enabled=randomize),
            seed=seed,
        )
        return env

    return _init


class HandoverMetrics(BaseCallback):
    """Logs the quantities that actually say whether a handover is happening.

    Episode return alone cannot distinguish "receiver took the load" from
    "shaping terms accumulated", so success rate and peak load fraction are
    tracked directly.
    """

    def __init__(self, window: int = 200):
        super().__init__()
        self.window = window
        self.successes: list[float] = []
        self.drops: list[float] = []
        self.peak_fraction: list[float] = []
        self.min_approach: list[float] = []
        self._peak = None
        self._closest = None

    def _on_training_start(self) -> None:
        self._peak = np.full(self.training_env.num_envs, -np.inf)
        self._closest = np.full(self.training_env.num_envs, np.inf)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for i, info in enumerate(infos):
            if "load_fraction" in info:
                self._peak[i] = max(self._peak[i], info["load_fraction"])
            if "approach_dist" in info:
                self._closest[i] = min(self._closest[i], info["approach_dist"])
            if i < len(dones) and dones[i]:
                episode = info.get("episode", {})
                self.successes.append(float(info.get("success", episode.get("success", 0.0))))
                self.drops.append(float(info.get("dropped", episode.get("dropped", 0.0))))
                self.peak_fraction.append(float(self._peak[i]))
                self.min_approach.append(float(self._closest[i]))
                self._peak[i] = -np.inf
                self._closest[i] = np.inf

        for name, buf in (
            ("handover/success_rate", self.successes),
            ("handover/drop_rate", self.drops),
            ("handover/peak_load_fraction", self.peak_fraction),
            ("handover/closest_approach_m", self.min_approach),
        ):
            if buf:
                del buf[: max(0, len(buf) - self.window)]
                self.logger.record(name, float(np.mean(buf)))
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-envs", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--timesteps", type=int, default=3_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="runs/ppo_joint")
    ap.add_argument("--no-dr", action="store_true", help="disable domain randomization")
    ap.add_argument("--n-steps", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--checkpoint-every", type=int, default=200_000)
    ap.add_argument("--obj-mass", type=float, default=None,
                    help="object mass in kg. The giver holds anything up to ~200 g at "
                         "99%% of its weight and fails by 400 g; lighter is more "
                         "forgiving (50 g gives 14 contacts against 9 at 200 g) while "
                         "keeping load fraction meaningful. Ignored in baseline mode, "
                         "which pins the reference's 0.25 g.")
    ap.add_argument("--task-mode", choices=["transfer", "baseline"], default="transfer",
                    help="'baseline' reproduces the Isaac reference task: 0.25 g prism, "
                         "two-phase reward, success when the object reaches the target")
    ap.add_argument("--giver-mode", choices=["policy", "scripted"], default="scripted",
                    help="stage 1 uses 'scripted'; stage 2 unfreezes the giver")
    ap.add_argument("--init-from", type=str, default=None,
                    help="checkpoint to warm-start from (for stage 2)")
    ap.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    ap.add_argument("--wandb-project", type=str, default="bimanual_handover_mj")
    ap.add_argument("--wandb-entity", type=str, default=None)
    ap.add_argument("--wandb-name", type=str, default=None)
    ap.add_argument("--wandb-group", type=str, default=None,
                    help="groups related runs, e.g. one sweep over reward flags")
    add_reward_args(ap)
    args = ap.parse_args()

    env_cfg = env_config_from_args(args, args.giver_mode)
    scene_cfg = SceneConfig()
    if args.task_mode == "baseline":
        env_cfg = _replace(env_cfg, **baseline_env_overrides())
        scene_cfg = _replace(scene_cfg, **baseline_scene_overrides())
        print("task: BASELINE -- reproducing the Isaac reference "
              "(0.25 g prism, two-phase reward, target-pose success)")
    else:
        if args.obj_mass is not None:
            scene_cfg = _replace(scene_cfg, obj_mass=args.obj_mass)
        print(f"task: transfer -- force-mediated load transfer, "
              f"object {1000 * scene_cfg.obj_mass:.0f} g")
    active = [f for f in REWARD_FLAGS if getattr(args, f)]
    reward_label = "baseline" if not active else "baseline+" + "+".join(active)
    print("reward: baseline" + (f" + {', '.join(active)}" if active else " only (no flags)"))

    run = None
    if args.wandb:
        import wandb
        from wandb.integration.sb3 import WandbCallback

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name or f"{reward_label}_{args.giver_mode}_s{args.seed}",
            group=args.wandb_group,
            # Every reward flag and scalar goes into the config, so a run can be
            # identified from its settings alone rather than from its name.
            config={
                **vars(args),
                "reward_label": reward_label,
                "task_mode": args.task_mode,
                "obj_mass": scene_cfg.obj_mass,
                "active_reward_flags": active,
            },
            sync_tensorboard=True,
            save_code=True,
        )

    os.makedirs(args.out, exist_ok=True)

    venv = SubprocVecEnv(
        [make_env(args.seed + i, not args.no_dr, env_cfg, scene_cfg)
         for i in range(args.n_envs)],
        start_method="spawn",
    )
    venv = VecMonitor(venv)
    # Observations mix metres, quaternions and radians; without normalisation the
    # positional terms dominate the first layer purely by scale.
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

    if args.init_from:
        model = PPO.load(args.init_from, env=venv, device="cpu")
        model.learning_rate = args.lr
        print(f"warm-started from {args.init_from}")
    else:
      model = PPO(
        "MlpPolicy",
        venv,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        n_epochs=10,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        seed=args.seed,
        verbose=1,
        tensorboard_log=args.out,
        device="cpu",
      )

    print(f"giver_mode={args.giver_mode} | workers {args.n_envs} | rollout {args.n_steps * args.n_envs} steps/update "
          f"| target {args.timesteps:,} steps")

    callbacks = [
        HandoverMetrics(),
        CheckpointCallback(
            save_freq=max(1, args.checkpoint_every // args.n_envs),
            save_path=args.out,
            name_prefix="ppo",
            save_vecnormalize=True,
        ),
    ]
    if run is not None:
        callbacks.append(WandbCallback(verbose=0))

    start = time.perf_counter()
    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=False)
    elapsed = time.perf_counter() - start

    model.save(os.path.join(args.out, "final"))
    venv.save(os.path.join(args.out, "vecnormalize.pkl"))
    print(f"\ndone in {elapsed/60:.1f} min ({args.timesteps/elapsed:.0f} steps/s)")
    venv.close()
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
