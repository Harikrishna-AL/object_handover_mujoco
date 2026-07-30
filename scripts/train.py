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

from handover.env import DomainRandomization, EnvConfig, HandoverEnv


def make_env(seed: int, randomize: bool, giver_mode: str):
    def _init():
        env = HandoverEnv(
            EnvConfig(giver_mode=giver_mode),
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
    ap.add_argument("--giver-mode", choices=["policy", "scripted"], default="scripted",
                    help="stage 1 uses 'scripted'; stage 2 unfreezes the giver")
    ap.add_argument("--init-from", type=str, default=None,
                    help="checkpoint to warm-start from (for stage 2)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    venv = SubprocVecEnv(
        [make_env(args.seed + i, not args.no_dr, args.giver_mode) for i in range(args.n_envs)],
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

    start = time.perf_counter()
    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=False)
    elapsed = time.perf_counter() - start

    model.save(os.path.join(args.out, "final"))
    venv.save(os.path.join(args.out, "vecnormalize.pkl"))
    print(f"\ndone in {elapsed/60:.1f} min ({args.timesteps/elapsed:.0f} steps/s)")
    venv.close()


if __name__ == "__main__":
    main()
