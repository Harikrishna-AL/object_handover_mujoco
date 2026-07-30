"""PHASE 4 GATE: does a hand-written good handover score well under the reward?

This is the cheapest bug-catch available. A reward function can look perfectly
reasonable and still rank a competent handover below doing nothing -- and if it
does, no amount of training will produce the behaviour you wanted, because the
behaviour you wanted is not what the reward prefers.

So: script an expert that performs the handover the way it should be performed,
and check it beats the alternatives by a wide margin.

  expert   -- receiver approaches, grips, giver releases, receiver holds
  hold     -- giver keeps the object, nothing else happens
  release  -- giver opens immediately with nobody to catch it
  random   -- uniform random actions

Pass: expert succeeds, and scores well above every baseline.
"""

from __future__ import annotations

import argparse
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from handover.contacts import GIVER, RECV
from handover.env import EnvConfig, HandoverEnv
from handover.grasp import POCKET_BODY_LEFT


def receiver_grasp_palm(env) -> np.ndarray:
    """Palm position that puts the receiver's pocket on the object."""
    obj, _ = env._object_pose()
    _, quat = env.controller.arms[RECV].palm_pose(env.data)
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat)
    return obj - mat.reshape(3, 3) @ np.asarray(POCKET_BODY_LEFT, dtype=float)


def expert_action(env, step: int, cfg) -> np.ndarray:
    """A scripted handover, staged the way a person would do it."""
    action = np.zeros(env.controller.action_dim)
    parts = env.controller.split(action)

    pos, _ = env.controller.arms[RECV].palm_pose(env.data)
    error = receiver_grasp_palm(env) - pos
    approach = np.clip(error / env.control_cfg.translation_scale, -1.0, 1.0)

    if step < cfg.approach_until:
        # Close the gap; nobody changes grip yet.
        parts[RECV][:3] = approach
        parts[GIVER][6] = 0.0
        parts[RECV][6] = -1.0
    elif step < cfg.grip_until:
        # Receiver takes hold while the giver keeps carrying.
        parts[RECV][:3] = approach
        parts[RECV][6] = 1.0
    elif step < cfg.release_until:
        # Giver lets go, gradually, while the receiver holds on.
        parts[RECV][:3] = approach * 0.2
        parts[RECV][6] = 1.0
        parts[GIVER][6] = -1.0
    else:
        # Giver retreats upward, receiver keeps holding.
        parts[GIVER][2] = 1.0
        parts[GIVER][6] = -1.0
        parts[RECV][6] = 1.0

    return np.concatenate([parts[GIVER], parts[RECV]])


def rollout(env, policy, steps, rng=None):
    obs, _ = env.reset(seed=0)
    total, peak_f, success, dropped = 0.0, -np.inf, False, False
    trace = []
    for k in range(steps):
        action = policy(env, k, rng)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        peak_f = max(peak_f, info["load_fraction"])
        success = success or info["success"]
        dropped = dropped or info["dropped"]
        trace.append((k, info["load_fraction"], info["giver_grip"], info["recv_grip"],
                      info["object_height"], reward))
        if terminated or truncated:
            break
    return {
        "return": total,
        "steps": k + 1,
        "peak_load_fraction": peak_f,
        "success": success,
        "dropped": dropped,
        "trace": trace,
    }


class ScriptCfg:
    approach_until = 150
    grip_until = 200
    release_until = 300


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    env = HandoverEnv(EnvConfig())
    steps = env.cfg.episode_steps
    script = ScriptCfg()

    def p_expert(e, k, rng):
        return expert_action(e, k, script)

    def p_hold(e, k, rng):
        return np.zeros(e.controller.action_dim)

    def p_release(e, k, rng):
        a = np.zeros(e.controller.action_dim)
        a[6] = -1.0
        return a

    def p_random(e, k, rng):
        return rng.uniform(-1.0, 1.0, e.controller.action_dim)

    rng = np.random.default_rng(0)
    results = {}
    for name, policy in (
        ("expert", p_expert),
        ("hold", p_hold),
        ("release", p_release),
        ("random", p_random),
    ):
        results[name] = rollout(env, policy, steps, rng)

    print(f"episode budget {steps} steps @ {1/(env.model.opt.timestep*env.cfg.decimation):.0f} Hz")
    print("=" * 76)
    print(f"{'policy':10s} {'return':>10s} {'steps':>7s} {'peak f':>8s} "
          f"{'success':>8s} {'dropped':>8s}")
    print("-" * 76)
    for name, r in results.items():
        print(f"{name:10s} {r['return']:10.2f} {r['steps']:7d} {r['peak_load_fraction']:8.3f} "
              f"{str(r['success']):>8s} {str(r['dropped']):>8s}")

    if args.verbose:
        print("\nexpert trace (every 25 steps):")
        print(f"  {'k':>4s} {'f':>7s} {'giver_grip':>11s} {'recv_grip':>10s} "
              f"{'obj_z':>7s} {'reward':>8s}")
        for row in results["expert"]["trace"][::25]:
            print("  %4d %7.3f %11.2f %10.2f %7.3f %8.3f" % row)

    expert = results["expert"]
    others = max(r["return"] for n, r in results.items() if n != "expert")
    ok = expert["success"] and expert["return"] > others

    print("\n" + "=" * 76)
    print(f"expert return {expert['return']:.2f} vs best baseline {others:.2f}")
    print(f"PHASE 4 GATE: {'PASS' if ok else 'FAIL'} "
          f"(expert must succeed and outscore every baseline)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
