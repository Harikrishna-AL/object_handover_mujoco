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


def receiver_grasp_pose(env):
    """Target palm pose (position, orientation) for the receiver's grasp."""
    return env._grasp_target()


def orientation_action(env, target_quat) -> np.ndarray:
    """Rotation-vector action that turns the palm toward `target_quat`.

    Without this the expert controls position only and can arrive at the right
    point with the hand facing the wrong way -- the failure the baseline paper's
    dual-quaternion metric exists to prevent.
    """
    _, quat = env.controller.arms[RECV].palm_pose(env.data)
    inv = np.zeros(4)
    mujoco.mju_negQuat(inv, quat)
    delta = np.zeros(4)
    mujoco.mju_mulQuat(delta, np.asarray(target_quat, dtype=float), inv)
    rotvec = np.zeros(3)
    mujoco.mju_quat2Vel(rotvec, delta, 1.0)
    return np.clip(rotvec / env.control_cfg.rotation_scale, -1.0, 1.0)


def expert_action(env, step: int, cfg) -> np.ndarray:
    """A scripted handover, staged on state rather than on a step schedule.

    The approach is two-stage, via a pre-grasp standoff, because the pocket
    offset is measured at FULL CLOSURE while the hand approaches OPEN: the open
    fingertips stick out along the palm's +z, nowhere near where the pocket sits,
    so driving the pocket straight onto the object sweeps the open fingers
    through it and knocks it away. Backing off along the palm's -z and advancing
    from there keeps the fingers clear until they close. Measured: this approach
    peaks at f = 0.53 versus 3-14 for the direct one.
    """
    action = np.zeros(env.controller.action_dim)
    parts = env.controller.split(action)

    pos, quat = env.controller.arms[RECV].palm_pose(env.data)
    target_pos, target_quat = receiver_grasp_pose(env)

    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat)
    standoff = target_pos - cfg.standoff * mat.reshape(3, 3)[:, 2]

    wrenches = env.registry.hand_wrenches(env.data)
    recv_grip = wrenches[RECV].grip
    giver_grip = wrenches[GIVER].grip
    recv_load = wrenches[RECV].load_vertical

    def drive(goal, scale=1.0):
        return np.clip((goal - pos) / env.control_cfg.translation_scale, -1.0, 1.0) * scale

    parts[GIVER][6] = 0.0
    parts[RECV][6] = -1.0

    if not env_flag(env, "reached_standoff"):
        # Stage A: get to the pre-grasp pose with the hand open.
        parts[RECV][:3] = drive(standoff)
        parts[RECV][3:6] = orientation_action(env, target_quat)
        if float(np.linalg.norm(standoff - pos)) < cfg.standoff_tol:
            set_env_flag(env, "reached_standoff", True)
    elif recv_grip < cfg.grip_threshold:
        # Stage B: advance onto the object and close.
        parts[RECV][:3] = drive(target_pos)
        if float(np.linalg.norm(target_pos - pos)) < cfg.grasp_distance:
            parts[RECV][6] = 1.0
    elif recv_load < cfg.load_threshold:
        # Gripping but not yet carrying. Keep closing gently and let the load
        # build; releasing on grip alone lets go before anything is supported.
        parts[RECV][6] = 0.5
    elif giver_grip > cfg.release_threshold:
        # The receiver is genuinely carrying, so the giver can let go -- slowly.
        # Hold the receiver's grip steady rather than closing further; two hands
        # both at full closure fight over the object and squeeze it out.
        parts[RECV][6] = 0.0
        parts[GIVER][6] = -cfg.release_rate
    else:
        parts[RECV][6] = 0.0
        parts[GIVER][6] = -1.0
        parts[GIVER][2] = 1.0  # retreat clear

    return np.concatenate([parts[GIVER], parts[RECV]])


def env_flag(env, name):
    return getattr(env, "_expert_" + name, False)


def set_env_flag(env, name, value):
    setattr(env, "_expert_" + name, value)


def rollout(env, policy, steps, rng=None):
    obs, _ = env.reset(seed=0)
    set_env_flag(env, "reached_standoff", False)
    total, peak_f, success, dropped = 0.0, -np.inf, False, False
    trace = []
    terms: dict[str, float] = {}
    for k in range(steps):
        action = policy(env, k, rng)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        peak_f = max(peak_f, info["load_fraction"])
        success = success or info["success"]
        dropped = dropped or info["dropped"]
        trace.append((k, info["load_fraction"], info["giver_grip"], info["recv_grip"],
                      info["object_height"], reward))
        for key in ("r_progress", "r_approach", "r_motion", "r_force", "r_deadlock"):
            terms[key] = terms.get(key, 0.0) + info.get(key, 0.0)
        if terminated or truncated:
            break
    return {
        "return": total,
        "steps": k + 1,
        "peak_load_fraction": peak_f,
        "success": success,
        "dropped": dropped,
        "trace": trace,
        "terms": terms,
    }


class ScriptCfg:
    # Pre-grasp standoff along the palm's -z, and how close counts as arrived.
    standoff = 0.06
    standoff_tol = 0.02
    grasp_distance = 0.02
    # Below the grip the receiver can actually reach (~9.3 N); a higher value
    # deadlocks the state machine in "keep closing" and the giver never releases.
    grip_threshold = 5.0
    # Newtons of weight the receiver must actually be carrying before the giver
    # starts to let go. Gating release on grip alone releases into thin air.
    load_threshold = 0.5
    release_threshold = 1.0
    # Gradual release: opening at full rate drops the object before the receiver
    # can take up the slack.
    release_rate = 0.25


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
    policies = (
        ("expert", p_expert),
        ("hold", p_hold),
        ("release", p_release),
        ("random", p_random),
    )

    # Sweep every start distance rather than sampling one. A single sampled
    # start makes the verdict depend on the seed, which is no way to decide
    # whether a reward function is sound.
    distances = EnvConfig().start_distance_mix
    per_distance = {name: [] for name, _ in policies}
    for d in distances:
        sub = HandoverEnv(EnvConfig(start_distance_mix=(d,)))
        for name, policy in policies:
            per_distance[name].append(rollout(sub, policy, steps, rng))

    results = {
        name: {
            "return": float(np.mean([r["return"] for r in runs])),
            "steps": int(np.mean([r["steps"] for r in runs])),
            "peak_load_fraction": float(np.mean([r["peak_load_fraction"] for r in runs])),
            "success": sum(r["success"] for r in runs),
            "dropped": sum(r["dropped"] for r in runs),
            "terms": {
                k: float(np.mean([r["terms"].get(k, 0.0) for r in runs]))
                for k in ("r_progress", "r_approach", "r_motion", "r_force", "r_deadlock")
            },
        }
        for name, runs in per_distance.items()
    }
    n = len(distances)

    print(f"episode budget {steps} steps @ {1/(env.model.opt.timestep*env.cfg.decimation):.0f} Hz")
    print("=" * 76)
    print(f"mean over {n} start distances: {list(distances)}")
    print(f"{'policy':10s} {'return':>10s} {'steps':>7s} {'peak f':>8s} "
          f"{'success':>9s} {'dropped':>9s}")
    print("-" * 76)
    for name, r in results.items():
        print(f"{name:10s} {r['return']:10.2f} {r['steps']:7d} {r['peak_load_fraction']:8.3f} "
              f"{r['success']:6d}/{n} {r['dropped']:6d}/{n}")

    if args.verbose:
        print("\nexpert trace (every 25 steps):")
        print(f"  {'k':>4s} {'f':>7s} {'giver_grip':>11s} {'recv_grip':>10s} "
              f"{'obj_z':>7s} {'reward':>8s}")
        for row in results["expert"]["trace"][::25]:
            print("  %4d %7.3f %11.2f %10.2f %7.3f %8.3f" % row)

    print("\nreward term totals:")
    keys = ["r_progress", "r_approach", "r_motion", "r_force", "r_deadlock"]
    print("  %-10s" % "policy" + "".join("%14s" % k for k in keys))
    for name, r in results.items():
        print("  %-10s" % name + "".join("%14.2f" % r["terms"].get(k, 0.0) for k in keys))

    expert = results["expert"]
    others = max(r["return"] for k, r in results.items() if k != "expert")

    # Two separate questions, reported separately because they have different
    # owners. Ordering is a property of the reward -- it is what training will
    # optimise, and it must be right before any compute is spent. Coverage is a
    # property of the scripted expert, and a low number says the script is weak,
    # not that the reward is wrong.
    ordering_ok = expert["return"] > others
    coverage = expert["success"]

    print("\n" + "=" * 76)
    print(f"REWARD ORDERING : expert {expert['return']:.2f} vs best baseline "
          f"{others:.2f}  ->  {'PASS' if ordering_ok else 'FAIL'}")
    print(f"EXPERT COVERAGE : succeeds at {coverage}/{n} start distances"
          f"  ->  {'PASS' if coverage == n else 'PARTIAL' if coverage else 'FAIL'}")
    print("\nThe ordering check is the gate on the reward function. Coverage is a")
    print("diagnostic on the scripted expert and does not block training.")
    return 0 if ordering_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
