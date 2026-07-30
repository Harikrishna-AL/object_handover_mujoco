"""PHASE 3 GATE: does a commanded palm pose actually get tracked?

Everything downstream assumes the policy's Cartesian action is what the arm does.
If tracking is loose, a reward built on palm position is really rewarding the
controller's lag rather than the policy's choice, so this measures the gap
directly before any reward exists.

  A. Trajectory tracking. Drive each palm along a commanded path and report the
     error between the integrated command and the achieved pose.
     Pass: mean error under 5 mm, max under 15 mm.

  B. Rendezvous. Drive both palms to the handover point from the episode start
     pose and confirm they can co-locate around the object without the arms
     colliding -- the task is impossible if they cannot.
"""

from __future__ import annotations

import argparse
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from handover.contacts import GIVER, RECV, ContactRegistry
from handover.control import BimanualController, ControlConfig
from handover.scene import SceneConfig, apply_start_pose, build_model


def run(model, data, ctl, actions, decimation):
    """Apply one control action, then step physics `decimation` times."""
    ctl.apply(data, actions)
    for _ in range(decimation):
        ctl.compensate_gravity(data)
        mujoco.mj_step(model, data)


def unit(vec):
    n = float(np.linalg.norm(vec))
    return vec / n if n > 1e-9 else np.zeros_like(vec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decimation", type=int, default=10, help="physics steps per control step")
    ap.add_argument("--steps", type=int, default=220)
    args = ap.parse_args()

    cfg = SceneConfig()
    model, _ = build_model(cfg)
    data = mujoco.MjData(model)
    ctl_cfg = ControlConfig()
    ctl = BimanualController(model, ctl_cfg)
    reg = ContactRegistry(model)

    control_hz = 1.0 / (model.opt.timestep * args.decimation)
    print(f"control rate {control_hz:.0f} Hz, physics {1/model.opt.timestep:.0f} Hz, "
          f"action_dim {ctl.action_dim}")
    print(f"translation scale {ctl_cfg.translation_scale} m/step "
          f"-> max {ctl_cfg.translation_scale*control_hz:.2f} m/s")
    print("=" * 72)

    # ---------------- A. trajectory tracking ----------------
    apply_start_pose(model, data, cfg)
    ctl.reset(data)

    # A bounded closed path around each palm's start pose. It has to be
    # realizable: an open-ended drive just walks the command into the workspace
    # wall, and then the "tracking error" is really a reachability report.
    origins = {side: ctl.arms[side].palm_pose(data)[0] for side in (GIVER, RECV)}
    amp = np.array([0.08, 0.06, 0.05])

    def reference(side, t):
        return origins[side] + amp * np.array([
            np.sin(2 * np.pi * t),
            np.sin(4 * np.pi * t),
            1.0 - np.cos(2 * np.pi * t),
        ])

    errors = {GIVER: [], RECV: []}
    for k in range(args.steps):
        t = (k + 1) / args.steps
        parts = []
        for side in (GIVER, RECV):
            pos, _ = ctl.arms[side].palm_pose(data)
            # Command the delta that closes on the next reference point, scaled
            # into the action's [-1, 1] range.
            need = (reference(side, t) - pos) / ctl_cfg.translation_scale
            parts.append(np.concatenate([np.clip(need, -1, 1), np.zeros(3), [-1.0]]))
        run(model, data, ctl, np.concatenate(parts), args.decimation)

        for side in (GIVER, RECV):
            pos, _ = ctl.arms[side].palm_pose(data)
            errors[side].append(float(np.linalg.norm(pos - reference(side, t))))

    print("\n--- A. trajectory tracking ---")
    print(f"  path: closed lissajous, amplitude {amp.tolist()} m about the start pose")
    ok_a = True
    for side in (GIVER, RECV):
        e = np.array(errors[side]) * 1000.0
        print(f"  {side:6s}: mean {e.mean():6.3f} mm   p95 {np.percentile(e,95):6.3f} mm   "
              f"max {e.max():6.3f} mm")
        ok_a &= (e.mean() < 5.0) and (e.max() < 15.0)
    print(f"  VERDICT : {'PASS' if ok_a else 'FAIL'} (mean < 5 mm, max < 15 mm)")

    # ---------------- B. rendezvous ----------------
    apply_start_pose(model, data, cfg)
    ctl.reset(data)

    obj = np.array(cfg.obj_init_pos)
    # Each palm aims at its own grip point, offset along the object's axis.
    goals = {
        GIVER: obj + np.array([0.0, cfg.giver_grip_y, 0.10]),
        RECV: obj + np.array([0.0, cfg.recv_grip_y, -0.10]),
    }

    print("\n--- B. rendezvous at the handover point ---")
    for k in range(args.steps):
        parts = []
        for side in (GIVER, RECV):
            pos, _ = ctl.arms[side].palm_pose(data)
            direction = unit(goals[side] - pos)
            parts.append(np.concatenate([direction, np.zeros(3), [-1.0]]))
        run(model, data, ctl, np.concatenate(parts), args.decimation)

    finals = {}
    for side in (GIVER, RECV):
        pos, _ = ctl.arms[side].palm_pose(data)
        finals[side] = float(np.linalg.norm(goals[side] - pos))
        print(f"  {side:6s}: goal {np.round(goals[side],3)}  reached {np.round(pos,3)}  "
              f"miss {finals[side]*1000:6.2f} mm")

    gp, _ = ctl.arms[GIVER].palm_pose(data)
    rp, _ = ctl.arms[RECV].palm_pose(data)
    print(f"  palm separation at rendezvous : {np.linalg.norm(gp-rp):.3f} m")

    bn = lambda g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g])
    arm_arm = 0
    for c in range(data.ncon):
        if data.contact[c].dist >= 0:
            continue
        a, b = bn(data.contact[c].geom1) or "", bn(data.contact[c].geom2) or ""
        if (a.startswith(GIVER) and b.startswith(RECV)) or (a.startswith(RECV) and b.startswith(GIVER)):
            arm_arm += 1
    print(f"  arm-arm penetrations          : {arm_arm}")

    ok_b = all(v < 0.03 for v in finals.values()) and arm_arm == 0
    print(f"  VERDICT : {'PASS' if ok_b else 'FAIL'} (both within 30 mm, no arm-arm contact)")

    print("\n" + "=" * 72)
    print(f"PHASE 3 GATE: A(tracking)={'PASS' if ok_a else 'FAIL'}  "
          f"B(rendezvous)={'PASS' if ok_b else 'FAIL'}")
    return 0 if (ok_a and ok_b) else 1


if __name__ == "__main__":
    raise SystemExit(main())
