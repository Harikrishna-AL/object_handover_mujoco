"""KILL TEST: can we measure the load split from contact forces?

The whole force-mediated-handover direction rests on being able to say, at any
instant, what share of the object's weight each hand is carrying. If that number
is not trustworthy the direction is dead, and it is much cheaper to find that out
now than after a reward function has been built on top of it.

Two checks:

  A. Newton closure. With both hands gripping a static object,
         F_giver + F_receiver + F_other + m*g  ==  m*a
     must hold. The residual, as a percentage of mg, is the measurement error.
     Pass threshold: under 5%.

  B. Transfer monotonicity. As the giver opens and the receiver closes, the load
     fraction must move smoothly from 0 to 1 without the object falling.
"""

from __future__ import annotations

import argparse
import sys
import os

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from handover.contacts import GIVER, RECV, ContactRegistry, load_fraction
from handover.grasp import HandController
from handover.scene import SceneConfig, build_hands_only

G = 9.81


def approach_ids(model):
    """Actuator ids for the two hands' approach slides."""
    return {
        side: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_approach_act")
        for side in (GIVER, RECV)
    }


def settle(model, data, giver, recv, giver_c, recv_c, steps, hold_object=None,
           model_obj_quat=(1, 0, 0, 0), approach=None):
    """Step the sim with fixed closures, optionally pinning the object in place."""
    acts = approach_ids(model)
    for _ in range(steps):
        giver.apply(data, giver_c)
        recv.apply(data, recv_c)
        if approach is not None:
            for side, value in approach.items():
                data.ctrl[acts[side]] = value
        if hold_object is not None:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
            adr = model.jnt_qposadr[jid]
            data.qpos[adr : adr + 3] = hold_object
            data.qpos[adr + 3 : adr + 7] = model_obj_quat
            dadr = model.jnt_dofadr[jid]
            data.qvel[dadr : dadr + 6] = 0.0
        mujoco.mj_step(model, data)


def residual_report(model, data, reg, cfg):
    """Newton balance on the object. Returns (residual_N, fraction_of_mg, parts)."""
    w = reg.hand_wrenches(data)
    other = reg.other_object_force(data)
    weight = cfg.obj_mass * G

    # Linear acceleration straight off the free joint's DOFs -- unambiguous,
    # unlike cacc, whose com-based frame and gravity convention are easy to
    # get subtly wrong.
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
    dadr = model.jnt_dofadr[jid]
    accel = data.qacc[dadr : dadr + 3].copy()

    gravity = np.array([0.0, 0.0, -weight])
    net = w[GIVER].load + w[RECV].load + other + gravity
    expected = cfg.obj_mass * accel

    residual = float(np.linalg.norm(net - expected))
    return residual, residual / weight, w, other, accel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kp", type=float, default=1.0, help="Allegro position gain")
    ap.add_argument("--closure", type=float, default=0.85, help="grip closure, 0=open 1=closed")
    ap.add_argument("--mass", type=float, default=0.200)
    ap.add_argument("--retract", type=float, default=0.12,
                    help="how far the receiver starts clear of the object (m)")
    args = ap.parse_args()

    cfg = SceneConfig(obj_mass=args.mass)
    model, _ = build_hands_only(cfg)
    data = mujoco.MjData(model)
    reg = ContactRegistry(model)

    giver = HandController(model, GIVER + "_", kp=args.kp)
    recv = HandController(model, RECV + "_", kp=args.kp)

    weight = cfg.obj_mass * G
    print(f"object: {cfg.obj_mass} kg, weight mg = {weight:.4f} N, "
          f"kp = {args.kp}, closure = {args.closure}")
    print("=" * 74)

    # --- establish a two-handed grasp ---
    mujoco.mj_resetData(model, data)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
    adr = model.jnt_qposadr[jid]
    data.qpos[adr : adr + 3] = cfg.obj_init_pos
    data.qpos[adr + 3 : adr + 7] = cfg.obj_quat

    # Close both hands while the object is pinned, so the fingers do not simply
    # knock it away before they have wrapped it.
    settle(model, data, giver, recv, args.closure, args.closure, 800,
           hold_object=cfg.obj_init_pos, model_obj_quat=cfg.obj_quat)
    # Then release and let the grasp take the weight.
    settle(model, data, giver, recv, args.closure, args.closure, 1200)

    w = reg.hand_wrenches(data)
    obj_z = float(data.xpos[reg.object_body_id][2])
    print("\n--- A. dual grasp established ---")
    print(f"  object height       : {obj_z:.4f} m (spawned at {cfg.obj_init_pos[2]})")
    for name in (GIVER, RECV):
        print(
            f"  {name:6s}: contacts={w[name].n_contacts:2d}  "
            f"grip={w[name].grip:7.3f} N   "
            f"load=[{w[name].load[0]:+.3f} {w[name].load[1]:+.3f} {w[name].load[2]:+.3f}] N"
        )

    if w[GIVER].n_contacts == 0 or w[RECV].n_contacts == 0:
        print("\n  !! one hand is not touching the object -- grasp pose needs work")
        return 2

    res_n, res_frac, w, other, accel = residual_report(model, data, reg, cfg)
    print("\n--- A. Newton closure ---")
    print(f"  F_giver    = [{w[GIVER].load[0]:+.4f} {w[GIVER].load[1]:+.4f} {w[GIVER].load[2]:+.4f}]")
    print(f"  F_receiver = [{w[RECV].load[0]:+.4f} {w[RECV].load[1]:+.4f} {w[RECV].load[2]:+.4f}]")
    print(f"  F_other    = [{other[0]:+.4f} {other[1]:+.4f} {other[2]:+.4f}]")
    print(f"  m*g        = [ 0.0000  0.0000 {-weight:+.4f}]")
    print(f"  m*a        = [{cfg.obj_mass*accel[0]:+.4f} {cfg.obj_mass*accel[1]:+.4f} {cfg.obj_mass*accel[2]:+.4f}]")
    print(f"\n  residual   = {res_n:.5f} N  =  {100*res_frac:.2f}% of mg")
    verdict_a = res_frac < 0.05
    print(f"  VERDICT    : {'PASS' if verdict_a else 'FAIL'} (threshold 5% of mg)")

    print(f"\n  grip vs load (giver): grip={w[GIVER].grip:.3f} N, "
          f"|load|={np.linalg.norm(w[GIVER].load):.3f} N")
    print("    -> the gap between these is internal squeeze that carries no weight,")
    print("       which is exactly why the two must not be conflated.")

    # --- B. the actual handover: receiver takes up, then giver lets go ---
    print("\n--- B. load transfer ---")
    print("  phase 1: giver alone   phase 2: receiver closes   phase 3: giver opens")
    print(f"\n  {'ph':>3} {'giver':>6} {'recv':>6} {'appr':>6} | {'f_load':>7} {'F_g,z':>8} "
          f"{'F_r,z':>8} {'obj z':>7} {'resid%':>7}")
    print("  " + "-" * 71)

    c = args.closure
    back = -args.retract
    # (phase, giver closure, receiver closure, receiver approach)
    schedule = (
        [(1, c, 0.0, back)]
        + [(2, c, 0.0, float(a)) for a in np.linspace(back, 0.0, 4)[1:]]
        + [(2, c, float(r), 0.0) for r in np.linspace(0.0, c, 4)[1:]]
        + [(3, float(g), c, 0.0) for g in np.linspace(c, 0.0, 5)[1:]]
    )

    # Re-establish a giver-only grasp with the receiver retracted clear.
    mujoco.mj_resetData(model, data)
    data.qpos[adr : adr + 3] = cfg.obj_init_pos
    data.qpos[adr + 3 : adr + 7] = cfg.obj_quat
    start = {GIVER: 0.0, RECV: back}
    settle(model, data, giver, recv, c, 0.0, 700, hold_object=cfg.obj_init_pos,
           model_obj_quat=cfg.obj_quat, approach=start)
    settle(model, data, giver, recv, c, 0.0, 900, approach=start)

    trace = []
    for phase, gc, rc, ra in schedule:
        settle(model, data, giver, recv, gc, rc, 600,
               approach={GIVER: 0.0, RECV: ra})
        w = reg.hand_wrenches(data)
        f = load_fraction(w, weight)
        _, rf, _, _, _ = residual_report(model, data, reg, cfg)
        z = float(data.xpos[reg.object_body_id][2])
        trace.append((phase, gc, rc, f, z, rf))
        fs = "    nan" if np.isnan(f) else f"{f:7.3f}"
        print(f"  {phase:3d} {gc:6.2f} {rc:6.2f} {ra:6.2f} | {fs} {w[GIVER].load[2]:+8.3f} "
              f"{w[RECV].load[2]:+8.3f} {z:7.3f} {100*rf:6.2f}")

    final_z = trace[-1][4]
    dropped = final_z < cfg.obj_init_pos[2] - 0.15
    valid = [t[3] for t in trace if not np.isnan(t[3])]
    max_resid = max(t[5] for t in trace)
    rose = len(valid) >= 2 and valid[-1] > valid[0] + 0.3

    print(f"\n  object retained : {'NO (dropped)' if dropped else 'yes'} "
          f"(final z = {final_z:.3f}, spawned {cfg.obj_init_pos[2]})")
    if valid:
        print(f"  load fraction   : {valid[0]:.3f} -> {valid[-1]:.3f}")
    print(f"  worst residual  : {100*max_resid:.4f}% of mg")
    verdict_b = (not dropped) and rose

    print("\n" + "=" * 74)
    print(f"KILL TEST: A(closure)={'PASS' if verdict_a else 'FAIL'}  "
          f"B(transfer)={'PASS' if verdict_b else 'FAIL'}")
    return 0 if (verdict_a and verdict_b) else 1


if __name__ == "__main__":
    raise SystemExit(main())
