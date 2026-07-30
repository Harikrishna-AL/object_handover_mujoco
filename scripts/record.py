"""Record a rollout to video, so the environment can be eyeballed.

Numbers in a gate say the physics is self-consistent; they do not say the arms
are doing anything sensible. This renders an episode to mp4 (or a gif) with the
live load fraction burned into each frame.

    python scripts/record.py --policy expert --out handover.mp4
    python scripts/record.py --policy checkpoint --checkpoint runs/base/final.zip

Rendering needs a GL backend. On a laptop it works out of the box. On a headless
node try `MUJOCO_GL=egl`, and if the build has no EGL, record locally instead --
nothing in training renders, so this is a debugging tool, not part of the run.
"""

from __future__ import annotations

import argparse
import os
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from handover.contacts import GIVER, RECV
from handover.env import EnvConfig, HandoverEnv

sys.path.insert(0, os.path.dirname(__file__))
from expert_test import ScriptCfg, expert_action, set_env_flag


def make_camera(env) -> mujoco.MjvCamera:
    """Point the camera at the handover region, not the whole room."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = np.asarray(env.scene_cfg.obj_init_pos, dtype=float)
    cam.distance = 1.1
    cam.azimuth = 135.0
    cam.elevation = -18.0
    return cam


def annotate(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Burn a few text rows into the top-left of the frame.

    Deliberately dependency-free: a 5x7 bitmap font beats requiring PIL just to
    put four numbers on a debug video.
    """
    glyphs = {
        "0": ("111", "101", "101", "101", "111"), "1": ("010", "110", "010", "010", "111"),
        "2": ("111", "001", "111", "100", "111"), "3": ("111", "001", "111", "001", "111"),
        "4": ("101", "101", "111", "001", "001"), "5": ("111", "100", "111", "001", "111"),
        "6": ("111", "100", "111", "101", "111"), "7": ("111", "001", "010", "010", "010"),
        "8": ("111", "101", "111", "101", "111"), "9": ("111", "101", "111", "001", "111"),
        ".": ("000", "000", "000", "000", "010"), "-": ("000", "000", "111", "000", "000"),
        ":": ("000", "010", "000", "010", "000"), " ": ("000", "000", "000", "000", "000"),
        "f": ("011", "100", "110", "100", "100"), "g": ("111", "100", "101", "101", "111"),
        "r": ("110", "101", "110", "100", "100"), "k": ("101", "110", "100", "110", "101"),
        "N": ("101", "111", "111", "101", "101"), "z": ("111", "001", "010", "100", "111"),
        "=": ("000", "111", "000", "111", "000"), "t": ("010", "111", "010", "010", "011"),
        "e": ("111", "100", "111", "100", "111"), "p": ("111", "101", "111", "100", "100"),
        "s": ("111", "100", "111", "001", "111"), "d": ("110", "101", "101", "101", "110"),
        "o": ("111", "101", "101", "101", "111"), "/": ("001", "001", "010", "100", "100"),
    }
    out = frame.copy()
    scale, pad = 2, 6
    for row, text in enumerate(lines):
        y0 = pad + row * (7 * scale)
        for col, ch in enumerate(text):
            g = glyphs.get(ch)
            if g is None:
                continue
            x0 = pad + col * (4 * scale)
            for dy, bits in enumerate(g):
                for dx, bit in enumerate(bits):
                    if bit == "1":
                        ys = slice(y0 + dy * scale, y0 + (dy + 1) * scale)
                        xs = slice(x0 + dx * scale, x0 + (dx + 1) * scale)
                        out[ys, xs] = 255
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["expert", "hold", "random", "checkpoint"],
                    default="expert")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--vecnormalize", type=str, default=None)
    ap.add_argument("--out", type=str, default="handover.mp4")
    ap.add_argument("--azimuth", type=float, default=135.0)
    ap.add_argument("--elevation", type=float, default=-18.0)
    ap.add_argument("--distance", type=float, default=1.1)
    ap.add_argument("--start-distance", type=float, default=None,
                    help="fix the receiver's start distance; default samples the mix")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--every", type=int, default=2, help="record every Nth control step")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = EnvConfig()
    if args.start_distance is not None:
        cfg = EnvConfig(start_distance_mix=(args.start_distance,))
    env = HandoverEnv(cfg)

    policy = None
    if args.policy == "checkpoint":
        if not args.checkpoint:
            ap.error("--policy checkpoint needs --checkpoint")
        from stable_baselines3 import PPO

        policy = PPO.load(args.checkpoint, device="cpu")

    obs, _ = env.reset(seed=args.seed)
    set_env_flag(env, "reached_standoff", False)
    script = ScriptCfg()
    rng = np.random.default_rng(args.seed)

    renderer = mujoco.Renderer(env.model, args.height, args.width)
    camera = make_camera(env)
    camera.azimuth, camera.elevation, camera.distance = (
        args.azimuth, args.elevation, args.distance)
    frames = []

    for step in range(env.cfg.episode_steps):
        if args.policy == "expert":
            action = expert_action(env, step, script)
        elif args.policy == "hold":
            action = np.zeros(env.controller.action_dim)
        elif args.policy == "random":
            action = rng.uniform(-1, 1, env.controller.action_dim)
        else:
            action, _ = policy.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        if step % args.every == 0:
            renderer.update_scene(env.data, camera=camera)
            frames.append(
                annotate(
                    renderer.render(),
                    [
                        f"t:{step:04d}",
                        f"f:{info['load_fraction']:6.2f}",
                        f"g:{info['giver_grip']:6.2f}N",
                        f"r:{info['recv_grip']:6.2f}N",
                        f"z:{info['object_height']:5.3f}",
                    ],
                )
            )

        if terminated or truncated:
            break

    outcome = "SUCCESS" if info["success"] else ("DROPPED" if info["dropped"] else "timeout")
    print(f"{args.policy}: {step + 1} steps, {outcome}, "
          f"peak load fraction {info['load_fraction']:.2f}")

    if args.out.endswith(".gif"):
        imageio.mimsave(args.out, frames, duration=1.0 / args.fps, loop=0)
    else:
        imageio.mimsave(args.out, frames, fps=args.fps, macro_block_size=1)
    print(f"wrote {args.out}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
