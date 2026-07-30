"""Scripted Allegro grasp poses, used to drive the validation experiments.

No policy here -- these are hand-written open/closed postures interpolated by a
scalar, so an experiment can say "giver at 100% closed, receiver at 40%" and
sweep the transfer without any learning in the loop.
"""

from __future__ import annotations

import mujoco
import numpy as np

# Joint suffixes in Allegro naming: ff/mf/rf are the fingers, th the thumb.
FINGERS = ("ff", "mf", "rf")

# A closed posture that wraps a rod-like object. The thumb has to rotate across
# (thj0) to oppose the fingers, otherwise the grasp has no opposing force.
# Measured free-space opening between fingertips and thumb tip at this posture
# is 4.9 cm, the tightest the Allegro reaches. The object must be wider than
# that for the fingers to press into it and generate grip force at all.
CLOSED = {
    "ffj0": 0.0, "ffj1": 1.45, "ffj2": 1.45, "ffj3": 1.05,
    "mfj0": 0.0, "mfj1": 1.45, "mfj2": 1.45, "mfj3": 1.05,
    "rfj0": 0.0, "rfj1": 1.45, "rfj2": 1.45, "rfj3": 1.05,
    "thj0": 1.20, "thj1": 0.60, "thj2": 0.70, "thj3": 1.00,
}

# Grasp pocket centre in the palm frame at full closure, measured from the model
# (see scripts/probe_pocket.py). The y component mirrors with hand chirality.
POCKET_RIGHT = (-0.031, -0.016, 0.072)
POCKET_LEFT = (-0.031, 0.016, 0.072)
GRIP_OPENING = 0.049

OPEN = {name: 0.0 for name in CLOSED}
OPEN["thj1"] = 0.263  # thj1's range starts at 0.263, not 0


class HandController:
    """Drives one hand's 16 position actuators between the open and closed poses."""

    def __init__(self, model: mujoco.MjModel, prefix: str, kp: float | None = None):
        self.model = model
        self.prefix = prefix
        self.act_ids: dict[str, int] = {}

        for joint, _ in CLOSED.items():
            # Actuator names follow the joint names: ffj0 -> ffa0.
            act = f"{prefix}hand_{joint.replace('j', 'a')}"
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act)
            if aid < 0:
                raise KeyError(f"no actuator {act!r} -- check the hand prefix")
            self.act_ids[joint] = aid

        if kp is not None:
            self.set_gain(kp)

    def set_gain(self, kp: float) -> None:
        """Override position-actuator stiffness.

        Menagerie ships the standalone Allegro with kp=1, which is far too soft
        to hold a 200 g object; a real grasp needs a stiffer position loop.
        """
        for aid in self.act_ids.values():
            self.model.actuator_gainprm[aid, 0] = kp
            self.model.actuator_biasprm[aid, 1] = -kp

    def target(self, closure: float) -> dict[str, float]:
        """Joint targets at a given closure, 0 = fully open, 1 = fully closed."""
        c = float(np.clip(closure, 0.0, 1.0))
        return {j: OPEN[j] + c * (CLOSED[j] - OPEN[j]) for j in CLOSED}

    def apply(self, data: mujoco.MjData, closure: float) -> None:
        """Write the interpolated posture into the control vector."""
        for joint, value in self.target(closure).items():
            data.ctrl[self.act_ids[joint]] = value
