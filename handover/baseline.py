"""Reproduction of the Isaac Lab baseline's task, for apples-to-apples comparison.

This is deliberately NOT the load-transfer task. It mirrors the reference
environment so a result here can be set beside the published one, and so the
rest of the codebase has a floor that is known to train.

What is reproduced:

* A 3.5 x 3.5 x 35 cm prism weighing 0.25 g. The mass matters more than anything
  else on this list: at a quarter of a gram the object needs essentially no grip
  force, so grasping is almost free and none of the force-transfer difficulty
  exists.
* Only the receiver learns. The giver holds a fixed pose for the episode,
  randomized between episodes, and opens its hand once the receiver has taken
  hold -- the reference hard-codes exactly that release.
* Two-phase reward on dual-quaternion pose distance: approach the object, then
  carry it to a target pose. Success is the object reaching that target.
* A 3 s episode.

What differs, and why:

* Observations stay ours (46-dim, hardware-realizable). The reference feeds the
  policy a 17-dim privileged vector. Ours is a superset of the same information,
  so it does not make the task easier in a way that would flatter the comparison.
* Contact weighting is per-digit (thumb / fingers / palm) rather than the
  reference's 18-element hand-tuned matrix over named collision sensors, which
  does not transfer to a different simulator's contact model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Reference values, read from bimanual_direct_env_cfg.py.
REW_SCALE_HAND_OBJ = 1.0
REW_SCALE_OBJ_TARGET = 12.0
BONUS_OBJ_REACH = 300.0
EPISODE_SECONDS = 3.0


@dataclass
class BaselineReward:
    """Two-phase reward: reach the object, then bring it to the target."""

    hand_obj_scale: float = REW_SCALE_HAND_OBJ
    obj_target_scale: float = REW_SCALE_OBJ_TARGET
    bonus: float = BONUS_OBJ_REACH

    # Per-digit contact credit, standing in for the reference's contact matrix.
    thumb_weight: float = 0.65
    finger_weight: float = 0.65
    palm_weight: float = 0.15
    contact_scale: float = 0.02

    def __call__(
        self,
        hand_obj_dist: float,
        obj_target_dist: float,
        approaching: bool,
        palm_leading: bool,
        grasped: bool,
        just_grasped: bool,
        reached_target: bool,
        wrench,
    ) -> tuple[float, dict]:
        """Reproduces the reference's reward composition.

        `approaching` is its `mod` term (closing on the object) and
        `palm_leading` its `pre_mod` (coming in palm-first rather than
        back-of-hand-first); both gate the approach reward the same way.
        """
        direction = 1.0 if approaching else -1.0
        divisor = 1.0 + 2.0 * (0.0 if palm_leading else 1.0)
        approach = direction * self.hand_obj_scale * float(np.exp(-2.0 * hand_obj_dist)) / divisor

        carry = self.obj_target_scale * float(np.exp(-2.0 * obj_target_dist))

        total = approach * (not grasped) + 10.0 * carry * grasped
        total += self.bonus * float(just_grasped) / 2.0
        total += self.bonus * float(reached_target)

        contact = self.contact_scale * (
            self.thumb_weight * wrench.thumb_grip
            + self.finger_weight * wrench.finger_grip
            + self.palm_weight * wrench.palm_grip
        )
        total += contact

        return total, {
            "r_approach_phase": approach * (not grasped),
            "r_carry_phase": 10.0 * carry * grasped,
            "r_contact": contact,
            "r_grasp_bonus": self.bonus * float(just_grasped) / 2.0,
            "r_target_bonus": self.bonus * float(reached_target),
        }


def baseline_scene_overrides() -> dict:
    """SceneConfig fields that make the object match the reference prism."""
    return {
        "obj_shape": "box",
        "obj_radius": 0.0175,      # half-width of the 3.5 cm square section
        "obj_half_length": 0.175,  # half of 35 cm
        "obj_mass": 0.00025,       # 0.25 g, as in the reference
    }


def baseline_env_overrides() -> dict:
    """EnvConfig fields that switch the task over to the reference formulation."""
    return {
        "task_mode": "baseline",
        "episode_steps": 150,      # 3 s at 50 Hz
        "giver_mode": "scripted",  # only the receiver learns
        "randomize_object": False, # the reference varies pose, not the object
        "randomize_start": True,
        "start_distance_mix": (1.0,),
        # A near-weightless object needs almost no grip, so the giver's initial
        # grasp is trivially stable and the carry-fraction filter is meaningless.
        "grasp_carry_low": 0.0,
        "grasp_carry_high": 1e9,
    }
