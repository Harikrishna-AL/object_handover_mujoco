"""The handover environment: force-mediated load transfer between two arms.

The episode starts with the giver already holding the object and the receiver
clear of it. Success is the receiver carrying the object's weight on its own,
with the giver fully released, held for long enough to be a grasp rather than a
moment of contact.

Two things about the design are deliberate and load-bearing:

* **Load fraction replaces a binary contact latch.** The Isaac environment's
  `obj_reached` fired off a contact threshold and never unfired, which is why a
  scripted release had to exist at all. `f = F_receiver,z / mg` is continuous,
  interpretable, and moves smoothly through the transfer, so the reward can
  shape the whole exchange rather than a single instant.

* **The actor never observes contact force.** Simulation hands us exact contact
  forces; the real Allegro has no fingertip sensing at all. Forces therefore
  appear in the reward and in the privileged critic state, both of which exist
  only at training time, and never in the actor's observation. Getting this
  wrong produces a policy that trains beautifully and cannot be deployed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .contacts import GIVER, RECV, ContactRegistry, load_fraction
from .control import BimanualController, ControlConfig
from .grasp import POCKET_BODY_LEFT, POCKET_BODY_RIGHT
from .scene import (
    SceneConfig,
    apply_start_pose,
    build_model,
    palm_pos_for_pocket,
)

G = 9.81


@dataclass
class EnvConfig:
    """Episode structure, reward weights, and success criteria."""

    # 8 s at 50 Hz. A handover needs approach, dual grasp, release and retreat;
    # the 3 s the Isaac task allowed is not enough for the full sequence.
    episode_steps: int = 400
    decimation: int = 10

    # --- how the giver's initial grasp is established ---
    # Must be full closure: the Allegro opening at 0.85 is 8.2 cm, wider than the
    # 6 cm object, so the fingers shut on air. The measured pocket offset is also
    # only valid at closure 1.0 -- it drifts by ~1.5 cm across the range.
    giver_grip_closure: float = 1.0
    settle_pinned_steps: int = 400
    settle_free_steps: int = 300

    # --- success ---
    # The receiver must carry essentially all the weight, the giver must be
    # genuinely off the object, and it has to persist -- a threshold crossed for
    # one step is a collision, not a grasp.
    success_load_fraction: float = 0.85
    success_giver_force: float = 0.15
    success_hold_steps: int = 25

    # --- failure ---
    drop_height: float = 0.25

    # --- reward weights ---
    w_progress: float = 12.0
    w_success: float = 50.0
    w_drop: float = 20.0
    w_approach: float = 1.5
    w_object_motion: float = 0.4
    w_excess_force: float = 0.5
    w_deadlock: float = 0.02

    # Grip force above which we start calling it crushing, in newtons. A full
    # closure on this object already sits near 33 N, so a lower threshold would
    # charge the giver for holding on normally. Measured
    # regime from the kill test: a sound grasp on this object sits at 5-15 N.
    grip_safe: float = 45.0

    # Both hands holding is the point of a handover, but only briefly; this
    # starts charging for it once the receiver has clearly taken the load.
    deadlock_after_fraction: float = 0.5


class HandoverEnv(gym.Env):
    """Single-agent (joint-policy) view of the two-arm handover.

    Both arms are driven by one action vector. This is the baseline the MARL
    version has to beat -- without it there is no way to say what decentralising
    the policy actually cost.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: EnvConfig | None = None,
        scene_cfg: SceneConfig | None = None,
        control_cfg: ControlConfig | None = None,
        seed: int | None = None,
    ):
        self.cfg = cfg or EnvConfig()
        self.scene_cfg = scene_cfg or SceneConfig()
        self.control_cfg = control_cfg or ControlConfig()

        self.model, _ = build_model(self.scene_cfg)
        self.data = mujoco.MjData(self.model)
        self.registry = ContactRegistry(self.model)
        self.controller = BimanualController(self.model, self.control_cfg)

        self.weight = self.scene_cfg.obj_mass * G
        self.object_joint = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_free"
        )
        self.object_qpos = self.model.jnt_qposadr[self.object_joint]
        self.object_dof = self.model.jnt_dofadr[self.object_joint]

        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._hold_count = 0
        self._prev_fraction = 0.0

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.controller.action_dim,), dtype=np.float32
        )
        obs = self._observe()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )
        self.state_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self._privileged_state().shape, dtype=np.float32
        )

    # ------------------------------------------------------------------ setup

    def _object_pose(self) -> tuple[np.ndarray, np.ndarray]:
        pos = self.data.qpos[self.object_qpos : self.object_qpos + 3].copy()
        quat = self.data.qpos[self.object_qpos + 3 : self.object_qpos + 7].copy()
        return pos, quat

    def _set_object_pose(self, pos: np.ndarray, quat: np.ndarray) -> None:
        self.data.qpos[self.object_qpos : self.object_qpos + 3] = pos
        self.data.qpos[self.object_qpos + 3 : self.object_qpos + 7] = quat
        self.data.qvel[self.object_dof : self.object_dof + 6] = 0.0

    def _grip_point(self, side: str) -> np.ndarray:
        """World position of a hand's grasp pocket, given its current palm pose."""
        arm = self.controller.arms[side]
        pos, quat = arm.palm_pose(self.data)
        pocket = POCKET_BODY_RIGHT if side == GIVER else POCKET_BODY_LEFT
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, quat)
        return pos + mat.reshape(3, 3) @ np.asarray(pocket, dtype=float)

    def _establish_giver_grasp(self) -> None:
        """Put the object in the giver's hand and close on it.

        The episode has to begin from a real grasp: spawning the object in free
        space and hoping the giver catches it would make every episode start
        with a different, mostly failed, precondition.
        """
        target = self._grip_point(GIVER)
        _, obj_quat = self._object_pose()
        obj_quat = np.asarray(self.scene_cfg.obj_quat, dtype=float)

        # Drive the grip rate hard positive for the giver and hard negative for
        # the receiver; the rate limiter saturates them at 1 and 0 respectively.
        parts = self.controller.split(np.zeros(self.controller.action_dim))
        parts[GIVER][6] = 1.0
        parts[RECV][6] = -1.0
        action = np.concatenate([parts[GIVER], parts[RECV]])

        # Pinned while the fingers wrap, so they cannot bat it away first.
        for _ in range(self.cfg.settle_pinned_steps):
            self.controller.apply(self.data, action)
            self._set_object_pose(target, obj_quat)
            self.controller.compensate_gravity(self.data)
            mujoco.mj_step(self.model, self.data)
            target = self._grip_point(GIVER)

        for _ in range(self.cfg.settle_free_steps):
            self.controller.apply(self.data, action)
            self.controller.compensate_gravity(self.data)
            mujoco.mj_step(self.model, self.data)

    # ----------------------------------------------------------- observations

    def _observe(self) -> np.ndarray:
        """Actor observation. Deliberately contains no contact force.

        Everything here is available on hardware: joint encoders, forward
        kinematics, and an external estimate of the object's pose.
        """
        obj_pos, obj_quat = self._object_pose()
        obj_vel = self.data.qvel[self.object_dof : self.object_dof + 3].copy()

        parts = [obj_pos, obj_quat, obj_vel]
        for side in (GIVER, RECV):
            arm = self.controller.arms[side]
            palm_pos, palm_quat = arm.palm_pose(self.data)
            joints = self.data.qpos[arm.ik.qpos_ids].copy()
            parts += [
                palm_pos,
                palm_quat,
                joints,
                # Where this hand would grip, relative to the object: the single
                # most useful geometric cue for closing the gap.
                self._grip_point(side) - obj_pos,
                [arm.hand_closure_command],
            ]
        parts.append([self._step_count / self.cfg.episode_steps])
        return np.concatenate([np.asarray(p, dtype=np.float64).ravel() for p in parts]).astype(
            np.float32
        )

    def _privileged_state(self) -> np.ndarray:
        """Critic state: the actor observation plus everything only sim knows."""
        wrenches = self.registry.hand_wrenches(self.data)
        extra = [
            [load_fraction(wrenches, self.weight)],
            [wrenches[GIVER].grip, wrenches[RECV].grip],
            wrenches[GIVER].load,
            wrenches[RECV].load,
            [float(wrenches[GIVER].n_contacts), float(wrenches[RECV].n_contacts)],
        ]
        return np.concatenate(
            [self._observe()] + [np.asarray(e, dtype=np.float64).ravel() for e in extra]
        ).astype(np.float32)

    # ---------------------------------------------------------------- reward

    def _reward(self, wrenches, fraction: float) -> tuple[float, dict]:
        cfg = self.cfg
        obj_pos, _ = self._object_pose()

        # Team term: progress along the transfer. Rewarding the increase rather
        # than the level keeps the policy from parking at a comfortable split.
        progress = float(np.clip(fraction, 0.0, 1.0) - np.clip(self._prev_fraction, 0.0, 1.0))
        r_progress = cfg.w_progress * progress

        # Per-agent shaping. The receiver is paid to close the gap between its
        # grasp pocket and the object; the giver is paid to hold it still while
        # that happens.
        approach = float(np.linalg.norm(self._grip_point(RECV) - obj_pos))
        r_approach = -cfg.w_approach * approach

        obj_speed = float(np.linalg.norm(self.data.qvel[self.object_dof : self.object_dof + 3]))
        r_motion = -cfg.w_object_motion * obj_speed

        # Crushing. Uses grip (the magnitude sum), which is the correct quantity
        # for squeeze -- load would be near zero for a hand that is crushing but
        # not lifting.
        excess = sum(max(0.0, w.grip - cfg.grip_safe) for w in wrenches.values())
        r_force = -cfg.w_excess_force * excess

        # Both hands holding indefinitely is the tug-of-war failure, so it costs
        # something once the receiver has clearly taken the load.
        both = wrenches[GIVER].n_contacts > 0 and wrenches[RECV].n_contacts > 0
        r_deadlock = (
            -cfg.w_deadlock if (both and fraction > cfg.deadlock_after_fraction) else 0.0
        )

        total = r_progress + r_approach + r_motion + r_force + r_deadlock
        return total, {
            "r_progress": r_progress,
            "r_approach": r_approach,
            "r_motion": r_motion,
            "r_force": r_force,
            "r_deadlock": r_deadlock,
        }

    def _success(self, wrenches, fraction: float) -> bool:
        obj_pos, _ = self._object_pose()
        return bool(
            fraction >= self.cfg.success_load_fraction
            and abs(wrenches[GIVER].load_vertical) <= self.cfg.success_giver_force
            and obj_pos[2] > self.cfg.drop_height
        )

    # ------------------------------------------------------------- gym API

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        apply_start_pose(self.model, self.data, self.scene_cfg)
        self.controller.reset(self.data)
        self._establish_giver_grasp()
        # Re-latch command targets: the settle moved the palms slightly. The
        # giver's grip command is carried over rather than zeroed, so the first
        # policy action modulates an existing grasp instead of inheriting one.
        self.controller.reset(
            self.data, closures={GIVER: self.cfg.giver_grip_closure, RECV: 0.0}
        )

        self._step_count = 0
        self._hold_count = 0
        wrenches = self.registry.hand_wrenches(self.data)
        self._prev_fraction = load_fraction(wrenches, self.weight)

        return self._observe(), {"state": self._privileged_state()}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        self.controller.apply(self.data, action)
        for _ in range(self.cfg.decimation):
            self.controller.compensate_gravity(self.data)
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        wrenches = self.registry.hand_wrenches(self.data)
        fraction = load_fraction(wrenches, self.weight)
        reward, terms = self._reward(wrenches, fraction)

        obj_pos, _ = self._object_pose()
        dropped = bool(obj_pos[2] < self.cfg.drop_height)

        if self._success(wrenches, fraction):
            self._hold_count += 1
        else:
            self._hold_count = 0
        held = self._hold_count >= self.cfg.success_hold_steps

        if held:
            reward += self.cfg.w_success
        if dropped:
            reward -= self.cfg.w_drop

        terminated = bool(held or dropped)
        truncated = bool(self._step_count >= self.cfg.episode_steps)

        self._prev_fraction = fraction

        info = {
            "state": self._privileged_state(),
            "load_fraction": fraction,
            "giver_grip": wrenches[GIVER].grip,
            "recv_grip": wrenches[RECV].grip,
            "giver_load_z": wrenches[GIVER].load_vertical,
            "recv_load_z": wrenches[RECV].load_vertical,
            "object_height": float(obj_pos[2]),
            "success": bool(held),
            "dropped": dropped,
            "hold_count": self._hold_count,
            **terms,
        }
        return self._observe(), float(reward), terminated, truncated, info
