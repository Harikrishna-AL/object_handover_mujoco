"""Per-arm Cartesian control: policy actions in, joint targets out.

Each arm gets its own `ArmController`, and each controller owns its own IK
solver. The Isaac environment shared a single DifferentialIKController between
both arms, which worked only because one arm was never actually commanded; the
moment both act in the same step the second `set_command` clobbers the first.
Making the solver a per-arm member removes that failure mode structurally rather
than relying on call ordering.

Action layout per arm (all components in [-1, 1]):

    [0:3]  translation delta of the palm, world frame
    [3:6]  rotation delta of the palm, world frame, as a scaled rotation vector
    [6:]   hand closure (see `hand_dof`)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

# Sides are keyed by the contact registry's labels ("giver"/"recv"), not by the
# scene's prefix strings ("giver_"/"recv_"). Keeping one vocabulary means a
# controller and a wrench reading can be looked up with the same key.
from .contacts import GIVER, RECV
from .grasp import CLOSED, OPEN, HandController
from .ik import ArmIK
from .scene import GIVER_ARM_JOINTS, RECV_ARM_JOINTS

# Which body each arm's Cartesian command refers to, and which joints move it.
ARM_JOINTS = {GIVER: GIVER_ARM_JOINTS, RECV: RECV_ARM_JOINTS}
PALM_BODY = {GIVER: f"{GIVER}_hand_palm", RECV: f"{RECV}_hand_palm"}

# Finger groups for the multi-DOF hand action.
FINGER_GROUPS = (("ff", "mf", "rf"), ("th",))

# How many velocity DOFs each joint type contributes.
_DOF_WIDTH = {
    mujoco.mjtJoint.mjJNT_FREE: 6,
    mujoco.mjtJoint.mjJNT_BALL: 3,
    mujoco.mjtJoint.mjJNT_SLIDE: 1,
    mujoco.mjtJoint.mjJNT_HINGE: 1,
}


def _dof_width(joint_type) -> int:
    return _DOF_WIDTH[mujoco.mjtJoint(joint_type)]


@dataclass
class ControlConfig:
    """Action scaling and solver effort for one control step."""

    # Per-step Cartesian limits. Small enough that a couple of IK iterations
    # converge, large enough to cross the workspace in a few seconds.
    translation_scale: float = 0.02
    rotation_scale: float = 0.15

    # Differential IK: the target is only a step away, so this warm-starts from
    # the current configuration and does not need to converge from scratch.
    ik_iters: int = 8
    ik_damping: float = 1e-2

    # Pulls redundant DOFs toward mid-range during control. Over a rollout the
    # 7-DOF receiver otherwise drifts into a joint limit and gets stuck there,
    # which reads as a control failure but is the solver picking a bad branch of
    # the redundancy. Applied only to arms that are actually redundant.
    ik_nullspace_gain: float = 0.2

    # 1 -> a single grip-closure scalar per hand. For load transfer the decision
    # that matters is "how hard am I holding", which is one-dimensional, and a
    # scalar keeps the exploration problem small. 2 -> fingers and thumb move
    # independently, which lets a policy modulate thumb opposition.
    #
    # This revises decision D2 (3-DOF aggregated, carried over from the Isaac
    # action mapping): that grouping was inherited from a reach-and-grasp task,
    # and nothing in the transfer problem needs three groups.
    hand_dof: int = 1

    # Grip is commanded as a RATE, not an absolute closure, so a zero action
    # means "hold the grip you have". With absolute closure the neutral action
    # maps to half-closed, which releases an established grasp on the very first
    # step -- every episode would begin by dropping the object, and the policy
    # would have to learn to hold on before it could learn anything else.
    # This also matches the Cartesian components, which are already deltas.
    closure_rate: float = 0.05

    # How far the integrated command may run ahead of where the palm actually
    # is. Without this leash a policy that pushes into a joint limit, or a hand
    # blocked by contact, accumulates an ever more distant target and then needs
    # seconds of opposite action to unwind it -- the command stops meaning
    # anything and the arm appears to ignore the policy. Contact blocking is the
    # normal case in a handover, so this is load-bearing, not a safety margin.
    target_leash: float = 0.05

    # Hard bounds on the commanded position, as (low, high) per axis. Keeps the
    # command inside the shared workspace of the two arms.
    workspace_low: tuple[float, float, float] = (0.20, -0.40, 0.15)
    workspace_high: tuple[float, float, float] = (0.95, 0.40, 1.00)


class ArmController:
    """Turns a normalised action into joint position targets for one arm."""

    def __init__(self, model: mujoco.MjModel, side: str, cfg: ControlConfig | None = None):
        if side not in ARM_JOINTS:
            raise KeyError(f"unknown side {side!r}; expected one of {list(ARM_JOINTS)}")

        self.model = model
        self.side = side
        self.cfg = cfg or ControlConfig()

        self.joint_names = ARM_JOINTS[side]
        self.palm_body = PALM_BODY[side]
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.palm_body)

        # This controller's own solver. Not shared.
        self.ik = ArmIK(
            model,
            self.palm_body,
            self.joint_names,
            damping=self.cfg.ik_damping,
            nullspace_gain=self.cfg.ik_nullspace_gain,
        )

        # Map each arm joint to the actuator that drives it. Resolving through
        # actuator_trnid rather than by name keeps this working across the two
        # arms, whose Menagerie models use different actuator naming schemes.
        self.arm_act_ids = []
        for name in self.joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            matches = [a for a in range(model.nu) if model.actuator_trnid[a, 0] == jid]
            if not matches:
                raise KeyError(f"joint {name!r} has no actuator driving it")
            self.arm_act_ids.append(matches[0])

        self.hand = HandController(model, f"{side}_")
        self._target_pos = np.zeros(3)
        self._target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        # Last commanded grip, in [0, 1]. Proprioceptive and available on real
        # hardware, unlike the contact forces it produces.
        self._closure_cmd = 0.0

    @property
    def action_dim(self) -> int:
        return 6 + self.cfg.hand_dof

    def reset(self, data: mujoco.MjData, closure: float = 0.0) -> None:
        """Latch the current palm pose as the command target.

        Also adopts the current joint configuration as the IK's rest posture.
        Biasing toward the range midpoints instead is actively wrong for the
        Gen3: four of its joints are continuous, so their "midpoint" is zero, and
        the bounded ones are centred at zero too -- an all-zero rest posture sits
        nowhere near any configuration the arm actually works in, and the null
        space pull drags the arm away from the task.
        """
        mujoco.mj_forward(self.model, data)
        self._target_pos = data.xpos[self.body_id].copy()
        self._target_quat = data.xquat[self.body_id].copy()
        self.ik.rest_posture = data.qpos[self.ik.qpos_ids].copy()
        self._closure_cmd = float(np.clip(closure, 0.0, 1.0))

    @property
    def hand_closure_command(self) -> float:
        """The grip closure most recently commanded, in [0, 1]."""
        return self._closure_cmd

    def palm_pose(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        return data.xpos[self.body_id].copy(), data.xquat[self.body_id].copy()

    def _integrate_target(self, data: mujoco.MjData, action: np.ndarray) -> None:
        """Advance the held target pose by the action's Cartesian delta.

        The target integrates rather than being re-derived from the measured pose
        each step: re-deriving would let contact push the command around, so a
        blocked hand would silently stop tracking what the policy asked for.

        It is then clamped to the workspace and leashed to the achieved pose, so
        the command can never run away from what the arm can actually do.
        """
        target = self._target_pos + action[:3] * self.cfg.translation_scale
        target = np.clip(target, self.cfg.workspace_low, self.cfg.workspace_high)

        actual = data.xpos[self.body_id]
        offset = target - actual
        distance = float(np.linalg.norm(offset))
        if distance > self.cfg.target_leash:
            target = actual + offset * (self.cfg.target_leash / distance)

        self._target_pos = target

        rotvec = action[3:6] * self.cfg.rotation_scale
        angle = float(np.linalg.norm(rotvec))
        if angle > 1e-9:
            delta = np.zeros(4)
            mujoco.mju_axisAngle2Quat(delta, rotvec / angle, angle)
            updated = np.zeros(4)
            mujoco.mju_mulQuat(updated, delta, self._target_quat)
            mujoco.mju_normalize4(updated)
            self._target_quat = updated

    def _hand_closure(self, action: np.ndarray) -> dict[str, float]:
        """Map the hand portion of the action onto joint targets.

        Actions arrive in [-1, 1] and closure lives in [0, 1], so this rescales
        rather than clipping -- clipping would make half the action range dead.
        """
        hand_action = np.clip(action[6:], -1.0, 1.0)
        self._closure_cmd = float(
            np.clip(self._closure_cmd + hand_action[0] * self.cfg.closure_rate, 0.0, 1.0)
        )
        closures = np.full(max(1, self.cfg.hand_dof), self._closure_cmd)
        if self.cfg.hand_dof > 1:
            closures = np.clip(closures + hand_action[1:] * self.cfg.closure_rate, 0.0, 1.0)

        if self.cfg.hand_dof == 1:
            return self.hand.target(float(closures[0]))

        targets = {}
        for group, closure in zip(FINGER_GROUPS, closures):
            for finger in group:
                for joint in (f"{finger}j{i}" for i in range(4)):
                    targets[joint] = OPEN[joint] + closure * (CLOSED[joint] - OPEN[joint])
        return targets

    def apply(self, data: mujoco.MjData, action: np.ndarray) -> None:
        """Write joint position targets for this arm's action."""
        action = np.asarray(action, dtype=float).reshape(-1)
        if action.shape[0] != self.action_dim:
            raise ValueError(
                f"{self.side} expected action of {self.action_dim}, got {action.shape[0]}"
            )
        action = np.clip(action, -1.0, 1.0)

        self._integrate_target(data, action)

        angles, _ = self.ik.solve(
            data,
            self._target_pos,
            target_quat=self._target_quat,
            iters=self.cfg.ik_iters,
            pos_tol=0.0,  # always run the full iteration budget
        )
        for act_id, angle in zip(self.arm_act_ids, angles):
            data.ctrl[act_id] = angle

        for joint, value in self._hand_closure(action).items():
            data.ctrl[self.hand.act_ids[joint]] = value

    def tracking_error(self, data: mujoco.MjData) -> float:
        """Distance from the achieved palm position to the commanded one."""
        return float(np.linalg.norm(data.xpos[self.body_id] - self._target_pos))


class BimanualController:
    """Both arms, each with its own solver; actions concatenate giver then receiver."""

    def __init__(self, model: mujoco.MjModel, cfg: ControlConfig | None = None):
        cfg = cfg or ControlConfig()
        self.model = model
        self.cfg = cfg
        self.arms = {side: ArmController(model, side, cfg) for side in (GIVER, RECV)}

        # Every DOF belonging to a robot, for gravity compensation. The object's
        # free joint is deliberately excluded -- it has to keep falling.
        self.robot_dofs = np.array(
            [
                model.jnt_dofadr[j] + k
                for j in range(model.njnt)
                for k in range(_dof_width(model.jnt_type[j]))
                if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "").startswith(
                    (f"{GIVER}_", f"{RECV}_")
                )
            ],
            dtype=int,
        )

    def compensate_gravity(self, data: mujoco.MjData) -> None:
        """Cancel gravity and Coriolis terms on the robot DOFs.

        Without this the Gen3's position servos droop by up to 2 cm under the
        arm's own weight, and the droop scales with extension -- an error no
        amount of IK effort can remove, because the IK solution is correct and
        the actuator simply is not reaching it. Both of these arms run gravity
        compensation in their real controllers, so modelling it is the accurate
        choice rather than a convenience.

        Must be called every physics step: the bias term is configuration
        dependent, so a value latched once per control step goes stale.
        """
        data.qfrc_applied[self.robot_dofs] = data.qfrc_bias[self.robot_dofs]

    @property
    def action_dim(self) -> int:
        return sum(arm.action_dim for arm in self.arms.values())

    def reset(self, data: mujoco.MjData, closures: dict[str, float] | None = None) -> None:
        closures = closures or {}
        for side, arm in self.arms.items():
            arm.reset(data, closure=closures.get(side, 0.0))

    def split(self, action: np.ndarray) -> dict[str, np.ndarray]:
        """Slice a joint action into per-arm actions."""
        action = np.asarray(action, dtype=float).reshape(-1)
        out, start = {}, 0
        for side in (GIVER, RECV):
            width = self.arms[side].action_dim
            out[side] = action[start : start + width]
            start += width
        return out

    def apply(self, data: mujoco.MjData, action: np.ndarray) -> None:
        for side, part in self.split(action).items():
            self.arms[side].apply(data, part)

    def tracking_errors(self, data: mujoco.MjData) -> dict[str, float]:
        return {side: arm.tracking_error(data) for side, arm in self.arms.items()}
