"""Damped least-squares inverse kinematics for a named body on one arm.

This is the same differential-IK idea the Isaac environment used, with one
deliberate difference: a solver instance is bound to a specific set of joints,
so two arms cannot share one controller and clobber each other's commands.
"""

from __future__ import annotations

import mujoco
import numpy as np


class ArmIK:
    """Solves for joint angles that place `body` at a target pose.

    Bound to one arm's joints at construction. Operates on a scratch MjData so
    solving never disturbs the caller's simulation state.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        body: str,
        joint_names: list[str],
        damping: float = 1e-2,
        nullspace_gain: float = 0.0,
        rest_posture: np.ndarray | None = None,
    ):
        self.model = model
        self.damping = damping
        self.nullspace_gain = nullspace_gain
        self._rest_posture = rest_posture
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        if self.body_id < 0:
            raise KeyError(f"unknown body {body!r}")

        self.joint_ids = []
        for name in joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise KeyError(f"unknown joint {name!r}")
            self.joint_ids.append(jid)

        # Column indices into the full Jacobian, and rows into qpos.
        self.dof_ids = np.array([model.jnt_dofadr[j] for j in self.joint_ids])
        self.qpos_ids = np.array([model.jnt_qposadr[j] for j in self.joint_ids])

        self.lower = model.jnt_range[self.joint_ids, 0].copy()
        self.upper = model.jnt_range[self.joint_ids, 1].copy()
        unlimited = ~model.jnt_limited[self.joint_ids].astype(bool)
        self.lower[unlimited] = -np.inf
        self.upper[unlimited] = np.inf

        # Posture the redundant DOFs relax toward: the midpoint of each limited
        # joint's range. On a 7-DOF arm the task leaves a null space, and without
        # a preference the solver happily parks a joint hard against its stop --
        # the arm then cannot move in that direction at all, producing a dead
        # zone that looks like a control failure but is a solver choice.
        if self._rest_posture is None:
            # Continuous joints (the Gen3 has four) have infinite range, so only
            # bounded joints get a midpoint; the rest relax toward zero.
            bounded = np.isfinite(self.lower) & np.isfinite(self.upper)
            rest = np.zeros(len(self.joint_ids))
            rest[bounded] = 0.5 * (self.lower[bounded] + self.upper[bounded])
            self._rest_posture = rest
        self.rest_posture = np.asarray(self._rest_posture, dtype=float)

    def solve(
        self,
        data: mujoco.MjData,
        target_pos: np.ndarray,
        target_quat: np.ndarray | None = None,
        iters: int = 200,
        pos_tol: float = 1e-4,
        step: float = 0.5,
    ) -> tuple[np.ndarray, float]:
        """Return (joint angles, final position error) for the target pose.

        `data` is read for the starting guess and left unmodified.
        """
        scratch = mujoco.MjData(self.model)
        scratch.qpos[:] = data.qpos
        scratch.qvel[:] = 0.0

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        err = np.zeros(6 if target_quat is not None else 3)

        for _ in range(iters):
            mujoco.mj_kinematics(self.model, scratch)
            mujoco.mj_comPos(self.model, scratch)

            err[:3] = target_pos - scratch.xpos[self.body_id]

            if target_quat is not None:
                cur_quat = scratch.xquat[self.body_id]
                cur_inv = np.zeros(4)
                mujoco.mju_negQuat(cur_inv, cur_quat)
                dq = np.zeros(4)
                mujoco.mju_mulQuat(dq, target_quat, cur_inv)
                mujoco.mju_quat2Vel(err[3:], dq, 1.0)

            if np.linalg.norm(err[:3]) < pos_tol:
                break

            mujoco.mj_jacBody(self.model, scratch, jacp, jacr, self.body_id)
            jac = np.vstack([jacp, jacr])[: len(err)][:, self.dof_ids]

            # Damped least squares: dq = J^T (J J^T + lambda^2 I)^-1 e
            jjt = jac @ jac.T
            jjt += (self.damping**2) * np.eye(jjt.shape[0])
            pinv = jac.T @ np.linalg.inv(jjt)
            dq = pinv @ err

            # Project a pull toward the rest posture through the null space, so
            # it steers redundant DOFs off their limits without disturbing the
            # task-space solution. Only meaningful when the arm has more joints
            # than the task constrains -- on a non-redundant arm the damped
            # projector is not exactly zero and the term just fights the task.
            if self.nullspace_gain > 0.0 and len(self.dof_ids) > len(err):
                current = scratch.qpos[self.qpos_ids]
                bias = self.nullspace_gain * (self.rest_posture - current)
                dq = dq + (np.eye(len(self.dof_ids)) - pinv @ jac) @ bias

            q = scratch.qpos[self.qpos_ids] + step * dq
            scratch.qpos[self.qpos_ids] = np.clip(q, self.lower, self.upper)

        mujoco.mj_kinematics(self.model, scratch)
        final_err = float(np.linalg.norm(target_pos - scratch.xpos[self.body_id]))
        return scratch.qpos[self.qpos_ids].copy(), final_err
