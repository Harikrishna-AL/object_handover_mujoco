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
    ):
        self.model = model
        self.damping = damping
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
            dq = jac.T @ np.linalg.solve(jjt, err)

            q = scratch.qpos[self.qpos_ids] + step * dq
            scratch.qpos[self.qpos_ids] = np.clip(q, self.lower, self.upper)

        mujoco.mj_kinematics(self.model, scratch)
        final_err = float(np.linalg.norm(target_pos - scratch.xpos[self.body_id]))
        return scratch.qpos[self.qpos_ids].copy(), final_err
