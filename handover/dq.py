"""Dual-quaternion pose distance, ported from the baseline's PyTorch version.

The baseline paper's headline result is that a dual-quaternion pose metric
converges the hand's orientation onto the target where other representations
did not, so the metric is reproduced here rather than replaced by a Euclidean
distance -- keeping the new work continuous with that claim, and letting the two
be compared directly (see `EnvConfig.approach_metric`).

The distance combines translation and rotation into one scalar: the difference
of two dual quaternions is the identity when the poses coincide, so its norm
away from the identity measures how far apart the frames are.

Two conventions are inherited deliberately from the original:

* The dual part is built as ``0.5 * q (x) t`` rather than the more common
  ``0.5 * t (x) q``. The original carries a TODO questioning this exact line.
  It is kept so numbers stay comparable with the baseline; the two agree at
  zero error, so it does not affect what the metric converges to, only the
  scaling along the way.
* The original's ``q_normalize`` returns its input unchanged, so quaternions are
  never actually renormalized. Reproduced, since MuJoCo hands us unit
  quaternions anyway and renormalizing would silently change the scale.
"""

from __future__ import annotations

import numpy as np


def q_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 (x) q2, both (..., 4) in (w, x, y, z)."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def q_conjugate(q: np.ndarray) -> np.ndarray:
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def pose_to_dq(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Pack a pose into a dual quaternion (8,) as [real(4), dual(4)]."""
    pos = np.asarray(pos, dtype=float)
    quat = np.asarray(quat, dtype=float)
    t_quat = np.concatenate([[0.0], pos])
    return np.concatenate([quat, 0.5 * q_mul(quat, t_quat)])


def dq_mul(dq1: np.ndarray, dq2: np.ndarray) -> np.ndarray:
    r1, d1 = dq1[:4], dq1[4:]
    r2, d2 = dq2[:4], dq2[4:]
    return np.concatenate([q_mul(r1, r2), q_mul(r1, d2) + q_mul(d1, r2)])


def dq_quaternion_conjugate(dq: np.ndarray) -> np.ndarray:
    return dq * np.array([1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0])


def dq_distance(dq_pred: np.ndarray, dq_real: np.ndarray) -> tuple[float, float, float]:
    """Return (total, translation, rotation) distance between two poses.

    Mirrors the baseline: multiply one pose by the other's conjugate, subtract
    the identity from the real scalar, and take the norms of the two halves.
    """
    res = dq_mul(dq_real, dq_quaternion_conjugate(dq_pred))
    res = res.copy()
    res[0] = abs(res[0]) - 1.0

    translation = float(np.linalg.norm(res[4:]))
    rotation = float(np.linalg.norm(res[:4]))
    return translation + rotation, translation, rotation


def pose_distance(
    pos1: np.ndarray, quat1: np.ndarray, pos2: np.ndarray, quat2: np.ndarray
) -> tuple[float, float, float]:
    """Dual-quaternion distance between two (position, quaternion) poses."""
    return dq_distance(pose_to_dq(pos1, quat1), pose_to_dq(pos2, quat2))
