"""Contact bookkeeping and the grip/load force decomposition.

Two things live here, and they exist because of specific failure modes:

1. `ContactRegistry` resolves every geom to a named owner once, at construction.
   Nothing downstream indexes contacts positionally. Positional slices into a
   sensor list are silent-failure bait the moment the scene gains a body.

2. `hand_wrenches` separates two quantities that are easy to conflate:

     grip force = sum of contact force MAGNITUDES
         What a hand squeezes with. Two fingers pressing 10 N on opposite faces
         gives 20 N. This is the right input to a crush penalty.

     load force = MAGNITUDE of the summed contact force vectors
         What a hand actually carries. Those same two fingers give ~0 N. This is
         the only one that can answer "who is holding the object up".

   A hand can squeeze hard and carry nothing, so a reward built on the first
   while meaning the second would be measuring the wrong thing entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

GIVER = "giver"
RECV = "recv"
OBJECT = "object"
WORLD = "world"


@dataclass
class HandWrench:
    """Contact summary for one hand against the object."""

    grip: float = 0.0
    """Sum of contact-force magnitudes -- how hard the hand squeezes."""

    load: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Net contact force on the object from this hand, in world coordinates."""

    torque: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Net moment about the object's centre of mass, in world coordinates."""

    n_contacts: int = 0

    @property
    def load_vertical(self) -> float:
        """Upward component of the net force -- the share of weight carried."""
        return float(self.load[2])


class ContactRegistry:
    """Resolves geoms to owners by name once, so nothing downstream guesses."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.geom_owner: list[str] = []

        for gid in range(model.ngeom):
            body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid])
            body = body or ""
            if body == OBJECT:
                self.geom_owner.append(OBJECT)
            elif body.startswith(f"{GIVER}_"):
                self.geom_owner.append(GIVER)
            elif body.startswith(f"{RECV}_"):
                self.geom_owner.append(RECV)
            else:
                self.geom_owner.append(WORLD)

        self.object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, OBJECT)
        if self.object_body_id < 0:
            raise KeyError("scene has no body named 'object'")

        counts = {k: self.geom_owner.count(k) for k in (GIVER, RECV, OBJECT, WORLD)}
        if counts[GIVER] == 0 or counts[RECV] == 0:
            raise ValueError(f"registry found no geoms for one of the hands: {counts}")
        self.counts = counts

    def hand_wrenches(self, data: mujoco.MjData) -> dict[str, HandWrench]:
        """Per-hand grip and load, summed over that hand's contacts with the object.

        Forces are expressed as acting ON the object, so they sum with gravity in
        Newton's balance.
        """
        out = {GIVER: HandWrench(), RECV: HandWrench()}
        obj_com = data.xipos[self.object_body_id]
        buf = np.zeros(6)

        for i in range(data.ncon):
            con = data.contact[i]
            own1 = self.geom_owner[con.geom1]
            own2 = self.geom_owner[con.geom2]

            # Only hand-object contacts carry load.
            if own1 == OBJECT and own2 in out:
                hand, object_is_geom1 = own2, True
            elif own2 == OBJECT and own1 in out:
                hand, object_is_geom1 = own1, False
            else:
                continue

            mujoco.mj_contactForce(self.model, data, i, buf)

            # buf[:3] is in the contact frame: [normal, tangent1, tangent2].
            # frame rows are the contact axes, so frame.T maps contact -> world.
            frame = con.frame.reshape(3, 3)
            force_world = frame.T @ buf[:3]

            # MuJoCo reports the force acting on geom2's body from geom1's body.
            # When the object is geom1, flip the sign to get the force on it.
            if object_is_geom1:
                force_world = -force_world

            wrench = out[hand]
            wrench.grip += float(np.linalg.norm(buf[:3]))
            wrench.load += force_world
            wrench.torque += np.cross(con.pos - obj_com, force_world)
            wrench.n_contacts += 1

        return out

    def other_object_force(self, data: mujoco.MjData) -> np.ndarray:
        """Net force on the object from anything that is not a hand (e.g. floor).

        Newton's balance must account for these or the residual is meaningless.
        """
        total = np.zeros(3)
        buf = np.zeros(6)

        for i in range(data.ncon):
            con = data.contact[i]
            own1 = self.geom_owner[con.geom1]
            own2 = self.geom_owner[con.geom2]

            if own1 == OBJECT and own2 == WORLD:
                object_is_geom1 = True
            elif own2 == OBJECT and own1 == WORLD:
                object_is_geom1 = False
            else:
                continue

            mujoco.mj_contactForce(self.model, data, i, buf)
            force_world = con.frame.reshape(3, 3).T @ buf[:3]
            total += -force_world if object_is_geom1 else force_world

        return total


def load_fraction(wrenches: dict[str, HandWrench], weight: float) -> float:
    """Share of the object's weight borne by the receiver.

    0 means the giver carries everything, 1 means the receiver does. This is the
    quantity a handover has to move smoothly from 0 to 1.

    Defined against the weight rather than against the two hands' sum. In statics
    the vertical components satisfy F_giver + F_receiver = mg exactly, so the two
    definitions agree -- but dividing by the sum degenerates whenever a hand pulls
    downward (a normal occurrence mid-transfer, and one that made an earlier
    version of this report a constant zero). Values outside [0, 1] are returned
    as they are: they mean one hand is actively pulling down, which is a real
    state worth seeing rather than clipping away.
    """
    return float(wrenches[RECV].load_vertical / weight)
