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

from dataclasses import dataclass, field, replace

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .contacts import GIVER, RECV, ContactRegistry, load_fraction
from .dq import pose_distance
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
class DomainRandomization:
    """Per-episode physics jitter, for sim-to-real robustness.

    A load-transfer policy is unusually sensitive to exactly these quantities:
    the learned behaviour is a force exchange, so friction, mass and contact
    compliance are first-order rather than nuisance parameters. Wired in from
    the start because retrofitting it means retraining.
    """

    enabled: bool = True
    mass_range: tuple[float, float] = (0.7, 1.4)
    friction_range: tuple[float, float] = (0.7, 1.3)
    # Contact softness: solref[0] is the time constant, larger being softer.
    solref_range: tuple[float, float] = (0.8, 1.3)
    # Finger stiffness stands in for tendon slack and actuator variation.
    hand_gain_range: tuple[float, float] = (0.7, 1.4)


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
    # Potential-based shaping telescopes to w_approach * (total distance closed),
    # so this is the whole budget for discovering the approach -- at 1.5 that was
    # 0.45 for the entire reach, invisible next to a 50-point success bonus. Only
    # the sum matters, and the potential form means raising it cannot reintroduce
    # a survival penalty.
    w_approach: float = 20.0

    # "dq"        : dual-quaternion pose distance, combining translation and
    #               orientation. Reproduces the baseline paper's headline
    #               contribution and is the default.
    # "euclidean" : position only, ignoring how the hand is oriented when it
    #               arrives. Kept so the two can be compared directly, which is
    #               the comparison the baseline paper makes.
    approach_metric: str = "dq"

    # ---- optional reward terms, all OFF by default ----
    #
    # Same structure as the Isaac environment: with no flags set the reward is
    # exactly the baseline, and each addition is attributable to one switch.
    #
    # NOTE ON CONSTANTS. The Isaac values (F_ref=0.15, F_safe=0.35,
    # F_threshold=0.10) were measured against that environment's summed filtered
    # contact forces. MuJoCo reports real newtons -- grips here run 9-33 N -- so
    # those numbers are ~100x too small and are NOT carried over. The values
    # below are set from the measured regime in this scene.
    use_motion_penalty: bool = False
    w_object_motion: float = 0.4

    use_deadlock_penalty: bool = False
    w_deadlock: float = 0.02

    # Velocity penalty near the object (Isaac: --vel_rew).
    vel_rew: bool = False
    v_min: float = 0.1
    lambda_vel: float = 5.0
    k_decay: float = 13.0

    # Force-based signals (Isaac: --use_force_rewards and the three signals).
    use_force_rewards: bool = False
    use_signal_1: bool = False
    use_signal_2: bool = False
    use_signal_instability: bool = False

    # Force constants, in newtons, from this scene's measured regime:
    # a sound grasp sits near 9 N (receiver) to 33 N (giver at full closure).
    F_ref: float = 10.0
    F_safe: float = 45.0
    F_threshold: float = 2.0

    lambda_firmness: float = 0.05
    lambda_balance: float = 0.1
    lambda_instability: float = 0.01
    lambda_force_excess: float = 0.5
    palm_weight: float = 0.3
    # Discount used inside the shaping term. Deliberately 1.0, not the learner's
    # 0.99: with gamma*phi(s') - phi(s) and a negative potential, the stationary
    # term is (gamma-1)*phi, which is POSITIVE and pays an agent ~0.06 per step
    # to sit still far from the object -- worth +22 over an episode, more than a
    # successful handover earned. At 1.0 the term telescopes exactly to
    # phi(end) - phi(start), so standing still is worth zero and only real
    # progress pays.
    shaping_gamma: float = 1.0

    # Both hands holding is the point of a handover, but only briefly; this
    # starts charging for it once the receiver has clearly taken the load.
    deadlock_after_fraction: float = 0.5

    # --- curriculum ---
    # "policy"   : the giver is controlled by the policy. The final task.
    # "scripted" : the giver holds, then opens once the receiver has taken hold.
    #
    # Stage 1 needs "scripted". A fresh policy random-walks the giver's grip to
    # zero within ~30 steps, the object drops, the episode ends, and the receiver
    # never sees enough of the task to learn the approach. This is the same
    # scripted release the Isaac environment hard-coded -- the difference is that
    # here it is an explicit, temporary stage that stage 2 removes, rather than a
    # permanent fixture standing in for the decision we actually want learned.
    giver_mode: str = "policy"

    # Receiver grip force at which the scripted giver begins to let go.
    scripted_release_grip: float = 5.0

    # Fractions of the way from the grasp pose to the nominal start that the
    # receiver may begin an episode at. Starting sometimes close is what makes
    # the task discoverable: from the full distance the policy has to execute a
    # long directed reach before it ever sees a grasp, so it converges instead on
    # standing still, which is safe and scores about 22. A snapshot is cached per
    # entry, so this costs a one-off build rather than per-episode work.
    start_distance_mix: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

    # Pre-grasp standoff along the palm's -z. Closest start sits here rather
    # than on the object itself.
    # 0.0 reproduces the measured-best configuration. Raising it moves the
    # closest start off the object, which is geometrically tidier but measured
    # WORSE end to end (0/5 expert successes vs 2/5), so it is left off pending
    # a proper sweep rather than adopted on the strength of the argument.
    pregrasp_standoff: float = 0.0

    # Whether to put the privileged critic state in the info dict. Off for the
    # single-agent baseline, which does not use it: the array is shipped across
    # a process boundary on every step of every worker, and paying that for
    # something nobody reads is a pointless throughput tax. MAPPO turns it on.
    include_state: bool = False


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
        randomization: DomainRandomization | None = None,
        seed: int | None = None,
    ):
        self.cfg = cfg or EnvConfig()
        self.scene_cfg = scene_cfg or SceneConfig()
        self.control_cfg = control_cfg or ControlConfig()
        self.randomization = randomization or DomainRandomization()

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

        self.object_body = self.registry.object_body_id
        self.object_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
        self._hand_act_ids = np.array(
            sorted(
                aid
                for arm in self.controller.arms.values()
                for aid in arm.hand.act_ids.values()
            ),
            dtype=int,
        )
        # Nominal values, so each episode randomizes from the same baseline
        # rather than compounding jitter on top of the previous draw.
        self._nominal = {
            "mass": float(self.model.body_mass[self.object_body]),
            "friction": self.model.geom_friction[self.object_geom].copy(),
            "solref": self.model.geom_solref[self.object_geom].copy(),
            "gain": self.model.actuator_gainprm[self._hand_act_ids, 0].copy(),
            "bias": self.model.actuator_biasprm[self._hand_act_ids, 1].copy(),
        }

        self._rng = np.random.default_rng(seed)
        # Snapshot of the settled post-grasp state, built once and restored on
        # every reset. Establishing the giver's grasp costs 700 physics steps
        # plus two IK solves -- half the wall time of an entire episode -- and
        # it is deterministic, so paying it per episode is pure waste.
        self._start_snapshots: list[dict[str, np.ndarray]] | None = None
        self._step_count = 0
        self._hold_count = 0
        self._prev_fraction = 0.0
        self._prev_potential = 0.0
        self._prev_contacts = {GIVER: 0, RECV: 0}
        self._prev_grip = {GIVER: 0.0, RECV: 0.0}

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

    # ---------------------------------------------------------- randomization

    def _randomize(self) -> dict[str, float]:
        """Draw fresh physics parameters. Returns what was drawn, for logging."""
        dr = self.randomization
        nom = self._nominal
        if not dr.enabled:
            self.weight = nom["mass"] * G
            return {}

        mass_scale = float(self._rng.uniform(*dr.mass_range))
        self.model.body_mass[self.object_body] = nom["mass"] * mass_scale
        # Load fraction is defined against weight, so it has to track the draw.
        self.weight = nom["mass"] * mass_scale * G

        friction_scale = float(self._rng.uniform(*dr.friction_range))
        self.model.geom_friction[self.object_geom] = nom["friction"] * friction_scale

        solref_scale = float(self._rng.uniform(*dr.solref_range))
        self.model.geom_solref[self.object_geom, 0] = nom["solref"][0] * solref_scale

        gain_scale = float(self._rng.uniform(*dr.hand_gain_range))
        self.model.actuator_gainprm[self._hand_act_ids, 0] = nom["gain"] * gain_scale
        self.model.actuator_biasprm[self._hand_act_ids, 1] = nom["bias"] * gain_scale

        return {
            "dr_mass": mass_scale,
            "dr_friction": friction_scale,
            "dr_solref": solref_scale,
            "dr_hand_gain": gain_scale,
        }

    # ------------------------------------------------------------------ setup

    def _object_pose(self) -> tuple[np.ndarray, np.ndarray]:
        pos = self.data.qpos[self.object_qpos : self.object_qpos + 3].copy()
        quat = self.data.qpos[self.object_qpos + 3 : self.object_qpos + 7].copy()
        return pos, quat

    def _set_object_pose(self, pos: np.ndarray, quat: np.ndarray) -> None:
        self.data.qpos[self.object_qpos : self.object_qpos + 3] = pos
        self.data.qpos[self.object_qpos + 3 : self.object_qpos + 7] = quat
        self.data.qvel[self.object_dof : self.object_dof + 6] = 0.0

    def _recv_grasp_palm(self) -> np.ndarray:
        """Nearest sane start: the PRE-GRASP standoff, not the grasp pose itself.

        The pocket offset is measured at full closure, but a hand approaching an
        episode start is open, and its fingertips stick out along the palm's +z.
        Interpolating start positions toward the exact grasp pose therefore puts
        the closest starts with their open fingers already inside the object,
        which knocks it loose within ~15 steps. Backing off along the palm's -z
        keeps every start on a clean approach line.
        """
        obj = np.asarray(self.scene_cfg.obj_init_pos, dtype=float)
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, np.asarray(self.scene_cfg.recv_start_quat, dtype=float))
        rot = mat.reshape(3, 3)
        return (
            obj
            - rot @ np.asarray(POCKET_BODY_LEFT, dtype=float)
            - self.cfg.pregrasp_standoff * rot[:, 2]
        )

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

    def _grasp_target(self) -> tuple[np.ndarray, np.ndarray]:
        """Pose the receiver's palm must reach to grasp the object where it is."""
        obj_pos, _ = self._object_pose()
        quat = np.asarray(self.scene_cfg.recv_start_quat, dtype=float)
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, quat)
        return obj_pos - mat.reshape(3, 3) @ np.asarray(POCKET_BODY_LEFT, dtype=float), quat

    def approach_error(self) -> tuple[float, float, float]:
        """(total, translation, rotation) error to the receiver's grasp pose."""
        palm_pos, palm_quat = self.controller.arms[RECV].palm_pose(self.data)
        target_pos, target_quat = self._grasp_target()
        if self.cfg.approach_metric == "euclidean":
            translation = float(np.linalg.norm(target_pos - palm_pos))
            return translation, translation, 0.0
        return pose_distance(palm_pos, palm_quat, target_pos, target_quat)

    def _potential(self, obj_pos: np.ndarray) -> float:
        """Shaping potential: closer to the grasp POSE is better.

        Position alone leaves the hand's orientation unconstrained, so a policy
        can arrive at the right point facing the wrong way and never grasp.
        Under "dq" this is the baseline's combined pose metric.
        """
        return -self.cfg.w_approach * self.approach_error()[0]

    def _palm_speed(self, side: str) -> float:
        """Linear speed of one hand's palm, in world coordinates."""
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.controller.arms[side].body_id,
            vel,
            0,  # world frame
        )
        return float(np.linalg.norm(vel[3:6]))

    def _reward(self, wrenches, fraction: float) -> tuple[float, dict]:
        """Baseline reward, plus whatever optional terms are switched on.

        Structured the same way as the Isaac environment: everything outside the
        baseline is behind its own flag and contributes exactly zero when off, so
        `--` with no reward flags reproduces the baseline bit for bit and each
        addition can be attributed to the flag that introduced it.
        """
        cfg = self.cfg
        obj_pos, _ = self._object_pose()
        terms: dict[str, float] = {}

        # ================= BASELINE (always on) =================

        # Progress along the transfer. Rewarding the increase rather than the
        # level keeps the policy from parking at a comfortable split.
        progress = float(np.clip(fraction, 0.0, 1.0) - np.clip(self._prev_fraction, 0.0, 1.0))
        terms["r_progress"] = cfg.w_progress * progress

        # Approach shaping, potential-based: gamma*phi(s') - phi(s).
        #
        # A plain -w*distance term is charged every step, so merely staying alive
        # costs ~180 over a full episode while dropping the object at step 28
        # costs ~33. Training duly discovered that dropping scores better than
        # persisting. The potential form telescopes, so it cannot create that
        # incentive, and it leaves the optimal policy unchanged (Ng, Harada &
        # Russell 1999).
        potential = self._potential(obj_pos)
        terms["r_approach"] = cfg.shaping_gamma * potential - self._prev_potential
        self._prev_potential = potential

        # ================= OPTIONAL TERMS (flag-gated) =================

        # --- object motion penalty ---
        if cfg.use_motion_penalty:
            obj_speed = float(
                np.linalg.norm(self.data.qvel[self.object_dof : self.object_dof + 3])
            )
            terms["r_motion"] = -cfg.w_object_motion * obj_speed

        # --- deadlock: both hands holding long after the receiver took the load ---
        if cfg.use_deadlock_penalty:
            both = wrenches[GIVER].n_contacts > 0 and wrenches[RECV].n_contacts > 0
            terms["r_deadlock"] = (
                -cfg.w_deadlock if (both and fraction > cfg.deadlock_after_fraction) else 0.0
            )

        # --- vel_rew: approach-speed penalty near the object ---
        if cfg.vel_rew:
            speed = self._palm_speed(RECV)
            excess_vel = max(0.0, speed - cfg.v_min)
            distance = float(np.linalg.norm(self._grip_point(RECV) - obj_pos))
            terms["r_velocity"] = (
                -cfg.lambda_vel * (excess_vel**2) * float(np.exp(-cfg.k_decay * distance))
            )

        # --- force-based signals ---
        if cfg.use_force_rewards:
            # Excess-force (crush) penalty. Uses grip, the magnitude sum, which
            # is the right quantity for squeeze: load would be near zero for a
            # hand that is crushing hard but not lifting.
            excess = sum(max(0.0, w.grip - cfg.F_safe) for w in wrenches.values())
            terms["r_force_excess"] = -cfg.lambda_force_excess * excess

        if cfg.use_signal_1:
            # Grasp firmness: reward committed contact, saturating so it cannot
            # be farmed by squeezing ever harder.
            firmness = sum(
                float(np.tanh(w.grip / cfg.F_ref)) for w in wrenches.values()
            )
            terms["r_signal_1_firmness"] = cfg.lambda_firmness * firmness

        if cfg.use_signal_2:
            # Thumb opposition balance: a stable grasp opposes the thumb against
            # the fingers, so penalise the ratio drifting away from parity.
            r_balance = 0.0
            for w in wrenches.values():
                opposing = w.finger_grip + cfg.palm_weight * w.palm_grip
                if w.grip > cfg.F_threshold:
                    ratio = w.thumb_grip / (opposing + 1e-6)
                    r_balance -= cfg.lambda_balance * abs(ratio - 1.0)
            terms["r_signal_2_balance"] = r_balance

        if cfg.use_signal_instability:
            # Contact churn paired with a force drop: fingers losing and
            # regaining the object rather than holding it.
            r_instability = 0.0
            for side, w in wrenches.items():
                churn = abs(w.n_contacts - self._prev_contacts[side])
                drop = max(0.0, self._prev_grip[side] - w.grip)
                r_instability -= cfg.lambda_instability * churn * drop
            terms["r_signal_instability"] = r_instability

        for side, w in wrenches.items():
            self._prev_contacts[side] = w.n_contacts
            self._prev_grip[side] = w.grip

        return float(sum(terms.values())), terms

    def _success(self, wrenches, fraction: float) -> bool:
        obj_pos, _ = self._object_pose()
        return bool(
            fraction >= self.cfg.success_load_fraction
            and abs(wrenches[GIVER].load_vertical) <= self.cfg.success_giver_force
            and obj_pos[2] > self.cfg.drop_height
        )

    # ------------------------------------------------------------- gym API

    def _randomize_off(self) -> None:
        nom = self._nominal
        self.model.body_mass[self.object_body] = nom["mass"]
        self.model.geom_friction[self.object_geom] = nom["friction"]
        self.model.geom_solref[self.object_geom] = nom["solref"]
        self.model.actuator_gainprm[self._hand_act_ids, 0] = nom["gain"]
        self.model.actuator_biasprm[self._hand_act_ids, 1] = nom["bias"]
        self.weight = nom["mass"] * G

    def _build_start_snapshot(self, distance: float) -> dict[str, np.ndarray]:
        """Settle a start state with the receiver `distance` of the way out."""
        scene = replace(
            self.scene_cfg,
            recv_start_palm=tuple(
                np.asarray(self._recv_grasp_palm(), dtype=float)
                + distance
                * (
                    np.asarray(self.scene_cfg.recv_start_palm, dtype=float)
                    - np.asarray(self._recv_grasp_palm(), dtype=float)
                )
            ),
        )
        apply_start_pose(self.model, self.data, scene)
        self.controller.reset(self.data)
        self._establish_giver_grasp()
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ctrl": self.data.ctrl.copy(),
            "act": self.data.act.copy(),
        }

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # The snapshot is built at nominal parameters so it is reusable; the
        # draw is applied on top and the grasp re-settles in a step or two.
        if self._start_snapshots is None:
            self._randomize_off()
            self._start_snapshots = [
                self._build_start_snapshot(d) for d in self.cfg.start_distance_mix
            ]

        snap = self._start_snapshots[int(self._rng.integers(len(self._start_snapshots)))]
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = snap["qpos"]
        self.data.qvel[:] = snap["qvel"]
        self.data.ctrl[:] = snap["ctrl"]
        if snap["act"].size:
            self.data.act[:] = snap["act"]

        # Drawn after the snapshot is restored, so every episode randomizes from
        # the same nominal baseline instead of compounding onto the last draw.
        draw = self._randomize()
        mujoco.mj_forward(self.model, self.data)

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
        self._prev_potential = self._potential(self._object_pose()[0])
        for side, w in wrenches.items():
            self._prev_contacts[side] = w.n_contacts
            self._prev_grip[side] = w.grip

        info = {**draw}
        if self.cfg.include_state:
            info["state"] = self._privileged_state()
        return self._observe(), info

    def _apply_curriculum(self, action: np.ndarray) -> np.ndarray:
        """Override the giver's action while it is on the scripted stage."""
        if self.cfg.giver_mode == "policy":
            return action

        wrenches = self.registry.hand_wrenches(self.data)
        parts = self.controller.split(action)
        giver = np.zeros_like(parts[GIVER])
        # Hold still, then release once the receiver has a real grip on it.
        giver[6] = -1.0 if wrenches[RECV].grip > self.cfg.scripted_release_grip else 0.0
        return np.concatenate([giver, parts[RECV]])

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        action = self._apply_curriculum(action)
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
            "load_fraction": fraction,
            "giver_grip": wrenches[GIVER].grip,
            "recv_grip": wrenches[RECV].grip,
            "giver_load_z": wrenches[GIVER].load_vertical,
            "recv_load_z": wrenches[RECV].load_vertical,
            "object_height": float(obj_pos[2]),
            "approach_dist": self.approach_error()[0],
            "approach_trans": self.approach_error()[1],
            "approach_rot": self.approach_error()[2],
            "success": bool(held),
            "dropped": dropped,
            "hold_count": self._hold_count,
            **terms,
        }
        if self.cfg.include_state:
            info["state"] = self._privileged_state()
        return self._observe(), float(reward), terminated, truncated, info
