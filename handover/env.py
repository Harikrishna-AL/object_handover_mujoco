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

    # Object mass is deliberately NOT randomized here. It is drawn per scenario
    # and the giver's grasp is settled under it; scaling it again at reset would
    # compound the two into a 5x spread and silently break grasps that were
    # established under a lighter object.
    friction_range: tuple[float, float] = (0.7, 1.3)
    # Contact softness: solref[0] is the time constant, larger being softer.
    solref_range: tuple[float, float] = (0.8, 1.3)
    # Finger stiffness stands in for tendon slack and actuator variation. Kept
    # narrow: the snapshot is settled at nominal gain, so a large cut here drops
    # an object the giver was holding perfectly well a moment earlier.
    hand_gain_range: tuple[float, float] = (0.88, 1.20)


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
    # Was 0.15 N, i.e. the giver had to be carrying under 8% of a 1.96 N object,
    # sustained for 25 steps. Combined with the receiver's ~9 N grip ceiling that
    # may simply have been unreachable, and an unreachable bonus is the same as
    # no bonus at all.
    success_giver_force: float = 0.40
    success_hold_steps: int = 15

    # --- failure ---
    drop_height: float = 0.25

    # Truncate if a joint runs away. The Isaac baseline had the same guard on
    # end-effector velocity. It is a safety net rather than a fix -- joint
    # armature is what actually keeps velocities sane -- but it stops a diverging
    # episode from poisoning a rollout, and makes runaways visible in the logs
    # instead of only as a MuJoCo warning on stderr.
    max_joint_velocity: float = 25.0

    # --- reward weights ---
    # These three are set against each other deliberately. The progress term
    # telescopes to w_progress * delta_f, so it is the ENTIRE budget for doing
    # the transfer; the drop penalty is one-shot. At the previous 12 vs 20, a
    # policy that completed the whole transfer and then fumbled scored -8 while
    # standing still scored 0 -- so trying was worse than not trying, and
    # training correctly settled at "do nothing" (ep_rew_mean plateaued at -5,
    # exactly -w_drop times the drop rate). Keep w_progress > w_drop.
    w_progress: float = 25.0

    # Progress is measured on a SMOOTHED load fraction, not the instantaneous
    # one. Raw f is clipped to [0, 1] for the progress term, so a single violent
    # step where the receiver strikes the object -- f was observed peaking at 20,
    # i.e. twenty times the object's weight -- registers as a completed transfer
    # and pays the whole w_progress. With the drop penalty at 8 that made
    # "slam the object, collect +25, eat the -8" worth +14.5 against 0 for doing
    # nothing: a reward hack that teaches exactly the opposite of a gentle
    # handover. (Under the earlier 12/20 weights the same exploit scored -18,
    # which is why it stayed latent until the valley was fixed.)
    #
    # An exponential average with this coefficient pays ~1.25 for a one-step
    # spike and the full 25 only for load genuinely held over ~60 steps.
    progress_smoothing: float = 0.05
    w_success: float = 50.0
    w_drop: float = 8.0

    # Paid per step while the receiver is carrying. Without it the success bonus
    # is a cliff: holding the object for 14 steps scores identically to never
    # touching it. This fills the gap so partial holds register, at a rate small
    # enough that loitering at f~1 for a whole episode (~20) stays well below
    # actually completing the handover (~50).
    w_carry: float = 0.05
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

    # Ceiling on the pose error the approach potential can see.
    #
    # The potential tracks the LIVE object, so when the object is dropped the
    # error explodes (0.11 -> 0.8 as it falls) and the telescoped shaping fires
    # a -5 to -15 hit on top of the -8 drop penalty. Measured: dropped episodes
    # were costing -13 to -22, so the effective drop cost was two to three times
    # what w_drop says, and the "trying is worse than standing still" valley came
    # straight back. It also violates the condition potential-based shaping needs
    # to be policy-preserving -- the potential must vanish at terminal states,
    # and an unbounded one is as far from that as possible.
    #
    # Clamping keeps the full gradient inside the working range (start errors run
    # 0.045-0.173) and stops a falling object from dominating the return.
    approach_error_cap: float = 0.25

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
    # standing still, which is safe and scores about zero.
    start_distance_mix: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

    # ---- episode variety (matches the Isaac baseline's reset randomization) ----
    #
    # The giver holds a FIXED pose for the whole episode but a DIFFERENT one each
    # episode, which is how the baseline worked: only the receiving arm learns,
    # and the object turns up somewhere new every time. Without this a policy can
    # memorise one trajectory and look excellent while having generalised nothing.
    #
    # Each draw needs its own settled snapshot (the grasp has to be established
    # under that pose and that object), so a pool is built once at construction
    # and sampled per episode. That keeps resets at ~0.5 ms instead of ~330 ms.
    # It is a discrete approximation of continuous randomization: pool_size
    # distinct scenarios rather than infinitely many. Raise it for more variety
    # at the cost of a longer one-off build.
    # A pool scenario is only kept if the giver genuinely carries the object.
    grasp_carry_low: float = 0.80
    grasp_carry_high: float = 1.30

    randomize_start: bool = True
    start_pool_size: int = 24
    giver_pos_jitter: tuple[float, float, float] = (0.15, 0.15, 0.10)
    giver_rot_jitter: float = 0.30

    # Object variety. The baseline paper's headline claim is that a policy
    # trained on one object transfers to other shapes and sizes; that claim
    # cannot be made from a single fixed cylinder.
    randomize_object: bool = True
    obj_radius_range: tuple[float, float] = (0.026, 0.034)
    obj_half_length_range: tuple[float, float] = (0.095, 0.130)
    obj_mass_range: tuple[float, float] = (0.12, 0.30)

    # Pre-grasp standoff along the palm's -z. Closest start sits here rather
    # than on the object itself.
    # 0.0 reproduces the measured-best configuration. Raising it moves the
    # closest start off the object, which is geometrically tidier but measured
    # WORSE end to end (0/5 expert successes vs 2/5), so it is left off pending
    # a proper sweep rather than adopted on the strength of the argument.
    pregrasp_standoff: float = 0.10

    # How far along the object's own axis each hand grips, measured from the
    # object's centre. The baseline holds the rod near one end and puts the
    # receiver's target 17.5 cm away along it; the gap between these two numbers
    # is what stops the two hands contesting the same volume.
    giver_grip_offset: float = 0.060
    recv_grip_offset: float = 0.080

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
        # Pool construction uses its own generator so the set of scenarios is the
        # same across workers; only which one an episode draws differs.
        self._pool_seed = 12345
        self._step_count = 0
        self._hold_count = 0
        self._prev_fraction = 0.0
        self._prev_potential = 0.0
        self._f_smooth = 0.0
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

        # Mass comes from the scenario, not from this draw.
        self.weight = nom["mass"] * G

        friction_scale = float(self._rng.uniform(*dr.friction_range))
        self.model.geom_friction[self.object_geom] = nom["friction"] * friction_scale

        solref_scale = float(self._rng.uniform(*dr.solref_range))
        self.model.geom_solref[self.object_geom, 0] = nom["solref"][0] * solref_scale

        gain_scale = float(self._rng.uniform(*dr.hand_gain_range))
        self.model.actuator_gainprm[self._hand_act_ids, 0] = nom["gain"] * gain_scale
        self.model.actuator_biasprm[self._hand_act_ids, 1] = nom["bias"] * gain_scale

        return {
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

    def _giver_pocket_for(self, palm: np.ndarray, quat: np.ndarray) -> np.ndarray:
        """Where the object will sit for a given giver palm pose.

        Needed before the scene is built: the receiver's start is defined
        relative to the object, and with the giver's pose randomized the object
        is no longer at the nominal spot. Deriving the receiver's start from the
        nominal position instead puts it in the wrong place entirely -- at the
        closest start distance, inside the object or nowhere near it.
        """
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=float))
        return np.asarray(palm, dtype=float) + mat.reshape(3, 3) @ np.asarray(
            POCKET_BODY_RIGHT, dtype=float
        )

    def _recv_grasp_palm(self, obj_pos: np.ndarray | None = None) -> np.ndarray:
        """Nearest sane start: the PRE-GRASP standoff, not the grasp pose itself.

        The pocket offset is measured at full closure, but a hand approaching an
        episode start is open, and its fingertips stick out along the palm's +z.
        Interpolating start positions toward the exact grasp pose therefore puts
        the closest starts with their open fingers already inside the object,
        which knocks it loose within ~15 steps. Backing off along the palm's -z
        keeps every start on a clean approach line.
        """
        obj = (
            np.asarray(self.scene_cfg.obj_init_pos, dtype=float)
            if obj_pos is None
            else np.asarray(obj_pos, dtype=float)
        )
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
        # Orient the rod to the GIVER'S PALM, not to the world.
        #
        # The Allegro's fingers spread along the palm's y and curl in its x-z
        # plane, so a cylinder is only properly wrapped when its axis lies along
        # the palm's y. A world-fixed spawn orientation means that alignment
        # changes with every randomized giver pose: measured, a horizontal rod
        # lost half its weight to something else, and a vertical one spawned
        # inside the fingers and produced 82 N across 24 contacts.
        _, palm_quat = self.controller.arms[GIVER].palm_pose(self.data)
        to_palm_y = np.zeros(4)
        mujoco.mju_axisAngle2Quat(to_palm_y, np.array([1.0, 0.0, 0.0]), -np.pi / 2)
        obj_quat = np.zeros(4)
        mujoco.mju_mulQuat(obj_quat, palm_quat, to_palm_y)
        mujoco.mju_normalize4(obj_quat)

        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, obj_quat)
        spawn_axis = mat.reshape(3, 3)[:, 2]
        target = self._grip_point(GIVER) + self.cfg.giver_grip_offset * spawn_axis

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
            target = self._grip_point(GIVER) + self.cfg.giver_grip_offset * spawn_axis

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

    def _object_axis(self, away_from_giver: bool = True) -> np.ndarray:
        """The rod's long axis in world coordinates (its own local z).

        The sign is resolved against where the giver is actually holding, so it
        points toward the FREE end. A fixed sign flips meaning as soon as the rod
        rotates, which sends the receiver at the end the giver is already on.
        """
        obj_pos, obj_quat = self._object_pose()
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, obj_quat)
        axis = mat.reshape(3, 3)[:, 2]
        if away_from_giver and np.dot(obj_pos - self._grip_point(GIVER), axis) < 0:
            axis = -axis
        return axis

    def _grasp_target(self) -> tuple[np.ndarray, np.ndarray]:
        """Pose the receiver's palm must reach to grasp the object.

        Aimed at a point along the rod offset from where the giver holds it, so
        the receiver has bare object to close on instead of the giver's fingers.
        """
        obj_pos, _ = self._object_pose()
        grip_point = obj_pos + self.cfg.recv_grip_offset * self._object_axis()
        quat = np.asarray(self.scene_cfg.recv_start_quat, dtype=float)
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, quat)
        return grip_point - mat.reshape(3, 3) @ np.asarray(POCKET_BODY_LEFT, dtype=float), quat

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
        error = min(self.approach_error()[0], self.cfg.approach_error_cap)
        return -self.cfg.w_approach * error

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
        clipped = float(np.clip(fraction, 0.0, 1.0))
        smoothed = (
            self._f_smooth + cfg.progress_smoothing * (clipped - self._f_smooth)
        )
        terms["r_progress"] = cfg.w_progress * (smoothed - self._f_smooth)
        self._f_smooth = smoothed

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

        # Carrying: continuous, so partial holds are worth something.
        obj_pos_now, _ = self._object_pose()
        if obj_pos_now[2] > cfg.drop_height:
            terms["r_carry"] = cfg.w_carry * float(np.clip(fraction, 0.0, 1.0))

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

    def _grasp_took(self) -> bool:
        """Is the giver actually CARRYING the object, not merely touching it?

        Checking contact and height alone is too permissive: an off-centre grip
        on a heavier rod can keep a few contacts while the rod pivots onto
        something else, and those scenarios then begin an episode already half
        failed. Requiring the giver's vertical load to account for most of the
        weight rejects them at pool-build time instead.
        """
        for _ in range(8):
            mujoco.mj_step(self.model, self.data)
        wrenches = self.registry.hand_wrenches(self.data)
        height = float(self.data.xpos[self.registry.object_body_id][2])
        carried = wrenches[GIVER].load_vertical / max(self.weight, 1e-6)
        return (
            wrenches[GIVER].n_contacts >= 3
            and height > self.cfg.drop_height
            and self.cfg.grasp_carry_low < carried < self.cfg.grasp_carry_high
        )

    def _build_pool(self, builder, max_tries: int = 12) -> list[dict]:
        """Build the scenario pool, keeping only scenarios the giver can hold.

        Some draws are simply not graspable -- a thin object at an awkward wrist
        angle -- and a snapshot of a failed grasp starts the episode with the
        object already falling, which reads as the policy dropping it. Rejecting
        them here keeps that out of the training signal.
        """
        pool, rejected = [], 0
        while len(pool) < max(1, self.cfg.start_pool_size):
            for _ in range(max_tries):
                snapshot = self._build_start_snapshot(self._sample_scenario(builder))
                if self._grasp_took():
                    pool.append(snapshot)
                    break
                rejected += 1
            else:
                # Fall back to the un-jittered nominal so the pool always fills.
                pool.append(self._build_start_snapshot(self._nominal_scenario()))
        self.pool_rejected = rejected
        return pool

    def _nominal_scenario(self) -> dict:
        scene = self.scene_cfg
        return {
            "distance": float(self.cfg.start_distance_mix[-1]),
            "radius": scene.obj_radius,
            "half_length": scene.obj_half_length,
            "mass": scene.obj_mass,
            "giver_palm": np.asarray(scene.giver_start_palm, dtype=float),
            "giver_quat": np.asarray(scene.giver_start_quat, dtype=float),
        }

    def _apply_object_params(self, radius: float, half_length: float, mass: float) -> None:
        """Resize and re-mass the object.

        Geometry is baked at compile time, so mass and inertia do not follow a
        size change automatically -- they have to be recomputed or the object
        keeps the inertia of whatever it used to be.
        """
        self.model.geom_size[self.object_geom, 0] = radius
        self.model.geom_size[self.object_geom, 1] = half_length
        self.model.body_mass[self.object_body] = mass
        # Solid cylinder about its own axis (local z) and the two transverse axes.
        transverse = mass * (3.0 * radius**2 + 4.0 * half_length**2) / 12.0
        axial = 0.5 * mass * radius**2
        self.model.body_inertia[self.object_body] = [transverse, transverse, axial]
        self._nominal["mass"] = mass

    def _sample_scenario(self, rng) -> dict:
        """Draw one episode scenario: giver pose, object, receiver start."""
        cfg, scene = self.cfg, self.scene_cfg
        scenario = {
            "distance": float(rng.choice(cfg.start_distance_mix)),
            "radius": scene.obj_radius,
            "half_length": scene.obj_half_length,
            "mass": scene.obj_mass,
            "giver_palm": np.asarray(scene.giver_start_palm, dtype=float),
            "giver_quat": np.asarray(scene.giver_start_quat, dtype=float),
        }
        if cfg.randomize_start:
            scenario["giver_palm"] = scenario["giver_palm"] + rng.uniform(
                -np.asarray(cfg.giver_pos_jitter), np.asarray(cfg.giver_pos_jitter)
            )
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis) + 1e-9
            angle = float(rng.uniform(-cfg.giver_rot_jitter, cfg.giver_rot_jitter))
            delta = np.zeros(4)
            mujoco.mju_axisAngle2Quat(delta, axis, angle)
            turned = np.zeros(4)
            mujoco.mju_mulQuat(turned, delta, scenario["giver_quat"])
            mujoco.mju_normalize4(turned)
            scenario["giver_quat"] = turned
        if cfg.randomize_object:
            scenario["radius"] = float(rng.uniform(*cfg.obj_radius_range))
            scenario["half_length"] = float(rng.uniform(*cfg.obj_half_length_range))
            scenario["mass"] = float(rng.uniform(*cfg.obj_mass_range))
        return scenario

    def _build_start_snapshot(self, scenario: dict) -> dict[str, np.ndarray]:
        """Settle a start state for one scenario and snapshot it."""
        self._apply_object_params(
            scenario["radius"], scenario["half_length"], scenario["mass"]
        )
        # Derive the receiver's start from where the object will actually be for
        # THIS scenario, not from the nominal spawn point.
        grasp = self._recv_grasp_palm(
            self._giver_pocket_for(scenario["giver_palm"], scenario["giver_quat"])
        )
        nominal = np.asarray(self.scene_cfg.recv_start_palm, dtype=float)
        scene = replace(
            self.scene_cfg,
            giver_start_palm=tuple(scenario["giver_palm"]),
            giver_start_quat=tuple(scenario["giver_quat"]),
            recv_start_palm=tuple(grasp + scenario["distance"] * (nominal - grasp)),
        )
        apply_start_pose(self.model, self.data, scene)
        self.controller.reset(self.data)
        self._establish_giver_grasp()
        return {
            "radius": scenario["radius"],
            "half_length": scenario["half_length"],
            "mass": scenario["mass"],
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
            builder = np.random.default_rng(self._pool_seed)
            self._start_snapshots = self._build_pool(builder)

        snap = self._start_snapshots[int(self._rng.integers(len(self._start_snapshots)))]
        # The object's size and mass are part of the scenario, so they have to be
        # restored alongside the state or the snapshot describes a different body.
        self._apply_object_params(snap["radius"], snap["half_length"], snap["mass"])
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
        self._f_smooth = 0.0
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
        runaway = bool(
            np.abs(self.data.qvel).max() > self.cfg.max_joint_velocity
            or not np.isfinite(self.data.qvel).all()
        )

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
        truncated = bool(self._step_count >= self.cfg.episode_steps or runaway)

        self._prev_fraction = fraction

        info = {
            "load_fraction": fraction,
            "load_fraction_smooth": self._f_smooth,
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
            "runaway": runaway,
            "max_qvel": float(np.abs(self.data.qvel).max()),
            "hold_count": self._hold_count,
            **terms,
        }
        if self.cfg.include_state:
            info["state"] = self._privileged_state()
        return self._observe(), float(reward), terminated, truncated, info
