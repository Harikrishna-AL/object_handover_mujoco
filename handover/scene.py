"""Compose the bimanual handover scene from MuJoCo Menagerie assets.

Two arms face each other, each with an Allegro hand:
    giver    = UR5e  + Allegro right hand   (prefix "giver_")
    receiver = Gen3  + Allegro left  hand   (prefix "recv_")

The object is a free-floating cylinder with realistic mass, spawned between them.
Everything is built with mjSpec so asset paths and name collisions are handled
for us, and the composed model is written out for inspection.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import mujoco
import numpy as np

from .ik import ArmIK

# Overridable so cluster jobs can point at a shared or scratch copy.
MENAGERIE = os.environ.get(
    "MUJOCO_MENAGERIE", os.path.expanduser("~/mujoco_menagerie")
)

UR5E_XML = f"{MENAGERIE}/universal_robots_ur5e/ur5e.xml"
GEN3_XML = f"{MENAGERIE}/kinova_gen3/gen3.xml"
ALLEGRO_RIGHT_XML = f"{MENAGERIE}/wonik_allegro/right_hand.xml"
ALLEGRO_LEFT_XML = f"{MENAGERIE}/wonik_allegro/left_hand.xml"

# Prefixes. Every body/joint/actuator name in the composed model carries one of
# these, which is what lets the contact registry attribute a contact to an arm.
GIVER = "giver_"
RECV = "recv_"


@dataclass
class SceneConfig:
    """Geometry and object properties for the handover scene."""

    # Base placement. The UR5e sits at the origin; the Gen3 faces it from +x.
    arm_separation: float = 1.10
    base_height: float = 0.0

    # Object: a graspable cylinder, bottle-sized. Short on purpose -- a long bar
    # makes the scalar load-split metric ambiguous against a pure moment (D4).
    # The diameter must exceed the Allegro's 4.9 cm minimum grip opening or the
    # fingers close on empty air and never generate force.
    obj_radius: float = 0.030
    obj_half_length: float = 0.075
    obj_mass: float = 0.200
    obj_friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)

    # Lying horizontally along y: a cylinder's axis is its local z, so this is a
    # 90 deg rotation about x. Both hands then grip it at different points along
    # its length, which is the geometry a real rod handover takes.
    obj_quat: tuple[float, float, float, float] = (0.70710678, 0.70710678, 0.0, 0.0)

    # Where the object starts, in world coordinates: the giver's grasp pocket,
    # so the static scene matches what the environment builds at reset.
    obj_init_pos: tuple[float, float, float] = (0.50, -0.02, 0.52)

    # Palm poses for the arm-free validation scene only. Set so the two hands
    # face each other across the object, gripping at different heights.
    # Arm-free validation scene. The giver grips from ABOVE (palm rolled 180 deg
    # about x) and the receiver from BELOW. The order matters: a hand underneath
    # the object stays a shelf when it opens, so a giver below can never actually
    # release. Palm positions are derived from the measured pocket offset.
    giver_palm_quat: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0)
    recv_palm_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    giver_grip_y: float = -0.045
    recv_grip_y: float = 0.045

    # Travel and gains for the validation scene's approach slide. The slide axis
    # is the palm's own z, so positive travel always means "toward the object".
    approach_travel: float = 0.30
    approach_kp: float = 400.0
    approach_kd: float = 40.0

    # Episode start: palm targets for the two arms, solved by IK. Chosen to keep
    # the arms well clear of each other so the approach is something the policy
    # has to do rather than something it starts inside.
    # The receiver's target is chosen for joint margin, not just reachability:
    # anywhere with x >= 0.76 folds the Gen3's elbow onto its -2.57 rad stop, and
    # an arm that starts saturated cannot move in that direction at all. This
    # start leaves 0.72 rad of margin while staying 0.36 m from the handover
    # point, so the approach remains something the policy has to do.
    giver_start_palm: tuple[float, float, float] = (0.531, -0.037, 0.591)
    recv_start_palm: tuple[float, float, float] = (0.62, 0.25, 0.55)

    # Start orientations matter as much as positions: the giver holds from above
    # (palm rolled 180 deg about x, fingers curling down) and the receiver comes
    # from below. Solving position-only leaves the palm at whatever roll the IK
    # happens to land on, which will not hold an object against gravity.
    # The grasp pocket sits along the palm body's +x, so "approach from above"
    # means rotating body +x to point down: +90 deg about y. Derived from the
    # corrected palm-frame pocket, not guessed.
    giver_start_quat: tuple[float, float, float, float] = (0.70710678, 0.0, 0.70710678, 0.0)
    # From below (-90 about y) with a -90 yaw. Of the four from-below variants
    # this is the only one the Gen3 reaches without pinning a joint on its stop.
    recv_start_quat: tuple[float, float, float, float] = (0.5, -0.5, -0.5, -0.5)

    # Standoff between the arm's tool flange and the Allegro palm. Mounting the
    # hand flush drives the finger proximals ~1 cm into the wrist geoms; real
    # setups use an adapter plate, and 2 cm is enough to clear it entirely.
    hand_mount_offset: float = 0.03

    # Solver settings. Elliptic cone + high impratio is what Menagerie's Allegro
    # ships with, and it matters for multi-contact grasping; the parent spec's
    # option block wins on attach, so we set it here explicitly.
    timestep: float = 0.002
    impratio: float = 10.0

    # Offscreen render buffer, used only by scripts/record.py.
    offscreen_width: int = 1280
    offscreen_height: int = 960


def _strip_keyframes(spec: mujoco.MjSpec) -> None:
    """Remove child keyframes; their qpos widths do not survive attachment."""
    for key in list(spec.keys):
        spec.delete(key)


def _set_contact_options(spec: mujoco.MjSpec, impratio: float) -> None:
    """Match the Allegro's contact settings so attaching does not warn."""
    spec.option.impratio = impratio
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC


def _build_arm(
    arm_xml: str, hand_xml: str, hand_prefix: str, site: str, impratio: float, mount_offset: float
) -> mujoco.MjSpec:
    """Attach an Allegro hand to an arm's tool site, returning the merged spec."""
    arm = mujoco.MjSpec.from_file(arm_xml)
    hand = mujoco.MjSpec.from_file(hand_xml)
    _strip_keyframes(arm)
    _strip_keyframes(hand)
    # The parent's option block wins on attach, so the arm must already agree
    # with the hand's elliptic-cone settings or the grasp contacts degrade.
    _set_contact_options(arm, impratio)
    frame = arm.attach(hand, prefix=hand_prefix, site=site)
    frame.pos = [0.0, 0.0, mount_offset]
    return arm


def build_spec(cfg: SceneConfig | None = None, hands_only: bool = False) -> mujoco.MjSpec:
    """Build the two-arm scene and return the uncompiled spec.

    With `hands_only`, the arms are omitted and the two Allegro hands are welded
    to the world facing each other. Newton's balance on the object involves only
    the hands' contact forces and gravity, so the arms are irrelevant to the
    force-measurement validation -- dropping them removes IK and arm dynamics as
    confounds. Contact naming is identical in both variants.
    """
    cfg = cfg or SceneConfig()

    world = mujoco.MjSpec()
    world.option.timestep = cfg.timestep
    world.option.impratio = cfg.impratio
    world.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    world.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    # Offscreen framebuffer for scripts/record.py. MuJoCo caps offscreen renders
    # at 640x480 unless the model declares a larger buffer, and the failure is
    # opaque -- the renderer just refuses.
    world.visual.global_.offwidth = cfg.offscreen_width
    world.visual.global_.offheight = cfg.offscreen_height

    # --- ground and lighting ---
    world.add_texture(
        name="grid",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=[0.1, 0.2, 0.3],
        rgb2=[0.2, 0.3, 0.4],
        width=300,
        height=300,
    )
    world.add_material(name="grid", textures=["", "grid"], texrepeat=[5, 5], reflectance=0.2)
    world.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[3, 3, 0.05],
        material="grid",
    )
    world.worldbody.add_light(
        pos=[0, 0, 3], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    )

    if hands_only:
        _attach_bare_hands(world, cfg)
        _add_object(world, cfg)
        return world

    # --- the two arms, facing each other across the x axis ---
    giver = _build_arm(
        UR5E_XML,
        ALLEGRO_RIGHT_XML,
        "hand_",
        site="attachment_site",
        impratio=cfg.impratio,
        mount_offset=cfg.hand_mount_offset,
    )
    recv = _build_arm(
        GEN3_XML,
        ALLEGRO_LEFT_XML,
        "hand_",
        site="pinch_site",
        impratio=cfg.impratio,
        mount_offset=cfg.hand_mount_offset,
    )

    giver_frame = world.worldbody.add_frame(pos=[0.0, 0.0, cfg.base_height])
    # 180 deg about z, so the Gen3 looks back down -x at the UR5e.
    recv_frame = world.worldbody.add_frame(
        pos=[cfg.arm_separation, 0.0, cfg.base_height], quat=[0.0, 0.0, 0.0, 1.0]
    )

    world.attach(giver, prefix=GIVER, frame=giver_frame)
    world.attach(recv, prefix=RECV, frame=recv_frame)

    # Menagerie's stock Gen3 reports a permanent base/shoulder overlap -- it is
    # present in the unmodified asset, not something the composition introduced.
    # Left in, it injects a constant spurious force into the receiver's base.
    world.add_exclude(
        name="recv_base_shoulder", bodyname1=f"{RECV}base_link", bodyname2=f"{RECV}shoulder_link"
    )

    _add_object(world, cfg)
    return world


def _add_object(world: mujoco.MjSpec, cfg: SceneConfig) -> None:
    """Add the free-floating handover object."""
    obj = world.worldbody.add_body(
        name="object", pos=list(cfg.obj_init_pos), quat=list(cfg.obj_quat)
    )
    obj.add_freejoint(name="object_free")
    obj.add_geom(
        name="object_geom",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[cfg.obj_radius, cfg.obj_half_length, 0.0],
        mass=cfg.obj_mass,
        friction=list(cfg.obj_friction),
        rgba=[0.15, 0.7, 0.25, 1.0],
    )


def palm_pos_for_pocket(
    pocket_world: np.ndarray, palm_quat: np.ndarray, pocket_local: np.ndarray
) -> np.ndarray:
    """Where to put a palm so its grasp pocket lands on `pocket_world`.

    The pocket offset is measured from the model rather than assumed, so this
    stays correct if the closed posture is retuned.
    """
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(palm_quat, dtype=float))
    return np.asarray(pocket_world, dtype=float) - mat.reshape(3, 3) @ np.asarray(
        pocket_local, dtype=float
    )


def bare_hand_palm_poses(cfg: SceneConfig) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Palm poses that put both grasp pockets on the object's axis."""
    from .grasp import POCKET_LEFT, POCKET_RIGHT

    obj = np.asarray(cfg.obj_init_pos, dtype=float)
    poses = {}
    for key, quat, pocket_local, grip_y in (
        (GIVER, cfg.giver_palm_quat, POCKET_RIGHT, cfg.giver_grip_y),
        (RECV, cfg.recv_palm_quat, POCKET_LEFT, cfg.recv_grip_y),
    ):
        # The object lies along y, so each hand grips at its own point on the axis.
        pocket_world = obj + np.array([0.0, grip_y, 0.0])
        quat_arr = np.asarray(quat, dtype=float)
        poses[key] = (palm_pos_for_pocket(pocket_world, quat_arr, pocket_local), quat_arr)
    return poses


def _attach_bare_hands(world: mujoco.MjSpec, cfg: SceneConfig) -> None:
    """Weld both Allegro hands to the world, gripping the object from opposite sides.

    The giver comes up from below and the receiver down from above, each taking a
    different point along the object's length -- the geometry a real rod handover
    takes, and it keeps the two palms clear of each other.
    """
    poses = bare_hand_palm_poses(cfg)
    for prefix, xml in ((GIVER, ALLEGRO_RIGHT_XML), (RECV, ALLEGRO_LEFT_XML)):
        pos, quat = poses[prefix]
        hand = mujoco.MjSpec.from_file(xml)
        _strip_keyframes(hand)

        # Each hand rides a vertical slide rather than being welded outright. A
        # hand parked at the grasp point supports the object passively even when
        # fully open, which silently contaminates any "one hand alone" baseline;
        # the slide lets a hand retract clear of the object entirely.
        mount = world.worldbody.add_body(name=f"{prefix}mount", pos=list(pos), quat=list(quat))
        mount.add_joint(
            name=f"{prefix}approach",
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=[0.0, 0.0, 1.0],
            range=[-cfg.approach_travel, cfg.approach_travel],
        )
        world.add_actuator(
            name=f"{prefix}approach_act",
            target=f"{prefix}approach",
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            # biastype must be set explicitly: it defaults to mjBIAS_NONE, which
            # discards biasprm and leaves an open-loop force source rather than a
            # position servo. The hand then just sinks under gravity.
            gaintype=mujoco.mjtGain.mjGAIN_FIXED,
            biastype=mujoco.mjtBias.mjBIAS_AFFINE,
            gainprm=[cfg.approach_kp] + [0.0] * 9,
            biasprm=[0.0, -cfg.approach_kp, -cfg.approach_kd] + [0.0] * 7,
            ctrlrange=[-cfg.approach_travel, cfg.approach_travel],
        )
        frame = mount.add_frame()
        world.attach(hand, prefix=f"{prefix}hand_", frame=frame)


def build_hands_only(cfg: SceneConfig | None = None) -> tuple[mujoco.MjModel, mujoco.MjSpec]:
    """Compile the arm-free validation scene."""
    cfg = cfg or SceneConfig()
    spec = build_spec(cfg, hands_only=True)
    return spec.compile(), spec


# Arm joint angles at episode start, by joint name. Set so the two arms face
# each other with their palms in the shared workspace around x ~ 0.55.
# Hand joints default to 0 (open); the grasp script drives them from there.
HOME_QPOS: dict[str, float] = {
    # UR5e -- reaching forward along +x toward the receiver. The UR5e base
    # carries a 180 deg z-rotation in its own model, hence pan = pi.
    "giver_shoulder_pan_joint": 3.14159265,
    "giver_shoulder_lift_joint": -1.20,
    "giver_elbow_joint": 1.40,
    "giver_wrist_1_joint": -1.75,
    "giver_wrist_2_joint": -1.5708,
    "giver_wrist_3_joint": 0.0,
    # Gen3 -- reaching back along -x toward the giver.
    "recv_joint_1": 0.0,
    "recv_joint_2": 0.35,
    "recv_joint_3": 3.14159265,
    "recv_joint_4": -2.05,
    "recv_joint_5": 0.0,
    "recv_joint_6": 0.95,
    "recv_joint_7": 1.5708,
}


def apply_home(model: mujoco.MjModel, data: mujoco.MjData, cfg: SceneConfig | None = None) -> None:
    """Put the arms in their home pose and the object at its spawn point.

    Everything is addressed by joint name rather than qpos index -- the index
    layout shifts whenever the scene changes, and silent misalignment there is
    exactly the class of bug we are trying to design out.
    """
    cfg = cfg or SceneConfig()
    mujoco.mj_resetData(model, data)

    for joint_name, angle in HOME_QPOS.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            raise KeyError(f"home pose references unknown joint {joint_name!r}")
        data.qpos[model.jnt_qposadr[jid]] = angle

    obj_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
    obj_adr = model.jnt_qposadr[obj_jid]
    data.qpos[obj_adr : obj_adr + 3] = cfg.obj_init_pos
    data.qpos[obj_adr + 3 : obj_adr + 7] = cfg.obj_quat

    # Hold position actuators at the home pose so the arms do not sag on step 0.
    mujoco.mj_forward(model, data)
    for act_id in range(model.nu):
        trn = model.actuator_trnid[act_id, 0]
        data.ctrl[act_id] = data.qpos[model.jnt_qposadr[trn]]


GIVER_ARM_JOINTS = [
    "giver_shoulder_pan_joint",
    "giver_shoulder_lift_joint",
    "giver_elbow_joint",
    "giver_wrist_1_joint",
    "giver_wrist_2_joint",
    "giver_wrist_3_joint",
]
RECV_ARM_JOINTS = [f"recv_joint_{i}" for i in range(1, 8)]


def apply_start_pose(
    model: mujoco.MjModel, data: mujoco.MjData, cfg: SceneConfig | None = None
) -> dict[str, float]:
    """Place both arms at the episode start via IK, and hold them there.

    Hand-picked joint angles put the two arms through each other; solving for
    palm targets instead keeps them clear and makes the start pose a property of
    the workspace rather than of six numbers someone guessed. Returns the IK
    residuals so a caller can assert the targets were actually reached.
    """
    cfg = cfg or SceneConfig()
    apply_home(model, data, cfg)

    errors = {}
    for key, joints, body, target, quat in (
        (GIVER, GIVER_ARM_JOINTS, "giver_hand_palm", cfg.giver_start_palm, cfg.giver_start_quat),
        (RECV, RECV_ARM_JOINTS, "recv_hand_palm", cfg.recv_start_palm, cfg.recv_start_quat),
    ):
        solver = ArmIK(model, body, joints)
        angles, err = solver.solve(
            data, np.asarray(target, dtype=float), target_quat=np.asarray(quat, dtype=float)
        )
        for name, value in zip(joints, angles):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            data.qpos[model.jnt_qposadr[jid]] = value
        errors[key] = err

    mujoco.mj_forward(model, data)
    for act_id in range(model.nu):
        trn = model.actuator_trnid[act_id, 0]
        data.ctrl[act_id] = data.qpos[model.jnt_qposadr[trn]]
    return errors


def build_model(cfg: SceneConfig | None = None) -> tuple[mujoco.MjModel, mujoco.MjSpec]:
    """Compile the scene, returning both the model and the spec that made it."""
    spec = build_spec(cfg)
    return spec.compile(), spec


if __name__ == "__main__":
    config = SceneConfig()
    model, spec = build_model(config)

    out = os.path.join(os.path.dirname(__file__), "..", "scene_composed.xml")
    with open(os.path.abspath(out), "w") as fh:
        fh.write(spec.to_xml())

    print(f"compiled ok: nq={model.nq} nv={model.nv} nu={model.nu} "
          f"nbody={model.nbody} ngeom={model.ngeom}")
    print(f"object mass = {config.obj_mass} kg -> weight = {config.obj_mass * 9.81:.4f} N")
    print(f"wrote {os.path.abspath(out)}")
