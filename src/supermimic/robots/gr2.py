"""Fourier GR2, fixed-head 27-DoF profile with the real-robot foot contact (UniLab SONIC asset).

The MJCF is ``gr2_sonic_27dof_realfoot.xml`` from UniLab (copied with its actuator XML and meshes
to ``supermimic/assets/gr2``). Its control contract is kept as is:

* actuators: the XML ``<position>`` elements (kp / kv / forcerange per joint), wrapped by mjlab's
  ``XmlActuatorCfg``; action scale ``0.25 * effort / kp`` as in UniLab;
* joints: XML damping 0.2, frictionloss 0.2 and per-joint armature;
* contact: foot meshes + thin sole boxes (condim 3, friction 1), link meshes colliding with the
  floor only. The feet get ``priority=1`` so their (randomised) friction owns the floor contact;
* default pose: all joints 0 (UniLab ``stand`` keyframe).

The XML ``<option>`` (timestep 0.001, Newton, ...) is not propagated by mjlab; the simulation
settings come from ``tracking_env_cfg`` (5 ms physics step, decimation 4, as UniLab's sim_dt).
Anchor and root are both ``base`` (pelvis), the SONIC convention for GR2.
"""

from __future__ import annotations

from pathlib import Path

from supermimic.robots import RobotSpec

ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "gr2"

GR2_JOINT_NAMES: tuple[str, ...] = (
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_pitch_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_pitch_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)

# Same 14 bodies as UniLab's GR2 SONIC config (G1 equivalents: pelvis, hip roll, knee, ankle
# roll, torso, shoulder roll, elbow, wrist).
GR2_TRACKED_BODIES: tuple[str, ...] = (
    "base",
    "left_thigh_roll_link",
    "left_shank_pitch_link",
    "left_foot_roll_link",
    "right_thigh_roll_link",
    "right_shank_pitch_link",
    "right_foot_roll_link",
    "waist_yaw_link",
    "left_upper_arm_roll_link",
    "left_lower_arm_pitch_link",
    "left_hand_yaw_link",
    "right_upper_arm_roll_link",
    "right_lower_arm_pitch_link",
    "right_hand_yaw_link",
)

FOOT_GEOM_REGEX = r"^[lr]_foot_(sole|mesh)$"


def gr2_xml_path() -> Path:
    p = ASSET_DIR / "gr2_sonic_27dof_realfoot.xml"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing (GR2 asset: xml, actuator xml, meshes in {ASSET_DIR})")
    return p


def _entity_cfg():
    import mujoco
    from mjlab.actuator import XmlActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
    from mjlab.utils.spec_config import GeomCfg

    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0), joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
        geoms=(GeomCfg(geom_names_expr=(FOOT_GEOM_REGEX,), priority=1),),
        spec_fn=lambda: mujoco.MjSpec.from_file(str(gr2_xml_path())),
        articulation=EntityArticulationInfoCfg(actuators=(XmlActuatorCfg(target_names_expr=(r".*_joint",)),), soft_joint_pos_limit_factor=0.9),
    )


def _action_scale() -> dict[str, float]:
    """0.25 * effort limit / kp of each XML position actuator, keyed by joint name."""
    import mujoco

    m = GR2.load_model()
    out = {}
    for i in range(m.nu):
        assert m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT
        joint = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[i, 0]))
        kp = float(m.actuator_gainprm[i, 0])
        effort = float(m.actuator_forcerange[i, 1])
        out[joint] = 0.25 * effort / kp
    assert set(out) == set(GR2_JOINT_NAMES), "every GR2 joint needs exactly one position actuator"
    return out


GR2 = RobotSpec(
    name="gr2",
    xml_path_fn=gr2_xml_path,
    joint_names=GR2_JOINT_NAMES,
    tracked_bodies=GR2_TRACKED_BODIES,
    anchor_body="base",
    root_body="base",
    foot_geom_regex=FOOT_GEOM_REGEX,
    com_body="waist_yaw_link",
    entity_cfg_fn=_entity_cfg,
    action_scale_fn=_action_scale,
    # Link meshes collide with the floor (G1 uses capsules); sized generously, to be checked on CUDA.
    nconmax=64,
    njmax=400,
)
