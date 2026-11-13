"""Unitree H2, fixed-head 29-DoF profile (UniLab asset, ``supermimic/assets/h2/h2.xml``).

The MJCF is UniLab's ``h2.xml`` (head welded to the torso: the head bodies and meshes stay so the head
renders, but it has no DoF, actuator or observation). Its control contract is kept as is:

* actuators: the XML ``<position>`` elements (kp / kv), wrapped by mjlab's ``XmlActuatorCfg``. The torque
  limit of a joint is the tighter of the actuator ``forcerange`` and the joint ``actuatorfrcrange``
  (ankle roll: 150 vs 19, ankle pitch: 150 vs 66.88); action scale ``0.25 * effort / kp`` as in UniLab's
  ``tracking_sonic.py``;
* joints: XML armature and frictionloss (no joint damping, the servo kv does the damping);
* contact: four 5 mm sphere corners per sole (condim 3, friction 1, priority 1); link meshes / capsules
  collide with the floor only (``conaffinity=0``);
* default pose: the UniLab ``stand`` keyframe (knee bent, arms slightly out), unlike GR2's zero pose.

Joint order is the model order (``npz`` order of ``g1_csv_to_h2_npz2.py``). It equals G1's except for the ankle
chain, which H2 builds roll -> pitch (G1: pitch -> roll); H2's waist is yaw -> roll -> pitch like G1's. Names are
otherwise identical, so the retargeter's name based mapping is exact (``scripts/data/check_h2_mapping.py``).

Tracked bodies: the 14 bodies of UniLab's H2 SONIC task (the G1 set) plus both ``ankle_pitch_link``. H2's
``ankle_roll_link`` is the *proximal* ankle link, so ankle pitch moves none of the 14 and the sole orientation is
untracked (the same hole as GR2's wrist pitch / roll, see ``GR2_WRIST``); ``ankle_pitch_link`` is the distal
foot body that carries the sole contact. Wrists need no extra bodies: ``wrist_yaw_link`` is already the
distal wrist link (chain roll -> pitch -> yaw). Anchor and root are both ``pelvis``.
"""

from __future__ import annotations

from pathlib import Path

from supermimic.robots import RobotSpec

ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "h2"

H2_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_roll_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# UniLab ``stand`` keyframe of scene_flat.xml (joint part), in H2_JOINT_NAMES order.
_LEG = (-0.312, 0.0, 0.0, 0.669, 0.0, -0.363)
H2_DEFAULT_JOINT_POS: tuple[float, ...] = (
    _LEG + _LEG + (0.0, 0.0, 0.0) + (0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0) + (0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0)
)

# G1 / UniLab SONIC 14 bodies + the distal foot bodies (16).
H2_TRACKED_BODIES: tuple[str, ...] = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
)

FOOT_GEOM_REGEX = r"^(left|right)_foot[1-4]_collision$"


def h2_xml_path() -> Path:
    p = ASSET_DIR / "h2.xml"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing (H2 asset: xml and meshes in {ASSET_DIR})")
    return p


def _entity_cfg():
    import mujoco
    from mjlab.actuator import XmlActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
    from mjlab.utils.spec_config import GeomCfg

    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0), joint_pos=dict(zip(H2_JOINT_NAMES, H2_DEFAULT_JOINT_POS)), joint_vel={".*": 0.0}
        ),
        geoms=(GeomCfg(geom_names_expr=(FOOT_GEOM_REGEX,), priority=1),),
        spec_fn=lambda: mujoco.MjSpec.from_file(str(h2_xml_path())),
        articulation=EntityArticulationInfoCfg(actuators=(XmlActuatorCfg(target_names_expr=(r".*_joint",)),), soft_joint_pos_limit_factor=0.9),
    )


def _action_scale() -> dict[str, float]:
    """0.25 * effort limit / kp of each XML position actuator, keyed by joint name.

    effort = min(actuator forcerange, joint actuatorfrcrange): the joint-level limit is what clamps the torque
    in MuJoCo and what UniLab's action scale is derived from."""
    import mujoco

    m = H2.load_model()
    out = {}
    for i in range(m.nu):
        assert m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT
        jid = int(m.actuator_trnid[i, 0])
        joint = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        kp = float(m.actuator_gainprm[i, 0])
        effort = float(m.actuator_forcerange[i, 1])
        if m.jnt_actfrclimited[jid]:
            effort = min(effort, float(m.jnt_actfrcrange[jid, 1]))
        out[joint] = 0.25 * effort / kp
    assert set(out) == set(H2_JOINT_NAMES), "every H2 joint needs exactly one position actuator"
    return out


H2 = RobotSpec(
    name="h2",
    xml_path_fn=h2_xml_path,
    joint_names=H2_JOINT_NAMES,
    tracked_bodies=H2_TRACKED_BODIES,
    anchor_body="pelvis",
    root_body="pelvis",
    foot_geom_regex=FOOT_GEOM_REGEX,
    com_body="torso_link",
    entity_cfg_fn=_entity_cfg,
    action_scale_fn=_action_scale,
    nconmax=64,
    njmax=400,
    default_joint_pos=H2_DEFAULT_JOINT_POS,
)
