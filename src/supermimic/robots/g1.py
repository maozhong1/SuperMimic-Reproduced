"""Unitree G1, 29 DoF (mjlab's asset and actuator configuration, unchanged)."""

from __future__ import annotations

from pathlib import Path

from supermimic.robots import RobotSpec

G1_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
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

G1_TRACKED_BODIES: tuple[str, ...] = (
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
)


def g1_xml_path() -> Path:
    import mjlab

    return Path(mjlab.__file__).parent / "asset_zoo/robots/unitree_g1/xmls/g1.xml"


def _entity_cfg():
    from mjlab.asset_zoo.robots import get_g1_robot_cfg

    return get_g1_robot_cfg()


def _action_scale() -> dict[str, float]:
    from mjlab.asset_zoo.robots import G1_ACTION_SCALE

    return G1_ACTION_SCALE


G1 = RobotSpec(
    name="g1",
    xml_path_fn=g1_xml_path,
    joint_names=G1_JOINT_NAMES,
    tracked_bodies=G1_TRACKED_BODIES,
    anchor_body="torso_link",
    root_body="pelvis",
    foot_geom_regex=r"^(left|right)_foot[1-7]_collision$",
    com_body="torso_link",
    entity_cfg_fn=_entity_cfg,
    action_scale_fn=_action_scale,
    nconmax=35,
    njmax=250,
)
