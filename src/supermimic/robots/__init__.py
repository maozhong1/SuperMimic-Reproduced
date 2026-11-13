"""Robot descriptions: one ``RobotSpec`` per humanoid, everything else derives its sizes from it.

A spec names the MJCF, the actuated joint order, the bodies the tracking reward uses and the
mjlab entity configuration. All tensor widths (state, observation, goal, context) follow from
the number of actuated hinges ``N``:

    state  s  = qpos(7+N) | qvel(6+N)          13 + 2N   (G1: 71,  GR2: 67)
    obs    o  = q, qd, w_b, g_b, a_prev          3N + 6   (G1: 93,  GR2: 87)
    goal   g  = q-q0, qd, g, w_b, v_h, h          2N + 10  (G1: 68,  GR2: 64) per future frame
    ctx    xi = friction, encoder bias, COM        N + 4   (G1: 33,  GR2: 31)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mujoco
    from mjlab.entity import EntityCfg


def state_dim(nj: int) -> int:
    return 13 + 2 * nj


def obs_dim(nj: int) -> int:
    return 3 * nj + 6


def goal_feat_dim(nj: int) -> int:
    return 2 * nj + 10


def xi_dim(nj: int) -> int:
    return nj + 4


def joints_from_state_dim(d: int) -> int:
    nj, r = divmod(d - 13, 2)
    assert r == 0 and nj > 0, f"not a state width: {d}"
    return nj


@dataclass(frozen=True)
class RobotSpec:
    name: str
    xml_path_fn: Callable[[], Path]  # robot-only MJCF (no floor; mjlab adds the terrain)
    joint_names: tuple[str, ...]  # actuated hinges in MjModel order (= qpos[7:], npz joint order)
    tracked_bodies: tuple[str, ...]  # bodies of the tracking reward (order defines the reward layout)
    anchor_body: str  # BeyondMimic anchor of the reward's re-anchoring and global root terms
    root_body: str  # free-joint body (terminations, goal root features)
    foot_geom_regex: str  # foot contact geoms (friction DR, xi friction)
    com_body: str  # body whose COM is randomised (xi COM offset)
    entity_cfg_fn: Callable[[], EntityCfg]
    action_scale_fn: Callable[[], dict[str, float]]
    nconmax: int
    njmax: int
    family: str = ""  # robots of one family share checkpoints / libraries (same joints and states), e.g. gr2_wrist -> gr2

    @property
    def compat(self) -> str:
        return self.family or self.name

    @property
    def xml_path(self) -> Path:
        return self.xml_path_fn()

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    @property
    def state_dim(self) -> int:
        return state_dim(self.num_joints)

    @property
    def obs_dim(self) -> int:
        return obs_dim(self.num_joints)

    @property
    def goal_feat_dim(self) -> int:
        return goal_feat_dim(self.num_joints)

    @property
    def xi_dim(self) -> int:
        return xi_dim(self.num_joints)

    @cached_property
    def _model(self) -> mujoco.MjModel:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
        assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE, f"{self.name}: first joint must be the free joint"
        assert tuple(names[1:]) == self.joint_names, f"{self.name}: joint order mismatch: {names[1:]}"
        assert model.nq == 7 + self.num_joints and model.nv == 6 + self.num_joints
        root = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.jnt_bodyid[0]))
        assert root == self.root_body, f"{self.name}: free joint is on {root}, spec says {self.root_body}"
        return model

    def load_model(self) -> mujoco.MjModel:
        """Standalone CPU MjModel (shared, do not mutate). Asserts the joint order of the spec."""
        return self._model


def stand_root_height(spec: RobotSpec, clearance: float = 0.001) -> float:
    """Root height that puts the lowest foot-contact point ``clearance`` above the floor at the
    zero joint pose (upright root). Handles box, capsule, sphere and mesh foot geoms."""
    import re

    import mujoco
    import numpy as np

    m = spec.load_model()
    d = mujoco.MjData(m)
    d.qpos[:] = 0.0
    d.qpos[3] = 1.0
    mujoco.mj_kinematics(m, d)
    lowest = np.inf
    for g in range(m.ngeom):
        if not re.match(spec.foot_geom_regex, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""):
            continue
        R, c, s, t = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g], m.geom_size[g], m.geom_type[g]
        if t == mujoco.mjtGeom.mjGEOM_BOX:
            z = c[2] - np.abs(R[2]) @ s
        elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            z = c[2] - abs(R[2, 2]) * s[1] - s[0]
        elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
            z = c[2] - s[0]
        elif t == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            z = (c[2] + (v @ R.T)[:, 2]).min()
        else:
            raise ValueError(f"unsupported foot geom type {t}")
        lowest = min(lowest, float(z))
    assert np.isfinite(lowest), f"no geom matches {spec.foot_geom_regex}"
    return -lowest + clearance


def get_robot(name: str) -> RobotSpec:
    if name == "g1":
        from supermimic.robots.g1 import G1

        return G1
    if name == "gr2":
        from supermimic.robots.gr2 import GR2

        return GR2
    if name == "gr2_wrist":
        from supermimic.robots.gr2 import GR2_WRIST

        return GR2_WRIST
    raise ValueError(f"unknown robot {name!r} (known: g1, gr2, gr2_wrist)")


ROBOT_NAMES = ("g1", "gr2", "gr2_wrist")
