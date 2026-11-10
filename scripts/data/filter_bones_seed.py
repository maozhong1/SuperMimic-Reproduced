"""Appendix E filters for BONES-SEED (plan D16): name filter + feasibility filter.

Reads converted npz clips (50 Hz, per-body link states incl. the robot root body) and writes
<npz_dir>/manifest_filtered.csv with one row per clip: name, frames, keep, reason.

  uv run python scripts/data/filter_bones_seed.py --npz-dir data/npz/bones_seed --workers 24
  uv run python scripts/data/filter_bones_seed.py --robot gr2 --npz-dir ~/work/data/gr2_npz

Robot-specific parts (root body name, standing-height window, foot collision geoms) come from the
RobotSpec; the standing window is the G1 one (0.60-0.85 m around a 0.793 m stand height) scaled by the
robot's stand height. The other thresholds are in metres/seconds and robot-independent.
"""

from __future__ import annotations

import csv
import multiprocessing as mp
from collections import Counter
from pathlib import Path

import numpy as np
import tyro

from supermimic.robots import get_robot, stand_root_height

# SONIC's keyword list (gear_sonic/data_process/filter_and_copy_bones_data.py) plus the paper's examples,
# underscore-delimited to avoid substring false positives (plan D16). The pure-acrobatics terms were
# dropped on 2026-10-21 to match the paper's post-name-filter count; the e8 full benchmark showed the paper
# does filter them (129,491 kept vs the paper's 129,514 with them removed), so they are back (EXPERIMENTS e8).
SONIC_KEYWORDS = [
    "bed", "bike", "chair", "climb", "com_up_50cm", "sitting", "step_on", "seat", "table", "_sit_", "sit_",
    "ladder", "crutch", "_bed_", "_ride_", "scooter", "stepdown", "box_hspu", "50cm_box_", "on_box",
    "fall_from", "on_1m", "form_box", "off_1m", "230m", "jump_over_obstacle_", "lift_crate_come_up_",
    "stair", "box_jump", "box_dips", "walking_on_edge", "push_obstacle",
]
SONIC_ACROBATICS = ["acrobatics_", "cartwheel", "handstand", "handstand_ff_", "safety_roll", "kozak_dance", "jump_to_shoulder_roll", "monkey_jump"]
PAPER_EXTRA = ["_box_", "box_", "_stair", "jump_off_50cm", "off_50cm", "fence", "look_over"]
KEYWORDS = sorted(set(SONIC_KEYWORDS + SONIC_ACROBATICS + PAPER_EXTRA))

# Feasibility thresholds (paper Appendix E + plan D16 choices).
ELEVATED_Z = 0.20
ELEVATED_S = 0.6
PENETRATE_Z_HARD = -0.12
PENETRATE_Z_SOFT = -0.08
PENETRATE_FRAC = 0.80
FLOAT_Z = 0.02
FLOAT_S = 0.5
UPRIGHT_GZ = 0.9
G1_STAND_H = 0.7928637524221748  # stand_root_height(G1); STAND_Z below is the G1 window
STAND_Z = (0.60, 0.85)
STILL_V = 0.1
STILL_QD = 0.5
FPS = 50


def name_filter(name: str) -> str | None:
    n = name.lower()
    for k in KEYWORDS:
        if k in n:
            return f"name:{k}"
    return None


def feasibility_filter(npz_path: str, robot: str = "g1") -> str | None:
    spec = get_robot(robot)
    stand_z = tuple(z * stand_root_height(spec) / G1_STAND_H for z in STAND_Z)
    with np.load(npz_path) as d:
        names = [str(x) for x in d["body_names"]]
        pos = d["body_pos_w"]  # (T, 30, 3)
        quat = d["body_quat_w"]  # (T, 30, 4) wxyz
        jv = d["joint_vel"]
        lin = d["body_lin_vel_w"]
    T = pos.shape[0]
    zmin = pos[:, :, 2].min(1)
    pel = names.index(spec.root_body)
    # elevated support: zmin > 0.20 for >= 0.6 s
    if _longest_run(zmin > ELEVATED_Z) >= ELEVATED_S * FPS:
        return "elevated_support"
    if (zmin < PENETRATE_Z_HARD).any() or (zmin < PENETRATE_Z_SOFT).mean() > PENETRATE_FRAC:
        return "ground_penetration"
    q = quat[:, pel]
    # gravity z-component in pelvis frame: -(R^T e_z)_z = -(1 - 2(x^2+y^2)) for wxyz quaternion
    gz = -(1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2))
    upright = gz < -UPRIGHT_GZ
    standing = (pos[:, pel, 2] > stand_z[0]) & (pos[:, pel, 2] < stand_z[1])
    still = (np.linalg.norm(lin[:, pel], axis=1) < STILL_V) & (np.sqrt((jv**2).mean(1)) < STILL_QD)
    gate = upright & standing & still
    if _longest_run(gate) >= FLOAT_S * FPS:
        # Exact lowest point of the foot collision capsules (paper: mj_geomDistance to the plane).
        foot_z = _foot_sole_height(npz_path, gate, robot)
        floating = gate.copy()
        floating[gate] = foot_z > FLOAT_Z
        if _longest_run(floating) >= FLOAT_S * FPS:
            return "floating_feet"
    return None


_MODELS: dict = {}


def _foot_geoms(spec):
    import re

    import mujoco

    m = spec.load_model()
    geoms = [g for g in range(m.ngeom) if re.match(spec.foot_geom_regex, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
    if spec.name == "g1":
        assert len(geoms) == 14 and all(m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE for g in geoms)
    assert geoms, f"no foot geoms match {spec.foot_geom_regex}"
    return m, geoms


def _geom_low_z(m, d, g) -> float:
    """Exact lowest world z of one foot collision geom (box, capsule, sphere or mesh)."""
    import mujoco

    R, c, s, t = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g], m.geom_size[g], m.geom_type[g]
    if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return c[2] - abs(R[2, 2]) * s[1] - s[0]
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        return c[2] - np.abs(R[2]) @ s
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return c[2] - s[0]
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        return (c[2] + (v @ R.T)[:, 2]).min()
    raise ValueError(f"unsupported foot geom type {t}")


def _foot_sole_height(npz_path: str, gate: np.ndarray, robot: str = "g1") -> np.ndarray:
    """Lowest point over both feet's collision geoms, for the gated frames (returns len(gate.sum()),)."""
    import mujoco

    spec = get_robot(robot)
    if robot not in _MODELS:
        _MODELS[robot] = _foot_geoms(spec)
    m, geoms = _MODELS[robot]
    d = mujoco.MjData(m)
    with np.load(npz_path) as z:
        names = [str(x) for x in z["body_names"]]
        pel = names.index(spec.root_body)
        root_pos = z["body_pos_w"][:, pel]
        root_quat = z["body_quat_w"][:, pel]
        jp = z["joint_pos"]
    idx = np.nonzero(gate)[0]
    out = np.empty(len(idx))
    for k, t in enumerate(idx):
        d.qpos[:3] = root_pos[t]
        d.qpos[3:7] = root_quat[t]
        d.qpos[7:] = jp[t]
        mujoco.mj_kinematics(m, d)
        out[k] = min(_geom_low_z(m, d, g) for g in geoms)
    return out


def _longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


def _one(args: tuple[str, str]) -> tuple[str, int, str | None]:
    path, robot = args
    name = Path(path).stem
    r = name_filter(name)
    frames = 0
    if r is None:
        r = feasibility_filter(path, robot)
    with np.load(path) as d:
        frames = int(d["joint_pos"].shape[0])
    return name, frames, r


def main(npz_dir: Path, workers: int = 24, out: Path | None = None, robot: str = "g1") -> None:
    files = sorted(str(p) for p in npz_dir.glob("*.npz"))
    print(f"{len(files)} clips")
    with mp.Pool(workers) as pool:
        rows = pool.map(_one, [(f, robot) for f in files], chunksize=64)
    out = out or npz_dir / "manifest_filtered.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "frames", "keep", "reason"])
        for name, frames, r in rows:
            w.writerow([name, frames, int(r is None), r or ""])
    reasons = Counter(r for _, _, r in rows)
    kept = sum(1 for _, _, r in rows if r is None)
    name_removed = sum(1 for _, _, r in rows if r and r.startswith("name:"))
    feas_removed = sum(1 for _, _, r in rows if r and not r.startswith("name:"))
    hours = sum(fr for _, fr, r in rows if r is None) / FPS / 3600
    print(f"raw {len(rows)} -> after name filter {len(rows) - name_removed} (paper 129,667) -> after feasibility {kept} (paper 129,514); {hours:.2f} h kept (paper 264.56)")
    print("by reason:", dict(reasons.most_common(40)))
    print("saved", out)


if __name__ == "__main__":
    tyro.cli(main)
