"""Check the G1 -> H2 joint mapping and sample-check H2 npz clips (from g1_csv_to_h2_npz2.py).

  uv run python scripts/data/check_h2_mapping.py --npz-dir ~/work/data/h2_npz --num-clips 150

1. mapping: joint-name sets, order differences and local joint axes (G1 vs H2 at zero pose, as seen by each
   parent link), actuator limits;
2. data: finite, 29 joints / 33 bodies, joint limit violations, FK consistency (joint_pos + root -> body_pos_w),
   lowest foot point over the floor, pelvis height.
"""

from __future__ import annotations

import random
from pathlib import Path

import mujoco
import numpy as np
import tyro

from supermimic.robots import get_robot


def mapping() -> None:
    g1, h2 = get_robot("g1"), get_robot("h2")
    gn, hn = list(g1.joint_names), list(h2.joint_names)
    print(f"name sets equal: {set(gn) == set(hn)} ({len(gn)} joints)")
    print("positions that differ:", [(i, a, b) for i, (a, b) in enumerate(zip(gn, hn)) if a != b])
    mg, mh = g1.load_model(), h2.load_model()
    for n in hn:
        ja, jb = (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for m in (mg, mh))
        same = np.allclose(mg.jnt_axis[ja], mh.jnt_axis[jb])
        lo_g, hi_g = mg.jnt_range[ja]
        lo_h, hi_h = mh.jnt_range[jb]
        print(f"  {n:28s} axis {'same' if same else 'DIFF'}  G1 [{lo_g:+.2f},{hi_g:+.2f}]  H2 [{lo_h:+.2f},{hi_h:+.2f}]")


def data(npz_dir: Path, num_clips: int, seed: int) -> None:
    spec = get_robot("h2")
    m = spec.load_model()
    d = mujoco.MjData(m)
    lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]
    gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""  # noqa: E731
    foot = [g for g in range(m.ngeom) if "_foot" in gname(g) and gname(g).endswith("_collision")]
    files = sorted(npz_dir.glob("*.npz"))
    random.Random(seed).shuffle(files)
    files = files[:num_clips]
    viol, frames, fk_err, low, rootz, bad = np.zeros(m.nq - 7), 0, 0.0, [], [], 0
    root = spec.npz_body_names().index(spec.root_body)
    for f in files:
        with np.load(f) as z:
            q, bp, bq = z["joint_pos"], z["body_pos_w"], z["body_quat_w"]
            bad += not all(np.isfinite(z[k]).all() for k in z.files)
        assert q.shape[1] == spec.num_joints and bp.shape[1] == m.nbody, (f, q.shape, bp.shape)
        viol += ((q < lo - 1e-4) | (q > hi + 1e-4)).sum(0)
        frames += len(q)
        rootz.append(bp[:, root, 2])
        for t in np.linspace(0, len(q) - 1, 5).astype(int):
            d.qpos[:3], d.qpos[3:7], d.qpos[7:] = bp[t, root], bq[t, root], q[t]
            mujoco.mj_kinematics(m, d)
            fk_err = max(fk_err, float(np.abs(d.xpos[1:] - bp[t, 1:]).max()))
            low.append(min(d.geom_xpos[g][2] - m.geom_size[g][0] for g in foot))
    rz = np.concatenate(rootz)
    print(f"{len(files)} clips, {frames} frames ({frames / 50 / 3600:.2f} h), non-finite files: {bad}")
    print(f"FK max body position error {fk_err:.2e} m")
    print(f"lowest foot point z: min {min(low):+.3f}  median {np.median(low):+.3f}  max {max(low):+.3f}")
    print(f"pelvis z p1/p50/p99: {np.percentile(rz, [1, 50, 99]).round(3)}")
    print("joint limit violations:", {n: int(v) for n, v in zip(spec.joint_names, viol) if v} or "none")


def main(npz_dir: Path | None = None, num_clips: int = 150, seed: int = 0) -> None:
    mapping()
    if npz_dir is not None:
        data(npz_dir.expanduser(), num_clips, seed)


if __name__ == "__main__":
    tyro.cli(main)
