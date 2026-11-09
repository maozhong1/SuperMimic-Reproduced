"""Pack filtered npz clips into one compact library file (fp16 (13+2N)-d reference states; G1: 71-d).

  uv run python scripts/data/build_library.py --npz-dir data/npz/bones_seed --manifest data/npz/bones_seed/manifest_filtered.csv --out data/library/bones_seed_compact.pt
  uv run python scripts/data/build_library.py --robot gr2 --npz-dir data/npz/bones_seed_gr2 --out data/library/bones_seed_gr2_compact.pt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import tyro
from tqdm import tqdm

from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.robots import ROBOT_NAMES, get_robot


def main(npz_dir: Path, out: Path, manifest: Path | None = None, workers: int = 16, robot: str = "g1") -> None:
    assert robot in ROBOT_NAMES, robot
    spec = get_robot(robot)
    files = discover_npz(npz_dir, manifest=manifest)
    print(f"{len(files)} clips")
    from concurrent.futures import ThreadPoolExecutor

    def load(f):
        with np.load(f) as d:
            if "joint_names" in d.files and tuple(str(n) for n in d["joint_names"]) != spec.joint_names:
                raise ValueError(f"{f}: joint order does not match the {robot} model order")
            names = [str(n) for n in d["body_names"]]
            r = names.index(spec.root_body)
            ref = np.concatenate([d["body_pos_w"][:, r], d["body_quat_w"][:, r], d["joint_pos"], d["body_lin_vel_w"][:, r], d["body_ang_vel_w"][:, r], d["joint_vel"]], axis=1)
        return f.stem, ref.astype(np.float16)

    with ThreadPoolExecutor(workers) as ex:
        items = list(tqdm(ex.map(load, files), total=len(files)))
    names = [n for n, _ in items]
    lengths = [int(r.shape[0]) for _, r in items]
    starts = np.cumsum([0] + lengths[:-1]).tolist()
    ref = torch.from_numpy(np.concatenate([r for _, r in items], 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    assert ref.shape[1] == spec.state_dim, (ref.shape, spec.state_dim)
    torch.save({"ref": ref, "names": names, "starts": starts, "lengths": lengths, "robot": robot}, out)
    print(f"saved {out}: {ref.shape[0]} frames ({ref.shape[0]/50/3600:.2f} h), {ref.numel()*2/1e9:.2f} GB fp16")


if __name__ == "__main__":
    tyro.cli(main)
