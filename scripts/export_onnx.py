"""Export a trained policy to ONNX and check it against PyTorch.

  uv run --with onnx --with onnxruntime python scripts/export_onnx.py \
      --ckpt logs/gr2_100k/ckpt_0100000.pt --out logs/gr2_100k/policy_100000.onnx

Graph: ``action = policy(obs_hist, goal)``, batch dimension dynamic.
  obs_hist [B, history * (3N+6)]  term-major history: joint_pos, joint_vel, base_ang_vel, projected_gravity, actions
  goal     [B, K * (2N+10)]       per frame: joint_pos N, joint_vel N, gravity 3, ang_vel_b 3, lin_vel_h 3, height 1
  action   [B, N]                 raw policy action; joint target = default_pose + action * action_scale (not in the graph)
The fixed input scalings are inside the graph. A sidecar ``<out>.json`` records dimensions, joint names, default pose
and the per-joint action scale needed for deployment.

Consistency check: PyTorch vs onnxruntime on random inputs at several batch sizes (fails if max |diff| > --tol).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import tyro

from supermimic.eval.benchmark import load_policy_from_checkpoint
from supermimic.robots import get_robot
from supermimic.world_model.heads import obs_terms


class _Wrapper(torch.nn.Module):
    def __init__(self, policy: torch.nn.Module) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.policy(obs_hist, goal, "motion")


@torch.no_grad()
def main(ckpt: Path, out: Path | None = None, opset: int = 17, tol: float = 1e-4, batches: list[int] = [1, 7, 256], seed: int = 0) -> None:
    import onnx
    import onnxruntime as ort

    out = out or ckpt.with_suffix(".onnx")
    policy, cfg = load_policy_from_checkpoint(ckpt, "cpu")
    policy.float().eval()
    pcfg = policy.cfg
    robot = get_robot(cfg["robot"])
    n = pcfg.num_joints
    obs_dim, goal_dim = pcfg.history * (3 * n + 6), pcfg.goal_dim
    model = _Wrapper(policy).eval()

    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = (torch.zeros(2, obs_dim), torch.zeros(2, goal_dim))
    torch.onnx.export(model, dummy, str(out), input_names=["obs_hist", "goal"], output_names=["action"], opset_version=opset,
                      dynamic_axes={"obs_hist": {0: "batch"}, "goal": {0: "batch"}, "action": {0: "batch"}}, dynamo=False)
    onnx.checker.check_model(onnx.load(str(out)))

    scales = robot.action_scale_fn()
    default_pose = [0.0] * n  # GR2 default pose is all zeros; G1 would need its own (see RobotSpec)
    meta = {
        "robot": cfg["robot"], "num_joints": n, "history": pcfg.history, "obs_dim": obs_dim, "goal_dim": goal_dim,
        "obs_terms": [{"name": k, "dim": d} for k, d in obs_terms(n)], "joint_names": list(robot.joint_names),
        "action_scale": [float(scales[j]) for j in robot.joint_names], "default_joint_pos": default_pose,
        "note": "joint target = default_joint_pos + action * action_scale",
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(seed)
    worst = 0.0
    for b in batches:
        o = rng.standard_normal((b, obs_dim)).astype(np.float32)
        g = rng.standard_normal((b, goal_dim)).astype(np.float32)
        ref = model(torch.from_numpy(o), torch.from_numpy(g)).numpy()
        got = sess.run(["action"], {"obs_hist": o, "goal": g})[0]
        assert got.shape == ref.shape == (b, n), (got.shape, ref.shape)
        d = float(np.abs(got - ref).max())
        worst = max(worst, d)
        print(f"batch {b:4d}: max |onnx - torch| = {d:.2e}   (|action| max {np.abs(ref).max():.2f})")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) and {out.with_suffix('.json')}")
    assert worst <= tol, f"ONNX output differs from PyTorch by {worst:.2e} > tol {tol:.0e}"
    print("consistency check passed")


if __name__ == "__main__":
    tyro.cli(main)
