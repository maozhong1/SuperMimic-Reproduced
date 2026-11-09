"""Robot contracts (G1, GR2) and a CPU end-to-end check of every robot-sized component.

The simulator itself (MuJoCo Warp) needs CUDA; everything here runs on CPU: model / entity
construction, motion library (full and compact), analytic heads, world model loss, imagined
rollout with BPTT, policy and critic shapes.
"""

import importlib.util
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from supermimic.robots import get_robot, stand_root_height
from supermimic.robots.gr2 import ASSET_DIR as GR2_ASSET_DIR

ROOT = Path(__file__).resolve().parents[1]
ROBOTS = ["g1", pytest.param("gr2", marks=pytest.mark.skipif(not GR2_ASSET_DIR.exists(), reason="GR2 asset not copied"))]
needs_gr2 = pytest.mark.skipif(not GR2_ASSET_DIR.exists(), reason="GR2 asset not copied")

EXPECTED = {  # N, state, obs, goal/frame, xi
    "g1": (29, 71, 93, 68, 33),
    "gr2": (27, 67, 87, 64, 31),
}


@pytest.mark.parametrize("robot", ROBOTS)
def test_dimensions_and_model(robot):
    spec = get_robot(robot)
    assert (spec.num_joints, spec.state_dim, spec.obs_dim, spec.goal_feat_dim, spec.xi_dim) == EXPECTED[robot]
    m = spec.load_model()
    assert m.nq == 7 + spec.num_joints and m.nv == 6 + spec.num_joints
    bodies = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)}
    assert set(spec.tracked_bodies) <= bodies and spec.anchor_body in spec.tracked_bodies
    assert spec.com_body in bodies
    assert len(spec.tracked_bodies) == 14


@needs_gr2
def test_gr2_contract():
    """Fixed head, anchor = root = base, and UniLab's action scale 0.25 * effort / kp."""
    spec = get_robot("gr2")
    m = spec.load_model()
    assert not any("head" in n for n in spec.joint_names)
    assert spec.anchor_body == spec.root_body == "base"
    kp = [200] + [300, 300, 100, 100, 50, 50, 50] * 2 + [90, 180, 120, 90, 30, 60] * 2
    effort = [74.45] + [74.45, 74.45, 42.75, 42.75, 17.325, 17.325, 17.325] * 2 + [224.4, 95.472, 43.83, 224.4, 43.83, 29.835] * 2
    scale = spec.action_scale_fn()
    for j, k, e in zip(spec.joint_names, kp, effort, strict=True):
        assert scale[j] == pytest.approx(0.25 * e / k, rel=1e-9), j
    assert m.nu == spec.num_joints


@pytest.mark.parametrize("robot", ROBOTS)
def test_entity_builds(robot):
    """mjlab entity (CPU spec compile): actuators, joint order, foot geoms with priority."""
    from mjlab.entity import Entity

    spec = get_robot(robot)
    ent = Entity(spec.entity_cfg_fn())
    m = ent.spec.compile()
    assert ent.joint_names == spec.joint_names
    assert m.nu == spec.num_joints
    ids, names = ent.find_geoms(spec.foot_geom_regex)
    assert len(ids) > 0
    for n in names:
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
        assert m.geom_priority[g] == 1 and m.geom_condim[g] == 3, n
    if robot == "gr2":
        # The XML collision structure is kept (GeomCfg patch, not CollisionCfg): same colliding geoms.
        raw = spec.load_model()
        colliding = lambda mm: int(((mm.geom_contype != 0) | (mm.geom_conaffinity != 0)).sum())  # noqa: E731
        assert colliding(m) == colliding(raw)


def test_stand_root_height():
    assert stand_root_height(get_robot("g1")) == pytest.approx(0.793, abs=2e-3)  # value used by eval_push before
    if GR2_ASSET_DIR.exists():
        assert 0.95 < stand_root_height(get_robot("gr2")) < 1.1


def _synthetic_clip(robot: str, path: Path, seconds: float = 3.0, fps: int = 50, joint_names=None) -> Path:
    """Walk-in-place-like clip: slow forward root drift and joint sinusoids inside the limits."""
    from supermimic.data.motion_convert import ResampledMotion, forward_kinematics

    spec = get_robot(robot)
    m = spec.load_model()
    T = int(seconds * fps)
    t = np.arange(T) / fps
    lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]
    mid, amp = 0.5 * (lo + hi), 0.25 * (hi - lo)
    phase = np.linspace(0, np.pi, spec.num_joints)
    q = mid + amp * np.sin(2 * np.pi * 0.5 * t[:, None] + phase)
    qd = np.gradient(q, 1.0 / fps, axis=0)
    yaw = 0.2 * t
    root_quat = np.stack([np.cos(yaw / 2), 0 * yaw, 0 * yaw, np.sin(yaw / 2)], -1)
    root_pos = np.stack([0.3 * t, 0 * t, stand_root_height(spec) + 0 * t], -1)
    motion = ResampledMotion(
        fps=fps, root_pos=root_pos, root_quat=root_quat, joint_pos=q,
        root_lin_vel_w=np.gradient(root_pos, 1.0 / fps, axis=0),
        root_ang_vel_w=np.stack([0 * t, 0 * t, 0.2 + 0 * t], -1), joint_vel=qd,
    )
    b = forward_kinematics(m, motion)
    np.savez(
        path, fps=np.array([fps], np.float32), joint_pos=q.astype(np.float32), joint_vel=qd.astype(np.float32),
        body_pos_w=b.pos_w.astype(np.float32), body_quat_w=b.quat_w.astype(np.float32),
        body_lin_vel_w=b.lin_vel_w.astype(np.float32), body_ang_vel_w=b.ang_vel_w.astype(np.float32),
        joint_names=np.array(joint_names or spec.joint_names), body_names=np.array(b.body_names),
    )
    return path


@pytest.mark.parametrize("robot", ROBOTS)
def test_pipeline_cpu(robot, tmp_path):
    """Library (full vs compact), analytic reward of the reference, WM loss, imagination + BPTT."""
    from supermimic.data.replay_buffer import StartBatch
    from supermimic.envs.mdp.motion_library import MotionLibrary
    from supermimic.imagination.compiled import ImagStep
    from supermimic.imagination.objective import policy_objective
    from supermimic.imagination.rollout import imagine
    from supermimic.kinematics.fk import HumanoidKinematics
    from supermimic.policy.nets import Critic, CriticCfg, Policy, PolicyCfg
    from supermimic.world_model.heads import AnalyticHeads
    from supermimic.world_model.state_spec import StateSpecCfg
    from supermimic.world_model.wm_trainer import reference_at
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg

    torch.manual_seed(0)
    spec = get_robot(robot)
    nj, K, H, n = spec.num_joints, 10, 4, 16
    files = [_synthetic_clip(robot, tmp_path / f"clip{i}.npz") for i in range(2)]
    q0 = torch.zeros(nj)
    lib = MotionLibrary(files, spec.tracked_bodies, q0, device="cpu", robot=robot)
    lib_c = MotionLibrary(files, spec.tracked_bodies, q0, device="cpu", robot=robot, compact=True, store_dtype=torch.float32)
    f = torch.randint(0, lib.num_frames - 20, (n,))
    for a, b in ((lib.bodies(f).pos_w, lib_c.bodies(f).pos_w), (lib.bodies(f).quat_w, lib_c.bodies(f).quat_w), (lib.goal_window(f, K), lib_c.goal_window(f, K))):
        np.testing.assert_allclose(a.numpy(), b.numpy(), atol=2e-4)
    assert lib.goal_window(f, K).shape == (n, K * spec.goal_feat_dim)

    m = spec.load_model()
    fk = HumanoidKinematics(m, device="cpu")
    lim = torch.as_tensor(m.jnt_range[1:], dtype=torch.float32)
    heads = AnalyticHeads(fk, spec.tracked_bodies, spec.anchor_body, spec.root_body, lim, q0)

    # The reference itself earns the maximal tracking reward (bodies / anchor indexing is consistent).
    qpos, qvel = lib.ref_qpos_qvel(f)
    s_ref = torch.cat([qpos, qvel], -1)
    assert s_ref.shape == (n, spec.state_dim)
    r = heads.reward(s_ref, torch.zeros(n, nj), torch.zeros(n, nj), reference_at(lib, f))
    np.testing.assert_allclose(r["total"].numpy(), heads.reward_cfg.max_step_reward, rtol=1e-4)

    wm = WorldModel(WorldModelCfg(hidden=(64, 64), state_spec=StateSpecCfg(num_joints=nj)), heads, "cpu")
    xi = torch.zeros(n, spec.xi_dim)
    xi[:, 0] = 0.75
    k = 3
    frames = f[None] + torch.arange(k + 1)[:, None]
    states = torch.stack([torch.cat(lib.ref_qpos_qvel(fr), -1) for fr in frames])
    refs = [reference_at(lib, fr) for fr in frames[:-1]]
    loss, _ = wm.window_loss(states, torch.zeros(k, n, nj), xi, torch.zeros(k, n), refs)
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in wm.parameters() if p.grad is not None)

    pcfg = PolicyCfg(encoder_hidden=(64,), decoder_hidden=(64,), num_joints=nj, goal_dim=K * spec.goal_feat_dim)
    policy = Policy(pcfg)
    critic = Critic(CriticCfg(hidden=(32,)), pcfg=pcfg)
    start = StartBatch(state=s_ref, xi=xi, frame=f, prev_action=torch.zeros(n, nj), hist_states=s_ref[None].expand(10, n, -1).contiguous(), hist_actions=torch.zeros(10, n, nj))
    wm.requires_grad_(False)
    ro = imagine(wm, policy, lib, start, H, K, step_fn=ImagStep(wm, q0))
    assert ro.states.shape == (H + 1, n, spec.state_dim) and ro.actions.shape == (H, n, nj)
    v = critic(wm.spec.canonical(ro.states[-1]), ro.goals[-1], wm.spec.normalize_xi(xi), ro.e_root[-1])
    J = policy_objective(ro, v, 0.99)
    (-J).backward()
    grads = [p.grad for p in policy.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads) and sum(float(g.abs().sum()) for g in grads) > 0


@needs_gr2
def test_npz_joint_order_is_checked(tmp_path):
    from supermimic.envs.mdp.motion_library import MotionLibrary

    spec = get_robot("gr2")
    wrong = list(spec.joint_names)
    wrong[0], wrong[1] = wrong[1], wrong[0]
    f = _synthetic_clip("gr2", tmp_path / "bad.npz", joint_names=wrong)
    with pytest.raises(ValueError, match="joint order"):
        MotionLibrary([f], spec.tracked_bodies, torch.zeros(spec.num_joints), device="cpu", robot="gr2")


def test_old_checkpoint_config_is_g1():
    """Checkpoints written before supermimic.robots have no num_joints: they rebuild as G1."""
    from supermimic.policy.nets import Policy, PolicyCfg

    old = {k: v for k, v in asdict(PolicyCfg()).items() if k != "num_joints"}
    p = Policy(PolicyCfg(**old))
    assert p(torch.zeros(2, 930), torch.zeros(2, 680)).shape == (2, 29)


@pytest.mark.parametrize("robot", ROBOTS)
def test_render_scene_compiles(robot):
    spec_path = ROOT / "scripts/media/render_rollout.py"
    mod_spec = importlib.util.spec_from_file_location("render_rollout", spec_path)
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)
    robot_spec = get_robot(robot)
    m = mujoco.MjModel.from_xml_string(mod.two_robot_xml(robot_spec.xml_path, robot_spec.root_body))
    assert m.nq == 2 * (7 + robot_spec.num_joints)
