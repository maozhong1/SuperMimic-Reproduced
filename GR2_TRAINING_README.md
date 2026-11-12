# SuperMimic on Fourier GR2: training, evaluation, visualization

Port of SuperMimic (world model + imagined rollouts for whole-body motion tracking) from Unitree G1 to Fourier GR2
(fixed head, 27 DoF, realfoot MJCF). The algorithm is unchanged; see `README.md`, `docs/PAPER_SPEC.md`,
`docs/EXPERIMENTS.md`. This file covers only what is GR2-specific.

## 1. Setup

Single RTX 5090 D (32 GB), CUDA 13, Python 3.12, `uv`. Prefix commands with `env -u PYTHONPATH` (a system
`PYTHONPATH`, e.g. ROS, can interfere). If `uv` is missing: `pip install --user --break-system-packages uv` and
`export PATH=$HOME/.local/bin:$PATH`.

```bash
cd ~/work/SuperMimic-Reproduced
env -u PYTHONPATH uv sync
env -u PYTHONPATH uv run pytest tests/    # 23 passed, 17 skipped (skipped ones need G1 LAFAN1 data)
```
The first run compiles MuJoCo Warp / torch.compile kernels (first ~50 iterations at ~0.2 it/s); steady state is
~4.4 it/s.

## 2. Implementation

`src/supermimic/robots/` holds one `RobotSpec` per robot (MJCF, joint order, tracked / anchor / root bodies, foot
geoms, COM body, mjlab entity, action scale, `nconmax/njmax`). Tensor widths follow from the joint count N: 
state 13+2N, observation 3N+6, goal 2N+10 per frame, context N+4 (G1 71/93/68/33, GR2 67/87/64/31). 
For GR2 the policy observation is 870-d (10-frame history), the goal 640-d, the action 27-d.

GR2 conventions (`robots/gr2.py`, assets in `src/supermimic/assets/gr2/`, from UniLab `gr2_sonic_27dof_realfoot`):
- XML position actuators; action scale `0.25 * effort / kp`; default pose all zeros.
- Foot geoms `^[lr]_foot_(sole|mesh)$` get `priority=1`; other collision as in the XML.
- **anchor = root = `base`** (SONIC convention; G1 uses `torso_link`).
- 14 tracked bodies as in UniLab (`base`, thigh_roll / shank_pitch / foot_roll, `waist_yaw_link`,
  upper_arm_roll / lower_arm_pitch / hand_yaw, left and right).
- Sim settings come from `tracking_env_cfg.py`: 5 ms step, decimation 4 (50 Hz). **The XML `<option>`
  (timestep 0.001, ...) is ignored by mjlab**, unlike UniLab's 0.001 s.
- `nconmax=64`, `njmax=400` are estimates, **not verified**. They are real buffer limits: when the total contacts
  exceed `num_envs * nconmax` the extra collision pairs are silently dropped (`if cid >= naconmax: return` in
  `mujoco_warp`), and `njmax` is a hard per-world constraint-row limit. No warning appears in the logs, but since the
  drop is silent that does not prove there is no overflow.

`sm-train --robot g1|gr2`; checkpoints and libraries record the robot (old ones are G1).
 Eval/play/record scripts follow the checkpoint's robot.

## 3. Data

```
BONES-SEED G1 CSV --(UniLab scripts/motion/g1_csv_to_gr2_npz.py)--> GR2 npz (50 Hz)
  --> filter_bones_seed.py --robot gr2 --> manifest_filtered.csv --> build_library.py --robot gr2 --> library .pt
```

npz keys: `fps, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w, joint_names,
body_names`. Checked (400 random clips): fps = 50, `joint_names` equals the RobotSpec order, 31 `body_names`
(incl. `world`, `base` and the 14 tracked bodies) in the same order. FK of the GR2 MJCF from `joint_pos`
matches the npz body poses (< 0.5 deg, < 1 mm; 5 clips x 8 frames).

`filter_bones_seed.py` had G1 assumptions and now takes `--robot`: root body from the spec, standing-height window
scaled by `stand_root_height` (G1 0.793 m, GR2 1.011 m), foot sole height from the spec's foot geoms (box / capsule /
sphere / mesh).

Data: `~/work/data/full_gr2_npz`, 51668 npz (103.33 h), already filtered by the user's own filter. Library `gr2_full.pt`:
18.6 M frames, 397,299 one-second bins, 2.49 GB, built in ~1 min.
```bash
D=~/work/data/full_gr2_npz
env -u PYTHONPATH uv run python scripts/data/build_library.py --robot gr2 --npz-dir $D --out ~/work/data/lib/gr2_full.pt
```
`filter_bones_seed.py --robot gr2 --npz-dir $D` (writes `manifest_filtered.csv`, pass it with `--manifest`) removed 0 more
by name and flagged 265 `floating_feet` clips.

## 4. Training

```bash
env -u PYTHONPATH uv run sm-train --robot gr2 --run-name gr2_full \
  --data.library ~/work/data/lib/gr2_full.pt --env.num-envs 4096 --policy-trainer.batch-size 2048 \
  --schedule.total-iterations 200000 --schedule.checkpoint-interval 2000 [--resume <ckpt>]
```
- Hyper-parameters are G1's (Table A.5/A.6) and were not tuned for GR2; `anchor_obs=False` (paper observation).
- One iteration = one 50 Hz control step of all 4096 envs. H=32 is the imagination horizon, UTD 0.4 means 0.4 gradient
  updates per iteration, gamma 0.99. About 4.4 it/s; 200000 it took 14.8 h, 17.7 GB GPU.
- **The learning rate depends on `total-iterations`**: cosine decay to 3% over the first 78% of it, then constant.
- `--resume` restores the networks, optimizers, sampler state and iteration; the replay buffer is refilled.
  Use a new `--run-name`.
- To survive SSH disconnects, detach the run:
  `setsid nohup env -u PYTHONPATH uv run sm-train ... > logs/<run>/stdout.log 2>&1 < /dev/null &`


## 5. Logs and metrics

```bash
env -u PYTHONPATH uv run python scripts/tb_tail.py logs/<run> --last 3 --filter policy/imag_r/
env -u PYTHONPATH uv run tensorboard --logdir logs --port 6006      # logs/<run>/tb
```
- `train/episode_norm_return` (`train_ret`): per finished training episode, reward sum / (0.1 * length), averaged.
  Training episodes have noise, domain randomization and reset perturbations and start where the sampler says, so it
  is not comparable to the evaluation return.
- `train/episode_len`: mean finished-episode length in 50 Hz steps (see the sampling artifact above).
- `policy/imag_r/<term>`: per-step reward terms in imagined rollouts (not the real simulator). Caps: body terms 0.02,
  root terms 0.01. These move with the replay-buffer content, so a dip is not necessarily a worse policy.
- Also: `policy/imag_survival_H`, `wm/online_nse_8step`, `wm/term_acc`, `sampler/entropy`.

Rewards are hard-coded (no config file) in two places that must be changed together: the simulator side in
`envs/tracking_env_cfg.py` and the imagination side `TrackingRewardCfg` in `envs/mdp/rewards.py` (weights: root pos/ori
0.5, body pos/ori/lin_vel/ang_vel 1.0, action_rate -0.1, joint_limit -2.0; stds 0.3/0.4/0.3/0.4/1.0/3.14; all times
dt = 0.02). Same as G1, not tuned for GR2.

## 6. Evaluation and visualization

Protocol: each clip starts at its first frame, deterministic, no noise, no domain randomization, played until the end
or a fall.

```bash
env -u PYTHONPATH uv run python scripts/eval_benchmark.py --ckpt logs/<run>/ckpt_latest.pt \
  --npz-dir ~/work/data/full_gr2_npz --clip-list <file with one clip stem per line> \
  --num-envs 500 --out logs/<run>/bench.json                 # 500 clips < 2 min; all 51668 clips ~3 h (not run); output is JSON
# single clips: --clips <name> --num-envs 1 instead of --clip-list

env -u PYTHONPATH uv run python scripts/play.py --ckpt logs/<run>/ckpt_latest.pt --npz-dir ~/work/data/full_gr2_npz \
  --clips <name> --num-envs 1 --viewer viser                 # browser: http://<host>:8080
```
`play.py` shows the policy tracking the reference (translucent = reference). Over SSH, if port 8080 is not reachable,
use `ssh -L 8080:localhost:8080 <user>@<host>`.

### Offline mp4

Record the rollouts once, then render. Verified on GR2 (needs a system `ffmpeg`, e.g. `apt install ffmpeg`).
```bash
env -u PYTHONPATH uv run python scripts/media/record_rollout.py --ckpt logs/<run>/ckpt_latest.pt \
  --npz-dir ~/work/data/full_gr2_npz --clips <stem1> <stem2> ... --out media/gr2_rollouts.npz
MUJOCO_GL=egl env -u PYTHONPATH uv run python scripts/media/render_rollout.py \
  --traj media/gr2_rollouts.npz --out-dir media/gr2_out --fps 25 --distance 4.5 --concat media/gr2_all.mp4
scp <user>@<host>:~/work/SuperMimic-Reproduced/media/gr2_all.mp4 .
```
- `--clips` takes file stems **without** `.npz`; `--traj` / `--out-dir` are options, not positional arguments.
- Solid grey = policy in simulation, translucent blue = reference; the label shows the normalized return.
- The camera is set when rendering, so changing it does not need a new recording. `--distance` default 2.7 makes the
  robot fill the frame; 4.5 shows the whole body with margin. Other options: `--elevation` (-14), `--azimuth` (140),
  `--width/--height` (1280x720). The look-at height (0.65 m) is hard-coded. `--seconds N` trims each clip; omit it for full length.

### Export to ONNX

`scripts/export_onnx.py` exports the policy (`action = policy(obs_hist, goal)`) with **batch size 1** (add `--dynamic`
for a dynamic batch dimension) and checks it against PyTorch on random inputs (fails if max |diff| > `--tol`, default
1e-4). `onnx` / `onnxruntime` are not project dependencies; `uv run --with` installs them temporarily.
```bash
env -u PYTHONPATH uv run --with onnx --with onnxruntime python scripts/export_onnx.py \
  --ckpt logs/gr2_full/ckpt_0200000.pt --out logs/gr2_full/policy_200000.onnx
```
It writes `policy_*.onnx` (34.8 MB) and a sidecar `policy_*.json` (dimensions, obs layout, joint names, default pose,
per-joint action scale). Verified on `gr2_full` ckpt_0200000: inputs `[1, 870]`, `[1, 640]`, output `[1, 27]`,
max |onnx - torch| 2.6e-6; onnxruntime rejects batch 2.

| | name | shape | content |
|---|---|---|---|
| in | `obs_hist` | [1, 870] | 10 frames, term-major: `joint_pos` 270, `joint_vel` 270, `base_ang_vel` 30, `projected_gravity` 30, previous `actions` 270 |
| in | `goal` | [1, 640] | 10 future frames x 64: `joint_pos` 27, `joint_vel` 27, gravity 3, ang_vel_b 3, lin_vel_h 3, height 1 |
| out | `action` | [1, 27] | raw action |

- The fixed input scalings (`obs_term_scale`, `goal_term_scale`) are inside the graph; do not scale again.
- The graph does **not** contain the action mapping: `joint target = default_joint_pos (all 0 for GR2) + action *
  action_scale[joint]`, with `action_scale = 0.25 * effort / kp` per joint (in the sidecar json).
- The consumer must build `obs_hist` / `goal` exactly as the training environment does (`envs/mdp/observations.py`,
  `world_model/heads.py:obs_terms`); that code is not part of the export.

## 7. Results

Evaluation: deterministic, first-frame start, no falls in any run below. Old = `gr2_100k` (5408 clips, 100k it); new =
`gr2_full` (51668 clips, 200k it, from scratch).

| 500 clips (seed 0) | Model | Falls | Mean return | Step-weighted |
|---|---|---|---|---|
| random from `gr2_5k` set | old | 0 | 0.849 | 0.834 |
| random from `gr2_5k` set | **new** | 0 | **0.901** | **0.882** |
| random from full set | new | 0 | 0.911 (median 0.942, p10 0.790) | 0.912 |

- Paired on the same 500 clips: new +0.052 (standard error 0.0015), better on 487 of 500; worse on 13, mainly
  `walk_fast_forward_injured_*` and `Turn_Start_Walk` (up to -0.13). The gain mixes more data and longer training.
- New model on the full-set sample by checkpoint: 0.870 (50k), 0.881 (100k), 0.909 (150k), 0.911 (200k). No gain after
  the lr floor (it 156250); 500 clips, one seed.
- Training: `train_ret` 0.66 (30k) -> 0.72 (200k), no NaN. `nse1` / `nse8` = world model normalized squared state error
  after 1 / 8 autoregressive steps on replay segments (lower is better): 0.0018 -> 0.0011 (`nse1`).
- Lowest returns are dance clips (`dance_hiphop_barbie_*_M` 0.468, `dance_latino_salsa_step_360_*_M` 0.517,
  `dance_hiphop_power_step_*_fast` 0.543); everyday motions (washing dishes, peeling vegetables) score >= 0.97. Causes
  were not analysed. Rollout video: `media/gr2_full_18.mp4`.
- Files: `logs/gr2_full/bench500*.json`, clip lists `eval500_clips.txt`, `eval500_old5k_clips.txt`.

## 8. Open items (not done / not verified)

- Evaluation of all 51668 clips; analysis of the low-return dance clips (cause unknown).
- Termination-reason counters in training; per-body error and triggering termination term for fallen clips.
- GR2-specific tuning (termination thresholds 0.25 / 0.8, reward stds, 0.001 s step): no evidence it is needed.

## 9. Layout

```
src/supermimic/robots/       RobotSpec (g1.py, gr2.py)      src/supermimic/assets/gr2/   GR2 MJCF + meshes
src/supermimic/envs/         env, rewards, terminations, reset sampler (mdp/commands.py)
src/supermimic/training/     loop.py, schedule.py           src/supermimic/eval/         benchmark.py
scripts/data/                filter_bones_seed.py, build_library.py, csv_to_npz_batch.py
scripts/                     eval_benchmark.py, play.py, export_onnx.py, tb_tail.py, media/
```
