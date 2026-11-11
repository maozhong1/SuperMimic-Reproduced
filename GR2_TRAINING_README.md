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

```bash
D=~/work/data/gr2_npz
env -u PYTHONPATH uv run python scripts/data/filter_bones_seed.py --robot gr2 --npz-dir $D --workers 16
env -u PYTHONPATH uv run python scripts/data/build_library.py --robot gr2 \
  --npz-dir $D --manifest $D/manifest_filtered.csv --out ~/work/data/lib/gr2_5k.pt
```
Current data: 5461 npz, 5408 kept (11.48 h; 49 `table` and 4 `_box_` clips removed by name; 
Library `gr2_5k.pt`: 2.07 M frames, 0.28 GB. After adding clips, re-run filter and `build_library.py`.

## 4. Training

```bash
env -u PYTHONPATH uv run sm-train --robot gr2 --run-name <run> \
  --data.library ~/work/data/lib/gr2_5k.pt --env.num-envs 4096 --policy-trainer.batch-size 2048 \
  --schedule.total-iterations <N> --schedule.checkpoint-interval <K> [--resume <ckpt>]
```
- Hyper-parameters are G1's (Table A.5/A.6) and were not tuned for GR2; `anchor_obs=False` (paper observation).
- One iteration = one 50 Hz control step of all 4096 envs. H=32 is the imagination horizon, UTD 0.4 means 0.4 gradient
  updates per iteration, gamma 0.99. About 4.4 it/s.
- **The learning rate depends on `total-iterations`**: cosine decay to 3% over the first 78% of it, then constant.
- `--resume` restores the networks, optimizers, sampler state and iteration; the replay buffer is refilled.
  Use a new `--run-name`.
- To survive SSH disconnects, detach the run:
  `setsid nohup env -u PYTHONPATH uv run sm-train ... > logs/<run>/stdout.log 2>&1 < /dev/null &`

### Reset sampler and `normalize_bin_return` (experimental, off by default, not recommended)

A *bin* is a 1 s (50-frame) segment of a clip (43950 bins for 5408 clips). Resets pick a bin with 50% uniform, 40%
`softmax(-25 * G_b / G_max)` (G_b = discounted return-to-go from that bin), 10% historical memory.
Finding: the last bin of each clip has a low G_b only because few steps remain. On the `gr2_50k` checkpoint it is 12.3%
of the bins but gets 54.5% of the resets (episodes of at most 50 steps), which explains `train/episode_len` of ~111
(expected ~102 without any fall, ~213 under uniform sampling). So **short training episodes are a sampling artifact,
not falling.**

`--env.normalize-bin-return` divides the return-to-go of clip-end episodes by `1 - gamma^remaining`. Result: almost all
bins then sit near `g_max`, so the softmax concentrates on the few lowest bins; the effective number of bins fell from
~9786 to 4-31, about half of all resets landed on a few dozen bins in some dance clips (largest single bin ~13%), and
those clips did not improve (`country_one_step` 0.586 -> 0.592, 6 falls before and after). Keep it off (the default
in `TrainConfig`, `SuperMimicEnvCfgArgs` and `MultiClipMotionCommandCfg`). When resuming with a different setting than
the checkpoint, `bin_return` / `bin_memory` are reset. 

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
- In TensorBoard: Scalars tab, search the tag, set Smoothing to 0, drag a box on the chart to zoom into a step range.

Rewards are hard-coded (no config file) in two places that must be changed together: the simulator side in
`envs/tracking_env_cfg.py` and the imagination side `TrackingRewardCfg` in `envs/mdp/rewards.py` (weights: root pos/ori
0.5, body pos/ori/lin_vel/ang_vel 1.0, action_rate -0.1, joint_limit -2.0; stds 0.3/0.4/0.3/0.4/1.0/3.14; all times
dt = 0.02). Same as G1, not tuned for GR2.

## 6. Evaluation and visualization

Protocol: each clip starts at its first frame, deterministic, no noise, no domain randomization, played until the end
or a fall.

```bash
env -u PYTHONPATH uv run python scripts/eval_benchmark.py --ckpt logs/<run>/ckpt_latest.pt \
  --npz-dir ~/work/data/gr2_npz --manifest ~/work/data/gr2_npz/manifest_filtered.csv \
  --num-envs 1024 --out logs/<run>/bench.json                # ~20 min for 5408 clips; output is JSON
# single clips: add  --clips <name> --num-envs 1  and drop --manifest

env -u PYTHONPATH uv run python scripts/play.py --ckpt logs/<run>/ckpt_latest.pt --npz-dir ~/work/data/gr2_npz \
  --clips <name> --num-envs 1 --viewer viser                 # browser: http://<host>:8080
```
`play.py` shows the policy tracking the reference (translucent = reference). Over SSH, if port 8080 is not reachable,
use `ssh -L 8080:localhost:8080 <user>@<host>`. Open-loop replay of npz joint angles is not supported (it has no
balance control and is not a policy test). `scripts/diag_drift.py` and `scripts/eval_push.py` have **not been run on
GR2**.

### Offline mp4

Record the rollouts once, then render. Verified on GR2 with `gr2_100k` (needs a system `ffmpeg`, e.g. `apt install ffmpeg`).
```bash
env -u PYTHONPATH uv run python scripts/media/record_rollout.py --ckpt logs/<run>/ckpt_latest.pt \
  --npz-dir ~/work/data/gr2_npz --clips <stem1> <stem2> ... --out media/gr2_rollouts.npz
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

`scripts/export_onnx.py` exports the policy (`action = policy(obs_hist, goal)`, batch dimension dynamic) and checks it
against PyTorch with random inputs at batch 1 / 7 / 256 (fails if max |diff| > `--tol`, default 1e-4). `onnx` and
`onnxruntime` are not project dependencies; `uv run --with` installs them temporarily and leaves `pyproject.toml` /
`uv.lock` untouched.
```bash
env -u PYTHONPATH uv run --with onnx --with onnxruntime python scripts/export_onnx.py \
  --ckpt logs/<run>/ckpt_0100000.pt --out logs/<run>/policy_100000.onnx
```
It writes `policy_*.onnx` (34.8 MB for GR2) and a sidecar `policy_*.json` (dimensions, obs layout, joint names,
default pose, per-joint action scale). Verified on `gr2_100k`: max |onnx - torch| ~3e-6.

| | name | shape | content |
|---|---|---|---|
| in | `obs_hist` | [B, 870] | 10 frames, term-major: `joint_pos` 270, `joint_vel` 270, `base_ang_vel` 30, `projected_gravity` 30, previous `actions` 270 |
| in | `goal` | [B, 640] | 10 future frames x 64: `joint_pos` 27, `joint_vel` 27, gravity 3, ang_vel_b 3, lin_vel_h 3, height 1 |
| out | `action` | [B, 27] | raw action |

- The fixed input scalings (`obs_term_scale`, `goal_term_scale`) are inside the graph; do not scale again.
- The graph does **not** contain the action mapping: `joint target = default_joint_pos (all 0 for GR2) + action *
  action_scale[joint]`, with `action_scale = 0.25 * effort / kp` per joint (in the sidecar json).
- The consumer must build `obs_hist` / `goal` exactly as the training environment does (`envs/mdp/observations.py`,
  `world_model/heads.py:obs_terms`); that code is not part of the export.
- **Not verified:** the check uses random inputs, not real observations from the simulator, and nothing was run on
  hardware or in another simulator. The `default_joint_pos` in the json is hard-coded to zeros (correct for GR2 only).
- The exporter uses the legacy TorchScript path (`dynamo=False`), which prints a deprecation warning in this PyTorch.

## 7. Results (5408 clips, same evaluation set)

| Checkpoint | Training | Falls | Mean return | Step-weighted |
|---|---|---|---|---|
| `gr2_50k` ckpt_final_50000 | + resume to 50000 | 18 (0.33%) | 0.820 | 0.806 |
| `gr2_78k` ckpt_0078000 | switch on | 14 (0.26%) | 0.824 | 0.812 |
| `gr2_100k` ckpt_0100000 | resume from `gr2_50k` to 100000 (lr floor from it 78125), switch off | **9 (0.17%)** | **0.8475** | **0.833** |

- Falls at 78000: 8 dance, 3 walk, 2 idle, 1 jog. Dance (258 clips) averages 0.58, far below the other categories.
  Return correlates with reference arm acceleration (Spearman -0.60 with `arm_acc_p99` from `retarget_manifest.csv`,
  1897 clips only; a correlation, not a cause).
- `gr2_100k` vs `gr2_78k`: +0.024 return, falls 14 -> 9; 4601 clips better, 807 worse, so unlike 50k -> 78k the gain is
  broad (e.g. `idle_turn` 0.926 -> 0.940, `walk_ff_loop` 0.779 -> 0.797, `jog_forward_start` 0.681 -> 0.705).
  `joint_limit` penalty fell from -8.1e-5 to -5.8e-5. Still one run, one seed; the gain cannot be split between
  training length and the lr floor, since no control run exists.
- Falls at 100k (9): 6 are `dance_western_country_one_step*` (steps 167-302); 3 are new `medium_big_light_*_behind_high`
  (A527, steps 199-215, return 0.70-0.72) that did not fall at 78k. 8 clips that fell at 78k are fixed
  (`idle_turn_360`, `jog_ff_loop`, `walk_ff_start`, ...). The cause of the A527 falls is not investigated.
- Visually checked in rollout videos (section 6): 10 clips (walk, turn, macarena, washing dishes, ...) all finish
  without falling; returns 0.76-0.95, the lowest is `walk_ff_stop_270_003` (0.763).

## 8. Open items (not done / not verified)

- Full BONES-SEED GR2 npz set (only 5461 files so far).
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
