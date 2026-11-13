# SuperMimic on Unitree H2: training notes

Port of the GR2 setup (`GR2_TRAINING_README.md`) to Unitree H2 (fixed head, 29 DoF). The algorithm is unchanged; this file
covers only what is H2-specific. Setup, evaluation and video workflow are the same as for GR2.

## 1. Implementation

`src/supermimic/robots/h2.py`, asset `src/supermimic/assets/h2/` (UniLab `h2.xml` + the 33 referenced meshes).

- Head: `head_pitch/yaw_link` are welded to `torso_link` (no joint, actuator or observation); they stay in the MJCF so
  rollouts render with a head. Their mass is part of the torso.
- 29 actuated joints, widths: state 71, observation 93 (x10 history), goal 68, context 33 (same as G1).
- XML position actuators, action scale `0.25 * effort / kp` with effort = min(actuator `forcerange`, joint
  `actuatorfrcrange`) = UniLab `tracking_sonic.py` (ankle roll 0.119, ankle pitch 0.239, hip/knee 0.45, waist 0.2,
  arms 0.338, wrist pitch/yaw 0.188). Using only `forcerange` would give ankle roll 0.94, 8x too large.
- Default pose = UniLab `stand` keyframe (knee 0.669, hip pitch -0.312, ankle pitch -0.363, arms (0.2, +-0.2, 0, 0.6)),
  not zeros. `RobotSpec.default_joint_pos` carries it (stand height, ONNX sidecar).

### 16 tracked bodies

The 14 UniLab / G1 bodies (`pelvis`, `*_hip_roll`, `*_knee`, `*_ankle_roll`, `torso`, `*_shoulder_roll`, `*_elbow`,
`*_wrist_yaw`) plus `left/right_ankle_pitch_link`. H2's ankle chain is **roll -> pitch** (G1: pitch -> roll), so
`ankle_roll_link` is the proximal link: ankle pitch moves none of the 14 bodies and the sole orientation is untracked.
`ankle_pitch_link` is the distal foot (carries the sole contacts). This is the same hole as GR2's wrist pitch/roll
(`gr2_wrist`). The wrists do not need extra bodies: the chain is roll -> pitch -> yaw and `wrist_yaw_link` is distal.

## 2. G1 <-> H2 joint mapping (`scripts/data/check_h2_mapping.py`)

- Same 29 names. Only the order differs at the ankles (H2 roll, pitch; G1 pitch, roll: index 4/5 and 10/11).
- `g1_csv_to_h2_npz2.py` maps CSV joints to H2 by name and writes `joint_pos` in H2 model order (= `H2_JOINT_NAMES`).
- Local joint axes and signs are the same for all 29 joints (pitch +y, roll +x, yaw +z; right side mirrored by limits).
  H2 differs in ranges (waist yaw +-1.75 vs +-2.62, wrist pitch +-0.44 vs +-1.61, ...), so the retargeter solves IK
  against H2 limits instead of copying angles. H2 hip pitch links are rotated 30 deg about x at the zero pose.
- Output npz has **no `body_names` / `joint_names`**: bodies are 33 in MuJoCo model order with `world` first
  (head_pitch/yaw at 17/18 shift all later indices vs G1). `RobotSpec.body_names_of` falls back to that order, with a
  body-count assertion; `motion_library`, `build_library.py` and `filter_bones_seed.py` use it.

## 3. Run

```bash
env -u PYTHONPATH uv run python scripts/data/check_h2_mapping.py --npz-dir ~/work/data/h2_npz
env -u PYTHONPATH uv run python scripts/data/filter_bones_seed.py --robot h2 --npz-dir ~/work/data/h2_npz --out ~/work/data/lib/h2_manifest_filtered.csv
env -u PYTHONPATH uv run python scripts/data/build_library.py --robot h2 --npz-dir ~/work/data/h2_npz \
  --manifest ~/work/data/lib/h2_manifest_filtered.csv --out ~/work/data/lib/h2.pt
env -u PYTHONPATH uv run sm-train --robot h2 --run-name h2 --data.library ~/work/data/lib/h2.pt \
  --env.num-envs 4096 --policy-trainer.batch-size 2048 --schedule.total-iterations 50000 --schedule.checkpoint-interval 3000
```

The filter drops 53 of 5461 clips by name (`table`, `_box_`: prop interaction) and none by feasibility, leaving 5408 clips
(11.5 h). The run above (50000 iterations, checkpoint every 3000) takes about 3.5 h at 4 it/s; adjust the iteration
count for longer runs.

## 4. Results (50000 iterations)

```bash
env -u PYTHONPATH uv run python scripts/eval_benchmark.py --ckpt logs/h2/ckpt_latest.pt --npz-dir ~/work/data/h2_npz \
  --clip-list logs/h2/eval_subset400.txt --num-envs 1024
```

- 400 random clips: mean normalized return 0.894 (step-weighted 0.869), 2 falls. The weakest clips are fast dances
  (slide, hiphop, vouge; 0.48-0.62).
- 8 hand-picked clips (macarena x2, walk x3, kitchen / cigarette): mean 0.888, no falls. Lowest is
  `victory_dance_macarena_change_direction` (0.794).
- Training end: train_ret 0.74, imag_ret 2.65.

Videos and ONNX export work as for GR2 (`record_rollout.py` / `render_rollout.py`, `export_onnx.py` writes the H2 default pose).
