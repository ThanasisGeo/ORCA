# grasp_rl: training the ORCA hand to grasp

Training code for the grasp-and-hold project (see `../orca_sim_parent/PROJECT_PLAN.md`).
The environment itself lives in orca_sim (`OrcaHandGrasp`); this folder only contains
**how we train and evaluate** on it. The RL playground (`../rl_playground`) stays separate.

Python env: the `orca` conda env (orca_sim installed in editable mode, plus PyTorch CPU,
Stable-Baselines3, TensorBoard, PyYAML).

```bash
conda activate orca
cd ~/Documents/ORCA/grasp_rl
```

## Files

| File | Purpose |
|---|---|
| `configs/residual_cube.yaml` | Stage 1: left hand, random cubes, reward = "don't drop it" |
| `configs/residual_cube_curriculum.yaml` | (older idea) same, then penalties switched on gradually |
| `configs/residual_cube_grip.yaml` | Stage 1b: fresh from scripted, weaker motors, 2.5 s hold, small object-relative grip penalty |
| `common.py` | config loading, `ResidualActionWrapper`, reward-term logger, reward curriculum |
| `train.py` | PPO training; one folder per run in `runs/` |
| `evaluate.py` | success rate per object, trained policy vs scripted baseline; viewer / video |
| `diagnose.py` | mass × friction grid: success, grip adaptation (decision vs outcome), "if slip then grip" reaction test; plots |
| `min_grip.py` | measured minimum grip: weakens the finger motors step by step until the object slips; plots next to the formula m·g/μ |
| `runs/`, `videos/` | outputs (git-ignored) |

## Workflow

```bash
# 0. Baseline: how good is the hand-coded grasp under these conditions?
python evaluate.py --scripted-only configs/residual_cube.yaml

# 1. Train (3M steps ~ 30 min at ~2k steps/s; check speed in the first minute)
python train.py configs/residual_cube.yaml
tensorboard --logdir runs          # second terminal -> http://localhost:6006

# 2. Evaluate and watch
python evaluate.py runs/<run>                          # vs scripted, 100 episodes
python evaluate.py runs/<run> --objects primitives     # generalization to unseen shapes
python evaluate.py runs/<run> --render                 # MuJoCo viewer
MUJOCO_GL=egl python evaluate.py runs/<run> --video    # videos/<run>.mp4
```

## Diagnostics (`diagnose.py`)

```bash
python diagnose.py --scripted configs/residual_cube_grip.yaml   # baseline -> runs/scripted_<config>/diagnose/
python diagnose.py runs/<run>                                   # -> runs/<run>/diagnose/
```

Runs one fixed shape (4 cm cube) over masses × frictions, and prints and plots:
- **success grid** (`success_grid.png`)
- **adaptation, decision** (`force_vs_required.png`, left): slope of the policy's finger-closing
  command against the force the object needs (m·g/μ). 0 = same command for everything.
  Judge adaptation by this one.
- **adaptation, outcome** (same plot, right): the resulting contact force. Partly physics: a heavier
  object presses harder into the fingers even with an identical command.
- **reaction test** (`reaction.png`): the same episode with and without an extra downward pull
  part-way through the hold. "If slip then grip" = the closing command rises after the pull.
- `timeseries.png`: force, command and slip over time, light/grippy vs heavy/slippery.

### Minimum grip (`min_grip.py`)

```bash
python min_grip.py runs/<run>                                   # -> runs/<run>/diagnose/min_grip*.png
python min_grip.py --scripted configs/residual_cube_grip.yaml
```

Per cell: normal episode until 1 s after table removal, then the finger motor cap goes
100% → 90% → … → 5% → 0% (0.5 s per step) while the policy keeps acting. The last step before
the object has moved > 5 mm gives the **minimum motor cap** (N·m; maps to the real servos'
current limit) and the **minimum grip force**. `min_grip.png` (a) is force_vs_required with the
measured minimum added; (b) is the motor cap needed. Caveat: joint damping makes fingers yield
slowly at very low torque, so values ≲ 0.015 N·m just mean "almost nothing".

"Grip force" everywhere = sum over all hand–object contacts of the **normal** contact force
(MuJoCo `mj_contactForce`), in N. Fingers pushing from opposite sides both count (that is
also what m·g/μ assumes). It is not motor torque: wedging can raise it while the motors get weaker.

Reference results (2026-11-16):

| Policy | Success | Decision slope | Reaction | Grip while holding |
|---|---|---|---|---|
| scripted (cap 0.3) | 48% (≤50 g only) | 0.000 | 0.000 | ~9 N |
| Stage 1 run `residual_cube_20261116-111625` | 100% | −0.006 | −0.003 | ~46 N (needs ~1 N), cube bounces ±1.5 cm |
| Stage 1b `residual_cube_grip_20261116-115513` (cap 0.3, grip penalty) | 96% (eval 91%) | +0.020 (mass only, not friction) | +0.014 (too weak to stop slip) | ~11 N; 150 g sags 2.6 cm |

Minimum grip (min_grip.py, 4 cm cube, slip > 5 mm):

| Policy | Actual grip | Measured minimum | Formula m·g/μ | Weakest cap that holds |
|---|---|---|---|---|
| scripted | ~9.4 N | ~6 N (50 g: 7–11 N); ≥100 g mostly slips at full cap | 0.2–3 N | 0.15 N·m avg, often 0.3 |
| Stage 1b | ~11 N | ~3.8 N (0.5–10 N), ≈ 3–4× formula | 0.2–3 N | 0.066 N·m avg (0–0.18) |

## How the residual policy works

```
joint_target = scripted_grasp(t) + 0.3 rad × policy_output        (policy_output in [-1, 1])
```

The scripted grasp (`orca_sim/grasp_policies.py`) is never modified by training. With a
zero output you get exactly the scripted grasp, so training starts from something that
works. `residual.scale` limits how far the policy can deviate. `tests/test_grasp.py` in
orca_sim checks that the scripted grasp still holds the nominal cube with both hands.

## What to watch in TensorBoard

| Curve | Meaning |
|---|---|
| `eval/success_rate` | **the main number**: deterministic policy, 32 episodes |
| `rollout/success_rate` | success while training (with exploration noise) |
| `terms/<name>` | per-episode sum of each reward term, **including terms with weight 0**, so you can see e.g. grip force before penalizing it |
| `curriculum/stage` | which penalty stage is active |
| `train/approx_kl`, `train/std` | update size, and exploration noise (should shrink slowly) |

## Reward terms (defined in `orca_sim/grasp_envs.py`)

Reward = Σ weight × term. Set weights in the config's `env.reward_weights`; unknown names are an error.

| Term | Raw value | Scripted grasp, typical per step* |
|---|---|---|
| `reach` | 0..1, fingertips near object | 0.4-0.55 |
| `hold` | 1 while held after table removal | 0 or 1 |
| `drop` / `success` | 1 once | |
| `action_rate` | Σ(Δ target)² | ~0.003 while closing |
| `table_contact` | # hand-table contacts | 1.5-4 while closing/holding |
| `self_contact` | # finger-finger contacts | 3-4 while holding |
| `grip_force` | total hand→object normal force (N), for logging | ~10-20 scripted; ~46 trained Stage 1 |
| `grip_excess` | N above `grip_safety_factor` × m·g/μ (object-specific) | |
| `grip_excess_sq` | `grip_excess`² | |
| `joint_limit` | # joints within 0.05 rad of a limit | ~0.3 while closing |

\*Measured on the left hand. The scripted grasp itself touches the table, bumps fingers and
squeezes hard, so penalizing these from the start would fight the base the residual builds on.
That's why they start at 0 and are added with the curriculum.

Note (2026-11-16): object friction only took effect after giving the object contact priority
in the scene. Before that, MuJoCo used max(hand 1.0, object μ), so Stage 1 never saw μ < 1.

## Adding objects

Add an `ObjectSpec` to `OBJECTS` in `orca_sim/src/orca_sim/grasp_objects.py`, put it in a set
in `OBJECT_SETS` (or list names directly in the config), and run the orca_sim tests (they
check every object spawns clear of the hand). Height limit: ≤ 4 cm on the table.

## Experiment log

Keep notes in `EXPERIMENTS.md`: run folder, what changed, eval success, what you saw in the video.
