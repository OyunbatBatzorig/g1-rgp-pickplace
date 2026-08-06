# G1 RGP — Reach, Grasp, Place

Unitree G1 humanoid pick-and-place, trained skill-by-skill in NVIDIA Isaac Lab and
verified sim2sim in MuJoCo before any real-hardware attempt.

**RGP** = **R**each → **G**rasp+lift → **P**lace+release, three policies trained and
verified independently, each handing off to the next:

```
Policy 1 (Reach)        Policy 2 (Grasp + Lift)      Policy 3 (Place + Release)
default ready pose  -->  cube grasped, lifted    -->  cube carried to goal marker,
gripper open,             15cm off the table,          released, episode ends
hand near cube             gripper closed
```

Same scene throughout: G1 + Dex1 gripper, a table, a 3cm cube, and (from Policy 3
on) a fixed goal marker. See [`g1_lift_rl/env_cfg_rgp_scene.py`](g1_lift_rl/env_cfg_rgp_scene.py).

## What this repo is (and isn't) for

The primary research question here is the **Isaac Lab → MuJoCo sim2sim transfer
gap** — how well a PPO policy trained in one physics engine carries over to
another, and why it does or doesn't. The trained policies are the *instrument*
for measuring that gap, not a polished product in their own right.

Concretely: if a checkpoint you train doesn't grasp reliably, lifts a bit
short, or looks rougher than you'd expect from a finished robotics demo, that's
a normal PPO training outcome — not necessarily something broken in your setup.
Reward shaping, thresholds, and training length in this repo are starting
points tuned enough to produce *some* transferable behavior to study, not
exhaustively optimized for task success. If you want a stronger policy, that's
a fair place to spend your own effort — see "Customization" below — but it's a
secondary goal here, not the point of the exercise.

## Status

| Policy | Env cfg | Trained | Verified | MuJoCo transfer |
|---|---|---|---|---|
| 1 — Reach | `env_cfg_rgp_reach.py` | done | done | done |
| 2 — Grasp + lift | `env_cfg_rgp_grasp.py` | in progress (15cm) | pending | pending |
| 3 — Place + release | not yet implemented | — | — | — |

## Quickstart

[INSTALL.md](INSTALL.md) has the full walkthrough — Isaac Lab setup, getting the robot
model, installing this package, training, and watching results. Short version, once
installed, everything runs through Isaac Lab's own launcher from the `IsaacLab/`
directory (a sibling of this repo):

```bash
conda activate env_isaaclab
cd ../IsaacLab

# Train a policy (checkpoints land in IsaacLab/logs/rsl_rl/<experiment_name>/<timestamp>/)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-G1-RGP-Reach-v0 --headless
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-G1-RGP-Grasp-v0 --headless

# Watch a trained policy (opens the GUI; --checkpoint defaults to the latest run)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-G1-RGP-Reach-Play-v0
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-G1-RGP-Grasp-Play-v0
```

Train policies **in order** — Policy 2's reset pose is measured from Policy 1's
actual trained behavior (see "Chaining policies" below), so it needs a Policy 1
checkpoint to exist first. `play.py` also exports a TorchScript policy to
`logs/rsl_rl/<experiment_name>/<run>/exported/policy.pt`, which the MuJoCo scripts load.

## Repository layout

```
g1_lift_ext/
├── g1_lift_rl/                    # installable package (pip install -e .)
│   ├── constants.py                # physical facts: robot USD/pose, joint names,
│   │                                #   table/cube geometry, GRASP_OFFSET, GOAL_POS
│   ├── env_cfg_rgp_scene.py        # shared scene: robot + table + cube
│   ├── env_cfg_rgp_reach.py        # Policy 1 env cfg
│   ├── env_cfg_rgp_grasp.py        # Policy 2 env cfg
│   ├── mdp/
│   │   ├── observations_rgp.py     # what the policy sees
│   │   ├── rewards_rgp.py          # reward shaping, one section per policy
│   │   ├── terminations_rgp.py     # episode-end conditions
│   │   └── events_rgp.py           # resets, incl. the Policy 1 -> 2 handoff
│   ├── agents/
│   │   └── rsl_rl_ppo_cfg_rgp.py   # PPO hyperparameters
│   └── __init__.py                 # gym.register() calls
├── mujoco_transfer/                 # sim2sim verification (see below)
└── setup.py
```

The old 4-policy chain (`env_cfg.py`, `env_cfg_policy2/3/4.py`,
`mdp/{rewards,observations,terminations,events}.py`) lives in the same package —
it's an earlier, independent iteration on the same task, kept for comparison.
The files above are the complete RGP-specific set; nothing else in `g1_lift_rl/`
is required to train or run RGP.

### Chaining policies

Each policy's reset pose for the *next* policy is **measured**, not guessed:
train Policy N, run a deterministic (no-exploration-noise) replay, record the
mean and per-joint std of its converged pose, and use that as Policy N+1's reset
distribution. `mdp/events_rgp.py`'s `RGP_POLICY1_ARM_POSE`/`_STD` is Policy 2's
copy of this measurement from Policy 1.

One subtlety worth knowing before writing a new policy's reset: Isaac Lab's
`JointPositionActionCfg(use_default_offset=True)` targets the articulation's
*static* `init_state.joint_pos`, which a reset event does **not** change. So the
reset event can move the robot anywhere, but the action config must keep
pointing at the same robot config (`RGP_G1_DEX1_CFG`) unmodified — only the
reset function differs between policies.

## Gym IDs

| ID | Env cfg | PPO cfg |
|---|---|---|
| `Isaac-G1-RGP-Reach-v0` | `G1RGPReachEnvCfg` (2048 envs) | `G1RGPReachPPORunnerCfg` |
| `Isaac-G1-RGP-Reach-Play-v0` | `G1RGPReachEnvCfg_PLAY` (16 envs) | same |
| `Isaac-G1-RGP-Grasp-v0` | `G1RGPGraspEnvCfg` (1024 envs) | `G1RGPGraspPPORunnerCfg` |
| `Isaac-G1-RGP-Grasp-Play-v0` | `G1RGPGraspEnvCfg_PLAY` (16 envs) | same |

Registered in [`g1_lift_rl/__init__.py`](g1_lift_rl/__init__.py) — adding a new
policy means adding its own `env_cfg_rgp_<name>.py` + PPO cfg, then two
`gym.register()` blocks (train + play) following the same pattern.

## MuJoCo sim2sim transfer

Before any real-hardware attempt, each policy's checkpoint is replayed in MuJoCo
(same robot, same masses/geometry) and compared step-by-step against its Isaac
Lab trajectory. Scripts live in [`mujoco_transfer/`](mujoco_transfer/):

1. **Build the MJCF scene** — `build_scene_rgp.py` generates
   `g1_rgp_reach_scene.xml` from the robot's native MJCF plus table/cube geoms
   matched to Isaac Lab's exact sizes and masses.
2. **Capture an Isaac Lab trajectory** — `capture_isaac_trajectory_rgp_reach.py`
   loads a checkpoint's exported `policy.pt`, runs it in Isaac Lab, and records
   per-step obs/action/joint-position to a `.npz` file.
3. **Replay in MuJoCo** — `run_rgp_reach.py` loads the same `policy.pt` and MJCF
   scene, reconstructs the same observation vector, and steps the same actions
   through MuJoCo (with a physically-calibrated gripper open/close mapping and
   defensive action clipping).
4. **Compare** — `compare_trajectories_rgp_reach.py` diffs the two `.npz` logs
   channel-by-channel and plots the result.

```bash
cd mujoco_transfer
python3 build_scene_rgp.py
python3 capture_isaac_trajectory_rgp_reach.py --checkpoint <path/to/exported/policy.pt>
python3 run_rgp_reach.py --save_trajectory
python3 compare_trajectories_rgp_reach.py
```

Policy 2 doesn't have its own `_rgp_grasp` variants of these scripts yet — copy
the `_rgp_reach` ones and point them at the Grasp task/checkpoint, following the
same four-step pattern. Unlike Policy 1's reset, Policy 2's MuJoCo-side initial
state must also match its reset pose (`RGP_POLICY1_ARM_POSE`), not the robot's
default USD pose — see "Chaining policies" above.

## Customization

- **Reward weights** — `g1_lift_rl/mdp/rewards_rgp.py`, weights set per-term in
  each policy's `RewardsCfg` (e.g. `env_cfg_rgp_grasp.py`).
- **Cube size** — `RGP_BLOCK_SIZE` in `env_cfg_rgp_scene.py`. Changing it also
  changes the cube's spawn height and every clearance/grasp threshold that's
  defined relative to it, so retrain after changing.
- **Cube spawn randomization** — `_RGP_REACH_JITTER` in `env_cfg_rgp_reach.py`.
- **Episode length / physics settings** — each env cfg's `__post_init__`.
