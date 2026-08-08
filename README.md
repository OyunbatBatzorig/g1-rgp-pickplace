# G1 RGP — Reach, Grasp, Place, Release

Unitree G1 humanoid pick-and-place, trained skill-by-skill in NVIDIA Isaac Lab and
verified sim2sim in MuJoCo before any real-hardware attempt.

**RGP** = **R**each → **G**rasp+lift → **P**lace → **R**elease+return, four policies
trained and verified independently, each handing off to the next:

```
Policy 1 (Reach)      Policy 2 (Grasp+Lift)     Policy 3 (Place)          Policy 4 (Release+Return)
default ready pose -->  cube grasped, lifted -->  cube carried to      -->  gripper opens, cube
gripper open,             15cm off the table,      goal marker,             stays at goal, arm
hand near cube              gripper closed          settled, still           returns to Policy 1's
                                                      gripping                own ready pose
```

Same scene throughout: G1 + Dex1 gripper, a table, a 3cm cube, and a yellow goal marker
(shared across all four policies, even Reach/Grasp which don't use it for anything — see
[`g1_lift_rl/env_cfg_rgp_scene.py`](g1_lift_rl/env_cfg_rgp_scene.py)). Policy 3 and 4 are
the only two whose observations/rewards actually reference the goal.

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
| 1 — Reach | `env_cfg_rgp_reach.py` | done | done (deterministic replay) | clean, no measurable gap |
| 2 — Grasp + lift | `env_cfg_rgp_grasp.py` | done | done | near-success (92% of lift height), grasp lost gradually ~step 150 |
| 3 — Place | `env_cfg_rgp_place.py` | done | done (32-env deterministic replay, 31/32 sustained) | grasp lost almost instantly (~6 steps); documented known gap, not further chased |
| 4 — Release + return | `env_cfg_rgp_release.py` | done | done (direct joint-space + EE-clearance diagnostics) | smallest joint-space gap of the chain (~0.14 rad); `arm_dist_to_ready` nearly identical between engines |

Policy 3 was originally designed as a combined "move to goal + release" policy; it was
split into Policy 3 (place-only, gripper stays closed) and Policy 4 (release-only, plus a
return-to-`READY_ARM_POSE` behavior — release alone would be a trivially easy task without
it) after the combined version exhibited a clinging failure mode: holding still paid more
reliably than the uncertain payoff of releasing. See "Reward design notes" below for the
two non-obvious bugs found and fixed in Policy 3 and 4's reward shaping.

## Quickstart

[INSTALL.md](INSTALL.md) has the full walkthrough — Isaac Lab setup, getting the robot
model, installing this package, training, and watching results. Short version, once
installed, everything runs through Isaac Lab's own launcher from the `IsaacLab/`
directory (a sibling of this repo):

```bash
conda activate env_isaaclab
cd ../IsaacLab

# Train a policy (checkpoints land in IsaacLab/logs/rsl_rl/<experiment_name>/<timestamp>/)
# -- train in order, each policy's reset is measured from the previous one's real
# convergence, so it needs that policy's checkpoint to already exist. See "Chaining
# policies" below.
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-G1-RGP-Reach-v0 --headless
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-G1-RGP-Grasp-v0 --headless
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-G1-RGP-Place-v0 --headless
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-G1-RGP-Release-v0 --headless

# Watch a trained policy (opens the GUI; --checkpoint defaults to the latest run)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-G1-RGP-Reach-Play-v0
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-G1-RGP-Grasp-Play-v0
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-G1-RGP-Place-Play-v0
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-G1-RGP-Release-Play-v0
```

`play.py` also exports a TorchScript policy to
`logs/rsl_rl/<experiment_name>/<run>/exported/policy.pt`, which the MuJoCo scripts load.

## Repository layout

```
g1_lift_ext/
├── g1_lift_rl/                        # installable package (pip install -e .)
│   ├── constants.py                    # physical facts: robot USD/pose, joint names,
│   │                                    #   table/cube geometry, GRASP_OFFSET, GOAL_POS
│   ├── env_cfg_rgp_scene.py            # shared scene: robot + table + cube + goal marker
│   ├── env_cfg_rgp_reach.py            # Policy 1 env cfg
│   ├── env_cfg_rgp_grasp.py            # Policy 2 env cfg
│   ├── env_cfg_rgp_place.py            # Policy 3 env cfg (own dedicated action baseline)
│   ├── env_cfg_rgp_release.py          # Policy 4 env cfg (own dedicated action baseline)
│   ├── policy2_held_states.pt          # captured Policy 2 grasp states -> Policy 3's reset
│   ├── policy3_settled_states.pt       # captured Policy 3 settled states -> Policy 4's reset
│   ├── mdp/
│   │   ├── observations_rgp.py         # what the policy sees
│   │   ├── rewards_rgp.py              # reward shaping, one section per policy
│   │   ├── terminations_rgp.py         # episode-end conditions
│   │   └── events_rgp.py               # resets, incl. every policy handoff
│   ├── agents/
│   │   └── rsl_rl_ppo_cfg_rgp.py       # PPO hyperparameters
│   └── __init__.py                     # gym.register() calls
├── capture_policy2_held_states.py      # captures Policy 2's real grasp states (run after training Policy 2)
├── capture_policy3_settled_states.py   # captures Policy 3's real settled states (run after training Policy 3)
├── mujoco_transfer/                    # sim2sim verification (see below)
└── setup.py
```

## Reward design notes

Two non-obvious reward-shaping bugs were found this project and are worth knowing about
before tuning your own weights or thresholds — both were invisible from the training curve
alone and only diagnosable by reading the reward code directly against a real measured
state:

- **Penalty valley (Policy 3).** `penalty_carry_height_rgp`/`penalty_table_clearance_rgp`
  were originally gated off at the same 0.06m radius as `settle`'s own success condition.
  Since reaching that radius requires descending *first*, the final approach into the goal
  passed through a band (0.06m–0.15m) where both penalties were still fully active, with no
  local incentive to keep closing the distance — training plateaued at exactly that band's
  edge across two independent runs with different exploration settings, which is what ruled
  out exploration collapse as the cause. Fixed by widening the penalty gate
  (`CARRY_PENALTY_GATE_RADIUS_RGP = 0.18`, above the lift-height cap) while leaving the
  0.06m success radius unchanged.
- **Tanh-saturated gradient (Policy 4).** `reward_return_to_ready_rgp`'s steepness constant
  (`RETURN_K_RGP`) was originally calibrated by metre-scale intuition (`3.0`) for a
  radian-scale quantity — real measured joint-space distances from the reset pose to
  `READY_ARM_POSE` are 2–4.5 rad, and `tanh(3.0 * 2..4.5)` is float32-saturated to `1.000000`
  across that entire range, meaning the term had **exactly zero gradient** for the whole
  1500-iteration run despite a nonzero weight. Fixed by recalibrating to `0.6`, which keeps
  `tanh(K*dist)` inside its responsive range across the observed distance span.

A third, smaller finding: once `return_to_ready`'s weight was raised to fix the above, the
arm's shortest joint-space path back to ready started swinging the gripper through the
just-placed cube. The existing `penalty_contact_disturbance_rgp` (reactive, only fires once
cube velocity actually spikes) wasn't strong enough on its own once competing against a
larger weight. `penalty_ee_near_object_after_release_rgp` (proactive, same style as
`penalty_base_clearance_rgp`) was added to fix it.

## Gym IDs

| ID | Env cfg | PPO cfg |
|---|---|---|
| `Isaac-G1-RGP-Reach-v0` | `G1RGPReachEnvCfg` (2048 envs) | `G1RGPReachPPORunnerCfg` |
| `Isaac-G1-RGP-Reach-Play-v0` | `G1RGPReachEnvCfg_PLAY` (16 envs) | same |
| `Isaac-G1-RGP-Grasp-v0` | `G1RGPGraspEnvCfg` (1024 envs) | `G1RGPGraspPPORunnerCfg` |
| `Isaac-G1-RGP-Grasp-Play-v0` | `G1RGPGraspEnvCfg_PLAY` (16 envs) | same |
| `Isaac-G1-RGP-Place-v0` | `G1RGPPlaceEnvCfg` (1024 envs) | `G1RGPPlacePPORunnerCfg` |
| `Isaac-G1-RGP-Place-Play-v0` | `G1RGPPlaceEnvCfg_PLAY` (16 envs) | same |
| `Isaac-G1-RGP-Release-v0` | `G1RGPReleaseEnvCfg` (1024 envs) | `G1RGPReleasePPORunnerCfg` |
| `Isaac-G1-RGP-Release-Play-v0` | `G1RGPReleaseEnvCfg_PLAY` (16 envs) | same |

Registered in [`g1_lift_rl/__init__.py`](g1_lift_rl/__init__.py) — adding a new
policy means adding its own `env_cfg_rgp_<name>.py` + PPO cfg, then two
`gym.register()` blocks (train + play) following the same pattern.

Reach/Grasp use a 36-dim observation (8 terms, no goal concept — neither task has a reward
term that references one); Place/Release use 39-dim (9 terms, adding `object_to_goal`).
This is a real architectural difference, not an oversight, and matters if you're writing
anything that loads more than one policy's checkpoint in the same process.

## MuJoCo sim2sim transfer

Before any real-hardware attempt, each policy's checkpoint is replayed in MuJoCo
(same robot, same masses/geometry) and compared step-by-step against its Isaac
Lab trajectory. Scripts live in [`mujoco_transfer/`](mujoco_transfer/), one set per policy
(`_rgp_reach`, `_rgp_grasp`, `_rgp_place`, `_rgp_release`), all following the same
four-step pattern:

1. **Build the MJCF scene** — `build_scene_rgp.py` generates
   `g1_rgp_reach_scene.xml` from the robot's native MJCF plus table/cube/goal-marker geoms
   matched to Isaac Lab's exact sizes, masses, and positions. One shared scene file for all
   four policies.
2. **Capture an Isaac Lab trajectory** — `capture_isaac_trajectory_rgp_<policy>.py`
   loads a checkpoint's exported `policy.pt`, runs it in Isaac Lab, and records
   per-step obs/action/joint-position to a `.npz` file.
3. **Replay in MuJoCo** — `run_rgp_<policy>.py` loads the same `policy.pt` and MJCF
   scene, reconstructs the same observation vector, and steps the same actions
   through MuJoCo (with a physically-calibrated gripper open/close mapping and
   defensive action clipping). Pass `--view` for the interactive viewer (add `--realtime`
   to pace it to real time instead of running flat out).
4. **Compare** — `compare_trajectories_rgp_<policy>.py` diffs the two `.npz` logs
   channel-by-channel, confirms the two rollouts actually started from an identical state
   (cube spawn offset, arm pose gap — both should read ~0) before attributing any
   divergence to physics, and plots the result.

```bash
cd mujoco_transfer
python3 build_scene_rgp.py
conda activate unitree_sim_env   # env_isaaclab doesn't have the mujoco package
python3 run_rgp_release.py --save_trajectory trajectory_mujoco_rgp_release.npz
conda activate env_isaaclab      # capture_isaac_trajectory_*.py needs Isaac Lab
cd ../../IsaacLab
./isaaclab.sh -p ../g1_lift_ext/mujoco_transfer/capture_isaac_trajectory_rgp_release.py
cd ../g1_lift_ext/mujoco_transfer
conda activate unitree_sim_env
python3 compare_trajectories_rgp_release.py
```

Findings so far (also in the Status table above): transfer difficulty tracks how much of a
policy's success depends on sustained, fine-margin contact rather than approach or
positioning. Reach (no contact dependency) transfers cleanly; Grasp+lift (first sustained
grip) loses it gradually around step 150; Place (grip inherited from Grasp, one handoff
further from the original contact) loses it almost instantly; Release+return (opens the
gripper deliberately, then moves through free space) shows the smallest joint-space gap of
the whole chain — consistent with contact-sensitivity, not overall task complexity, being
the actual driver of the transfer gap.

## Customization

- **Reward weights** — `g1_lift_rl/mdp/rewards_rgp.py`, weights set per-term in
  each policy's `RewardsCfg` (e.g. `env_cfg_rgp_grasp.py`). Before changing a
  threshold/steepness constant, check it against a real measured range of the quantity it
  applies to first — see "Reward design notes" above for what happens when that's skipped.
- **Cube size** — `RGP_BLOCK_SIZE` in `env_cfg_rgp_scene.py`. Changing it also
  changes the cube's spawn height and every clearance/grasp threshold that's
  defined relative to it, so retrain after changing.
- **Cube spawn randomization** — `_RGP_REACH_JITTER` in `env_cfg_rgp_reach.py`.
- **Episode length / physics settings** — each env cfg's `__post_init__`.
