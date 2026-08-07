#!/usr/bin/env python3
"""Run the full RGP chain (Reach -> Grasp+lift -> Place -> Release+return)
back-to-back in one continuous rollout, picking up each policy's own latest
trained checkpoint automatically.

The environment is reset ONCE at the start; only the acting policy network
swaps between phases, so the physical state (robot pose, cube position) carries
over exactly the way the real chain hands off -- this is not three separate
episodes stitched together, it's one continuous rollout.

Two structural mismatches between phases have to be handled explicitly,
because they're not visible from the printed reward/obs tables alone:

  1. Action baseline. JointPositionActionCfg(use_default_offset=True) computes
     target = default_joint_pos + scale*action, and default_joint_pos is
     snapshotted ONCE into the action term's own _offset tensor at env
     construction (isaaclab/envs/mdp/actions/joint_actions.py:195) -- it is
     never re-read from the articulation afterward. Reach and Grasp share one
     baseline (RGP_G1_DEX1_CFG); Place and Release each have their OWN
     dedicated baseline (RGP_G1_DEX1_PLACE_CFG / RGP_G1_DEX1_RELEASE_CFG, see
     env_cfg_rgp_place.py / env_cfg_rgp_release.py) because a fresh policy
     can't output the large one-shot correction needed to just stay put
     otherwise (the same bug diagnosed and fixed for Policy 3's own solo
     training). So _offset is swapped by hand at the start of each phase --
     see PHASE_ROBOT_CFG below.
  2. Observation width. Reach/Grasp have no goal concept (36-dim, 8 terms).
     Place and Release both add object_to_goal (39-dim, 9 terms) right before
     last_action. The whole rollout runs inside ONE env built from a
     Reach/Grasp-shaped task (--task), so the extra 3 values don't exist in
     the env's own observation output -- they're computed by calling the REAL
     object_to_goal_rgp() function directly (not reimplemented) and spliced
     into the 36-dim obs at the same position Place/Release's own
     ObservationsCfg puts it, only when one of their policies is acting.

This is a demo/visualization tool, not an evaluation script -- see README.md's
"What this repo is (and isn't) for": individual phases may not look fully
polished, since the point of this project is the Isaac Lab -> MuJoCo transfer
gap, not a maximally-tuned policy.

Run from the IsaacLab/ directory, same as any other script in this repo:
    ./isaaclab.sh -p ../g1_lift_ext/play_chain.py
"""
import argparse
import glob
import os
import sys
from importlib import metadata

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-G1-RGP-Grasp-Play-v0",
                     help="Env cfg used for the whole rollout (scene is shared across all RGP "
                          "policies, including the goal marker -- see env_cfg_rgp_scene.py). Only "
                          "determines the initial reset event and the env's NATIVE obs width "
                          "(36-dim, Reach/Grasp-shaped); Place's extra obs dims and action "
                          "baseline are patched in per-phase regardless of this choice.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--reach_steps", type=int, default=400)
parser.add_argument("--grasp_steps", type=int, default=400)
parser.add_argument("--place_steps", type=int, default=500)
parser.add_argument("--release_steps", type=int, default=500)
parser.add_argument("--reach_checkpoint", type=str, default=None,
                     help="Explicit path, overrides latest_checkpoint()'s mtime-based pick "
                          "(mtime can silently favor a newer-but-not-actually-confirmed run).")
parser.add_argument("--grasp_checkpoint", type=str, default=None,
                     help="Explicit path, overrides latest_checkpoint()'s mtime-based pick.")
parser.add_argument("--place_checkpoint", type=str, default=None,
                     help="Explicit path, overrides latest_checkpoint()'s mtime-based pick.")
parser.add_argument("--release_checkpoint", type=str, default=None,
                     help="Explicit path, overrides latest_checkpoint()'s mtime-based pick.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
from tensordict import TensorDict
import g1_lift_rl  # noqa: F401
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from g1_lift_rl.agents.rsl_rl_ppo_cfg_rgp import (
    G1RGPReachPPORunnerCfg, G1RGPGraspPPORunnerCfg, G1RGPPlacePPORunnerCfg, G1RGPReleasePPORunnerCfg,
)
from g1_lift_rl.constants import ARM_JOINTS
from g1_lift_rl.env_cfg_rgp_scene import RGP_G1_DEX1_CFG
from g1_lift_rl.env_cfg_rgp_place import RGP_G1_DEX1_PLACE_CFG
from g1_lift_rl.env_cfg_rgp_release import RGP_G1_DEX1_RELEASE_CFG
from g1_lift_rl.mdp import observations_rgp as obs_rgp

RSL_RL_VERSION = metadata.version("rsl-rl-lib")

# Phase order: (label, experiment_name, agent cfg, step budget, checkpoint
# override). checkpoint override is None unless --*_checkpoint is passed -- otherwise
# falls back to latest_checkpoint()'s mtime-based pick, which can silently
# favor a newer-but-not-actually-confirmed run (e.g. Policy 1 has a newer
# Aug 5 run that trains fine but isn't the one confirmed good -- the July 30
# run is).
PHASE_SPECS = [
    ("Reach", "g1_rgp_reach", G1RGPReachPPORunnerCfg(), args_cli.reach_steps, args_cli.reach_checkpoint),
    ("Grasp + lift", "g1_rgp_grasp", G1RGPGraspPPORunnerCfg(), args_cli.grasp_steps, args_cli.grasp_checkpoint),
    ("Place", "g1_rgp_place", G1RGPPlacePPORunnerCfg(), args_cli.place_steps, args_cli.place_checkpoint),
    ("Release + return", "g1_rgp_release", G1RGPReleasePPORunnerCfg(), args_cli.release_steps, args_cli.release_checkpoint),
]

# experiment_name -> the ArticulationCfg whose init_state.joint_pos that
# policy's actions are offset from. Reach/Grasp share the plain baseline;
# Place and Release each needed their own (see module docstring point 1).
PHASE_ROBOT_CFG = {
    "g1_rgp_reach": RGP_G1_DEX1_CFG,
    "g1_rgp_grasp": RGP_G1_DEX1_CFG,
    "g1_rgp_place": RGP_G1_DEX1_PLACE_CFG,
    "g1_rgp_release": RGP_G1_DEX1_RELEASE_CFG,
}

# experiment_name -> native obs width that policy was trained with.
PHASE_OBS_DIM = {
    "g1_rgp_reach": 36,
    "g1_rgp_grasp": 36,
    "g1_rgp_place": 39,
    "g1_rgp_release": 39,
}

# experiment_name -> that policy's own registered Play task id, used only to
# build a (config-only, no live scene) env_cfg object for the shape stub below.
PHASE_TASK_ID = {
    "g1_rgp_reach": "Isaac-G1-RGP-Reach-Play-v0",
    "g1_rgp_grasp": "Isaac-G1-RGP-Grasp-Play-v0",
    "g1_rgp_place": "Isaac-G1-RGP-Place-Play-v0",
    "g1_rgp_release": "Isaac-G1-RGP-Release-Play-v0",
}

ACTION_DIM = 8  # arm(7) + gripper(1) -- identical across every RGP policy (see
                 # every printed "Active Action Terms (shape: 8)" table this project has produced)


class _NetworkShapeStub:
    """Stand-in for a VecEnv, used ONLY to size and load a policy network --
    never to actually simulate anything. OnPolicyRunner's actor/critic network
    gets constructed from whatever env it's given (env.get_observations() is
    called at construction to size the network -- rsl_rl/runners/
    on_policy_runner.py:36), so loading Place's 39-dim checkpoint requires
    building against something 39-dim, not the shared 36-dim demo env.

    A first version of this script used a REAL throwaway gym.make() env per
    policy for this. That works, but Isaac Sim's viewport is tied to the
    whole process (AppLauncher's headless flag is set once, globally) --
    every one of those three throwaway envs, even though only ever reset and
    never stepped, rendered its own static scene to the GUI window before the
    next one replaced it. Watching that sequence live is indistinguishable
    from "the demo is stuck." This stub reports only the attributes
    OnPolicyRunner.__init__ / PPO.construct_algorithm / Logger.__init__
    actually read (checked directly against rsl-rl-lib 5.0.1's source, not
    guessed) -- num_envs, num_actions, cfg, device, get_observations() -- so
    no real environment, and no extra viewport flash, is ever created."""

    def __init__(self, obs_dim: int, num_actions: int, device: str, cfg):
        self.num_envs = 1
        self.num_actions = num_actions
        self.device = device
        self.cfg = cfg
        self._obs = TensorDict({"policy": torch.zeros(1, obs_dim, device=device)}, batch_size=[1])

    def get_observations(self):
        return self._obs


def latest_checkpoint(experiment_name: str) -> str | None:
    """Most recently WRITTEN model_*.pt across every run of `experiment_name`
    (by file mtime, not run-directory name) -- avoids a fresh, barely-started
    run shadowing a fully-trained checkpoint from an interrupted earlier run."""
    checkpoints = glob.glob(os.path.join("logs", "rsl_rl", experiment_name, "*", "model_*.pt"))
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)


def set_action_baseline(env, arm_names: list[str], robot_cfg) -> None:
    """Overwrite the arm action term's _offset in place to match robot_cfg's
    own init_state.joint_pos -- see module docstring point 1 for why this
    can't be done by just writing robot.data.default_joint_pos instead."""
    target = torch.tensor([robot_cfg.init_state.joint_pos[n] for n in arm_names], device=env.unwrapped.device)
    env.unwrapped.action_manager.get_term("arm")._offset[:] = target


def adapt_obs(obs: TensorDict, target_dim: int, env, num_envs: int) -> TensorDict:
    """Reshape the env's native 36-dim obs to whatever target_dim the acting
    policy actually expects -- see module docstring point 2."""
    full = obs["policy"]
    if full.shape[-1] == target_dim:
        return obs
    if full.shape[-1] == 36 and target_dim == 39:
        # Splice in the REAL object_to_goal_rgp() value (not reimplemented)
        # at the same position Place's own ObservationsCfg puts it: right
        # before last_action, i.e. after the shared first 28 dims.
        object_to_goal = obs_rgp.object_to_goal_rgp(env.unwrapped)
        padded = torch.cat([full[:, :28], object_to_goal, full[:, 28:]], dim=-1)
        return TensorDict({"policy": padded}, batch_size=[num_envs])
    raise ValueError(f"no adapter for {full.shape[-1]}-dim obs -> {target_dim}-dim")


def print_state(env, label: str):
    scene = env.unwrapped.scene
    robot, obj = scene["robot"], scene["object"]
    env_origins = scene.env_origins
    ee_ids, _ = robot.find_bodies(["right_hand_Link1_3", "right_hand_Link2_3"])
    ee_pos = robot.data.body_pos_w[:, ee_ids, :].mean(dim=1)
    cube_pos = obj.data.root_pos_w
    dist = torch.norm(ee_pos - cube_pos, dim=-1)
    cube_z = (cube_pos[:, 2] - env_origins[:, 2])
    print(f"  [{label}] end state -- EE-to-cube dist: mean={dist.mean():.4f}m  "
          f"cube height: mean={cube_z.mean():.4f}m std={cube_z.std():.4f}m")


def load_phase_policy(experiment_name: str, agent_cfg, ckpt: str, device: str):
    """Builds the actor/critic network sized to THIS phase's own observation
    width, loads the checkpoint into it, and returns the inference callable
    -- via _NetworkShapeStub, no real environment or GUI window involved."""
    tmp_cfg = parse_env_cfg(PHASE_TASK_ID[experiment_name], device=device, num_envs=1)
    stub_env = _NetworkShapeStub(
        obs_dim=PHASE_OBS_DIM[experiment_name], num_actions=ACTION_DIM, device=device, cfg=tmp_cfg)
    runner = OnPolicyRunner(stub_env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(ckpt)
    return runner.get_inference_policy(device=device)


@hydra_task_config(args_cli.task, None)
def main(env_cfg: ManagerBasedRLEnvCfg, _agent_cfg):
    device = env_cfg.sim.device

    phases = []
    for label, experiment_name, agent_cfg, steps, ckpt_override in PHASE_SPECS:
        ckpt = ckpt_override or latest_checkpoint(experiment_name)
        if ckpt is None:
            print(f"[chain] skipping {label}: no checkpoint found for '{experiment_name}' yet")
            continue
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, RSL_RL_VERSION)
        policy = load_phase_policy(experiment_name, agent_cfg, ckpt, device)
        phases.append((label, ckpt, policy, steps, experiment_name))
        print(f"[chain] {label}: {ckpt} ({steps} steps)")

    if not phases:
        print("[chain] no trained checkpoints found -- train Policy 1 first (see INSTALL.md).")
        return

    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)
    num_envs = args_cli.num_envs

    robot = env.unwrapped.scene["robot"]
    arm_ids, arm_names = robot.find_joints(ARM_JOINTS, preserve_order=True)

    def step_phase(obs, policy, experiment_name):
        obs_for_policy = adapt_obs(obs, PHASE_OBS_DIM[experiment_name], env, num_envs)
        with torch.no_grad():
            actions = policy(obs_for_policy)
        return env.step(actions)

    obs, _ = env.reset()
    for label, ckpt, policy, steps, experiment_name in phases:
        print(f"\n[chain] === {label} ===")
        set_action_baseline(env, arm_names, PHASE_ROBOT_CFG[experiment_name])
        for _ in range(steps):
            obs, _, _, _ = step_phase(obs, policy, experiment_name)
        print_state(env, label)

    label, _, policy, _, experiment_name = phases[-1]
    print(f"\n[chain] holding on final phase ({label}) -- close the window or Ctrl+C to stop")
    while simulation_app.is_running():
        obs, _, _, _ = step_phase(obs, policy, experiment_name)

    env.close()


main()
simulation_app.close()
