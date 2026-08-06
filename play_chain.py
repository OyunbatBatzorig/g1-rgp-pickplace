#!/usr/bin/env python3
"""Run the full RGP chain (Reach -> Grasp+lift -> Place+release) back-to-back in
one continuous rollout, picking up each policy's own latest trained checkpoint
automatically.

The environment is reset ONCE at the start; only the acting policy network
swaps between phases, so the physical state (robot pose, cube position) carries
over exactly the way the real chain hands off -- this is not three separate
episodes stitched together, it's one continuous rollout.

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
                     help="Env cfg used for the whole rollout (scene/actions/observations are "
                          "identical across RGP policies by design). Switch to the Place task's "
                          "Play cfg here once Policy 3 exists, so the goal marker is in the scene.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--reach_steps", type=int, default=400)
parser.add_argument("--grasp_steps", type=int, default=400)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import g1_lift_rl  # noqa: F401
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config

from g1_lift_rl.agents.rsl_rl_ppo_cfg_rgp import G1RGPReachPPORunnerCfg, G1RGPGraspPPORunnerCfg

RSL_RL_VERSION = metadata.version("rsl-rl-lib")

# Phase order: (label, experiment_name, agent cfg, step budget). Add Policy 3
# here once it's trained (agents/rsl_rl_ppo_cfg_rgp.py + its own experiment_name)
# -- everything else below picks it up automatically.
PHASE_SPECS = [
    ("Reach", "g1_rgp_reach", G1RGPReachPPORunnerCfg(), args_cli.reach_steps),
    ("Grasp + lift", "g1_rgp_grasp", G1RGPGraspPPORunnerCfg(), args_cli.grasp_steps),
]


def latest_checkpoint(experiment_name: str) -> str | None:
    """Most recently WRITTEN model_*.pt across every run of `experiment_name`
    (by file mtime, not run-directory name) -- avoids a fresh, barely-started
    run shadowing a fully-trained checkpoint from an interrupted earlier run."""
    checkpoints = glob.glob(os.path.join("logs", "rsl_rl", experiment_name, "*", "model_*.pt"))
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)


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


@hydra_task_config(args_cli.task, None)
def main(env_cfg: ManagerBasedRLEnvCfg, _agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)
    device = env.unwrapped.device

    phases = []
    for label, experiment_name, agent_cfg, steps in PHASE_SPECS:
        ckpt = latest_checkpoint(experiment_name)
        if ckpt is None:
            print(f"[chain] skipping {label}: no checkpoint found for '{experiment_name}' yet")
            continue
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, RSL_RL_VERSION)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=str(device))
        runner.load(ckpt)
        policy = runner.get_inference_policy(device=device)
        phases.append((label, ckpt, policy, steps))
        print(f"[chain] {label}: {ckpt} ({steps} steps)")

    if not phases:
        print("[chain] no trained checkpoints found -- train Policy 1 first (see INSTALL.md).")
        env.close()
        return

    obs, _ = env.reset()
    with torch.no_grad():
        for label, ckpt, policy, steps in phases:
            print(f"\n[chain] === {label} ===")
            for _ in range(steps):
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)
            print_state(env, label)

        label, _, policy, _ = phases[-1]
        print(f"\n[chain] holding on final phase ({label}) -- close the window or Ctrl+C to stop")
        while simulation_app.is_running():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

    env.close()


main()
simulation_app.close()
