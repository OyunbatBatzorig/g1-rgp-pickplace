#!/usr/bin/env python3
"""Captures a library of REAL, physically-verified "settled at the goal,
still gripping" states from Policy 3's own trained checkpoint -- for Policy
4's reset to sample from directly, same technique as
capture_policy2_held_states.py used for the Policy 2 -> 3 handoff (see that
file's own docstring for why: a reconstructed state is fragile, a real
captured one is safe by construction).

Success filter matches reward_place_rgp's own gate exactly: settled (near
goal AND slow) AND still grasping.

Run (from IsaacLab/, like every other script in this repo) whenever Policy 3
is retrained, to keep the library matched to the current checkpoint:
    ./isaaclab.sh -p ../g1_lift_ext/capture_policy3_settled_states.py \
        --checkpoint logs/rsl_rl/g1_rgp_place/<run>/model_<N>.pt \
        --out ../g1_lift_ext/g1_lift_rl/policy3_settled_states.pt --headless
"""
import argparse
import sys
from importlib import metadata

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-G1-RGP-Place-Play-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=700)
parser.add_argument("--out", type=str, required=True)
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

from g1_lift_rl.constants import ARM_JOINTS, GRIPPER_JOINTS
from g1_lift_rl.mdp import rewards_rgp as rew_rgp


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)
    device = env.unwrapped.device
    e = env.unwrapped

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=str(device))
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=device)

    scene = e.scene
    robot = scene["robot"]
    obj = scene["object"]
    env_origins = scene.env_origins
    arm_ids, arm_names = robot.find_joints(ARM_JOINTS, preserve_order=True)
    gripper_ids, _ = robot.find_joints(GRIPPER_JOINTS)

    N = args_cli.num_envs

    obs, _ = env.reset()
    for i in range(args_cli.steps):
        with torch.no_grad():
            actions = policy(obs)
        obs, _, dones, extras = env.step(actions)

    settled = rew_rgp._settled_rgp(e)
    grasping = rew_rgp._is_grasping_rgp(e)
    success = settled & grasping

    n_success = success.sum().item()
    print(f"\nsuccessful settled states: {n_success}/{N}")

    if n_success == 0:
        print("!!! no successful envs -- nothing to save")
        env.close()
        simulation_app.close()
        return

    s = success
    arm_pose = robot.data.joint_pos[s][:, arm_ids].cpu()
    gripper_pose = robot.data.joint_pos[s][:, gripper_ids].cpu()
    cube_pos_w = obj.data.root_pos_w
    cube_pos_local = (cube_pos_w[s] - env_origins[s]).cpu()

    torch.save({
        "arm_joint_names": arm_names,
        "arm_pose": arm_pose,
        "gripper_pose": gripper_pose,
        "cube_pos_local": cube_pos_local,
        "source_checkpoint": args_cli.checkpoint,
    }, args_cli.out)
    print(f"saved {n_success} settled states to {args_cli.out}")
    print(f"  arm_pose range per joint (min/max, rad):")
    for j, name in enumerate(arm_names):
        print(f"    {name:28s} [{arm_pose[:, j].min():.4f}, {arm_pose[:, j].max():.4f}]")
    dist_to_goal = rew_rgp._dist_to_goal_rgp(e)[s].cpu()
    print(f"  dist_to_goal range: [{dist_to_goal.min():.4f}, {dist_to_goal.max():.4f}]")

    env.close()


main()
simulation_app.close()
