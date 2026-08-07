#!/usr/bin/env python3
"""Captures a library of REAL, physically-verified "genuinely holding the
cube, lifted" states from Policy 2's own trained checkpoint -- for Policy 3's
reset to sample from directly, instead of reconstructing a state via FK +
offset math (which proved fragile: a single-frame teleport into contact is a
much more violent operation for the physics engine than the gradual real
closing motion Policy 2 itself performs over many simulated steps).

Same success filter as the deterministic verification (EE-to-cube < GRASP_DIST
AND gripper closed AND lifted >2cm), run across many envs so the saved
library has real diversity -- no synthetic noise needed, this IS the real
distribution of how Policy 2 actually ends up holding the cube.

Run (from IsaacLab/, like every other script in this repo) whenever Policy 2
is retrained, to keep the library matched to the current checkpoint:
    ./isaaclab.sh -p ../g1_lift_ext/capture_policy2_held_states.py \
        --checkpoint logs/rsl_rl/g1_rgp_grasp/<run>/exported/policy.pt \
        --out ../g1_lift_ext/g1_lift_rl/policy2_held_states.pt --headless
"""
import argparse
import sys
from importlib import metadata

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-G1-RGP-Grasp-Play-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=512)
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
from g1_lift_rl.mdp.rewards_rgp import GRASP_DIST_RGP, GRIP_CLOSED_THRESHOLD_RGP


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)
    device = env.unwrapped.device

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=str(device))
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=device)

    scene = env.unwrapped.scene
    robot = scene["robot"]
    obj = scene["object"]
    env_origins = scene.env_origins
    ee_frame_ids = robot.find_bodies(["right_hand_Link1_3", "right_hand_Link2_3"])[0]
    arm_ids, arm_names = robot.find_joints(ARM_JOINTS, preserve_order=True)
    gripper_ids, _ = robot.find_joints(GRIPPER_JOINTS)

    N = args_cli.num_envs
    steps = 400

    obs, _ = env.reset()
    for i in range(steps):
        with torch.no_grad():
            actions = policy(obs)
        obs, _, dones, extras = env.step(actions)

    ee_pos = robot.data.body_pos_w[:, ee_frame_ids, :].mean(dim=1)
    cube_pos_w = obj.data.root_pos_w
    grip_mean = robot.data.joint_pos[:, gripper_ids].mean(dim=-1)
    final_dist = torch.norm(ee_pos - cube_pos_w, dim=-1)
    lift_height = cube_pos_w[:, 2] - env_origins[:, 2]

    is_grasping = (final_dist < GRASP_DIST_RGP) & (grip_mean > GRIP_CLOSED_THRESHOLD_RGP)
    success = is_grasping & (lift_height > 0.7)  # sanity floor, well below any real lift height

    n_success = success.sum().item()
    print(f"\nsuccessful held states: {n_success}/{N}")

    if n_success == 0:
        print("!!! no successful envs -- nothing to save")
        env.close()
        simulation_app.close()
        return

    s = success
    arm_pose = robot.data.joint_pos[s][:, arm_ids].cpu()          # (n_success, 7)
    gripper_pose = robot.data.joint_pos[s][:, gripper_ids].cpu()  # (n_success, 2)
    cube_pos_local = (cube_pos_w[s] - env_origins[s]).cpu()       # (n_success, 3)

    torch.save({
        "arm_joint_names": arm_names,
        "arm_pose": arm_pose,
        "gripper_pose": gripper_pose,
        "cube_pos_local": cube_pos_local,
        "source_checkpoint": args_cli.checkpoint,
    }, args_cli.out)
    print(f"saved {n_success} held states to {args_cli.out}")
    print(f"  arm_pose range per joint (min/max, rad):")
    for j, name in enumerate(arm_names):
        print(f"    {name:28s} [{arm_pose[:, j].min():.4f}, {arm_pose[:, j].max():.4f}]")
    print(f"  lift height range: [{(cube_pos_local[:, 2]).min():.4f}, {(cube_pos_local[:, 2]).max():.4f}]")

    env.close()


main()
simulation_app.close()
