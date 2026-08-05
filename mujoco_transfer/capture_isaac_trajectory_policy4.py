"""Captures Policy 4's deterministic Isaac Lab rollout (33-dim obs, same
layout as Policy 3) for comparison against run_policy4.py's MuJoCo rollout.
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--policy_path", type=str,
                     default="/home/virtual-acc/projects/g1_lift_ext/logs/rsl_rl/g1_policy4_release_return/2026-07-20_15-58-36/exported/policy.pt")
parser.add_argument("--steps", type=int, default=795)
parser.add_argument("--out", type=str,
                     default="/home/virtual-acc/projects/g1_lift_ext/mujoco_transfer/trajectory_isaac_policy4.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
import g1_lift_rl  # noqa
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from g1_lift_rl.constants import ARM_JOINTS

task = "Isaac-G1-Policy4-Ext-Play-v0"
env_cfg = parse_env_cfg(task, device=args.device, num_envs=1)
env = gym.make(task, cfg=env_cfg, render_mode=None)
robot = env.unwrapped.scene["robot"]
arm_ids, _ = robot.find_joints(ARM_JOINTS, preserve_order=True)

policy = torch.jit.load(args.policy_path, map_location="cpu")
policy.eval()

obs, _ = env.reset()
obs_tensor = obs["policy"].cpu()

cube_pos0 = env.unwrapped.scene["object"].data.root_pos_w[0].cpu().numpy().copy()

obs_dim = obs_tensor.shape[-1]
obs_log = np.zeros((args.steps, obs_dim), dtype=np.float32)
action_log = np.zeros((args.steps, 8), dtype=np.float32)
arm_abs_log = np.zeros((args.steps, 7), dtype=np.float32)

for t in range(args.steps):
    with torch.no_grad():
        action = policy(obs_tensor)
    obs_log[t] = obs_tensor[0].numpy()
    action_log[t] = action[0].numpy()
    arm_abs_log[t] = robot.data.joint_pos[0, arm_ids].cpu().numpy()
    obs, rew, terminated, truncated, info = env.step(action.to(env.unwrapped.device))
    obs_tensor = obs["policy"].cpu()

np.savez(args.out, obs=obs_log, action=action_log, arm_abs=arm_abs_log, cube_pos0=cube_pos0)

with open(args.out + ".done.txt", "w") as f:
    f.write(f"saved {args.out}\ncube_pos0={cube_pos0.tolist()}\nobs_dim={obs_dim}\nsteps={args.steps}\n")
    f.flush()

env.close()
simulation_app.close()
