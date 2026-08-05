"""Captures Policy 2's deterministic Isaac Lab rollout (obs + action + absolute
arm joint positions, per control step) to an .npz file, for direct comparison
against run_policy2.py's MuJoCo rollout via compare_trajectories.py.

Same approach as capture_isaac_trajectory.py (Policy 1): uses the SAME
exported policy.pt (torch.jit) as the MuJoCo side, fed the env's raw
obs["policy"] tensor directly. Only difference from that script: the task id
(Policy 2's own gym task, whose reset event -- reset_robot_then_couple_cube --
handles the arm-then-cube coupling automatically, unlike run_policy2.py's
MuJoCo side which has to replicate that coupling manually).
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--policy_path", type=str,
                     default="/home/virtual-acc/projects/g1_lift_ext/logs/rsl_rl/g1_policy2_grasp_carry/2026-07-20_16-52-28/exported/policy.pt")
parser.add_argument("--steps", type=int, default=795)
parser.add_argument("--out", type=str,
                     default="/home/virtual-acc/projects/g1_lift_ext/mujoco_transfer/trajectory_isaac_policy2.npz")
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

task = "Isaac-G1-Policy2-Ext-Play-v0"
env_cfg = parse_env_cfg(task, device=args.device, num_envs=1)
env = gym.make(task, cfg=env_cfg, render_mode=None)
robot = env.unwrapped.scene["robot"]
arm_ids, _ = robot.find_joints(ARM_JOINTS, preserve_order=True)

policy = torch.jit.load(args.policy_path, map_location="cpu")
policy.eval()

obs, _ = env.reset()
obs_tensor = obs["policy"].cpu()

cube_pos0 = env.unwrapped.scene["object"].data.root_pos_w[0].cpu().numpy().copy()

obs_log = np.zeros((args.steps, 39), dtype=np.float32)
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
    f.write(f"saved {args.out}\ncube_pos0={cube_pos0.tolist()}\nsteps={args.steps}\n")
    f.flush()

env.close()
simulation_app.close()
