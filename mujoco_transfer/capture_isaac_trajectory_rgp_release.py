"""Captures RGP Policy 4's (release + return-to-ready) deterministic Isaac Lab
rollout (obs + action + absolute arm joint positions, per control step) to an
.npz file, for direct comparison against run_rgp_release.py's MuJoCo rollout.

39-dim obs (same 9-term layout as Policy 3), gym task
Isaac-G1-RGP-Release-Play-v0. Policy 4's reset is already fully deterministic
(ONE fixed real state from policy3_settled_states.pt, not sampled per-episode
-- see mdp/events_rgp.py's reset_robot_then_couple_cube_settled_rgp), so like
capture_isaac_trajectory_rgp_place.py there's no need for an --arm_pose
override: run_rgp_release.py starts from the exact same fixed state by
construction, once its hardcoded constants are updated from this script's
printed arm_pose0/cube_pos0.
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--policy_path", type=str,
                     default="/home/virtual-acc/projects/IsaacLab/logs/rsl_rl/g1_rgp_release/2026-08-07_13-41-01/exported/policy.pt")
parser.add_argument("--steps", type=int, default=795)
parser.add_argument("--out", type=str,
                     default="/home/virtual-acc/projects/g1_lift_ext/mujoco_transfer/trajectory_isaac_rgp_release.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
import g1_lift_rl  # noqa
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from g1_lift_rl.constants import ARM_JOINTS, GRIPPER_JOINTS

task = "Isaac-G1-RGP-Release-Play-v0"
env_cfg = parse_env_cfg(task, device=args.device, num_envs=1)
env = gym.make(task, cfg=env_cfg, render_mode=None)
robot = env.unwrapped.scene["robot"]
arm_ids, _ = robot.find_joints(ARM_JOINTS, preserve_order=True)
gripper_ids, _ = robot.find_joints(GRIPPER_JOINTS)

policy = torch.jit.load(args.policy_path, map_location="cpu")
policy.eval()

obs, _ = env.reset()
obs_tensor = obs["policy"].cpu()

cube_pos0 = env.unwrapped.scene["object"].data.root_pos_w[0].cpu().numpy().copy()
arm_pose0 = robot.data.joint_pos[0, arm_ids].cpu().numpy().copy()
gripper_pose0 = robot.data.joint_pos[0, gripper_ids].cpu().numpy().copy()

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

np.savez(args.out, obs=obs_log, action=action_log, arm_abs=arm_abs_log,
         cube_pos0=cube_pos0, arm_pose0=arm_pose0)

with open(args.out + ".done.txt", "w") as f:
    f.write(f"saved {args.out}\ncube_pos0={cube_pos0.tolist()}\narm_pose0={arm_pose0.tolist()}\n"
            f"gripper_pose0={gripper_pose0.tolist()}\nsteps={args.steps}\n")
    f.flush()

print(f"arm_pose0 (Policy 4's fixed reset state, for reference): {' '.join(f'{v:.6f}' for v in arm_pose0)}")
print(f"gripper_pose0: {' '.join(f'{v:.6f}' for v in gripper_pose0)}")
print(f"cube_pos0: {' '.join(f'{v:.6f}' for v in cube_pos0)}")

env.close()
simulation_app.close()
