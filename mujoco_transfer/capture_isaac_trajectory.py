"""Captures Policy 1's deterministic Isaac Lab rollout (obs + action + absolute
arm joint positions, per control step) to an .npz file, for direct comparison
against run_policy1.py's MuJoCo rollout via compare_trajectories.py.

Uses the SAME exported policy.pt (torch.jit) as the MuJoCo side, fed the env's
raw obs["policy"] tensor directly -- this isolates observation-reconstruction
differences (physics/geometry approximations) as the only source of
behavioral divergence between the two sims, since the network forward pass
itself is identical on both sides.

Note: "gripper never closes" is NOT a bug (see the correction this script was
built to investigate further) -- Policy 1's own RewardsCfg only rewards
reach-and-hold at PRE_GRASP_ARM_POSE with the gripper open; grasping is
Policy 2's job (penalty_early_close exists specifically to stop Policy 1 from
closing early). The real transfer question is whether MuJoCo's arm actually
converges to PRE_GRASP_ARM_POSE the way Isaac Lab's does.
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--policy_path", type=str,
                     default="/home/virtual-acc/projects/g1_lift_ext/logs/rsl_rl/g1_lift_policy1/2026-07-16_09-18-57/exported/policy.pt")
parser.add_argument("--steps", type=int, default=795)
parser.add_argument("--out", type=str,
                     default="/home/virtual-acc/projects/g1_lift_ext/mujoco_transfer/trajectory_isaac.npz")
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

task = "Isaac-G1-Lift-Ext-Play-v0"
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

# Explicit file write (not print()) -- simulation_app.close() below exits the
# process abruptly, before Python's normal exit-time stdout flush runs; this
# bug has already been hit and fixed the same way several times this session.
with open(args.out + ".done.txt", "w") as f:
    f.write(f"saved {args.out}\ncube_pos0={cube_pos0.tolist()}\nsteps={args.steps}\n")
    f.flush()

env.close()
simulation_app.close()
