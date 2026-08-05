"""Second snapshot, at GRIPPER_CLOSE instead of the default GRIPPER_OPEN, to
isolate Joint1_1/Joint2_1's actual translation axis+distance from the fixed
internal geometry of each 3-piece rigid finger assembly (confirmed via
query_hand_kinematics.py: Link1_2/Link1_3/Link2_2/Link2_3 are not independent
joints, so their offset from Link1_1/Link2_1 should be IDENTICAL in both
snapshots if they're truly rigidly fixed -- this run checks that directly).
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import g1_lift_rl  # noqa
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

RESULT_FILE = "hand_kinematics_result2.txt"
f = open(RESULT_FILE, "w")
def log(msg=""):
    print(msg, flush=True)
    f.write(str(msg) + "\n")
    f.flush()

task = "Isaac-G1-Lift-Ext-Play-v0"
env_cfg = parse_env_cfg(task, device=args.device, num_envs=1)
env = gym.make(task, cfg=env_cfg, render_mode=None)
robot = env.unwrapped.scene["robot"]

HAND_BODIES = [
    "right_wrist_yaw_link", "right_hand_base_link",
    "right_hand_Link1_1", "right_hand_Link1_2", "right_hand_Link1_3",
    "right_hand_Link2_1", "right_hand_Link2_2", "right_hand_Link2_3",
]
body_ids, body_names = robot.find_bodies(HAND_BODIES, preserve_order=True)
gripper_joint_ids, _ = robot.find_joints(["right_hand_Joint1_1", "right_hand_Joint2_1"])

env.reset()
# Drive the gripper joints directly to GRIPPER_CLOSE and let PD settle.
GRIPPER_CLOSE = 0.0245
jp = robot.data.joint_pos.clone()
jp[:, gripper_joint_ids] = GRIPPER_CLOSE
robot.write_joint_state_to_sim(jp, robot.data.joint_vel.clone())
robot.set_joint_position_target(jp)
robot.write_data_to_sim()
for _ in range(50):
    env.unwrapped.sim.step()
    env.unwrapped.scene.update(dt=env.unwrapped.sim.get_physics_dt())

pos_w = robot.data.body_pos_w[0, body_ids, :]
quat_w = robot.data.body_quat_w[0, body_ids, :]
log("Body world positions/orientations at GRIPPER_CLOSE:")
for i, name in enumerate(body_names):
    log(f"  {name:24s} pos_w={pos_w[i].tolist()}  quat_w(wxyz)={quat_w[i].tolist()}")

log(f"\nActual gripper joint pos reached: {robot.data.joint_pos[0, gripper_joint_ids].tolist()}")

f.close()
env.close()
simulation_app.close()
