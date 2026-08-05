"""Read-only query of the REAL Dex1 hand's kinematic structure from Isaac Lab
(training's actual USD) -- body transforms and joint types/axes for the 7 real
hand bodies. Used to author an accurate MuJoCo hand replacement; does not
modify Isaac Lab, the USD, or any training config.
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

RESULT_FILE = "hand_kinematics_result.txt"
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
    "right_wrist_yaw_link",  # parent reference
    "right_hand_base_link",
    "right_hand_Link1_1", "right_hand_Link1_2", "right_hand_Link1_3",
    "right_hand_Link2_1", "right_hand_Link2_2", "right_hand_Link2_3",
]
body_ids, body_names = robot.find_bodies(HAND_BODIES, preserve_order=True)

env.reset()
log("Body world positions/orientations at reset (READY_ARM_POSE):")
pos_w = robot.data.body_pos_w[0, body_ids, :]
quat_w = robot.data.body_quat_w[0, body_ids, :]
for i, name in enumerate(body_names):
    log(f"  {name:24s} pos_w={pos_w[i].tolist()}  quat_w(wxyz)={quat_w[i].tolist()}")

log("\nRelative to right_wrist_yaw_link (world-frame offset, NOT yet rotated into wrist-local frame):")
wrist_pos = pos_w[0]
for i, name in enumerate(body_names):
    log(f"  {name:24s} offset_from_wrist_world={(pos_w[i]-wrist_pos).tolist()}")

log("\nJoint info for the finger joints:")
finger_joints = ["right_hand_Joint1_1", "right_hand_Joint2_1"]
for jn in finger_joints:
    jids, jnames = robot.find_joints([jn])
    jid = jids[0]
    log(f"  {jn}: joint_pos_limits={robot.data.joint_pos_limits[0, jid].tolist()}")

log("\nAll joint names (to see if there's more than 1 joint per finger, e.g. coupled/mimic joints):")
log(str(robot.joint_names))

f.close()
env.close()
simulation_app.close()
