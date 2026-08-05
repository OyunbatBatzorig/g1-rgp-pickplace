import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import g1_lift_rl  # noqa
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

task = "Isaac-G1-Lift-Ext-Play-v0"
env_cfg = parse_env_cfg(task, device=args.device, num_envs=1)
env = gym.make(task, cfg=env_cfg, render_mode=None)
robot = env.unwrapped.scene["robot"]

HAND_BODIES = [
    "right_hand_base_link",
    "right_hand_Link1_1", "right_hand_Link1_2", "right_hand_Link1_3",
    "right_hand_Link2_1", "right_hand_Link2_2", "right_hand_Link2_3",
]
body_ids, body_names = robot.find_bodies(HAND_BODIES, preserve_order=True)
masses = robot.root_physx_view.get_masses()
inertias = robot.root_physx_view.get_inertias()  # (num_instances, num_bodies, 9) flattened 3x3

f = open("hand_mass_result.txt", "w")
def log(msg=""):
    print(msg, flush=True)
    f.write(str(msg) + "\n")
    f.flush()

log("REAL_MASS_START")
for i, name in enumerate(body_names):
    m = masses[0, body_ids[i]].item()
    inertia_flat = inertias[0, body_ids[i]].tolist()
    log(f"{name}: mass={m:.6f} kg  inertia_flat_3x3={inertia_flat}")
log("REAL_MASS_END")
f.close()

env.close()
simulation_app.close()
