"""Runs Policy 2's trained checkpoint (2026-07-20_16-52-28: grasp + carry to
inspection) in MuJoCo. Same 9-term, 39-dim observation and 8-dim action as
Policy 1 -- env_cfg_policy2.py's ObservationsCfg/ActionsCfg are IDENTICAL to
Policy 1's, term-for-term -- so all the transfer machinery built for Policy 1
(joint mapping, PD gains, real-hand geometry, physically-anchored gripper
conversion, action clip, cube-velocity cap) carries over unchanged. Only two
things differ: which checkpoint is loaded, and how the episode starts.

STARTING STATE differs from Policy 1's (table-resting cube, default arm
pose): Policy 2 starts already holding the pre-grasp position with the cube
COUPLED to wherever the arm actually is (env_cfg_policy2.py's
reset_robot_then_couple_cube: arm -> PRE_GRASP_ARM_POSE, then
cube = actual_ee_fk - GRASP_OFFSET, not an independent spawn point). This
script replicates that: set the arm to PRE_GRASP_ARM_POSE_AT_TRAIN_TIME below
deterministically (no noise, matching every other deterministic replay this
project uses), read MuJoCo's own resulting fingertip position via
mj_forward, then place the cube at ee_pos - GRASP_OFFSET -- coupled the same
way, not read off a fixed constant.

PRE_GRASP_ARM_POSE_AT_TRAIN_TIME is deliberately NOT the current
PRE_GRASP_ARM_POSE in constants.py -- checkpoint 2026-07-20_16-52-28 was
trained before that value was replaced with the fresh teleop measurement, so
using the current one here would start this checkpoint from a pose it never
saw. When Policy 2 gets retrained against the new pose, update this constant
to match (and swap --policy_path) -- everything else in this file is
checkpoint-agnostic and doesn't need to change. That's the reusable part.
"""
import argparse
import sys
import time

import mujoco
import numpy as np
import torch

sys.path.insert(0, ".")
from joint_mapping import MUJOCO_JOINT_ORDER
from pd_gains import DEFAULT_QPOS_MJC, ARM_JOINTS, GRIPPER_OPEN

parser = argparse.ArgumentParser()
parser.add_argument("--policy_path", type=str,
                     default="/home/virtual-acc/projects/g1_lift_ext/logs/rsl_rl/g1_policy2_grasp_carry/2026-07-20_16-52-28/exported/policy.pt")
parser.add_argument("--steps", type=int, default=795)
parser.add_argument("--view", action="store_true", help="open the interactive viewer")
parser.add_argument("--realtime", action="store_true", help="pace --view to real time (0.01s/step) instead of running flat out")
parser.add_argument("--wait_for_signal", type=str, default=None,
                     help="with --view, hold the static starting frame indefinitely (syncing the viewer) until this file path exists, then start stepping -- avoids guessing a fixed delay")
parser.add_argument("--log_every", type=int, default=50)
parser.add_argument("--action_clip", type=float, default=3.0,
                     help="clip raw policy actions to +-this before applying/feeding back (prevents out-of-distribution runaway; 0 disables)")
parser.add_argument("--save_trajectory", type=str, default=None,
                     help="if set, log obs/action/absolute-arm-qpos every step and save to this .npz path (for compare_trajectories.py)")
parser.add_argument("--arm_pose", type=float, nargs=7, default=None,
                     help="override the 7 ARM_JOINTS starting values (order: pitch,roll,yaw,elbow,wrist_roll,wrist_pitch,wrist_yaw) "
                          "-- used to match Isaac Lab's actual sampled post-reset-noise pose for a fair trajectory comparison, "
                          "since PRE_GRASP_ARM_POSE_AT_TRAIN_TIME below is deterministic (no noise)")
args = parser.parse_args()

GRIPPER_CLOSE = 0.0245
GRASP_OFFSET = (-0.010, 0.027, 0.012)
INSPECT_POS = np.array([-0.037, -0.222, 0.997])

# Matches checkpoint 2026-07-20_16-52-28's actual training-time value (the
# 2026-07-16_09-18-57-measured PRE_GRASP_ARM_POSE, pre-teleop-update) -- see
# module docstring. Update when Policy 2 is retrained against the new pose.
PRE_GRASP_ARM_POSE_AT_TRAIN_TIME = {
    "right_shoulder_pitch_joint": +0.1751,
    "right_shoulder_roll_joint": +0.0587,
    "right_shoulder_yaw_joint": +0.1028,
    "right_elbow_joint": -0.3644,
    "right_wrist_roll_joint": +0.1334,
    "right_wrist_pitch_joint": +0.5374,
    "right_wrist_yaw_joint": +0.1710,
}

# Training's REAL hand's own measured fingertip separation at each extreme --
# see run_policy1.py's docstring for the full physically-anchored-conversion
# rationale, identical here.
TRAIN_SEP_AT_OPEN = 0.09007   # meters, at GRIPPER_OPEN
TRAIN_SEP_AT_CLOSE = 0.00117  # meters, at GRIPPER_CLOSE

ARM_MJC_IDS = [MUJOCO_JOINT_ORDER.index(n) for n in ARM_JOINTS]

EE_BODY_NAMES = ["right_hand_Link1_3", "right_hand_Link2_3"]
HAND_BASE_BODY_NAME = "right_hand_base_link"
GRIPPER_MJC_NAMES = ["right_hand_Joint1_1", "right_hand_Joint2_1"]
GRIPPER_MJC_IDS = [MUJOCO_JOINT_ORDER.index(n) for n in GRIPPER_MJC_NAMES]

m = mujoco.MjModel.from_xml_path("g1_lift_scene.xml")
d = mujoco.MjData(m)

ee_body_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in EE_BODY_NAMES]
hand_base_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, HAND_BASE_BODY_NAME)
object_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
arm_joint_ids_full = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]
assert -1 not in ee_body_ids and hand_base_body_id != -1 and object_body_id != -1, \
    "one or more body names not found in the MuJoCo model -- check names above"

ee_tip_local_offset = []
for bid in ee_body_ids:
    geom_ids = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid]
    tip_geom = max(geom_ids, key=lambda g: np.linalg.norm(m.geom_pos[g]))
    ee_tip_local_offset.append(m.geom_pos[tip_geom].copy())
    print(f"  EE body {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid)}: "
          f"tip local offset = {m.geom_pos[tip_geom]} (magnitude {np.linalg.norm(m.geom_pos[tip_geom]):.4f}m)")

default_qpos = np.array(DEFAULT_QPOS_MJC)
d.qpos[:33] = default_qpos
d.ctrl[:33] = default_qpos

# 1. Arm -> PRE_GRASP_ARM_POSE_AT_TRAIN_TIME (deterministic, no reset noise),
# unless --arm_pose overrides it with Isaac Lab's actual sampled reset pose.
start_pose = dict(PRE_GRASP_ARM_POSE_AT_TRAIN_TIME)
if args.arm_pose is not None:
    start_pose = dict(zip(ARM_JOINTS, args.arm_pose))
    print(f"arm start pose overridden (matching Isaac Lab's actual post-reset sample): {args.arm_pose}")
for name, val in start_pose.items():
    idx = MUJOCO_JOINT_ORDER.index(name)
    d.qpos[idx] = val
    d.ctrl[idx] = val
mujoco.mj_forward(m, d)

# 2. Couple the cube to the ACTUAL resulting fingertip position, same formula
# as env_cfg_policy2.py's reset_robot_then_couple_cube (cube = ee - GRASP_OFFSET)
# -- not an independent spawn point the way Policy 1's BLOCK_INIT_POS is.
tip_world0 = [
    d.xpos[bid] + d.xmat[bid].reshape(3, 3) @ local_offset
    for bid, local_offset in zip(ee_body_ids, ee_tip_local_offset)
]
ee_pos0 = np.mean(tip_world0, axis=0)
cube_pos0 = ee_pos0 - np.array(GRASP_OFFSET)
d.qpos[33:36] = cube_pos0
print(f"coupled cube spawn (ee_fk - GRASP_OFFSET): {cube_pos0}")

mujoco.mj_forward(m, d)


def _measure_separation(raw_value):
    saved_qpos = d.qpos[:33].copy()
    d.qpos[GRIPPER_MJC_IDS] = raw_value
    mujoco.mj_forward(m, d)
    tip_world = [
        d.xpos[bid] + d.xmat[bid].reshape(3, 3) @ local_offset
        for bid, local_offset in zip(ee_body_ids, ee_tip_local_offset)
    ]
    sep = float(np.linalg.norm(tip_world[0] - tip_world[1]))
    d.qpos[:33] = saved_qpos
    mujoco.mj_forward(m, d)
    return sep


def _lerp(x, x0, x1, y0, y1):
    if x1 == x0:
        return y0
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


_sep_at_open_cmd = _measure_separation(GRIPPER_OPEN)
_sep_at_close_cmd = _measure_separation(GRIPPER_CLOSE)
if _sep_at_open_cmd >= _sep_at_close_cmd:
    MJC_RAW_OPEN, MJC_SEP_OPEN = GRIPPER_OPEN, _sep_at_open_cmd
    MJC_RAW_CLOSE, MJC_SEP_CLOSE = GRIPPER_CLOSE, _sep_at_close_cmd
else:
    MJC_RAW_OPEN, MJC_SEP_OPEN = GRIPPER_CLOSE, _sep_at_close_cmd
    MJC_RAW_CLOSE, MJC_SEP_CLOSE = GRIPPER_OPEN, _sep_at_open_cmd
print(f"gripper calibration: raw={GRIPPER_OPEN:+.4f} -> sep={_sep_at_open_cmd*100:.2f}cm  "
      f"raw={GRIPPER_CLOSE:+.4f} -> sep={_sep_at_close_cmd*100:.2f}cm")
print(f"  => this model's OPEN: raw={MJC_RAW_OPEN:+.4f} (sep={MJC_SEP_OPEN*100:.2f}cm)   "
      f"CLOSE: raw={MJC_RAW_CLOSE:+.4f} (sep={MJC_SEP_CLOSE*100:.2f}cm)")
print(f"  training's OPEN: sep={TRAIN_SEP_AT_OPEN*100:.2f}cm   CLOSE: sep={TRAIN_SEP_AT_CLOSE*100:.2f}cm\n")


def to_mjc_gripper(isaac_value):
    is_open = abs(isaac_value - GRIPPER_OPEN) < abs(isaac_value - GRIPPER_CLOSE)
    return MJC_RAW_OPEN if is_open else MJC_RAW_CLOSE


def to_isaac_gripper(mjc_value):
    sep = _lerp(mjc_value, MJC_RAW_OPEN, MJC_RAW_CLOSE, MJC_SEP_OPEN, MJC_SEP_CLOSE)
    return _lerp(sep, TRAIN_SEP_AT_OPEN, TRAIN_SEP_AT_CLOSE, GRIPPER_OPEN, GRIPPER_CLOSE)


policy = torch.jit.load(args.policy_path, map_location="cpu")
policy.eval()

last_action = np.zeros(8, dtype=np.float32)


def build_observation():
    """Identical term order/content to run_policy1.py's -- Policy 2's
    ObservationsCfg is term-for-term the same as Policy 1's."""
    arm_joint_pos_rel = d.qpos[ARM_MJC_IDS] - default_qpos[ARM_MJC_IDS]
    arm_joint_vel = d.qvel[ARM_MJC_IDS]
    gripper_joint_pos = to_isaac_gripper(d.qpos[GRIPPER_MJC_IDS])

    object_pos = d.xpos[object_body_id].copy()
    tip_world = [
        d.xpos[bid] + d.xmat[bid].reshape(3, 3) @ local_offset
        for bid, local_offset in zip(ee_body_ids, ee_tip_local_offset)
    ]
    ee_pos = np.mean(tip_world, axis=0)
    ee_to_object = object_pos - ee_pos
    object_to_inspect = INSPECT_POS - object_pos
    hand_base_pos = d.xpos[hand_base_body_id].copy()
    hand_base_to_object = object_pos - hand_base_pos

    obs = np.concatenate([
        arm_joint_pos_rel, arm_joint_vel, gripper_joint_pos,
        object_pos, ee_pos, ee_to_object, object_to_inspect, hand_base_to_object,
        last_action,
    ]).astype(np.float32)
    assert obs.shape == (39,), obs.shape
    return obs, ee_pos, object_pos


print(f"Policy 2 checkpoint: {args.policy_path}")
print(f"Running {args.steps} control steps (decimation=2, 0.01s each -> {args.steps*0.01:.1f}s)\n")

viewer = None
if args.view:
    import mujoco.viewer
    viewer = mujoco.viewer.launch_passive(m, d)
    print("viewer open, is_running=", viewer.is_running())
    if args.wait_for_signal:
        import os
        print(f"holding static starting frame, waiting for signal file: {args.wait_for_signal}")
        while not os.path.exists(args.wait_for_signal):
            viewer.sync()
            time.sleep(0.05)
        print("signal received, starting now")

obs_log = np.zeros((args.steps, 39), dtype=np.float32) if args.save_trajectory else None
action_log = np.zeros((args.steps, 8), dtype=np.float32) if args.save_trajectory else None
arm_abs_log = np.zeros((args.steps, 7), dtype=np.float32) if args.save_trajectory else None

dist_history = []
inspect_dist_history = []
start_wall = time.time()
for t in range(args.steps):
    obs, ee_pos, object_pos = build_observation()
    with torch.no_grad():
        action = policy(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
    if args.action_clip > 0:
        action = np.clip(action, -args.action_clip, args.action_clip)
    if args.save_trajectory:
        obs_log[t] = obs
        action_log[t] = action
        arm_abs_log[t] = d.qpos[ARM_MJC_IDS]
    last_action[:] = action

    target_arm = default_qpos[ARM_MJC_IDS] + action[:7] * 0.5
    d.ctrl[ARM_MJC_IDS] = target_arm
    isaac_wants_open = action[7] >= 0
    gripper_target = to_mjc_gripper(GRIPPER_OPEN if isaac_wants_open else GRIPPER_CLOSE)
    d.ctrl[GRIPPER_MJC_IDS] = gripper_target

    step_wall_start = time.time()
    for _ in range(2):  # decimation=2
        mujoco.mj_step(m, d)
        cube_speed = np.linalg.norm(d.qvel[33:36])
        if cube_speed > 1.0:
            d.qvel[33:36] *= 1.0 / cube_speed
    if viewer is not None:
        viewer.sync()
        if not viewer.is_running():
            break
        if args.realtime:
            remaining = 0.01 - (time.time() - step_wall_start)
            if remaining > 0:
                time.sleep(remaining)

    dist = float(np.linalg.norm(ee_pos - object_pos))
    inspect_dist = float(np.linalg.norm(INSPECT_POS - object_pos))
    dist_history.append(dist)
    inspect_dist_history.append(inspect_dist)
    if t % args.log_every == 0 or t == args.steps - 1:
        print(f"  step {t:4d}  EE-object dist={dist:.4f}  object-to-inspect={inspect_dist:.4f}  "
              f"gripper={'OPEN' if isaac_wants_open else 'CLOSE'}  arm_action_norm={np.linalg.norm(action[:7]):.3f}")

print(f"\nfinal EE-object dist: {dist_history[-1]:.4f}")
print(f"final object-to-inspect dist: {inspect_dist_history[-1]:.4f}")
print(f"min object-to-inspect dist over episode: {min(inspect_dist_history):.4f}")
print(f"wall time: {time.time()-start_wall:.1f}s for {args.steps} steps")

if args.save_trajectory:
    np.savez(args.save_trajectory, obs=obs_log, action=action_log, arm_abs=arm_abs_log, cube_pos0=cube_pos0)
    print(f"trajectory saved to {args.save_trajectory}")

if viewer is not None:
    print("holding window open, ctrl+C to exit")
    try:
        while viewer.is_running():
            mujoco.mj_step(m, d)
            viewer.sync()
    except KeyboardInterrupt:
        pass
