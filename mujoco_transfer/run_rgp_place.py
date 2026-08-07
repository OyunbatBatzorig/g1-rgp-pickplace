"""Runs RGP Policy 3's (place) trained checkpoint in MuJoCo, reconstructing
its exact 9-term, 39-dim observation and applying its 8-dim action.

Adapted from run_rgp_grasp.py -- same robot, same real-hand-geometry
reconstruction and gripper calibration, same ACTION_CLIP/cube-velocity-cap
transfer-safety measures, same g1_rgp_reach_scene.xml scene.

Two real structural differences from Policy 2:

1. Reset state. Policy 3 does NOT start from RGP_POLICY1_ARM_POSE -- it starts
   from ONE FIXED real state sampled from Policy 2's own captured rollout
   (policy2_held_states.pt's "closest to the library's own mean" pick, same
   selection Isaac Lab's reset_robot_then_couple_cube_grasping_rgp makes).
   PLACE_ARM_POSE/PLACE_GRIPPER_POSE/PLACE_CUBE_POS below are that state's
   values, extracted once and hardcoded (no --arm_pose override needed --
   unlike Policy 2's randomized-per-episode reset, this one is already fully
   deterministic by construction, so there's nothing to match against a
   particular Isaac Lab rollout).

2. Action baseline. Policy 3's Isaac Lab config gives the arm a DEDICATED
   default pose (RGP_G1_DEX1_PLACE_CFG) equal to PLACE_ARM_POSE itself, NOT
   the shared READY_ARM_POSE every other policy in this chain uses --
   JointPositionActionCfg(use_default_offset=True) reads that as its action
   baseline, and getting this wrong on the Isaac Lab side (reusing the shared
   default) was the actual root cause of Policy 3's first training failure
   this session. default_qpos below is DEFAULT_QPOS_MJC with the arm entries
   overwritten to match, for the same reason.
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

# Policy 2's own converged "held, lifted" state (fixed_idx=185 of 506,
# closest to the library's own mean arm pose) -- Policy 3's reset target AND
# its dedicated action baseline, both on the Isaac Lab side and here.
PLACE_ARM_POSE = {
    "right_shoulder_pitch_joint": -0.2807234823703766,
    "right_shoulder_roll_joint": 0.034067295491695404,
    "right_shoulder_yaw_joint": 0.9662142395973206,
    "right_elbow_joint": 0.32968395948410034,
    "right_wrist_roll_joint": -1.3125163316726685,
    "right_wrist_pitch_joint": 0.7132918238639832,
    "right_wrist_yaw_joint": -1.4258983135223389,
}
PLACE_GRIPPER_POSE = [0.006509182043373585, 0.0055356924422085285]  # real, mechanically-blocked value
PLACE_CUBE_POS = [0.019256591796875, -0.21521103382110596, 0.9783000946044922]

# env-local == world-frame here (single-env MuJoCo scene). See
# env_cfg_rgp_scene.py: RGP_GOAL_POS = (GOAL_POS[0], GOAL_POS[1],
# TABLE_TOP_Z + RGP_BLOCK_SIZE/2).
RGP_GOAL_POS = np.array([0.070, -0.389, 0.825])

parser = argparse.ArgumentParser()
parser.add_argument("--policy_path", type=str,
                     default="/home/virtual-acc/projects/IsaacLab/logs/rsl_rl/g1_rgp_place/2026-08-06_16-11-00/exported/policy.pt")
parser.add_argument("--steps", type=int, default=795)
parser.add_argument("--view", action="store_true", help="open the interactive viewer")
parser.add_argument("--realtime", action="store_true", help="pace --view to real time (0.01s/step) instead of running flat out")
parser.add_argument("--wait_for_signal", type=str, default=None,
                     help="with --view, hold the static starting frame indefinitely (syncing the viewer) until this file path exists, then start stepping")
parser.add_argument("--log_every", type=int, default=50)
parser.add_argument("--action_clip", type=float, default=3.0,
                     help="clip raw policy actions to +-this before applying/feeding back (prevents out-of-distribution runaway; 0 disables)")
parser.add_argument("--save_trajectory", type=str, default=None,
                     help="if set, log obs/action/absolute-arm-qpos every step and save to this .npz path (for compare_trajectories_rgp_place.py)")
args = parser.parse_args()

GRIPPER_CLOSE = 0.0245

TRAIN_SEP_AT_OPEN = 0.09007   # meters, at GRIPPER_OPEN
TRAIN_SEP_AT_CLOSE = 0.00117  # meters, at GRIPPER_CLOSE

ARM_MJC_IDS = [MUJOCO_JOINT_ORDER.index(n) for n in ARM_JOINTS]

EE_BODY_NAMES = ["right_hand_Link1_3", "right_hand_Link2_3"]
HAND_BASE_BODY_NAME = "right_hand_base_link"
GRIPPER_MJC_NAMES = ["right_hand_Joint1_1", "right_hand_Joint2_1"]
GRIPPER_MJC_IDS = [MUJOCO_JOINT_ORDER.index(n) for n in GRIPPER_MJC_NAMES]

m = mujoco.MjModel.from_xml_path("g1_rgp_reach_scene.xml")
d = mujoco.MjData(m)

ee_body_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in EE_BODY_NAMES]
hand_base_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, HAND_BASE_BODY_NAME)
object_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
assert -1 not in ee_body_ids and hand_base_body_id != -1 and object_body_id != -1, \
    "one or more body names not found in the MuJoCo model -- check names above"

ee_tip_local_offset = []
for bid in ee_body_ids:
    geom_ids = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid]
    tip_geom = max(geom_ids, key=lambda g: np.linalg.norm(m.geom_pos[g]))
    ee_tip_local_offset.append(m.geom_pos[tip_geom].copy())
    print(f"  EE body {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid)}: "
          f"tip local offset = {m.geom_pos[tip_geom]} (magnitude {np.linalg.norm(m.geom_pos[tip_geom]):.4f}m)")

# Action-target baseline: DEFAULT_QPOS_MJC (shared READY_ARM_POSE/LEFT_ARM_STOW
# default) with the right-arm entries overwritten to PLACE_ARM_POSE -- Policy
# 3's own DEDICATED baseline, matching RGP_G1_DEX1_PLACE_CFG on the Isaac Lab
# side. Must NOT be the plain shared default here; see module docstring.
default_qpos = np.array(DEFAULT_QPOS_MJC)
for name, val in PLACE_ARM_POSE.items():
    default_qpos[MUJOCO_JOINT_ORDER.index(name)] = val

# --- Reset state: PLACE_ARM_POSE, gripper at its real captured (blocked)
# value, cube at PLACE_CUBE_POS. STATE only -- the drive TARGET below is set
# to full GRIPPER_CLOSE, not this value, so there's a real continuous holding
# force from frame 0 (same guard as Isaac Lab's own reset event: target==state
# gives zero corrective force and the cube falls).
d.qpos[:33] = default_qpos
d.ctrl[:33] = default_qpos
for name, val in PLACE_ARM_POSE.items():
    idx = MUJOCO_JOINT_ORDER.index(name)
    d.qpos[idx] = val
    d.ctrl[idx] = val
d.qpos[GRIPPER_MJC_IDS] = PLACE_GRIPPER_POSE
d.ctrl[GRIPPER_MJC_IDS] = GRIPPER_CLOSE
d.qpos[33:36] = PLACE_CUBE_POS
mujoco.mj_forward(m, d)
print(f"reset cube position (fixed, from policy2_held_states.pt): {PLACE_CUBE_POS}")


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
    """39-dim observation, same term order as env_cfg_rgp_place.py's
    ObservationsCfg: arm_joint_pos_rel(7), arm_joint_vel(7),
    gripper_joint_pos(2), object_position(3), ee_position(3),
    ee_to_object(3), hand_base_to_object(3), object_to_goal(3), last_action(8).
    object_to_goal is the ONE addition over Policy 1/2's 36-dim layout."""
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
    hand_base_pos = d.xpos[hand_base_body_id].copy()
    hand_base_to_object = object_pos - hand_base_pos
    object_to_goal = RGP_GOAL_POS - object_pos

    obs = np.concatenate([
        arm_joint_pos_rel, arm_joint_vel, gripper_joint_pos,
        object_pos, ee_pos, ee_to_object, hand_base_to_object, object_to_goal,
        last_action,
    ]).astype(np.float32)
    assert obs.shape == (39,), obs.shape
    return obs, ee_pos, object_pos


print(f"RGP Policy 3 checkpoint: {args.policy_path}")
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
dist_to_goal_history = []
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
    dist_to_goal = float(np.linalg.norm(RGP_GOAL_POS - object_pos))
    dist_history.append(dist)
    dist_to_goal_history.append(dist_to_goal)
    if t % args.log_every == 0 or t == args.steps - 1:
        print(f"  step {t:4d}  EE-object dist={dist:.4f}  dist_to_goal={dist_to_goal:.4f}  "
              f"gripper={'OPEN' if isaac_wants_open else 'CLOSE'}  arm_action_norm={np.linalg.norm(action[:7]):.3f}")

print(f"\nfinal EE-object dist: {dist_history[-1]:.4f}")
print(f"final dist_to_goal: {dist_to_goal_history[-1]:.4f}  min dist_to_goal: {min(dist_to_goal_history):.4f}")
print(f"wall time: {time.time()-start_wall:.1f}s for {args.steps} steps")

if args.save_trajectory:
    np.savez(args.save_trajectory, obs=obs_log, action=action_log, arm_abs=arm_abs_log, cube_pos0=np.array(PLACE_CUBE_POS))
    print(f"trajectory saved to {args.save_trajectory}")

if viewer is not None:
    print("holding window open, ctrl+C to exit")
    try:
        while viewer.is_running():
            mujoco.mj_step(m, d)
            viewer.sync()
    except KeyboardInterrupt:
        pass
