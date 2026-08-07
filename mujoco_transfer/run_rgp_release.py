"""Runs RGP Policy 4's (release + return-to-ready) trained checkpoint in
MuJoCo, reconstructing its exact 9-term, 39-dim observation and applying its
8-dim action.

Adapted from run_rgp_place.py -- same robot, same real-hand-geometry
reconstruction and gripper calibration, same ACTION_CLIP/cube-velocity-cap
transfer-safety measures, same g1_rgp_reach_scene.xml scene, same
state-vs-target gripper reset guard.

Two real structural differences from Policy 3:

1. Reset state. Policy 4 starts from ONE FIXED real state sampled from Policy
   3's own captured rollout (policy3_settled_states.pt's "closest to the
   library's own mean" pick, same selection Isaac Lab's
   reset_robot_then_couple_cube_settled_rgp makes) -- cube already AT the
   goal, gripper still closed. RELEASE_ARM_POSE/RELEASE_GRIPPER_POSE/
   RELEASE_CUBE_POS below are that state's values, captured via
   capture_isaac_trajectory_rgp_release.py (not guessed).

2. Action baseline. Policy 4's Isaac Lab config gives the arm a DEDICATED
   default pose (RGP_G1_DEX1_RELEASE_CFG) equal to RELEASE_ARM_POSE itself,
   NOT Policy 3's own PLACE_ARM_POSE baseline -- getting this wrong was
   Policy 3's own root cause the first time around, so it's set correctly
   from the start here. default_qpos below is DEFAULT_QPOS_MJC with the arm
   entries overwritten to match.
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

# Policy 3's own converged "settled at goal, still gripping" state (see
# capture_isaac_trajectory_rgp_release.py's printed/saved values) -- Policy
# 4's reset target AND its dedicated action baseline, both on the Isaac Lab
# side and here.
RELEASE_ARM_POSE = {
    "right_shoulder_pitch_joint": -0.8246623277664185,
    "right_shoulder_roll_joint": 0.18693634867668152,
    "right_shoulder_yaw_joint": 0.6653597950935364,
    "right_elbow_joint": 1.4457058906555176,
    "right_wrist_roll_joint": 0.33834579586982727,
    "right_wrist_pitch_joint": -0.7330788969993591,
    "right_wrist_yaw_joint": 0.12768231332302094,
}
RELEASE_GRIPPER_POSE = [0.007776439189910889, 0.007330344058573246]  # real, mechanically-blocked value
RELEASE_CUBE_POS = [0.07231330871582031, -0.37784862518310547, 0.8447996377944946]  # already at goal

# env-local == world-frame here (single-env MuJoCo scene). Same constant
# Policy 3 uses -- see env_cfg_rgp_scene.py: RGP_GOAL_POS.
RGP_GOAL_POS = np.array([0.070, -0.389, 0.825])

parser = argparse.ArgumentParser()
parser.add_argument("--policy_path", type=str,
                     default="/home/virtual-acc/projects/IsaacLab/logs/rsl_rl/g1_rgp_release/2026-08-07_13-41-01/exported/policy.pt")
parser.add_argument("--steps", type=int, default=795)
parser.add_argument("--view", action="store_true", help="open the interactive viewer")
parser.add_argument("--realtime", action="store_true", help="pace --view to real time (0.01s/step) instead of running flat out")
parser.add_argument("--wait_for_signal", type=str, default=None,
                     help="with --view, hold the static starting frame indefinitely (syncing the viewer) until this file path exists, then start stepping")
parser.add_argument("--log_every", type=int, default=50)
parser.add_argument("--action_clip", type=float, default=3.0,
                     help="clip raw policy actions to +-this before applying/feeding back (prevents out-of-distribution runaway; 0 disables)")
parser.add_argument("--save_trajectory", type=str, default=None,
                     help="if set, log obs/action/absolute-arm-qpos every step and save to this .npz path (for compare_trajectories_rgp_release.py)")
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
# default) with the right-arm entries overwritten to RELEASE_ARM_POSE --
# Policy 4's own DEDICATED baseline, matching RGP_G1_DEX1_RELEASE_CFG on the
# Isaac Lab side. Must NOT be Policy 3's PLACE_ARM_POSE or the plain shared
# default here; see module docstring.
default_qpos = np.array(DEFAULT_QPOS_MJC)
for name, val in RELEASE_ARM_POSE.items():
    default_qpos[MUJOCO_JOINT_ORDER.index(name)] = val

# --- Reset state: RELEASE_ARM_POSE, gripper at its real captured (blocked)
# value, cube at RELEASE_CUBE_POS (already at goal). STATE only -- the drive
# TARGET below is set to full GRIPPER_CLOSE, not this value, so there's a
# real continuous holding force from frame 0 (same guard as Isaac Lab's own
# reset event and Policy 3's own MuJoCo runner: target==state gives zero
# corrective force and the cube falls immediately).
d.qpos[:33] = default_qpos
d.ctrl[:33] = default_qpos
for name, val in RELEASE_ARM_POSE.items():
    idx = MUJOCO_JOINT_ORDER.index(name)
    d.qpos[idx] = val
    d.ctrl[idx] = val
d.qpos[GRIPPER_MJC_IDS] = RELEASE_GRIPPER_POSE
d.ctrl[GRIPPER_MJC_IDS] = GRIPPER_CLOSE
d.qpos[33:36] = RELEASE_CUBE_POS
mujoco.mj_forward(m, d)
print(f"reset cube position (fixed, from policy3_settled_states.pt): {RELEASE_CUBE_POS}")


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
    """39-dim observation, same term order as env_cfg_rgp_release.py's
    ObservationsCfg (identical layout to Policy 3's): arm_joint_pos_rel(7),
    arm_joint_vel(7), gripper_joint_pos(2), object_position(3), ee_position(3),
    ee_to_object(3), hand_base_to_object(3), object_to_goal(3), last_action(8)."""
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


def arm_dist_to_ready():
    """Same quantity reward_return_to_ready_rgp tracks on the Isaac Lab side
    -- L2 joint-space distance from the current right-arm pose to
    READY_ARM_POSE (Policy 1's own reset pose). Reused here purely for
    logging/comparison, not fed to the policy."""
    ready = np.array([DEFAULT_QPOS_MJC[MUJOCO_JOINT_ORDER.index(n)] for n in ARM_JOINTS])
    # DEFAULT_QPOS_MJC's arm entries are the shared READY_ARM_POSE/LEFT_ARM_STOW
    # default (unmodified) -- default_qpos above is the OVERWRITTEN copy, so use
    # the original array, not default_qpos, to get READY_ARM_POSE itself.
    return float(np.linalg.norm(d.qpos[ARM_MJC_IDS] - ready))


print(f"RGP Policy 4 checkpoint: {args.policy_path}")
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
arm_ready_history = []
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
    arm_ready_history.append(arm_dist_to_ready())
    if t % args.log_every == 0 or t == args.steps - 1:
        print(f"  step {t:4d}  EE-object dist={dist:.4f}  dist_to_goal={dist_to_goal:.4f}  "
              f"gripper={'OPEN' if isaac_wants_open else 'CLOSE'}  arm_dist_to_ready={arm_ready_history[-1]:.4f}  "
              f"arm_action_norm={np.linalg.norm(action[:7]):.3f}")

print(f"\nfinal EE-object dist: {dist_history[-1]:.4f}")
print(f"final dist_to_goal: {dist_to_goal_history[-1]:.4f}")
print(f"final arm_dist_to_ready: {arm_ready_history[-1]:.4f}  min arm_dist_to_ready: {min(arm_ready_history):.4f}")
print(f"wall time: {time.time()-start_wall:.1f}s for {args.steps} steps")

if args.save_trajectory:
    np.savez(args.save_trajectory, obs=obs_log, action=action_log, arm_abs=arm_abs_log, cube_pos0=np.array(RELEASE_CUBE_POS))
    print(f"trajectory saved to {args.save_trajectory}")

if viewer is not None:
    print("holding window open, ctrl+C to exit")
    try:
        while viewer.is_running():
            mujoco.mj_step(m, d)
            viewer.sync()
    except KeyboardInterrupt:
        pass
