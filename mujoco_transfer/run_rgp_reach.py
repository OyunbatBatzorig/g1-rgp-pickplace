"""Runs RGP Policy 1's (reach) trained checkpoint in MuJoCo, reconstructing
its exact 8-term, 36-dim observation and applying its 8-dim action (7 arm
joint targets + 1 binary gripper command) each control step.

Adapted from run_policy1.py (old chain) -- same underlying task shape
(reach + hold, gripper open throughout), same robot, same real-hand-geometry
reconstruction and gripper physically-anchored calibration technique, same
ACTION_CLIP/cube-velocity-cap transfer-safety measures (see run_policy1.py's
own docstring for the full history of why each of these exists -- not
re-derived here, same robot/mechanism). Two real differences from Policy 1:
  1. Scene: g1_rgp_reach_scene.xml (3cm cube, not the old chain's 6cm).
  2. Observation: 36-dim, NO object_to_inspect term -- RGP Policy 1 has no
     "inspect" concept at all (see mdp/observations_rgp.py).
  3. Reset pose: RGP Policy 1's actual reset pose IS its action-target
     baseline (both READY_ARM_POSE, via reset_robot_to_default_rgp with no
     override) -- unlike old Policy 2, no --arm_pose override mechanism is
     needed here; DEFAULT_QPOS_MJC already matches Isaac Lab's actual
     post-reset state directly.
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
                     default="/home/virtual-acc/projects/IsaacLab/logs/rsl_rl/g1_rgp_reach/2026-07-30_18-58-09/exported/policy.pt")
parser.add_argument("--steps", type=int, default=795)
parser.add_argument("--view", action="store_true", help="open the interactive viewer")
parser.add_argument("--realtime", action="store_true", help="pace --view to real time (0.01s/step) instead of running flat out")
parser.add_argument("--wait_for_signal", type=str, default=None,
                     help="with --view, hold the static starting frame indefinitely (syncing the viewer) until this file path exists, then start stepping")
parser.add_argument("--log_every", type=int, default=50)
parser.add_argument("--cube_pos", type=float, nargs=3, default=None,
                     help="override the cube's spawn xyz for this MuJoCo run only (Isaac Lab/RGP_BLOCK_INIT_POS untouched)")
parser.add_argument("--action_clip", type=float, default=3.0,
                     help="clip raw policy actions to +-this before applying/feeding back (prevents out-of-distribution runaway; 0 disables)")
parser.add_argument("--save_trajectory", type=str, default=None,
                     help="if set, log obs/action/absolute-arm-qpos every step and save to this .npz path (for compare_trajectories.py)")
args = parser.parse_args()

GRIPPER_CLOSE = 0.0245

# Training's REAL hand's own measured fingertip separation at each extreme --
# same physical hand as the old chain (SAME robot USD/asset), so these are
# the same measured constants, not re-derived.
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

default_qpos = np.array(DEFAULT_QPOS_MJC)
d.qpos[:33] = default_qpos
d.ctrl[:33] = default_qpos

if args.cube_pos is not None:
    d.qpos[33:36] = args.cube_pos
    print(f"cube position overridden (MuJoCo-only, RGP_BLOCK_INIT_POS untouched): {args.cube_pos}")

cube_pos0 = d.qpos[33:36].copy()

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
    """Returns the 36-dim observation vector in RGP Policy 1's exact term
    order (mdp/observations_rgp.py): arm_joint_pos_rel(7), arm_joint_vel(7),
    gripper_joint_pos(2), object_position(3), ee_position(3),
    ee_to_object(3), hand_base_to_object(3), last_action(8). NO
    object_to_inspect -- that's the one structural difference from Policy
    1's own 39-dim layout."""
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

    obs = np.concatenate([
        arm_joint_pos_rel, arm_joint_vel, gripper_joint_pos,
        object_pos, ee_pos, ee_to_object, hand_base_to_object,
        last_action,
    ]).astype(np.float32)
    assert obs.shape == (36,), obs.shape
    return obs, ee_pos, object_pos


print(f"RGP Policy 1 checkpoint: {args.policy_path}")
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

obs_log = np.zeros((args.steps, 36), dtype=np.float32) if args.save_trajectory else None
action_log = np.zeros((args.steps, 8), dtype=np.float32) if args.save_trajectory else None
arm_abs_log = np.zeros((args.steps, 7), dtype=np.float32) if args.save_trajectory else None

dist_history = []
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
    dist_history.append(dist)
    if t % args.log_every == 0 or t == args.steps - 1:
        print(f"  step {t:4d}  EE-object dist={dist:.4f}  gripper={'OPEN' if isaac_wants_open else 'CLOSE'}  "
              f"arm_action_norm={np.linalg.norm(action[:7]):.3f}")

print(f"\nfinal EE-object dist: {dist_history[-1]:.4f}")
print(f"min EE-object dist over episode: {min(dist_history):.4f}")
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
