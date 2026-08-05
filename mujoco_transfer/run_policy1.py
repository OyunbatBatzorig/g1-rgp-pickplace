"""Runs Policy 1's actual trained checkpoint (2026-07-16_09-18-57) in MuJoCo,
reconstructing its exact 9-term, 39-dim observation and applying its 8-dim
action (7 arm joint targets + 1 binary gripper command) each control step.

RE-ENABLED the reconstructed real 7-body hand (build_scene.py's
replace_right_hand_with_real_geometry) -- testing whether real collision
geometry closes the ~20cm trajectory gap left after tightening contact
solref (see build_scene.py's solref comment). Body/joint names now match
Isaac Lab's own EE_LINKS/HAND_BASE_LINK/GRIPPER_JOINTS constants directly, no
translation needed. The substitute-hand version (2 flat prismatic sliders, no
mounting-plate body) is what the comments below still describe in places for
history -- swap EE_BODY_NAMES/HAND_BASE_BODY_NAME/GRIPPER_MJC_NAMES back and
re-comment the build_scene.py call to revert.

ee_position/ee_to_object use the midpoint of the two real fingertip bodies
(right_hand_Link1_3/Link2_3, each with a single attached collision geom, so
the "largest local-frame offset" tip-selection logic below still applies but
is trivial here); hand_base_to_object uses the real mounting plate
(right_hand_base_link)'s own origin -- the actual body training's own
penalty_base_clearance targets, unlike the substitute's stand-in
(right_wrist_yaw_link) which had no equivalent body at all.

Gripper sign convention was inverted on the SUBSTITUTE mechanism (confirmed
empirically: commanding training's GRIPPER_OPEN(-0.02) yields ~1.8cm
separation/closed, GRIPPER_CLOSE(+0.0245) yields ~10.7cm/open) -- the real
hand's calibration is measured fresh at startup below regardless, so this is
historical context, not an assumption this script relies on. Fixed with a
PHYSICALLY-ANCHORED conversion (not a coincidence-of-matching-raw-ranges
reflection): at startup this model's own fingertip separation is measured at
both raw command values, and to_mjc_gripper/to_isaac_gripper interpolate
through each simulator's own measured open/close separation rather than
assuming the two raw joint ranges agree. This makes the conversion correct
by construction even if a future hand model's raw range doesn't numerically
coincide with Isaac Lab's (-0.02 to 0.0245) the way this substitute's does.

Control period matches training exactly: sim.dt=0.005s, decimation=2 ->
0.01s per policy step, 795 steps ~ 8s episode (env_cfg.py's episode_length_s).

TRANSFER CHALLENGE (found + fixed): the policy's raw output has no saturating
activation (typical for a Gaussian PPO head) -- unbounded during training
because observations always stayed in-distribution there. Once anything pushes
MuJoCo's observations slightly out-of-distribution (residual approximation
error in the substitute hand geometry, or simply a genuinely-different-but-
correct gripper state training never saw), the raw action can grow without
bound: confirmed directly, elbow_joint's *target* reached -10.2 rad by step
380 (its real range is [-1.0472, 2.0944]) with no clip anywhere in the loop,
producing a 213m runaway. ACTION_CLIP below (a standard, defensible real-
deployment safety practice even though training itself never used one) fixes
this completely.

A second, independent transfer challenge: hard finger-cube contact can launch
the cube at an unphysical velocity (confirmed separately with the real hand:
15.6 m/s and climbing, ballistic free-fall, after a single contact event).
Training's own env_cfg.py guards against exactly this with
RigidBodyPropertiesCfg(max_depenetration_velocity=1.0) on both robot and cube.
MuJoCo has no identical named parameter; clamping the cube's velocity directly
after each physics step is the same pattern training's own DISTURBANCE_VEL_CAP
already uses at the observation/reward level, just applied here at the physics
level instead. Kept as a standing safeguard regardless of which hand is active.
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
                     default="/home/virtual-acc/projects/g1_lift_ext/logs/rsl_rl/g1_lift_policy1/2026-07-16_09-18-57/exported/policy.pt")
parser.add_argument("--steps", type=int, default=795)
parser.add_argument("--view", action="store_true", help="open the interactive viewer")
parser.add_argument("--realtime", action="store_true", help="pace --view to real time (0.01s/step) instead of running flat out")
parser.add_argument("--wait_for_signal", type=str, default=None,
                     help="with --view, hold the static starting frame indefinitely (syncing the viewer) until this file path exists, then start stepping")
parser.add_argument("--log_every", type=int, default=50)
parser.add_argument("--cube_pos", type=float, nargs=3, default=None,
                     help="override the cube's spawn xyz for this MuJoCo run only (Isaac Lab/BLOCK_INIT_POS untouched)")
parser.add_argument("--action_clip", type=float, default=3.0,
                     help="clip raw policy actions to +-this before applying/feeding back (prevents out-of-distribution runaway; 0 disables)")
parser.add_argument("--save_trajectory", type=str, default=None,
                     help="if set, log obs/action/absolute-arm-qpos every step and save to this .npz path (for compare_trajectories.py)")
args = parser.parse_args()

GRIPPER_CLOSE = 0.0245
GRASP_OFFSET = (-0.010, 0.027, 0.012)
INSPECT_POS = np.array([-0.037, -0.222, 0.997])

# Training's REAL hand's own measured fingertip separation at each extreme
# (query_hand_kinematics.py/query_hand_kinematics2.py, right_hand_Link1_3/
# right_hand_Link2_3 world positions, read-only from Isaac Lab). These are the
# physical ground truth the gripper conversion is anchored to, instead of
# assuming Isaac Lab's and MuJoCo's raw joint ranges coincide (see module
# docstring / the conversation this was built from: the previous reflection-
# based version happened to reduce to the same formula only because both
# ranges are numerically -0.02 to 0.0245 -- a coincidence of this particular
# substitute, not something guaranteed to hold for a different hand model).
TRAIN_SEP_AT_OPEN = 0.09007   # meters, at GRIPPER_OPEN
TRAIN_SEP_AT_CLOSE = 0.00117  # meters, at GRIPPER_CLOSE

ARM_MJC_IDS = [MUJOCO_JOINT_ORDER.index(n) for n in ARM_JOINTS]

# RE-ENABLED real hand geometry (build_scene.py's
# replace_right_hand_with_real_geometry) -- names match Isaac Lab's own
# EE_LINKS/HAND_BASE_LINK/GRIPPER_JOINTS constants directly.
EE_BODY_NAMES = ["right_hand_Link1_3", "right_hand_Link2_3"]
HAND_BASE_BODY_NAME = "right_hand_base_link"
GRIPPER_MJC_NAMES = ["right_hand_Joint1_1", "right_hand_Joint2_1"]
GRIPPER_MJC_IDS = [MUJOCO_JOINT_ORDER.index(n) for n in GRIPPER_MJC_NAMES]

m = mujoco.MjModel.from_xml_path("g1_lift_scene.xml")
d = mujoco.MjData(m)

ee_body_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in EE_BODY_NAMES]
hand_base_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, HAND_BASE_BODY_NAME)
object_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
assert -1 not in ee_body_ids and hand_base_body_id != -1 and object_body_id != -1, \
    "one or more body names not found in the MuJoCo model -- check names above"

# The finger bodies' own origin sits at the slider's MOUNTING point, not the
# fingertip -- the actual collision geometry extends ~8-11cm further out
# along each finger's local frame. Pick, per finger body, whichever attached
# geom has the largest local-frame offset magnitude (the far/thin tip piece,
# not the near/thick rail piece) and use ITS position (transformed into world
# frame via the body's rotation each step), not the body origin.
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
    # Cube's freejoint qpos is [x,y,z, qw,qx,qy,qz] at indices 33:40 -- only
    # override position (33:36), leave CUBE_ROT's orientation (36:40) as-is.
    d.qpos[33:36] = args.cube_pos
    print(f"cube position overridden (MuJoCo-only, BLOCK_INIT_POS untouched): {args.cube_pos}")

cube_pos0 = d.qpos[33:36].copy()

mujoco.mj_forward(m, d)


def _measure_separation(raw_value):
    """Command both gripper joints to raw_value, forward-kinematics only (no
    contact/dynamics needed -- this is an unobstructed open/close sweep), and
    return the resulting fingertip separation in meters. Restores qpos after."""
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


# Physically-anchored gripper calibration: measure THIS model's own fingertip
# separation at each of the two raw values the policy can command (rather
# than assuming its raw joint range/sign matches Isaac Lab's -- see module
# docstring). Whichever raw value yields the larger separation is this
# mechanism's own "open"; identify it empirically instead of assuming it's
# whichever constant is literally named GRIPPER_OPEN.
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
    """Isaac Lab's binary gripper command (always exactly GRIPPER_OPEN or
    GRIPPER_CLOSE) -> this MuJoCo model's own raw joint value for that same
    semantic state (open/close), from the calibration above -- correct
    whether or not the two simulators' raw ranges/signs happen to agree."""
    is_open = abs(isaac_value - GRIPPER_OPEN) < abs(isaac_value - GRIPPER_CLOSE)
    return MJC_RAW_OPEN if is_open else MJC_RAW_CLOSE


def to_isaac_gripper(mjc_value):
    """This MuJoCo model's actual (possibly in-transit) raw joint value ->
    the Isaac Lab raw joint value that would produce the SAME physical
    fingertip separation, via each simulator's own calibrated open/close
    separation (not by assuming raw joint ranges/signs match)."""
    sep = _lerp(mjc_value, MJC_RAW_OPEN, MJC_RAW_CLOSE, MJC_SEP_OPEN, MJC_SEP_CLOSE)
    return _lerp(sep, TRAIN_SEP_AT_OPEN, TRAIN_SEP_AT_CLOSE, GRIPPER_OPEN, GRIPPER_CLOSE)


policy = torch.jit.load(args.policy_path, map_location="cpu")
policy.eval()

last_action = np.zeros(8, dtype=np.float32)


def build_observation():
    """Returns the 39-dim observation vector in Isaac Lab's exact term order.

    Policy 1's observations/actions only ever touch ARM_JOINTS (7) and
    GRIPPER_JOINTS (2) -- never the other 24 joints (legs/waist/left arm),
    which just sit PD-held at their defaults the whole episode, exactly as in
    training. So this only needs ARM_MJC_IDS/GRIPPER_MJC_IDS (already resolved
    by joint NAME, in ARM_JOINTS'/GRIPPER order -- the same order Isaac Lab's
    own arm_joint_pos_rel/gripper_joint_pos functions produce), not the full
    33-joint ISAAC_TO_MJC/MJC_TO_ISAAC reordering.
    """
    arm_joint_pos_rel = d.qpos[ARM_MJC_IDS] - default_qpos[ARM_MJC_IDS]
    arm_joint_vel = d.qvel[ARM_MJC_IDS]
    gripper_joint_pos = to_isaac_gripper(d.qpos[GRIPPER_MJC_IDS])

    object_pos = d.xpos[object_body_id].copy()
    # World position of each finger's actual tip, not its body origin: offset
    # transformed by the body's own rotation matrix (d.xmat, row-major 3x3).
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


print(f"Policy 1 checkpoint: {args.policy_path}")
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
    # action[7]>=0 means Isaac Lab's BinaryJointAction wants OPEN -- command
    # this model's actual open value (GRIPPER_CLOSE numerically, per the
    # inverted mechanism), not GRIPPER_OPEN literally.
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
            remaining = 0.01 - (time.time() - step_wall_start)  # 0.01s = decimation*sim.dt
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
