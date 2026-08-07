# g1_lift_rl/mdp/rewards_rgp.py
"""Reward functions for the RGP 3-policy chain. Policy 1 (reach) rewards come
first; Policy 2 (grasp + lift) rewards follow in their own section.

Staged-shaping approach: dense hover -> gated approach -> orientation signal
-> grasp -> lift. No term anywhere pulls toward a fixed target JOINT POSE --
see env_cfg_rgp_reach.py's module docstring for why.
"""
from __future__ import annotations

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply

from ..constants import ARM_JOINTS, EE_LINKS, GRIPPER_JOINTS, GRIPPER_OPEN, GRIPPER_CLOSE, HAND_BASE_LINK, TABLE_TOP_Z, GRASP_OFFSET, READY_ARM_POSE
# Cube size comes from the scene file (single source of truth), not
# constants.py's BLOCK_SIZE, which belongs to the old chain's own 6cm cube.
from ..env_cfg_rgp_scene import RGP_BLOCK_SIZE

# ---------------------------------------------------------------------------
# Policy 1 (reach) thresholds
# ---------------------------------------------------------------------------
K_RGP = 5.0                        # tanh steepness for metre-scale position errors
ALIGN_XY_RGP = 0.05                # m -- xy error under which approach/orient pay;
                                    # tightening this hurts discoverability during training
HOVER_OFFSET_RGP = 0.03            # m -- waypoint height above cube centre
HOVER_BACK_OFFSET_RGP = 0.06       # m -- waypoint pulled toward the robot's own body,
                                    # so the natural hover-to-descend path doesn't dive onto the cube's top face
ORIENT_Z_SCALE_RGP = 0.02          # m -- decay scale for the fingertip-level orientation check
EARLY_CLOSE_DIST_RGP = 0.04        # m -- distance beyond which closing the gripper is premature
BASE_DANGER_XY_RADIUS_RGP = 0.03   # m -- xy distance under which the mounting plate counts as "over" the cube
BASE_CLEARANCE_MARGIN_RGP = 0.015  # m -- required clearance above the cube's top surface
DISTURBANCE_VEL_THRESHOLD_RGP = 0.05  # m/s -- cube velocity above this while the gripper is open counts as a knock
DISTURBANCE_VEL_CAP_RGP = 2.0      # m/s -- hard cap before computing the penalty, so a pathological
                                    # contact-event velocity never reaches the value function directly

# ---------------------------------------------------------------------------
# Policy 2 (grasp + lift 7cm) thresholds
# ---------------------------------------------------------------------------
GRASP_DIST_RGP = 0.04               # m -- enveloped: EE within this of cube centre
GRIP_CLOSED_THRESHOLD_RGP = -0.018  # rad -- measured gripper position when physically
                                     # blocked closed by the cube (the joint never reaches its own hard limit)
LIFT_CAP_RGP = 0.15                 # m -- lift reward saturates at 15cm real height
_CUBE_REST_Z_RGP = TABLE_TOP_Z + RGP_BLOCK_SIZE / 2.0  # cube's resting centre height,
                                     # the lift-height baseline (never TABLE_TOP_Z directly,
                                     # or the reward pays out before any real lift happens)

# ---------------------------------------------------------------------------
# Policy 3 (place + release) thresholds
# ---------------------------------------------------------------------------
SETTLE_NEAR_RADIUS_RGP = 0.06   # m -- distance to goal under which settle/place engage.
                                 # Was 0.15m; too close to this reset's own start distance
                                 # (0.237m, from wherever Policy 2's captured hold happens to
                                 # be) -- "settled" was reachable with ~9cm of real travel,
                                 # letting release_gradient pay out almost immediately without
                                 # genuine carrying. 0.06m (~2x the cube's own 3cm size) forces
                                 # real arrival first.
SETTLE_SLOW_VEL_RGP = 0.05      # m/s -- cube speed under which "settled" (not just nearby) counts
RELEASE_HOLD_STEPS_RGP = 15     # control steps (~0.15s at decimation=2/sim.dt=0.005) the
                                 # settled+open condition must sustain to count as a genuine
                                 # release rather than a single-frame flicker
CARRY_MIN_HEIGHT_RGP = LIFT_CAP_RGP  # m -- reuses Policy 2's own validated lift height as
                                 # the "carried, not dragged along the table" threshold in transit
CARRY_PENALTY_GATE_RADIUS_RGP = 0.18  # m -- carry_height/table_clearance turn off past this
                                 # distance, NOT SETTLE_NEAR_RADIUS_RGP (0.06m). Diagnosed via two
                                 # independent full training runs (different entropy_coef/
                                 # init_noise_std) both converging to the cube parked ~0.144-0.151m
                                 # from goal -- suspiciously close to CARRY_MIN_HEIGHT_RGP (0.15m).
                                 # Root cause: with the gate at 0.06m, the final descent from carry
                                 # height into the goal passes through a zone (0.06m-0.15m away)
                                 # where these penalties are still fully active, since reaching 0.06m
                                 # itself requires descending first. That's a penalty valley with no
                                 # local incentive to cross, not an exploration problem -- raising
                                 # entropy_coef/init_noise_std (see agents/rsl_rl_ppo_cfg_rgp.py)
                                 # didn't help. Set above CARRY_MIN_HEIGHT_RGP so the final approach
                                 # is never penalized at all; SETTLE_NEAR_RADIUS_RGP (0.06m) stays
                                 # the threshold for settle/place themselves, unchanged.


def _is_grasping_rgp(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Enveloped (EE within GRASP_DIST_RGP of the cube) AND gripper closed past
    GRIP_CLOSED_THRESHOLD_RGP."""
    return (_ee_obj_dist(env) < GRASP_DIST_RGP) & (_grip_mean(env) > GRIP_CLOSED_THRESHOLD_RGP)


def reward_descend_rgp(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Fine positioning: dense pull to the verified grasp point (cube +
    GRASP_OFFSET), gated on xy-alignment. Smaller role than Policy 1's own
    reward_approach_rgp -- Policy 2 starts already roughly positioned, this
    only needs to close the last few cm precisely."""
    ee, cube = _ee_pos_w(env), _cube_pos_w(env)
    grasp_offset = torch.tensor(GRASP_OFFSET, device=env.device)
    target = cube + grasp_offset
    return _xy_aligned(env) * (1.0 - torch.tanh(K_RGP * torch.norm(ee - target, dim=-1)))


def reward_close_gradient_rgp(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Dense bridge between descend and grasp: rewards closing PROGRESSIVELY
    as the hand approaches, instead of only paying at the exact grasp instant.
    Falloff tied to GRASP_DIST_RGP (linear ramp to 0 by 2x GRASP_DIST_RGP,
    ~8cm) so closing can't become net-rewarding too early. Gated on
    xy-alignment like descend -- can't reward closing from an off-centre angle."""
    proximity = torch.clamp(1.0 - _ee_obj_dist(env) / (2.0 * GRASP_DIST_RGP), 0.0, 1.0)
    return _xy_aligned(env) * proximity * _closedness(env)


def reward_grasp_rgp(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Binary bonus while enveloping AND closing -- the gate. Rewards exactly
    the condition reward_lift_rgp itself requires."""
    return _is_grasping_rgp(env).float()


def reward_lift_rgp(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Height above the cube's own resting height (capped at LIFT_CAP_RGP =
    7cm), gated on grasping -- continuous, so it also keeps the grasp held
    once lifting starts."""
    height = torch.clamp(_cube_pos_w(env)[:, 2] - _CUBE_REST_Z_RGP, 0.0, LIFT_CAP_RGP)
    return (height / LIFT_CAP_RGP) * _is_grasping_rgp(env).float()

# Mounting-plate mesh bounding box (measured), local frame -- the plate's
# origin alone undercounts its real reach, so clearance checks use corners.
_HAND_BASE_BBOX_MIN_RGP = (-0.035, 0.000, -0.035)
_HAND_BASE_BBOX_MAX_RGP = (0.035, 0.0738, 0.089)
_HAND_BASE_CORNERS_LOCAL_RGP = [
    (x, y, z)
    for x in (_HAND_BASE_BBOX_MIN_RGP[0], _HAND_BASE_BBOX_MAX_RGP[0])
    for y in (_HAND_BASE_BBOX_MIN_RGP[1], _HAND_BASE_BBOX_MAX_RGP[1])
    for z in (_HAND_BASE_BBOX_MIN_RGP[2], _HAND_BASE_BBOX_MAX_RGP[2])
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _ee_pos_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    ids, _ = robot.find_bodies(EE_LINKS)
    return robot.data.body_pos_w[:, ids, :].mean(dim=1)


def _cube_pos_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    obj: RigidObject = env.scene["object"]
    return obj.data.root_pos_w


def _ee_obj_dist(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.norm(_ee_pos_w(env) - _cube_pos_w(env), dim=-1)


def _grip_mean(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    gids, _ = robot.find_joints(GRIPPER_JOINTS)
    return robot.data.joint_pos[:, gids].mean(dim=-1)


def _closedness(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Gripper position normalized to 0 (open) .. 1 (closed)."""
    return torch.clamp((_grip_mean(env) - GRIPPER_OPEN) / (GRIPPER_CLOSE - GRIPPER_OPEN), 0.0, 1.0)


def _hand_base_closest_corner_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """World position of whichever corner of the mounting plate's actual mesh
    extent is closest, in xy, to the cube. (N, 3)"""
    robot: Articulation = env.scene["robot"]
    ids, _ = robot.find_bodies([HAND_BASE_LINK])
    pos = robot.data.body_pos_w[:, ids[0], :]
    quat = robot.data.body_quat_w[:, ids[0], :]
    cube_pos = _cube_pos_w(env)

    corners_local = torch.tensor(_HAND_BASE_CORNERS_LOCAL_RGP, device=env.device)
    n = pos.shape[0]
    q = quat.unsqueeze(1).expand(n, 8, 4).reshape(n * 8, 4)
    c = corners_local.unsqueeze(0).expand(n, 8, 3).reshape(n * 8, 3)
    world_offsets = quat_apply(q, c).reshape(n, 8, 3)
    world_corners = pos.unsqueeze(1) + world_offsets

    xy_dist = torch.norm(world_corners[..., :2] - cube_pos[:, None, :2], dim=-1)
    closest_idx = xy_dist.argmin(dim=-1, keepdim=True)
    idx_expanded = closest_idx.unsqueeze(-1).expand(-1, -1, 3)
    return torch.gather(world_corners, 1, idx_expanded).squeeze(1)


def _xy_aligned(env: ManagerBasedRLEnv) -> torch.Tensor:
    ee, cube = _ee_pos_w(env), _cube_pos_w(env)
    xy_err = torch.norm(ee[:, :2] - cube[:, :2], dim=-1)
    return (xy_err < ALIGN_XY_RGP).float()


# ---------------------------------------------------------------------------
# Policy 1 rewards
# ---------------------------------------------------------------------------
def reward_hover_rgp(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Dense pull to a waypoint above and behind the cube. Always on."""
    target = _cube_pos_w(env).clone()
    target[:, 1] += HOVER_BACK_OFFSET_RGP
    target[:, 2] += HOVER_OFFSET_RGP
    return 1.0 - torch.tanh(K_RGP * torch.norm(_ee_pos_w(env) - target, dim=-1))


def reward_approach_rgp(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Dense pull of the EE toward the cube centre, gated on xy-alignment (fingers
    can straddle) so it only fires once the hand is genuinely over the cube, not
    during lateral transit."""
    ee, cube = _ee_pos_w(env), _cube_pos_w(env)
    return _xy_aligned(env) * (1.0 - torch.tanh(K_RGP * torch.norm(ee - cube, dim=-1)))


def reward_orient_align_rgp(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Encourages the two-finger straddle to stay roughly HORIZONTAL, not
    tilted/palm-down -- a loose, EE-geometry-only orientation signal.
    Deliberately NOT a fixed joint-pose target (see module docstring): scores
    the fingertip pair's own relative height only, so any joint configuration
    that achieves a level straddle scores equally well. Gated the same way as
    reward_approach_rgp -- orientation only matters once actually near the cube."""
    robot: Articulation = env.scene["robot"]
    ids, _ = robot.find_bodies(EE_LINKS)
    p1 = robot.data.body_pos_w[:, ids[0], :]
    p2 = robot.data.body_pos_w[:, ids[1], :]
    z_gap = torch.abs(p1[:, 2] - p2[:, 2])
    return _xy_aligned(env) * torch.exp(-z_gap / ORIENT_Z_SCALE_RGP)


def penalty_early_close_rgp(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Closed gripper while far from the cube = penalized -- Policy 1 must arrive
    with the gripper open, not pre-empt Policy 2 by closing early."""
    return _closedness(env) * (_ee_obj_dist(env) > EARLY_CLOSE_DIST_RGP).float()


def penalty_contact_disturbance_rgp(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalizes cube linear velocity above threshold while the gripper is still
    open -- discourages knocking the cube during approach. NaN-guarded and
    magnitude-capped before use: an extreme/NaN velocity from a pathological
    contact event must never reach the value function directly."""
    obj: RigidObject = env.scene["object"]
    lin_vel = torch.norm(obj.data.root_lin_vel_w, dim=-1)
    lin_vel = torch.nan_to_num(lin_vel, nan=0.0, posinf=DISTURBANCE_VEL_CAP_RGP, neginf=0.0)
    lin_vel = torch.clamp(lin_vel, max=DISTURBANCE_VEL_CAP_RGP)
    excess = torch.clamp(lin_vel - DISTURBANCE_VEL_THRESHOLD_RGP, min=0.0)
    still_open = (_closedness(env) < 0.5).float()
    return excess * still_open


def penalty_base_clearance_rgp(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalizes the solid mounting plate sitting over AND too low relative to
    the cube -- uses the plate's CLOSEST CORNER (full mesh extent), not its
    origin, since the origin point alone undercounts the plate's real reach."""
    corner = _hand_base_closest_corner_w(env)
    cube_pos = _cube_pos_w(env)
    xy_dist = torch.norm(corner[:, :2] - cube_pos[:, :2], dim=-1)
    cube_top_z = cube_pos[:, 2] + RGP_BLOCK_SIZE / 2.0
    clearance = corner[:, 2] - (cube_top_z + BASE_CLEARANCE_MARGIN_RGP)
    over_cube = (xy_dist < BASE_DANGER_XY_RADIUS_RGP).float()
    return torch.clamp(-clearance, min=0.0) * over_cube


def penalty_action_rate_rgp(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
    )


def penalty_joint_vel_rgp(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    ids, _ = robot.find_joints(ARM_JOINTS)
    return torch.sum(torch.square(robot.data.joint_vel[:, ids]), dim=-1)


# ---------------------------------------------------------------------------
# Policy 3 helpers
# ---------------------------------------------------------------------------
def _dist_to_goal_rgp(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Cube-to-goal distance in WORLD frame. RGP_GOAL_POS is an env-LOCAL
    constant (same convention as BLOCK_INIT_POS etc.) -- must add
    env.scene.env_origins before comparing against _cube_pos_w (genuinely
    world-frame). Missing this made every distance off by each env's own
    grid offset (1.6-4m observed with env_spacing=2.5), saturating
    1-tanh(K*dist) to ~0 regardless of true proximity -- root cause of
    move_to_goal/settle/place/release/release_gradient all reading exactly
    0.0000 through an entire training run. object_to_goal_rgp (the
    observation term) already does this conversion correctly; this helper
    didn't match it."""
    from ..env_cfg_rgp_scene import RGP_GOAL_POS
    goal = torch.tensor(RGP_GOAL_POS, device=env.device) + env.scene.env_origins
    return torch.norm(_cube_pos_w(env) - goal, dim=-1)


def _settled_rgp(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Cube is close to the goal AND moving slowly -- arrived, not just passing through."""
    obj: RigidObject = env.scene["object"]
    speed = torch.norm(obj.data.root_lin_vel_w, dim=-1)
    return (_dist_to_goal_rgp(env) < SETTLE_NEAR_RADIUS_RGP) & (speed < SETTLE_SLOW_VEL_RGP)


def _release_hold_counter_rgp(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Per-env running count of consecutive settled+open steps, stored on the
    env instance. Self-resets: any step where the condition breaks zeroes it
    (see reward_release_rgp), including right after a scene reset (freshly
    handed-off state is gripping, not settled-open, by construction)."""
    if not hasattr(env, "_release_hold_counter_rgp"):
        env._release_hold_counter_rgp = torch.zeros(env.num_envs, device=env.device)
    return env._release_hold_counter_rgp


# ---------------------------------------------------------------------------
# Policy 3 rewards
# ---------------------------------------------------------------------------
def reward_move_to_goal_rgp(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Dense pull of the cube toward the goal, gated on actually grasping --
    no reward for shoving the cube around ungripped."""
    return _is_grasping_rgp(env).float() * (1.0 - torch.tanh(K_RGP * _dist_to_goal_rgp(env)))


def reward_settle_rgp(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Dense bonus for being near the goal AND slow -- bridges move_to_goal's
    coarse approach into a controlled, non-crashing arrival."""
    obj: RigidObject = env.scene["object"]
    speed = torch.norm(obj.data.root_lin_vel_w, dim=-1)
    proximity = torch.clamp(1.0 - _dist_to_goal_rgp(env) / SETTLE_NEAR_RADIUS_RGP, 0.0, 1.0)
    slowness = torch.clamp(1.0 - speed / (2.0 * SETTLE_SLOW_VEL_RGP), 0.0, 1.0)
    return _is_grasping_rgp(env).float() * proximity * slowness


def reward_place_rgp(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Bonus for genuinely settled at the goal, still gripping."""
    return (_settled_rgp(env) & _is_grasping_rgp(env)).float()


def reward_release_rgp(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Bonus for a SUSTAINED open gripper while settled at the goal -- gated on
    a running per-env step counter (RELEASE_HOLD_STEPS_RGP), not an instant
    check, so a single-frame flicker can't collect this reward."""
    settled_open = _settled_rgp(env) & (~_is_grasping_rgp(env)) & (_closedness(env) < 0.5)
    counter = _release_hold_counter_rgp(env)
    counter[:] = torch.where(settled_open, counter + 1, torch.zeros_like(counter))
    return (counter >= RELEASE_HOLD_STEPS_RGP).float()


def reward_release_gradient_rgp(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Dense bridge for the release problem: once settled at the goal,
    continuously reward PROGRESSIVELY opening the gripper, instead of only
    paying at the exact sustained-release instant reward_release_rgp checks.
    Mirrors reward_close_gradient_rgp's role for grasping, inverted -- gated
    so it can't compete with move_to_goal/settle during transit (only active
    once settled)."""
    return _settled_rgp(env).float() * (1.0 - _closedness(env))


def penalty_carry_height_rgp(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalizes carrying the cube too low (dragged/skimmed along the table)
    during transit -- excluded near the goal, where coming down is correct.
    Gated on grasping. Uses CARRY_PENALTY_GATE_RADIUS_RGP (0.18m), NOT
    SETTLE_NEAR_RADIUS_RGP (0.06m) -- see that constant's comment: gating at
    0.06m created a penalty valley between 0.06m-0.15m that blocked the final
    descent into the goal from ever being locally rewarding."""
    height = _cube_pos_w(env)[:, 2] - _CUBE_REST_Z_RGP
    shortfall = torch.clamp(CARRY_MIN_HEIGHT_RGP - height, min=0.0) / CARRY_MIN_HEIGHT_RGP
    away_from_goal = (_dist_to_goal_rgp(env) > CARRY_PENALTY_GATE_RADIUS_RGP).float()
    return shortfall * _is_grasping_rgp(env).float() * away_from_goal


def penalty_table_clearance_rgp(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Same mounting-plate-over-surface penalty as Policy 1's
    penalty_base_clearance_rgp, excluded near the goal (the plate MUST come
    close to the table there to place the cube). Uses
    CARRY_PENALTY_GATE_RADIUS_RGP -- see penalty_carry_height_rgp."""
    corner = _hand_base_closest_corner_w(env)
    cube_pos = _cube_pos_w(env)
    xy_dist = torch.norm(corner[:, :2] - cube_pos[:, :2], dim=-1)
    cube_top_z = cube_pos[:, 2] + RGP_BLOCK_SIZE / 2.0
    clearance = corner[:, 2] - (cube_top_z + BASE_CLEARANCE_MARGIN_RGP)
    over_cube = (xy_dist < BASE_DANGER_XY_RADIUS_RGP).float()
    away_from_goal = (_dist_to_goal_rgp(env) > CARRY_PENALTY_GATE_RADIUS_RGP).float()
    return torch.clamp(-clearance, min=0.0) * over_cube * away_from_goal


# ---------------------------------------------------------------------------
# Policy 4 (release + return) thresholds and rewards
# ---------------------------------------------------------------------------
RETURN_K_RGP = 0.6  # tanh steepness for the joint-space return term. The first full
                     # training run (model_1499.pt, 2026-08-06) used 3.0 here --
                     # direct checkpoint verification (diag_release_return.py) showed
                     # arm_dist_to_ready sitting at 2.1-4.4 rad throughout the episode
                     # (release pose is far from READY_ARM_POSE, and nothing pulled it
                     # closer), so tanh(3.0*dist) was already saturated to 1.000000 in
                     # float32 at every single measured point -- reward_return_to_ready
                     # had EXACTLY zero gradient for the entire 1500-iteration run
                     # (Episode_Reward/return_to_ready logged 0.0000 start to finish).
                     # 0.6 keeps tanh(K*dist) inside its responsive range across the
                     # observed 0-4.5 rad span (K*4.4 = 2.6, tanh=0.989, still off the
                     # saturation plateau) so there's a real gradient everywhere the
                     # arm actually visits, not just near the very end of the return.

def _arm_dist_to_ready_rgp(env: ManagerBasedRLEnv) -> torch.Tensor:
    """L2 distance in joint space from the current right-arm pose to
    READY_ARM_POSE (Policy 1's own reset pose -- see constants.py). Reused by
    both reward_return_to_ready_rgp and (once Policy 4 has a checkpoint) its
    own deterministic-replay verification."""
    robot: Articulation = env.scene["robot"]
    arm_ids, arm_names = robot.find_joints(ARM_JOINTS, preserve_order=True)
    target = torch.tensor(
        [READY_ARM_POSE[n] for n in arm_names], device=env.device
    )
    return torch.norm(robot.data.joint_pos[:, arm_ids] - target, dim=-1)


def reward_return_to_ready_rgp(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Dense pull of the right arm back toward READY_ARM_POSE -- Policy 1's
    own starting pose, matching the old chain's own move+release -> move+place
    / release+return split (see env_cfg_rgp_release.py's module docstring).

    Gated on NOT grasping (release_gradient/reward_release_rgp are the
    incentive to let go; this term only starts pulling once the gripper has
    actually opened past the grasp-distance/closed-threshold gate that
    _is_grasping_rgp checks). Deliberately a plain boolean gate, not a
    distance-based one -- CARRY_PENALTY_GATE_RADIUS_RGP's own history this
    session is the reason: a spatial cutoff can create a penalty valley
    between "not yet past the gate" and "already at the goal." A boolean
    "already released" gate has no such dead zone -- the moment the grasp
    condition clears, this term activates smoothly from wherever the arm
    already is, with no intermediate region being actively fought."""
    already_released = (~_is_grasping_rgp(env)).float()
    return already_released * (1.0 - torch.tanh(RETURN_K_RGP * _arm_dist_to_ready_rgp(env)))


RETURN_CLEARANCE_RADIUS_RGP = 0.08  # m -- EE-to-cube distance under which the return path
                                     # counts as "too close." Found via direct GUI observation
                                     # (2026-08-07, weight=4.0 checkpoint): the shortest joint-space
                                     # path back to READY_ARM_POSE swings the gripper straight through
                                     # the just-placed cube, knocking it off the goal marker.
                                     # penalty_contact_disturbance_rgp (velocity-triggered, weight
                                     # -0.1) didn't stop this -- it's REACTIVE (only fires once the
                                     # cube is already moving fast), and was easily dominated once
                                     # return_to_ready's own weight was raised to 4.0. This is
                                     # PROACTIVE instead, same style as penalty_base_clearance_rgp /
                                     # penalty_table_clearance_rgp -- discourages the path from
                                     # passing near the object at all, continuous (not a step), so it
                                     # doesn't fight the moment of release itself (EE is necessarily
                                     # right at the cube the instant already_released flips true; the
                                     # penalty is at its own max there and smoothly relaxes as the arm
                                     # moves away, same direction reward_return_to_ready_rgp already
                                     # pulls -- no dead zone between the two).


def penalty_ee_near_object_after_release_rgp(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalizes the end-effector being close to the cube while returning to
    ready -- gated on the same already_released condition
    reward_return_to_ready_rgp uses, so it only applies post-release (being
    close to the cube during approach/place is obviously correct there)."""
    already_released = (~_is_grasping_rgp(env)).float()
    closeness = torch.clamp(RETURN_CLEARANCE_RADIUS_RGP - _ee_obj_dist(env), min=0.0)
    return closeness * already_released
