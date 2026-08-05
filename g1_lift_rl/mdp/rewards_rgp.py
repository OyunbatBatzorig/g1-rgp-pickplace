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

from ..constants import ARM_JOINTS, EE_LINKS, GRIPPER_JOINTS, GRIPPER_OPEN, GRIPPER_CLOSE, HAND_BASE_LINK, TABLE_TOP_Z, GRASP_OFFSET
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
LIFT_CAP_RGP = 0.10                 # m -- lift reward saturates at 10cm real height
_CUBE_REST_Z_RGP = TABLE_TOP_Z + RGP_BLOCK_SIZE / 2.0  # cube's resting centre height,
                                     # the lift-height baseline (never TABLE_TOP_Z directly,
                                     # or the reward pays out before any real lift happens)


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
