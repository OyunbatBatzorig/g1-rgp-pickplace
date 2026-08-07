# g1_lift_rl/mdp/events_rgp.py
"""Reset/event functions for the RGP chain."""
from __future__ import annotations

import os

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

from ..constants import ARM_JOINTS, GRIPPER_JOINTS, GRIPPER_OPEN, GRIPPER_CLOSE, EE_LINKS, GRASP_OFFSET

# Policy 1's own measured convergence pose (deterministic replay, mean +
# per-joint std across 32 envs) -- Policy 2's reset target.
#
# Guard: this pose is ONLY ever used as a RESET target (the joint STATE
# written at episode start). It must never become the action config's
# default-offset baseline -- JointPositionActionCfg(use_default_offset=True)
# reads the articulation's static init_state, which a reset event does not
# change. Policy 2's ActionsCfg keeps pointing at the same RGP_G1_DEX1_CFG as
# Policy 1, unmodified; only the reset event differs.
RGP_POLICY1_ARM_POSE = {
    "right_shoulder_pitch_joint": 0.0194,
    "right_shoulder_roll_joint": 0.0350,
    "right_shoulder_yaw_joint": 0.3514,
    "right_elbow_joint": 0.1271,
    "right_wrist_roll_joint": -0.0574,
    "right_wrist_pitch_joint": -0.1853,
    "right_wrist_yaw_joint": -0.2790,
}
RGP_POLICY1_ARM_POSE_STD = {
    "right_shoulder_pitch_joint": 0.0798,
    "right_shoulder_roll_joint": 0.0179,
    "right_shoulder_yaw_joint": 0.0367,
    "right_elbow_joint": 0.1034,
    "right_wrist_roll_joint": 0.0069,
    "right_wrist_pitch_joint": 0.0261,
    "right_wrist_yaw_joint": 0.0457,
}

# Policy 3's reset does NOT reconstruct a held state from a measured pose +
# offset formula -- see reset_robot_then_couple_cube_grasping_rgp's own
# docstring below for why (a reconstruction proved fragile: teleporting into
# an already-closed, already-contacting configuration is a much more violent
# operation for the physics engine than the real, gradual closing motion
# Policy 2 performs). Instead it uses ONE fixed state from a library of REAL
# captured states (produced once by capture_policy2_held_states.py, loaded
# lazily by _load_policy2_held_states below), not a fresh random sample per
# episode.
#
# Why fixed, not sampled (found the hard way): a first version sampled a
# different one of 506 real states every episode. Training completely failed
# -- move_to_goal stayed at exactly 0.0000 for all 1500 iterations, action_std
# collapsed 0.60->0.13 (exploration collapse, not task discovery). The old
# chain's own Policy 3 (env_cfg_policy3.py) faces a similarly large
# action-baseline mismatch (its reset pose, INSPECT_ARM_POSE, is also ~2 rad
# from the shared READY_ARM_POSE action baseline every JointPositionActionCfg
# in this codebase uses) and trains successfully -- but from a SINGLE fixed
# deterministic pose every episode, never randomized. Sampling across 506
# genuinely different real poses turns "learn one large corrective action"
# into "learn a corrective action that generalizes across many different
# starting configurations," which is a much harder problem for the same
# training budget, on top of RGP's own baseline gap being somewhat larger
# (2.4 vs 2.0 rad) and concentrated on two joints (wrist_roll, wrist_yaw)
# instead of one. Matching the old chain's own proven approach: fixed pose.
_POLICY2_HELD_STATES_PATH = os.path.join(os.path.dirname(__file__), "..", "policy2_held_states.pt")
_policy2_held_states_cache = None


def _load_policy2_held_states(device):
    global _policy2_held_states_cache
    if _policy2_held_states_cache is None:
        data = torch.load(_POLICY2_HELD_STATES_PATH, map_location=device)
        arm_pose = data["arm_pose"].to(device)
        # the single fixed state used every episode: whichever real captured
        # state is closest to the library's own mean arm pose (a real,
        # physically-valid state, not an arbitrary index or a synthetic average).
        mean_pose = arm_pose.mean(dim=0, keepdim=True)
        fixed_idx = torch.argmin(torch.norm(arm_pose - mean_pose, dim=-1)).item()
        _policy2_held_states_cache = {
            "arm_joint_names": data["arm_joint_names"],
            "arm_pose": arm_pose,
            "gripper_pose": data["gripper_pose"].to(device),
            "cube_pos_local": data["cube_pos_local"].to(device),
            "fixed_idx": fixed_idx,
        }
    return _policy2_held_states_cache


def reset_robot_to_default_rgp(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset joints to the default pose AND write that pose into the drive target.

    Writing the drive target for ALL joints (not just actuated ones) is what
    holds passive joints -- left arm, waist, legs -- at rest; otherwise the PD
    controller drives them back toward the USD's raw default every step, since
    only ARM_JOINTS/GRIPPER_JOINTS are ever commanded by the policy's actions.
    """
    robot: Articulation = env.scene[asset_cfg.name]

    default_pos = robot.data.default_joint_pos[env_ids]
    default_vel = robot.data.default_joint_vel[env_ids]

    robot.write_joint_state_to_sim(default_pos, default_vel, env_ids=env_ids)
    robot.set_joint_position_target(default_pos, env_ids=env_ids)
    robot.write_data_to_sim()


def reset_robot_then_couple_cube_rgp(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Policy 2 reset: arm to RGP_POLICY1_ARM_POSE +- its measured per-joint
    noise, gripper open, THEN cube coupled to the arm's ACTUAL resulting
    fingertip position (cube = ee_fk - GRASP_OFFSET) rather than sampled
    independently -- independently jittering both risks a bad relative
    geometry that depenetrates violently at reset.

    Starts from all joints at default (same as reset_robot_to_default_rgp) so
    legs/waist/left-arm are held the same way, then overwrites the right arm
    + gripper on top.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    n = len(env_ids)
    device = robot.device

    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = robot.data.default_joint_vel[env_ids].clone()

    arm_ids, arm_names = robot.find_joints(ARM_JOINTS, preserve_order=True)
    mean = torch.tensor([RGP_POLICY1_ARM_POSE[n_] for n_ in arm_names], device=device)
    std = torch.tensor([RGP_POLICY1_ARM_POSE_STD[n_] for n_ in arm_names], device=device)
    noise = torch.randn(n, len(arm_ids), device=device) * std
    joint_pos[:, arm_ids] = mean.unsqueeze(0) + noise

    gripper_ids, _ = robot.find_joints(GRIPPER_JOINTS)
    joint_pos[:, gripper_ids] = GRIPPER_OPEN

    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)
    robot.write_data_to_sim()

    # Refresh kinematics without stepping physics, so body_pos_w reflects the
    # arm pose just written.
    env.sim.forward()
    env.scene.update(dt=0.0)

    ee_ids, _ = robot.find_bodies(EE_LINKS)
    ee_pos = robot.data.body_pos_w[env_ids][:, ee_ids, :].mean(dim=1)
    grasp_offset = torch.tensor(GRASP_OFFSET, device=device)
    cube_pos = ee_pos - grasp_offset

    cube_state = obj.data.default_root_state[env_ids].clone()
    cube_state[:, :3] = cube_pos
    cube_state[:, 7:] = 0.0  # zero velocity
    obj.write_root_state_to_sim(cube_state, env_ids=env_ids)


def reset_robot_then_couple_cube_grasping_rgp(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Policy 3 reset: samples a REAL, physically-verified held state from
    Policy 2's own deterministic replay (captured once via
    capture_policy2_held_states.py -> policy2_held_states.pt) -- genuinely
    holding the cube, lifted, from the very first frame.

    NOT a reconstruction (measured arm pose + measured cube offset, rotated
    by the current orientation). An earlier version of this reset did that
    and it was fragile: teleporting directly into an already-closed,
    already-contacting configuration in one frame is a much more violent
    operation for the physics engine than the gradual closing motion Policy 2
    itself performs over many real simulated steps -- which is why THAT
    never launches anything, every single training episode. Sampling a real
    recorded (arm pose, gripper pose, cube position) triple sidesteps
    reconstruction entirely: each saved state actually occurred in
    simulation, so it's geometrically consistent and safe by construction,
    and the library's own natural diversity (506 real distinct episodes)
    replaces synthetic per-joint noise.

    Guard specific to a grasping handoff: the gripper's STATE is written at
    the recorded (real, mechanically-blocked-by-the-cube) value, but the
    drive TARGET is written at GRIPPER_CLOSE (full nominal closure), not the
    same value. If target == state here, the controller sees "already at
    target" and stops pushing -- with nothing counteracting gravity, the
    cube falls within ~20 steps. The ongoing push against the blocked target
    is what generates real holding force.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    n = len(env_ids)
    device = robot.device

    states = _load_policy2_held_states(device)
    idx = torch.full((n,), states["fixed_idx"], device=device, dtype=torch.long)

    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = robot.data.default_joint_vel[env_ids].clone()

    arm_ids, arm_names = robot.find_joints(ARM_JOINTS, preserve_order=True)
    col_for_name = {name: i for i, name in enumerate(states["arm_joint_names"])}
    cols = [col_for_name[n_] for n_ in arm_names]
    joint_pos[:, arm_ids] = states["arm_pose"][idx][:, cols]

    gripper_ids, _ = robot.find_joints(GRIPPER_JOINTS)
    joint_pos[:, gripper_ids] = states["gripper_pose"][idx]

    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    target_pos = joint_pos.clone()
    target_pos[:, gripper_ids] = GRIPPER_CLOSE
    robot.set_joint_position_target(target_pos, env_ids=env_ids)
    robot.write_data_to_sim()

    env.sim.forward()
    env.scene.update(dt=0.0)

    env_origins = env.scene.env_origins[env_ids]
    cube_pos = states["cube_pos_local"][idx] + env_origins

    cube_state = obj.data.default_root_state[env_ids].clone()
    cube_state[:, :3] = cube_pos
    cube_state[:, 7:] = 0.0
    obj.write_root_state_to_sim(cube_state, env_ids=env_ids)


# Policy 4's reset uses the exact same "sample a real captured state, fixed
# not randomized" technique as Policy 3's own reset above -- see
# reset_robot_then_couple_cube_grasping_rgp's docstring for why (a
# reconstruction is fragile; sampling from real deterministic-replay states
# is safe by construction; a single fixed state avoids the exploration
# collapse a randomized version caused the first time this pattern was
# tried). Library built by capture_policy3_settled_states.py once Policy 3
# has a trained checkpoint -- states filtered to settled+grasping, matching
# reward_place_rgp's own success gate.
_POLICY3_SETTLED_STATES_PATH = os.path.join(os.path.dirname(__file__), "..", "policy3_settled_states.pt")
_policy3_settled_states_cache = None


def _load_policy3_settled_states(device):
    global _policy3_settled_states_cache
    if _policy3_settled_states_cache is None:
        data = torch.load(_POLICY3_SETTLED_STATES_PATH, map_location=device)
        arm_pose = data["arm_pose"].to(device)
        mean_pose = arm_pose.mean(dim=0, keepdim=True)
        fixed_idx = torch.argmin(torch.norm(arm_pose - mean_pose, dim=-1)).item()
        _policy3_settled_states_cache = {
            "arm_joint_names": data["arm_joint_names"],
            "arm_pose": arm_pose,
            "gripper_pose": data["gripper_pose"].to(device),
            "cube_pos_local": data["cube_pos_local"].to(device),
            "fixed_idx": fixed_idx,
        }
    return _policy3_settled_states_cache


def reset_robot_then_couple_cube_settled_rgp(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Policy 4 reset: arm/gripper/cube to ONE fixed real state sampled from
    Policy 3's own converged rollout (settled at the goal, still gripping) --
    same technique, same guards (target-vs-state split for the gripper drive
    so holding force is real, not zero-corrective-force) as
    reset_robot_then_couple_cube_grasping_rgp. Only the source library and
    file differ."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    n = len(env_ids)
    device = robot.device

    states = _load_policy3_settled_states(device)
    idx = torch.full((n,), states["fixed_idx"], device=device, dtype=torch.long)

    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = robot.data.default_joint_vel[env_ids].clone()

    arm_ids, arm_names = robot.find_joints(ARM_JOINTS, preserve_order=True)
    col_for_name = {name: i for i, name in enumerate(states["arm_joint_names"])}
    cols = [col_for_name[n_] for n_ in arm_names]
    joint_pos[:, arm_ids] = states["arm_pose"][idx][:, cols]

    gripper_ids, _ = robot.find_joints(GRIPPER_JOINTS)
    joint_pos[:, gripper_ids] = states["gripper_pose"][idx]

    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    target_pos = joint_pos.clone()
    target_pos[:, gripper_ids] = GRIPPER_CLOSE
    robot.set_joint_position_target(target_pos, env_ids=env_ids)
    robot.write_data_to_sim()

    env.sim.forward()
    env.scene.update(dt=0.0)

    env_origins = env.scene.env_origins[env_ids]
    cube_pos = states["cube_pos_local"][idx] + env_origins

    cube_state = obj.data.default_root_state[env_ids].clone()
    cube_state[:, :3] = cube_pos
    cube_state[:, 7:] = 0.0
    obj.write_root_state_to_sim(cube_state, env_ids=env_ids)
