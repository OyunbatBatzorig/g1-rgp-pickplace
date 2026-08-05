# g1_lift_rl/mdp/events_rgp.py
"""Reset/event functions for the RGP chain."""
from __future__ import annotations

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

from ..constants import ARM_JOINTS, GRIPPER_JOINTS, GRIPPER_OPEN, EE_LINKS, GRASP_OFFSET

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
