# g1_lift_rl/mdp/observations_rgp.py
"""Observation functions for the RGP 3-policy chain. Policy 1 (reach) only for now."""
from __future__ import annotations

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv

from ..constants import ARM_JOINTS, GRIPPER_JOINTS, EE_LINKS, HAND_BASE_LINK

_OBS_POS_CAP = 5.0  # m -- generously larger than any legitimate scene coordinate


def _safe(t: torch.Tensor) -> torch.Tensor:
    t = torch.nan_to_num(t, nan=0.0, posinf=_OBS_POS_CAP, neginf=-_OBS_POS_CAP)
    return torch.clamp(t, min=-_OBS_POS_CAP, max=_OBS_POS_CAP)


def _ee_pos_w_rgp(env: ManagerBasedRLEnv) -> torch.Tensor:
    """World-frame grasp centre = midpoint of the two fingertip links. (N, 3)"""
    robot: Articulation = env.scene["robot"]
    ids, _ = robot.find_bodies(EE_LINKS)
    return robot.data.body_pos_w[:, ids, :].mean(dim=1)


def arm_joint_pos_rel_rgp(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Right-arm joint positions relative to default pose. (N, 7)"""
    robot: Articulation = env.scene["robot"]
    ids, _ = robot.find_joints(ARM_JOINTS)
    return _safe((robot.data.joint_pos - robot.data.default_joint_pos)[:, ids])


def arm_joint_vel_rgp(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Right-arm joint velocities. (N, 7)"""
    robot: Articulation = env.scene["robot"]
    ids, _ = robot.find_joints(ARM_JOINTS)
    return _safe(robot.data.joint_vel[:, ids])


def gripper_joint_pos_rgp(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Raw gripper finger joint positions (continuous open/closed signal). (N, 2)"""
    robot: Articulation = env.scene["robot"]
    ids, _ = robot.find_joints(GRIPPER_JOINTS)
    return robot.data.joint_pos[:, ids]


def object_position_rgp(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Cube position in the env-local frame. (N, 3)"""
    obj: RigidObject = env.scene["object"]
    return _safe(obj.data.root_pos_w - env.scene.env_origins)


def ee_position_rgp(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Grasp-centre (EE) position in the env-local frame. (N, 3)"""
    return _safe(_ee_pos_w_rgp(env) - env.scene.env_origins)


def ee_to_object_rgp(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Vector from EE to cube (origin-independent). (N, 3)"""
    obj: RigidObject = env.scene["object"]
    return _safe(obj.data.root_pos_w - _ee_pos_w_rgp(env))


def hand_base_to_object_rgp(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Vector from the gripper mounting-plate origin to the cube. (N, 3)"""
    obj: RigidObject = env.scene["object"]
    robot: Articulation = env.scene["robot"]
    ids, _ = robot.find_bodies([HAND_BASE_LINK])
    base_pos = robot.data.body_pos_w[:, ids[0], :]
    return _safe(obj.data.root_pos_w - base_pos)


def object_to_goal_rgp(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Vector from the cube to the goal position. (N, 3) Policy 3 only --
    Reach/Grasp have no goal concept."""
    from ..env_cfg_rgp_scene import RGP_GOAL_POS
    obj: RigidObject = env.scene["object"]
    goal = torch.tensor(RGP_GOAL_POS, device=env.device)
    return _safe(goal - obj.data.root_pos_w + env.scene.env_origins)


def last_action_rgp(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Previous step's raw action vector, defensively clipped -- a runaway raw
    action value has been observed to feed back through this term before."""
    return _safe(env.action_manager.action)
