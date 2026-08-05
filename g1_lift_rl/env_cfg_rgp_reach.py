# g1_lift_rl/env_cfg_rgp_reach.py
"""RGP chain, Policy 1: reach from the robot's default ready pose to the cube,
gripper open, hold there for the full episode.

Design note: no reward term pulls the arm toward a fixed target JOINT POSE.
reward_orient_align_rgp constrains orientation on EE geometry (fingertip pair
level) instead, so the policy is free to converge to whatever joint
configuration reaches a good EE pose. Policy 2's reset pose is built by
*measuring* that convergence after training (see mdp/events_rgp.py), not by
constraining Policy 1 to hit one pre-chosen pose.
"""
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.utils import configclass

from . import mdp
from .mdp import observations_rgp as obs_rgp
from .mdp import rewards_rgp as rew_rgp
from .mdp import terminations_rgp as term_rgp
from .mdp import events_rgp as evt_rgp
from .env_cfg_rgp_scene import RGPSceneCfg
from .constants import ARM_JOINTS, GRIPPER_JOINTS, GRIPPER_OPEN, GRIPPER_CLOSE

_RGP_REACH_JITTER = 0.005  # m -- isotropic XY cube-spawn jitter around BLOCK_INIT_POS


@configclass
class ActionsCfg:
    arm = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=ARM_JOINTS, scale=0.5, use_default_offset=True)
    gripper = mdp.BinaryJointPositionActionCfg(
        asset_name="robot", joint_names=GRIPPER_JOINTS,
        open_command_expr={j: GRIPPER_OPEN for j in GRIPPER_JOINTS},
        close_command_expr={j: GRIPPER_CLOSE for j in GRIPPER_JOINTS},
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        arm_joint_pos_rel = ObsTerm(func=obs_rgp.arm_joint_pos_rel_rgp)
        arm_joint_vel = ObsTerm(func=obs_rgp.arm_joint_vel_rgp)
        gripper_joint_pos = ObsTerm(func=obs_rgp.gripper_joint_pos_rgp)
        object_position = ObsTerm(func=obs_rgp.object_position_rgp)
        ee_position = ObsTerm(func=obs_rgp.ee_position_rgp)
        ee_to_object = ObsTerm(func=obs_rgp.ee_to_object_rgp)
        hand_base_to_object = ObsTerm(func=obs_rgp.hand_base_to_object_rgp)
        last_action = ObsTerm(func=obs_rgp.last_action_rgp)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    hover = RewTerm(func=rew_rgp.reward_hover_rgp, weight=1.0)
    approach = RewTerm(func=rew_rgp.reward_approach_rgp, weight=1.5)
    orient_align = RewTerm(func=rew_rgp.reward_orient_align_rgp, weight=1.0)
    early_close = RewTerm(func=rew_rgp.penalty_early_close_rgp, weight=-0.5)
    contact_disturbance = RewTerm(func=rew_rgp.penalty_contact_disturbance_rgp, weight=-0.1)
    # Deliberately only one clearance penalty -- stacking multiple clearance
    # terms has triggered exploration collapse (action_std -> 0) before.
    base_clearance = RewTerm(func=rew_rgp.penalty_base_clearance_rgp, weight=-3.0)
    action_rate = RewTerm(func=rew_rgp.penalty_action_rate_rgp, weight=-0.01)
    joint_vel = RewTerm(func=rew_rgp.penalty_joint_vel_rgp, weight=-1.0e-4)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    dropped = DoneTerm(func=term_rgp.object_dropped_rgp, time_out=False)
    launched = DoneTerm(func=term_rgp.object_launched_rgp, time_out=False)
    # No success termination -- must reach AND hold for the full episode, so
    # Policy 2 gets a stable state to reset from.


@configclass
class EventCfg:
    reset_robot = EventTermCfg(
        func=evt_rgp.reset_robot_to_default_rgp, mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reset_object = EventTermCfg(
        func=mdp.reset_root_state_uniform, mode="reset",
        params={
            "pose_range": {"x": (-_RGP_REACH_JITTER, _RGP_REACH_JITTER),
                            "y": (-_RGP_REACH_JITTER, _RGP_REACH_JITTER)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class G1RGPReachEnvCfg(ManagerBasedRLEnvCfg):
    scene = RGPSceneCfg(num_envs=2048, env_spacing=2.5, replicate_physics=True)
    observations = ObservationsCfg()
    actions = ActionsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    events = EventCfg()
    commands = None
    curriculum = None

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 8.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.003
        self.sim.physx.enable_ccd = True
        self.sim.physx.num_position_iterations = 12
        self.sim.physx.num_velocity_iterations = 4


@configclass
class G1RGPReachEnvCfg_PLAY(G1RGPReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
