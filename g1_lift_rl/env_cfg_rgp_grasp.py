# g1_lift_rl/env_cfg_rgp_grasp.py
"""RGP chain, Policy 2: grasp the cube and lift it 10cm above the table.

Resets the arm to Policy 1's own measured convergence pose
(RGP_POLICY1_ARM_POSE in mdp/events_rgp.py) +- its measured per-joint noise,
then couples the cube to the arm's actual resulting fingertip position
(cube = ee_fk - GRASP_OFFSET) so reset-time geometry is always
self-consistent -- independently jittering arm and cube risks a bad relative
geometry that depenetrates violently at reset.

Same scene and observation functions as Policy 1 (no "inspect" concept in
this chain). Rewards/terminations are fresh, following a grasp+lift shaping
ladder (descend -> close_gradient -> grasp -> lift) with a 10cm lift cap.

Guard: the action config below still points at the same RGP_G1_DEX1_CFG as
Policy 1, unmodified. JointPositionActionCfg(use_default_offset=True) reads
the articulation's static init_state as its action baseline -- a reset event
that moves the robot's live joint state elsewhere does not change that
baseline. Only the reset event differs between policies; the action config
never does.
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
    descend = RewTerm(func=rew_rgp.reward_descend_rgp, weight=1.0)
    close_gradient = RewTerm(func=rew_rgp.reward_close_gradient_rgp, weight=1.0)
    grasp = RewTerm(func=rew_rgp.reward_grasp_rgp, weight=2.0)
    lift = RewTerm(func=rew_rgp.reward_lift_rgp, weight=4.0)
    early_close = RewTerm(func=rew_rgp.penalty_early_close_rgp, weight=-0.5)
    contact_disturbance = RewTerm(func=rew_rgp.penalty_contact_disturbance_rgp, weight=-0.1)
    base_clearance = RewTerm(func=rew_rgp.penalty_base_clearance_rgp, weight=-3.0)
    action_rate = RewTerm(func=rew_rgp.penalty_action_rate_rgp, weight=-0.01)
    joint_vel = RewTerm(func=rew_rgp.penalty_joint_vel_rgp, weight=-1.0e-4)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    dropped = DoneTerm(func=term_rgp.object_dropped_rgp, time_out=False)
    launched = DoneTerm(func=term_rgp.object_launched_rgp, time_out=False)


@configclass
class EventCfg:
    reset_robot_and_cube = EventTermCfg(
        func=evt_rgp.reset_robot_then_couple_cube_rgp, mode="reset",
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "object_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class G1RGPGraspEnvCfg(ManagerBasedRLEnvCfg):
    scene = RGPSceneCfg(num_envs=1024, env_spacing=2.5, replicate_physics=True)
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
class G1RGPGraspEnvCfg_PLAY(G1RGPGraspEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
