# g1_lift_rl/env_cfg_rgp_release.py
"""RGP chain, Policy 4: release the cube at the goal, then return the right
arm to READY_ARM_POSE -- the same pose the whole chain starts from at Policy
1's own reset (see constants.py). Not "release only": an earlier draft of
this policy considered release alone, but that makes the task nearly free
(the gripper has one degree of freedom to move and no real cost to opening
it once already settled at the goal) -- adding the return leg gives Policy 4
an actual second skill to learn, mirroring the old chain's own split of an
original combined move+release policy into move+place (holds) and
release+return (releases, then goes home).

Resets to ONE fixed, real already-settled state captured from Policy 3's own
checkpoint (mdp/events_rgp.py's reset_robot_then_couple_cube_settled_rgp,
library built by capture_policy3_settled_states.py) -- same technique, same
rationale as the Policy 2 -> 3 handoff: a reconstructed state is fragile, a
real captured one is safe by construction, and a single fixed state (not
sampled per-episode) avoids the exploration collapse that a randomized
version caused the first time this pattern was tried for Policy 3.

Dedicated action baseline (RGP_G1_DEX1_RELEASE_CFG below), same lesson as
Policy 3's own fix: JointPositionActionCfg(use_default_offset=True) reads the
articulation's STATIC default_joint_pos as its action baseline, which does
NOT track reset events. Without a baseline matching this reset pose,
action~=0 would mean "snap back toward READY_ARM_POSE" from the very first
step -- exactly the mismatch that stalled Policy 3 until it got its own
dedicated config.

Subclasses RGPPlaceSceneCfg (reuses its goal marker) -- Policy 4 still needs
to know where the goal is for its observations, even though the cube starts
already there.
"""
import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
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
from .env_cfg_rgp_place import RGPPlaceSceneCfg
from .env_cfg_rgp_scene import RGP_G1_DEX1_CFG
from .constants import ARM_JOINTS, GRIPPER_JOINTS, GRIPPER_OPEN, GRIPPER_CLOSE

# Same "closest real state to the library's own mean" pick used by
# reset_robot_then_couple_cube_settled_rgp -- computed independently here (at
# config-build time) so the action baseline below matches the reset target
# exactly, by construction.
_settled = torch.load(os.path.join(os.path.dirname(__file__), "policy3_settled_states.pt"), map_location="cpu")
_settled_arm_pose_all = _settled["arm_pose"]
_settled_fixed_idx = torch.argmin(
    torch.norm(_settled_arm_pose_all - _settled_arm_pose_all.mean(dim=0, keepdim=True), dim=-1)
).item()
RELEASE_ARM_POSE = dict(zip(_settled["arm_joint_names"], _settled_arm_pose_all[_settled_fixed_idx].tolist()))

_release_joint_pos = dict(RGP_G1_DEX1_CFG.init_state.joint_pos)
_release_joint_pos.update(RELEASE_ARM_POSE)
RGP_G1_DEX1_RELEASE_CFG = ArticulationCfg(
    prim_path=RGP_G1_DEX1_CFG.prim_path,
    spawn=RGP_G1_DEX1_CFG.spawn,
    init_state=ArticulationCfg.InitialStateCfg(
        pos=RGP_G1_DEX1_CFG.init_state.pos,
        rot=RGP_G1_DEX1_CFG.init_state.rot,
        joint_pos=_release_joint_pos,
        joint_vel={".*": 0.0},
    ),
    actuators=RGP_G1_DEX1_CFG.actuators,
)


@configclass
class RGPReleaseSceneCfg(RGPPlaceSceneCfg):
    robot: ArticulationCfg = RGP_G1_DEX1_RELEASE_CFG


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
        object_to_goal = ObsTerm(func=obs_rgp.object_to_goal_rgp)
        last_action = ObsTerm(func=obs_rgp.last_action_rgp)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    release = RewTerm(func=rew_rgp.reward_release_rgp, weight=6.0)
    release_gradient = RewTerm(func=rew_rgp.reward_release_gradient_rgp, weight=2.0)
    # Was 2.0. With the RETURN_K_RGP=0.6 gradient fix (rewards_rgp.py) and
    # 2300 total training iterations (1500 + an 800-iter resume), the arm
    # reliably closes from ~2.1 rad (reset) to ~0.46-0.5 rad post-release and
    # holds there -- a real, direct-checkpoint-verified improvement over the
    # earlier ~1.6 rad plateau, but still visibly short of READY_ARM_POSE in
    # the GUI. Doubled to 4.0 to push harder on closing that last margin;
    # verify via direct diagnostic (diag_release_return.py) after training,
    # not the curve alone -- same discipline as every other fix this session.
    return_to_ready = RewTerm(func=rew_rgp.reward_return_to_ready_rgp, weight=4.0)
    # New: proactive clearance penalty. The weight=4.0 checkpoint above showed
    # the return path swinging the gripper straight through the just-placed
    # cube (GUI-observed, 2026-08-07) -- contact_disturbance below is
    # velocity-triggered/reactive and wasn't enough on its own. See
    # RETURN_CLEARANCE_RADIUS_RGP's own comment in rewards_rgp.py.
    ee_clearance_on_return = RewTerm(func=rew_rgp.penalty_ee_near_object_after_release_rgp, weight=-2.0)
    contact_disturbance = RewTerm(func=rew_rgp.penalty_contact_disturbance_rgp, weight=-0.1)
    action_rate = RewTerm(func=rew_rgp.penalty_action_rate_rgp, weight=-0.01)
    joint_vel = RewTerm(func=rew_rgp.penalty_joint_vel_rgp, weight=-1.0e-4)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    dropped = DoneTerm(func=term_rgp.object_dropped_rgp, time_out=False)
    launched = DoneTerm(func=term_rgp.object_launched_rgp, time_out=False)
    # No success termination -- same rationale as every other policy in this
    # chain: dense rewards accumulate more signal over a full episode than a
    # policy that learns to end things early.


@configclass
class EventCfg:
    reset_robot_and_cube = EventTermCfg(
        func=evt_rgp.reset_robot_then_couple_cube_settled_rgp, mode="reset",
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "object_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class G1RGPReleaseEnvCfg(ManagerBasedRLEnvCfg):
    scene = RGPReleaseSceneCfg(num_envs=1024, env_spacing=2.5, replicate_physics=True)
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
class G1RGPReleaseEnvCfg_PLAY(G1RGPReleaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
