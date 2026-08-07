# g1_lift_rl/env_cfg_rgp_place.py
"""RGP chain, Policy 3: carry the grasped cube to the goal marker and place it
there, gripper CLOSED the whole episode -- no release. Release is a separate
Policy 4 (not yet built), matching the old chain's own split of an original
combined move+release policy into move+place (holds) and release+return
(releases) -- done there for the same reason: a policy that's already
reliably grasping+carrying has little pressure to risk letting go, since
clinging pays a similar reward to releasing right up until release actually
happens. Rather than keep tuning reward weights against that conflict (tried:
reward_release_gradient_rgp, a dense bridge for progressively opening the
gripper -- removed from this policy's reward set, kept in mdp/rewards_rgp.py
for Policy 4 to use), separating the skills removes the conflict at its root:
Policy 3 has zero reward for opening the gripper at all now, so nothing
competes with move_to_goal/settle/place.

Resets to ONE fixed, real already-holding state captured from Policy 2's own
checkpoint (mdp/events_rgp.py's reset_robot_then_couple_cube_grasping_rgp,
library built by capture_policy2_held_states.py) -- gripper closed, cube
genuinely grasped, from the very first frame. No re-grasp phase: an earlier
version tried reconstructing a held state from a measured pose + offset
formula, which proved fragile (a single-frame teleport into contact is a far
more violent event for the physics engine than the real gradual closing
motion Policy 2 itself performs).

Fixed, not randomly sampled across the library: a sampled-every-episode
version trained to total failure (move_to_goal stuck at exactly 0.0 for all
1500 iterations, action_std collapsed 0.60->0.13).

Root cause, isolated directly (not just inferred from the training curve):
holding the arm at the exact reset pose via a manually-computed "cancel the
baseline pull" action, for 200 steps, kept the grasp perfectly solid in
16/16 envs (EE-to-cube distance identical to 4 decimal places before and
after) -- the reset geometry itself was never the problem. The actual issue
is JointPositionActionCfg(use_default_offset=True) reading RGP_G1_DEX1_CFG's
shared READY_ARM_POSE-based default as its action baseline, ~2.4 rad from
this reset pose -- an untrained policy (action~=0) can't yet output the
large, precise correction needed just to stay put, so it loses the grasp
before it can learn anything about the actual task. Policy 1/2 don't hit
this because they start at or near their own action baseline already.

Fixed via RGP_G1_DEX1_PLACE_CFG below: a dedicated robot config whose default
arm pose IS this reset pose, so action~=0 means "stay gripping" instead of
"snap back to standing-reach pose." Sampling across the library remains
future work once this baseline fix is confirmed to help at all (the old
chain's own Policy 3 trains successfully from a single fixed pose with a
similarly large ~2.0 rad gap from ITS OWN shared baseline, so a fixed pose is
not inherently the limiting factor either way).

Subclasses RGPSceneCfg for its own dedicated action-baseline robot config
(RGP_G1_DEX1_PLACE_CFG) -- the goal marker itself now lives on the shared
base scene (env_cfg_rgp_scene.py), so all four policies show it, not just
this one.
"""
import os

import torch

from isaaclab.actuators import ImplicitActuatorCfg
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
from .env_cfg_rgp_scene import RGPSceneCfg, RGP_G1_DEX1_CFG
from .constants import ARM_JOINTS, GRIPPER_JOINTS, GRIPPER_OPEN, GRIPPER_CLOSE

# Same "closest real state to the library's own mean" pick used by
# reset_robot_then_couple_cube_grasping_rgp (mdp/events_rgp.py) -- computed
# independently here (at config-build time, no simulation needed yet) so the
# action baseline below matches the reset target exactly, by construction.
_held = torch.load(os.path.join(os.path.dirname(__file__), "policy2_held_states.pt"), map_location="cpu")
_held_arm_pose_all = _held["arm_pose"]
_held_fixed_idx = torch.argmin(
    torch.norm(_held_arm_pose_all - _held_arm_pose_all.mean(dim=0, keepdim=True), dim=-1)
).item()
PLACE_ARM_POSE = dict(zip(_held["arm_joint_names"], _held_arm_pose_all[_held_fixed_idx].tolist()))

# Same robot as RGP_G1_DEX1_CFG in every respect except the right arm's
# DEFAULT pose -- this is the action-space baseline JointPositionActionCfg
# reads (use_default_offset=True), not just a reset target. Matching it to
# the reset pose means action~=0 means "stay gripping," not "snap back to
# READY_ARM_POSE" -- see this module's own docstring for why that distinction
# turned out to matter.
_place_joint_pos = dict(RGP_G1_DEX1_CFG.init_state.joint_pos)
_place_joint_pos.update(PLACE_ARM_POSE)
RGP_G1_DEX1_PLACE_CFG = ArticulationCfg(
    prim_path=RGP_G1_DEX1_CFG.prim_path,
    spawn=RGP_G1_DEX1_CFG.spawn,
    init_state=ArticulationCfg.InitialStateCfg(
        pos=RGP_G1_DEX1_CFG.init_state.pos,
        rot=RGP_G1_DEX1_CFG.init_state.rot,
        joint_pos=_place_joint_pos,
        joint_vel={".*": 0.0},
    ),
    actuators=RGP_G1_DEX1_CFG.actuators,
)


@configclass
class RGPPlaceSceneCfg(RGPSceneCfg):
    robot: ArticulationCfg = RGP_G1_DEX1_PLACE_CFG
    # Goal marker now lives on the shared RGPSceneCfg base (env_cfg_rgp_scene.py)
    # so it shows up for Policy 1/2 too, not just here -- inherited as-is.


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
    # No re-grasp bridge needed: the reset (mdp/events_rgp.py's
    # reset_robot_then_couple_cube_grasping_rgp) samples a real, already-
    # holding state from Policy 2's own captured rollout, so the cube is
    # genuinely grasped from the first frame. Straight to the actual task.
    move_to_goal = RewTerm(func=rew_rgp.reward_move_to_goal_rgp, weight=3.0)
    settle = RewTerm(func=rew_rgp.reward_settle_rgp, weight=1.5)
    place = RewTerm(func=rew_rgp.reward_place_rgp, weight=2.0)
    # No release/release_gradient here -- Policy 3 keeps the gripper CLOSED the
    # whole episode by design (see module docstring). Both reward functions
    # stay defined in mdp/rewards_rgp.py for Policy 4 (release-only) to reuse.
    carry_height = RewTerm(func=rew_rgp.penalty_carry_height_rgp, weight=-2.0)
    table_clearance = RewTerm(func=rew_rgp.penalty_table_clearance_rgp, weight=-3.0)
    contact_disturbance = RewTerm(func=rew_rgp.penalty_contact_disturbance_rgp, weight=-0.1)
    action_rate = RewTerm(func=rew_rgp.penalty_action_rate_rgp, weight=-0.01)
    joint_vel = RewTerm(func=rew_rgp.penalty_joint_vel_rgp, weight=-1.0e-4)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    dropped = DoneTerm(func=term_rgp.object_dropped_rgp, time_out=False)
    launched = DoneTerm(func=term_rgp.object_launched_rgp, time_out=False)
    # No success termination -- runs the full episode either way, same
    # rationale as Policy 1/2: dense rewards accumulate more signal over a
    # full episode than a policy that learns to end things early.


@configclass
class EventCfg:
    reset_robot_and_cube = EventTermCfg(
        func=evt_rgp.reset_robot_then_couple_cube_grasping_rgp, mode="reset",
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "object_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class G1RGPPlaceEnvCfg(ManagerBasedRLEnvCfg):
    scene = RGPPlaceSceneCfg(num_envs=1024, env_spacing=2.5, replicate_physics=True)
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
class G1RGPPlaceEnvCfg_PLAY(G1RGPPlaceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
