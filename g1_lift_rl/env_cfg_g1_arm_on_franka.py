# g1_lift_rl/env_cfg_g1_arm_on_franka.py
"""True isolated-arm ablation (2026-07-28): forks Franka's own official
FrankaCubeLiftEnvCfg (config/franka/joint_pos_env_cfg.py) one-for-one,
swapping ONLY the robot -- Franka's FRANKA_PANDA_CFG replaced with G1's real
right arm + Dex1 gripper, extracted as a standalone FIXED-BASE USD
(g1_right_arm_dex1_only.usd) via Isaac Sim's GUI (Stage panel: deleted every
non-arm link/joint, kept torso_link as the new base, welded torso_link to
world with a Fixed Joint, Body0 empty -- confirmed `is_fixed_base=True`,
15 bodies, 9 joints, stable under 60 steps of free-fall sim, no NaN).

Franka's OWN table, cube, scene, reward/curriculum/PPO are reused entirely
unmodified -- only the robot asset + the action/observation/command wiring
that has to reference G1's own joint and link names differ. This is the
direct, structural answer to the isolation objection raised against
env_cfg_g1_franka_parity.py (which reused G1's whole floating-body): this
robot IS a true fixed-base articulation, exactly like Franka's own
panda_link0, not an approximation via mass scaling.

Mounting pose (torso_link position/rotation): placed visually in the Isaac
Sim GUI against Franka's real table/cube (referenced in via the Script
Editor at their actual task positions), then confirmed numerically
(check_g1_arm_on_franka.py): EE-to-nominal-cube distance 0.35m using the
arm's raw default joint pose (0.28m once the ee_frame offset below was
measured and applied) -- G1's own READY_ARM_POSE was tried first and
produced a folded, 0.86m-away configuration, since it was tuned for G1's old
body orientation and doesn't transfer to this new mount.

Starting joint pose: G1_ARM_HOME_POSE below, a deliberate pre-grasp pose
derived by jogging each arm joint live in the Isaac Sim GUI (Play mode,
dragging each joint's drive target) until the arm extended toward the table
instead of resting folded near the chest -- replaces the raw-default
placeholder used while the mount pose itself was being confirmed.
"""
import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg, TerminationTermCfg as DoneTerm
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.agents.rsl_rl_ppo_cfg import LiftCubePPORunnerCfg

from .constants import ARM_JOINTS, BLOCK_SIZE, GRIPPER_JOINTS, GRIPPER_OPEN, GRIPPER_CLOSE, HAND_BASE_LINK
from .mdp.rewards import GRASP_DIST, GRIP_CLOSED_THRESHOLD, EARLY_CLOSE_DIST, ALIGN_XY

# g1_dex1_arm_only.usd (renamed from the original "g1_right_arm_dex1_only.usd"
# during a 2026-07-30 GUI file-management mishap): torso_link at its original,
# unrotated, file-authored orientation. A rotated sibling variant also exists
# (g1_right_arm_with_torso.usd, torso_link pre-rotated -90deg about X with a
# freshly-rebaked FixedJoint anchor) but ISN'T used here -- this file applies
# the mount rotation at spawn time instead (below), which is the
# already-validated approach (measurements, sanity checks, and the PPO smoke
# test earlier in this project's history all used this file + this rotation).
#
# 2026-07-30 debugging note: this file briefly failed to spawn at ALL (any
# position/rotation, including pure identity) with `RuntimeError: Failed to
# find an articulation... ArticulationRootAPI`, and a misleading
# "disjointed body transforms" warning that turned out to be an unrelated
# red herring. Root cause: the file was genuinely missing
# UsdPhysics.ArticulationRootAPI (a casualty of the same file-mixup above,
# not a false alarm like an earlier, similar-looking scare in this project's
# history) -- fixed by re-applying the API and re-saving in place.
G1_ARM_USD = (
    "/home/virtual-acc/projects/unitree_sim_isaaclab/assets/robots/"
    "g1-29dof-dex1-base-fix-usd/g1_dex1_arm_only.usd"
)

# MEASURED VISUALLY (Isaac Sim GUI, 2026-07-28): positioned interactively
# against Franka's real table/cube (referenced in via Script Editor at their
# real task positions), gripper visually landing right next to the cube,
# then confirmed numerically. Orient (-90, 0, 0) deg about X -> quat below.
G1_ARM_MOUNT_POS = (0.00511, -0.29861, -0.10323)
G1_ARM_MOUNT_ROT = (0.7071, -0.7071, 0.0, 0.0)

# MEASURED (Isaac Sim GUI, jogged live in Play mode, 2026-07-28): a deliberate
# "home"/pre-grasp pose, replacing the raw USD default (which just rested
# folded near the chest -- see docstring above). Degrees -> radians:
#   right_shoulder_pitch_joint =  90.0 deg
#   right_shoulder_roll_joint  = -90.0 deg
#   right_shoulder_yaw_joint   = -90.0 deg
#   right_elbow_joint          = -22.0 deg
# Wrist joints (roll/pitch/yaw) intentionally left unset (raw default) --
# not respecified as part of this pose.
G1_ARM_HOME_POSE = {
    "right_shoulder_pitch_joint": math.radians(90.0),
    "right_shoulder_roll_joint": math.radians(-90.0),
    "right_shoulder_yaw_joint": math.radians(-90.0),
    "right_elbow_joint": math.radians(-22.0),
}

# ALTERNATE MOUNT (2026-07-30): natural/upright torso orientation (identity
# rotation) instead of the -90-about-X laid-down mount above, positioned near
# where Franka's own arm base sits. GUI-confirmed (scratch scene
# g1_29dof_dex1_base_with_table_and_cube.usd) with the arm left at its raw
# USD default joint pose (no jogged home-pose override) -- already brings
# the hand down toward the table/cube with no joint tuning at all, unlike
# the original mount where the raw default rested folded near the chest.
G1_ARM_MOUNT_POS_UPRIGHT = (-0.03077, 0.0, -0.044)
G1_ARM_MOUNT_ROT_UPRIGHT = (1.0, 0.0, 0.0, 0.0)


def _ee_obj_dist(env) -> torch.Tensor:
    """EE (ee_frame sensor, same field access as Isaac Lab's own stock
    object_ee_distance) to cube-centre distance."""
    object_: RigidObject = env.scene["object"]
    ee_frame: FrameTransformer = env.scene["ee_frame"]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    return torch.norm(ee_pos_w - object_.data.root_pos_w, dim=-1)


def _grip_mean(env) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    gids, _ = robot.find_joints(GRIPPER_JOINTS)
    return robot.data.joint_pos[:, gids].mean(dim=-1)


def _closedness(env) -> torch.Tensor:
    """Gripper position normalized to 0 (GRIPPER_OPEN) .. 1 (GRIPPER_CLOSE) --
    same normalization as g1_lift_rl.mdp's own _closedness()."""
    return torch.clamp((_grip_mean(env) - GRIPPER_OPEN) / (GRIPPER_CLOSE - GRIPPER_OPEN), 0.0, 1.0)


def _grasp_gate(env) -> torch.Tensor:
    """Enveloped (EE within GRASP_DIST of the cube) AND gripper closed past
    GRIP_CLOSED_THRESHOLD. Reimplemented locally rather than reusing
    g1_lift_rl.mdp's own _is_grasping()/reward_goal_tracking_* -- those
    assume a fixed INSPECT_POS goal, but this ablation reuses Franka's own
    randomized object_pose command as the goal, so only the GRASP_DIST/
    GRIP_CLOSED_THRESHOLD *thresholds* are shared, not the reward functions
    themselves.
    """
    return (_ee_obj_dist(env) < GRASP_DIST) & (_grip_mean(env) > GRIP_CLOSED_THRESHOLD)


def object_out_of_bounds(env, x_range, y_range, below_height) -> torch.Tensor:
    """Terminate when the cube gets knocked/slid out of the workable zone
    while still at table level -- once it leaves the arm's reachable area
    there is nothing left to learn from the rest of the episode. Gated on
    z < below_height so a legitimately grasped-and-carried cube (which may
    leave the spawn zone laterally on its way to the goal command, whose
    stock ranges reach x 0.6 / y +-0.25) is never affected -- a lifted cube
    is exempt by construction."""
    object_: RigidObject = env.scene["object"]
    pos = object_.data.root_pos_w - env.scene.env_origins
    out_x = (pos[:, 0] < x_range[0]) | (pos[:, 0] > x_range[1])
    out_y = (pos[:, 1] < y_range[0]) | (pos[:, 1] > y_range[1])
    at_table_level = pos[:, 2] < below_height
    return (out_x | out_y) & at_table_level


def penalty_early_close(env) -> torch.Tensor:
    """Closed gripper while far from the cube = penalized -- keeps the
    approach open-handed instead of closing from the start of the episode.
    Reused verbatim (same EARLY_CLOSE_DIST threshold, same formula) from
    g1_lift_rl.mdp.rewards.penalty_early_close, already proven across
    env_cfg.py/env_cfg_policy2.py/env_cfg_combined.py (weight=-0.5 in all
    three) -- reimplemented locally rather than imported directly since it
    depends on _ee_obj_dist/_closedness, which this ablation computes via
    its own ee_frame sensor (see _grasp_gate docstring)."""
    return _closedness(env) * (_ee_obj_dist(env) > EARLY_CLOSE_DIST).float()


def reward_close_gradient(env) -> torch.Tensor:
    """Dense pay for closing PROGRESSIVELY while xy-aligned and near the
    cube -- the positive counterpart penalty_early_close was missing in this
    config. Ported from g1_lift_rl.mdp.rewards.reward_close_gradient (the
    proven Policy-2 version, incl. its documented fix history: falloff tied
    to GRASP_DIST -- linear ramp to 0 by 2x GRASP_DIST ~ 8cm -- NOT a slower
    shared tanh constant, so closing is only net-rewarding right at the
    cube and early_close keeps winning everywhere else; gated on xy
    alignment so an off-centre closing angle can't collect it either).
    Added (2026-07-30) after full runs 1-3 ALL ended with lifting_object=
    0.0000 and action std collapsed to ~0.06: with only a penalty on the
    gripper channel and the first positive gripper signal requiring a
    complete 6.5cm lift, "never close" is the exact local optimum PPO
    reliably converged to, three times."""
    ee = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    cube = env.scene["object"].data.root_pos_w
    aligned = (torch.norm(ee[:, :2] - cube[:, :2], dim=-1) < ALIGN_XY).float()
    proximity = torch.clamp(1.0 - _ee_obj_dist(env) / (2.0 * GRASP_DIST), 0.0, 1.0)
    return aligned * proximity * _closedness(env)


def reward_grasp(env) -> torch.Tensor:
    """Binary bonus while the grasp is actually achieved (enveloped AND
    closed onto the cube) -- the middle rung between close_gradient above
    and the full lift bonus. Same _grasp_gate the lift/goal rewards already
    use, so what it pays for is by construction exactly what they require.
    Mirrors g1_lift_rl.mdp.rewards.reward_grasp (Policy 2, weight 2.0)."""
    return _grasp_gate(env).float()


LIFT_CAP = 0.12          # m -- matches g1_lift_rl.mdp.rewards.LIFT_CAP
_UPRIGHT_CUBE_REST_Z = 0.012  # env-local z of the cube's resting CENTER (v2/v3 zone)


def reward_lift_dense(env) -> torch.Tensor:
    """Dense per-height lift reward, gated on grasping -- ported from
    g1_lift_rl.mdp.rewards.reward_lift (Policy 2, weight 4.0), incl. its
    documented rest-center-baseline fix (height must be genuinely 0 at
    rest, no free credit for grasping in place). Added (2026-07-30) after
    run 4's deterministic replay (diagnose_grasp_run4.py) PHYSICALLY
    confirmed genuine enveloped grasps (10/32 envs, finger joints blocked
    at grip_mean~-0.010, zero air-closed fakes) and even one real 8.3cm
    lift -- but the binary 6.5cm lifting_object bonus alone left "hold on
    the table" as the stable optimum (runs' lifting term averaged 0.0000).
    This term pays proportionally for every mm of height while holding, so
    those rare real lift events become a smooth gradient instead of a
    1-in-32 cliff crossing."""
    obj: RigidObject = env.scene["object"]
    height = torch.clamp(
        (obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]) - _UPRIGHT_CUBE_REST_Z,
        0.0, LIFT_CAP,
    )
    return (height / LIFT_CAP) * _grasp_gate(env).float()


def reward_object_lifted_grasp_gated(env, minimal_height: float, object_cfg=SceneEntityCfg("object")) -> torch.Tensor:
    """Same as stock object_is_lifted, but a swat/launch with an open or
    uncommitted gripper can no longer collect this reward -- only a genuine
    held-and-lifted cube counts. See _grasp_gate() docstring for why this
    fix (already proven in g1_lift_rl/mdp/rewards.py for a different
    ablation, 2026-07-27) is reimplemented locally here instead of reused
    directly."""
    base = mdp.object_is_lifted(env, minimal_height=minimal_height, object_cfg=object_cfg)
    return base * _grasp_gate(env).float()


def reward_goal_distance_grasp_gated(
    env, std: float, minimal_height: float, command_name: str,
    robot_cfg=SceneEntityCfg("robot"), object_cfg=SceneEntityCfg("object"),
) -> torch.Tensor:
    """Same as stock object_goal_distance, additionally gated on
    _grasp_gate() -- see reward_object_lifted_grasp_gated docstring."""
    base = mdp.object_goal_distance(
        env, std=std, minimal_height=minimal_height, command_name=command_name,
        robot_cfg=robot_cfg, object_cfg=object_cfg,
    )
    return base * _grasp_gate(env).float()


@configclass
class G1RightArmDex1LiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # --- robot: G1's real right arm + Dex1, extracted as a standalone
        # fixed-base asset (see module docstring) ---
        self.scene.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=G1_ARM_USD,
                # FIXED (2026-07-30): was False. g1_lift_ext's own proven
                # G1_DEX1_CFG (env_cfg.py) uses True -- self-collision matters
                # for realistic finger/palm/forearm contact behavior during
                # grasping, which this ablation needs now that grasping (not
                # just reaching) is the point.
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True),
                # Same contact-resolution-speed cap as this project's own proven
                # G1_DEX1_CFG (env_cfg.py) -- prevents an uncapped depenetration
                # response from launching the cube on finger contact.
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_linear_velocity=1000.0, max_angular_velocity=1000.0, max_depenetration_velocity=1.0
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=G1_ARM_MOUNT_POS, rot=G1_ARM_MOUNT_ROT,
                joint_pos={**G1_ARM_HOME_POSE, GRIPPER_JOINTS[0]: GRIPPER_OPEN, GRIPPER_JOINTS[1]: GRIPPER_OPEN},
                joint_vel={".*": 0.0},
            ),
            actuators={
                # Same gains as this project's own proven G1_DEX1_CFG (env_cfg.py).
                "arm": ImplicitActuatorCfg(joint_names_expr=ARM_JOINTS, stiffness=150.0, damping=10.0),
                "gripper": ImplicitActuatorCfg(joint_names_expr=GRIPPER_JOINTS, stiffness=800.0, damping=3.0),
            },
        )

        # --- actions: G1's own joint names, same action-term TYPES as Franka ---
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=ARM_JOINTS, scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=GRIPPER_JOINTS,
            open_command_expr={j: GRIPPER_OPEN for j in GRIPPER_JOINTS},
            close_command_expr={j: GRIPPER_CLOSE for j in GRIPPER_JOINTS},
        )
        # Set the body name for the end effector -- Dex1's mounting-plate
        # body, the structural analog of Franka's "panda_hand".
        self.commands.object_pose.body_name = HAND_BASE_LINK

        # --- cube: SWAPPED from Franka's own DexCube (scale=0.8 -> 4.8cm) to
        # G1's own proven cube physics (env_cfg.py: static_friction=10.0,
        # dynamic_friction=1.5) -- FLAGGED DEVIATION from "reuse Franka's
        # scene unmodified": visually confirmed in the Isaac Sim GUI that
        # Dex1's gripper cannot reliably close on Franka's smaller, smooth,
        # low-friction DexCube. This is a hard geometric/physical blocker,
        # not a style choice -- an ungraspable object makes any training
        # attempt moot regardless of reward/curriculum design, so it
        # overrides the "keep everything but the robot identical" ablation
        # purity goal.
        #
        # Size: G1's own BLOCK_SIZE (6cm) halved to 3cm (2:1) -- G1's own
        # established size, at THIS gripper approach angle/mount, still
        # looked too large; not reusing BLOCK_SIZE directly here, just its
        # proven friction/material properties. Position kept at Franka's own
        # (0.5, 0) x/y; z recomputed for the new half-size (table surface
        # ~0.031, computed from Franka's original 4.8cm-cube spawn z=0.055
        # minus its own half-size -> +0.015 for 3cm's half-size = 0.046).
        # CUBE_ROT (G1's own approach-angle tuning) NOT applied yet -- that
        # was tuned for G1's OLD scene geometry, not this new mount; using
        # identity rotation until re-measured.
        ARM_ON_FRANKA_CUBE_SIZE = BLOCK_SIZE / 2.0
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=sim_utils.CuboidCfg(
                size=(ARM_ON_FRANKA_CUBE_SIZE,) * 3,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False, retain_accelerations=False, max_depenetration_velocity=1.0
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True, contact_offset=0.01, rest_offset=0.0
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="max", restitution_combine_mode="min",
                    static_friction=10.0, dynamic_friction=1.5, restitution=0.0,
                ),
            ),
            # MOVED CLOSER (2026-07-30, user's direct GUI observation while
            # watching training): center of the desired range
            # x:[0.2,0.35] y:[-0.15,0.15] z:0.013 (table-relative, same frame
            # as this pos) -- (0.5, 0, 0.046) put the cube too far for this
            # arm's actual coordinated reach from the home pose. Center here,
            # pose_range below spans out to the edges of that observed range.
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.275, 0.0, 0.013], rot=[1, 0, 0, 0]),
        )

        # --- ee_frame: rooted at torso_link (this asset's fixed base, the
        # structural analog of Franka's panda_link0), target = HAND_BASE_LINK.
        # Offset MEASURED (measure_ee_offset.py, 2026-07-28): fingertip
        # (EE_LINKS) midpoint relative to HAND_BASE_LINK's own local frame =
        # (0.0, 0.0973, 0.0142), magnitude 0.0984m -- matches this project's
        # own earlier independent estimate ("~9.8cm", constants.py) almost
        # exactly, a good consistency check.
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/" + HAND_BASE_LINK,
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0973, 0.0142]),
                ),
            ],
        )

        # --- cube spawn randomization: NARROWED from Franka's own default
        # (x: +-0.1m, y: +-0.25m) -- this arm's coordinated reach from the
        # home pose doesn't cover nearly as wide an area as Franka's own arm
        # does. Range/center MOVED (2026-07-30, user's direct GUI
        # observation while watching training): desired absolute range
        # x:[0.2,0.35] y:[-0.15,0.15] -> center (0.275, 0.0) is the object's
        # own init_state.pos above, so pose_range here spans +-0.075 (x) and
        # +-0.15 (y) around that center to reach the same absolute edges.
        self.events.reset_object_position.params["pose_range"] = {
            "x": (-0.075, 0.075), "y": (-0.15, 0.15), "z": (0.0, 0.0)
        }

        # --- reward fix (2026-07-30): this class inherits RewardsCfg unmodified
        # from Isaac Lab's own stock LiftEnvCfg, whose lifting_object/
        # object_goal_tracking* terms gate purely on cube world-z height --
        # no check for WHY the cube is up. The trained checkpoint from this
        # exact config (2026-07-30_11-38-33) confirmed via GUI playback that
        # the policy exploits this: it swats/launches the cube with
        # right_hand_base_link rather than grasping it, since a launch clears
        # the height gate identically to a real lift. g1_lift_ext's own
        # combined-policy config (mdp/rewards.py) already found and fixed this
        # identical exploit on 2026-07-27 via an added grasp gate (EE within
        # GRASP_DIST of the cube AND gripper closed past
        # GRIP_CLOSED_THRESHOLD). Reimplemented locally above
        # (reward_object_lifted_grasp_gated/reward_goal_distance_grasp_gated)
        # rather than reused directly, since this ablation's goal-tracking
        # uses Franka's own randomized object_pose command, not the fixed
        # INSPECT_POS goal g1_lift_rl's own versions assume -- only swapping
        # .func, params stay exactly as inherited (std/minimal_height/
        # command_name all still apply, my functions take the same names).
        self.rewards.lifting_object.func = reward_object_lifted_grasp_gated
        self.rewards.object_goal_tracking.func = reward_goal_distance_grasp_gated
        self.rewards.object_goal_tracking_fine_grained.func = reward_goal_distance_grasp_gated

        # --- early-close penalty (2026-07-30, user's own GUI observation
        # watching the upright-mount checkpoint play back): the gripper was
        # closing from the very start of the episode, before ever reaching
        # the cube. Stock Franka reward has no term discouraging this at
        # all. g1_lift_ext's own successful configs (env_cfg.py,
        # env_cfg_policy2.py, env_cfg_combined.py) all carry the identical
        # fix, same weight (-0.5) -- adding it here too, RewardsCfg doesn't
        # declare this field statically but RewardManager builds its term
        # list from cfg.__dict__, so a new dynamically-added RewTerm works
        # the same as a declared one.
        self.rewards.penalty_early_close = RewTerm(func=penalty_early_close, weight=-0.5)


@configclass
class G1RightArmDex1LiftEnvCfg_PLAY(G1RightArmDex1LiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False


@configclass
class G1RightArmDex1PPORunnerCfg(LiftCubePPORunnerCfg):
    experiment_name = "g1_right_arm_dex1_on_franka_table"


@configclass
class G1RightArmDex1UprightLiftEnvCfg(G1RightArmDex1LiftEnvCfg):
    """Alternate-mount ablation (2026-07-30): natural/upright torso
    orientation (identity rotation) instead of the -90-about-X laid-down
    mount above, at a near-Franka-base position (G1_ARM_MOUNT_POS_UPRIGHT),
    with a GUI-derived fingers-down home pose (all 7 arm joints, wrists
    included -- see joint_pos below for derivation history; earlier
    raw-default and shoulder-pitch-only variants both failed to ever fire
    lifting_object). Also carries Franka-parity gripper drive gains and a
    tightened cube spawn range -- see inline notes. Base scene/actions/
    grasp-gated rewards inherited from the base OnFrankaTable ablation.
    """
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.init_state.pos = G1_ARM_MOUNT_POS_UPRIGHT
        self.scene.robot.init_state.rot = G1_ARM_MOUNT_ROT_UPRIGHT

        # --- gripper gains (2026-07-30): Franka-parity squeeze/damping.
        # Deep-dive comparison against Franka's own working lift setup found
        # Dex1's fingers are the SAME mechanism class as panda_hand's (two
        # prismatic parallel-jaw joints, same 200N USD maxForce) but were
        # driven far softer: our inherited 800/3 vs Franka's 2000/100 --
        # ~12N vs ~48N sustained squeeze on the blocking cube, and damping 3
        # lets finger-cube contact ring where Franka's 100 is dead-stable.
        # These are NOT Franka-specific numbers blindly copied onto foreign
        # hardware: effort_limit_sim=200 is Dex1's own USD-authored maxForce,
        # and stiffness/damping are drive-tuning free parameters, not
        # physical properties of the hand. Runs 1-2 both ended with
        # lifting_object=0.0000 throughout; weak grip is one of the two
        # suspected causes (the other: home-pose hand orientation, fixed
        # separately).
        self.scene.robot.actuators["gripper"].stiffness = 2000.0
        # FIXED (2026-07-30, after run-5 NaN crash at iter 281): damping
        # 100 -> 30. Franka's damping=100 is tuned for its ~100-150g
        # fingers; Dex1's finger links weigh 8-28 GRAMS (measured from the
        # USD), making 100 roughly 10x overdamped there (critical damping
        # 2*sqrt(k*m) ~ 8-15 for these masses). Run 5 died with value-loss
        # NaN -> policy-std NaN mid-grasp-learning -- the more grasping per
        # iteration, the more contact-solver dice rolls. 30 matches
        # Franka's DAMPING RATIO at Dex1's masses instead of copying its
        # absolute number; stiffness (the squeeze strength that made
        # grasping work) stays 2000.
        self.scene.robot.actuators["gripper"].damping = 30.0
        self.scene.robot.actuators["gripper"].effort_limit_sim = 200.0
        # FIXED (2026-07-30, runs 5+7 NaN crashes): Dex1's USD authors
        # maxJointVelocity=2119 m/s on the PRISMATIC finger joints --
        # physically absurd (4.45cm travel). Runs 5/7 both died with value
        # loss 0.02 -> 1.5e17 -> inf -> nan in 3 iterations: a single
        # contact-impulse velocity spike while squeezing blows up the
        # velocity observations/returns. 1 m/s is far above any legitimate
        # finger motion and far below the explosion regime.
        self.scene.robot.actuators["gripper"].velocity_limit_sim = 1.0
        # Also match Franka's own solver iteration count (its ArticulationCfg
        # sets position=8; we'd been running the default 4) -- more contact
        # solver iterations, same stability intent as the damping fix.
        self.scene.robot.spawn.articulation_props.solver_position_iteration_count = 8

        self.scene.robot.init_state.joint_pos = {
            # Home position v2 (2026-07-30, second GUI derivation): fingers
            # pointing down at the tabletop, Franka-ready-pose-style. The
            # deep-dive Franka comparison identified hand orientation as the
            # likely reason lifting_object never fired in runs 1-2: Franka's
            # own init pose starts the hand already pointing straight down
            # (grasp-straddle nearly automatic), while both our earlier poses
            # (raw default = jammed in table; v1 shoulder_pitch=-90 only =
            # hand pointing up/away at z~0.43) left finger orientation for
            # the policy to discover on its own -- and no reward term ever
            # teaches it. All 7 arm joints jogged live in the GUI this time
            # (wrists included -- v1 left them at 0), user-derived and
            # user-confirmed visually against the table/cube scene. User
            # note: this is "90 degrees facing down" -- if it underperforms,
            # a fully-face-down variant may follow.
            "right_shoulder_pitch_joint": math.radians(-92.9),
            "right_shoulder_roll_joint": math.radians(-62.1),
            "right_shoulder_yaw_joint": math.radians(81.7),
            "right_elbow_joint": math.radians(27.8),
            "right_wrist_roll_joint": math.radians(103.5),
            # v3 (2026-07-30, after run-4 playback): 32.2 -> 77.0 deg -- the
            # user's planned "directly facing down" fallback. v2's ~90-deg-
            # sideways gripper learned to grasp (run 4) but plateaued at
            # grasp-and-hold; with fingers now vertical over the cube,
            # hoisting is a straight upward pull instead of a held side-grip.
            # Within the joint's real +-92.5 deg limit.
            "right_wrist_pitch_joint": math.radians(77.0),
            "right_wrist_yaw_joint": 0.0,
            GRIPPER_JOINTS[0]: GRIPPER_OPEN, GRIPPER_JOINTS[1]: GRIPPER_OPEN,
        }

        # --- cube: this mount's reach geometry is completely different from
        # the -90-about-X mount above (different arm direction entirely), so
        # the old mount's cube position/range doesn't carry over.
        # NARROWED v2 (2026-07-30, user, together with home pose v2): absolute
        # spawn range x:[0.3,0.4] y:[-0.21,-0.05] z fixed at 0.012 -- tighter
        # than v1's x:[0.22,0.4] y:[-0.28,-0.05], concentrated under the new
        # fingers-down home pose to make the grasp easier to discover.
        # init below = range center; pose_range is the same absolute range
        # re-expressed relative to init (Isaac Lab's reset_object_position
        # event samples relative to init_state.pos).
        self.scene.object.init_state.pos = [0.35, -0.13, 0.012]
        self.events.reset_object_position.params["pose_range"] = {
            "x": (-0.05, 0.05), "y": (-0.08, 0.08), "z": (0.0, 0.0)
        }

        # --- cube mass (2026-07-30, user request): 0.1 -> 0.2 kg. Watching
        # the smoke tests, the 0.1kg cube got punted out of the zone by the
        # lightest finger graze; doubling the mass makes it stay put under
        # incidental contact. Grip-security check: the new Franka-parity
        # gains sustain ~48N/finger of squeeze and the contact pair runs at
        # the cube's own friction (static 10.0 via combine_mode=max), so
        # 0.2kg (~2N of weight) is nowhere near slip territory.
        self.scene.object.spawn.mass_props = sim_utils.MassPropertiesCfg(mass=0.2)

        # --- observation clipping (2026-07-30, runs 5+7 NaN crashes):
        # second line of defense behind the velocity-limit fix above. No
        # legitimate state comes near these bounds (joint_pos < 3.2 rad,
        # nominal joint_vel < ~20, scene spans < 1.5m), so clipping is
        # inert in normal operation and only truncates explosion spikes
        # before they can reach the value function (0.02 -> 1.5e17 -> nan
        # was observed to happen within a single iteration's rollout).
        self.observations.policy.joint_pos.clip = (-10.0, 10.0)
        self.observations.policy.joint_vel.clip = (-50.0, 50.0)
        self.observations.policy.object_position.clip = (-5.0, 5.0)
        self.observations.policy.target_object_position.clip = (-5.0, 5.0)
        self.observations.policy.actions.clip = (-10.0, 10.0)

        # --- goal command box (2026-07-30 audit): stock ranges (pos_x
        # 0.4-0.6, pos_y +-0.25, pos_z 0.25-0.5, root frame) are sized for
        # Franka's ~0.85m arm. G1's right arm measures 0.5175m max
        # straight-line extension (shoulder->fingertip-mid, from the USD
        # link chain), leaving only ~13% of the stock box realistically
        # reachable (Monte-Carlo vs measured shoulder position; mean stock
        # goal sat at 0.546m -- beyond even theoretical max). Narrowed to a
        # box just above the cube spawn zone, every corner <=0.44m from the
        # shoulder ("lift and hold nearby"): env-local x 0.28-0.38,
        # y -0.20-0.0, z 0.21-0.32, re-expressed in the command's robot-root
        # frame (root at (-0.0308, 0, -0.044), identity rotation).
        self.commands.object_pose.ranges.pos_x = (0.31, 0.41)
        self.commands.object_pose.ranges.pos_y = (-0.20, 0.0)
        self.commands.object_pose.ranges.pos_z = (0.25, 0.36)

        # --- cube solver iterations (2026-07-30 audit): Franka's own
        # DexCube carried solver_position_iteration_count=16 /
        # velocity=1 -- silently lost when this config swapped in a custom
        # cuboid (fell back to defaults 4/0). Restored to Franka's values:
        # more contact-solver iterations on the grasped body, same
        # stability intent as the run-5-NaN fixes.
        self.scene.object.spawn.rigid_props.solver_position_iteration_count = 16
        self.scene.object.spawn.rigid_props.solver_velocity_iteration_count = 1

        # --- episode length (2026-07-30, user request): 5s -> 8s. The
        # stock 5s is Franka-paced; gives the slower grasp-ladder sequence
        # (reach -> align -> close -> hoist 6.5cm -> carry to goal) more
        # room per episode. 8s also matches the main g1_lift_ext task's own
        # episode length. At decimation=2/dt=0.01 this is 400 control steps.
        self.episode_length_s = 8.0
        # Coupled: stock resampling (5,5)s == old episode length, i.e. one
        # goal per episode. Keep that semantic at 8s -- without this the
        # goal would silently start switching mid-episode.
        self.commands.object_pose.resampling_time_range = (8.0, 8.0)

        # --- boundary termination (2026-07-30, user request): end the
        # episode when the cube gets knocked/slid out of the workable zone
        # while still at table level (smoke testing showed ~50% of episodes
        # ending in object_dropping -- lots of cube-punting during early
        # exploration; once out of reach the rest of the episode teaches
        # nothing). Bounds = the v2 spawn range +-0.1m margin. below_height
        # =0.04 matches the stock lift threshold (minimal_height), so any
        # cube counted as "lifted" by the reward can never trip this.
        self.terminations.object_out_of_bounds = DoneTerm(
            func=object_out_of_bounds,
            params={"x_range": (0.2, 0.5), "y_range": (-0.31, 0.05), "below_height": 0.04},
        )

        # --- grasp-attempt ladder (2026-07-30, after runs 1-3 all converged
        # to reach-and-hover with lifting_object=0.0000): the two proven
        # Policy-2 shaping rungs (see the functions' docstrings above),
        # at Policy 2's own weights (env_cfg_policy2.py: close_gradient=1.0,
        # grasp=2.0). Completes the incentive chain reach(1.0) ->
        # close-while-aligned(1.0) -> hold-grasp(2.0) -> lift(15) ->
        # carry-to-goal(16+5); before this, the gripper channel's only
        # learnable signal was the early-close penalty.
        self.rewards.close_gradient = RewTerm(func=reward_close_gradient, weight=1.0)
        self.rewards.grasp = RewTerm(func=reward_grasp, weight=2.0)
        self.rewards.lift_dense = RewTerm(func=reward_lift_dense, weight=4.0)

        # --- early-close penalty REMOVED for this config (2026-07-30, user
        # request -- Franka-style experiment): Franka's task has NO gripper
        # shaping at all and discovers grasping because its untrained policy
        # randomly closes ~50% of episodes over an auto-straddled cube.
        # Our penalty (inherited from the base class) suppressed exactly
        # that discovery engine. With the close_gradient/grasp ladder above
        # now pointing closing incentives at the cube, the penalty's
        # original job (stop closed-fist approaches) is covered by carrots
        # instead of the stick -- so let random closing run free, as Franka
        # does. (RewardManager skips None-valued terms.)
        self.rewards.penalty_early_close = None

        # --- lift threshold (2026-07-30, user request -- the long-deferred
        # "raise LIFT_MINIMAL_HEIGHT toward 6-7cm" item, now applied to this
        # config): stock minimal_height=0.04 is ABSOLUTE env-local z; the
        # cube's rest center sits at z=0.012, so 0.04 demanded only 2.8cm of
        # real lift -- low enough for tilt/pinch geometry to cross without a
        # clean grasp (the same exploit ceiling documented in the main
        # g1_lift_ext task). 0.077 = rest center + 6.5cm, the middle of the
        # user's requested 6-7cm band. Applied to all three height-gated
        # reward terms. NOTE: object_out_of_bounds.below_height deliberately
        # stays at 0.04 -- its job is only "on the table vs carried", and
        # raising it with this would wrongly terminate genuine low carries.
        for _term in (
            self.rewards.lifting_object,
            self.rewards.object_goal_tracking,
            self.rewards.object_goal_tracking_fine_grained,
        ):
            _term.params["minimal_height"] = 0.077

        # --- reach reward std (2026-07-30): stock reaching_object (std=0.1,
        # inherited unmodified, same as the base OnFrankaTable class above)
        # gives a near-vanishing tanh-kernel gradient at this pose's ~0.46m
        # starting EE-to-cube distance (dist/std=4.6, sech^2~0.0002) -- the
        # 50-iteration smoke test confirmed this: reaching_object flat at
        # 0.0003 the whole run, no trend, the same signature that motivated
        # bumping REACH_STD 0.1->0.3 in env_cfg_combined.py for an even
        # closer 0.55m starting distance. Not applied to the base
        # OnFrankaTable class -- that mount's tuned 0.28m start already gets
        # a real (if weak) gradient at std=0.1 and its run is a completed,
        # locked-in comparison baseline.
        self.rewards.reaching_object.params["std"] = 0.3


@configclass
class G1RightArmDex1UprightLiftEnvCfg_PLAY(G1RightArmDex1UprightLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False


@configclass
class G1RightArmDex1UprightPPORunnerCfg(LiftCubePPORunnerCfg):
    experiment_name = "g1_right_arm_dex1_on_franka_table_upright"
