# g1_lift_rl/env_cfg_rgp_scene.py
"""Shared scene for the RGP (Reach -> Grasp+lift -> Place+release) 3-policy
chain: robot, table, cube. Independent of the old 4-policy chain's env_cfg.py
(no subclassing), built from the same physical constants in constants.py.

The goal marker (yellow disc at RGP_GOAL_POS, visual-only) lives here so all
four policies show it during playback, even though only Policy 3/4's rewards
actually reference RGP_GOAL_POS -- purely for consistent GUI viewing across
the chain, no effect on Policy 1/2's observations, rewards, or trained
checkpoints (kinematic, massless, no collision, not in any obs term).
"""
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from .constants import (
    ROBOT_USD, ROBOT_POS, ROBOT_ROT, READY_ARM_POSE, LEFT_ARM_STOW,
    GRIPPER_OPEN, TABLE_POS, TABLE_SIZE, BLOCK_SIZE, BLOCK_INIT_POS, CUBE_ROT,
    TABLE_TOP_Z, GOAL_POS,
)

RGP_BLOCK_SIZE = 0.03  # cube edge length (m)
RGP_BLOCK_INIT_POS = (BLOCK_INIT_POS[0], BLOCK_INIT_POS[1], TABLE_TOP_Z + RGP_BLOCK_SIZE / 2.0)

# Goal xy reused from the old chain's own GOAL_POS (an arbitrary but already-
# established table placement, not tied to cube size); z recomputed for THIS
# scene's 3cm cube (old GOAL_POS's z was calibrated for the old chain's 6cm
# cube) so a resting cube's centre at the goal matches RGP_BLOCK_SIZE.
RGP_GOAL_POS = (GOAL_POS[0], GOAL_POS[1], TABLE_TOP_Z + RGP_BLOCK_SIZE / 2.0)

# ---------------------------------------------------------------------------
# Robot
# ---------------------------------------------------------------------------
_DEFAULT_JOINTS_RGP = {
    ".*_hip_.*": 0.0, ".*_knee_joint": 0.0, ".*_ankle_.*": 0.0,
    "waist_.*": 0.0,
    "left_hand_Joint.*": GRIPPER_OPEN, "right_hand_Joint.*": GRIPPER_OPEN,
    **READY_ARM_POSE,
    **LEFT_ARM_STOW,
}

RGP_G1_DEX1_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=ROBOT_USD,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            max_linear_velocity=1000.0, max_angular_velocity=1000.0, max_depenetration_velocity=1.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=ROBOT_POS, rot=ROBOT_ROT,
        joint_pos=_DEFAULT_JOINTS_RGP, joint_vel={".*": 0.0},
    ),
    actuators={
        "body": ImplicitActuatorCfg(joint_names_expr=["(?!waist_).*_joint"], stiffness=150.0, damping=10.0),
        "waist": ImplicitActuatorCfg(joint_names_expr=["waist_.*_joint"], stiffness=10000.0, damping=10000.0),
        # The USD authors an absurd ~2119 m/s velocity limit on the finger
        # joints; uncapped, a contact impulse during grasping can send joint
        # velocity into a value-loss NaN. Cap it explicitly.
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[".*_hand_Joint.*"], stiffness=800.0, damping=3.0,
            velocity_limit_sim=1.0,
        ),
    },
)


@configclass
class RGPSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(1.0, 1.0, 1.0)),
    )
    robot: ArticulationCfg = RGP_G1_DEX1_CFG
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.55, 0.5)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=TABLE_POS),
    )
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=(RGP_BLOCK_SIZE, RGP_BLOCK_SIZE, RGP_BLOCK_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, retain_accelerations=False, max_depenetration_velocity=1.0
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.01, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max", restitution_combine_mode="min",
                static_friction=10.0, dynamic_friction=1.5, restitution=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=RGP_BLOCK_INIT_POS, rot=CUBE_ROT),
    )
    # Visual-only marker (flat disc, kinematic, massless, no collision) at the
    # goal's xy -- purely a reference for where reward_place_rgp/
    # reward_release_rgp actually check, can't be pushed or interacted with.
    # Sits at TABLE_TOP_Z (not RGP_GOAL_POS's own z, which is the cube's
    # resting CENTRE height) so the disc lies flush on the table rather than
    # floating half the cube's height above it.
    goal = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Goal",
        spawn=sim_utils.CylinderCfg(
            radius=0.05, height=0.002,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(RGP_GOAL_POS[0], RGP_GOAL_POS[1], TABLE_TOP_Z)),
    )
