# g1_lift_rl/env_cfg_franka_g1_scene.py
"""Franka-in-G1-scene ablation (2026-07-27): alternative to the mass-override
arm-isolation experiment (env_cfg_g1_arm_isolated.py). Instead of putting a
(mass-hacked) G1 into the scene, this keeps FRANKA's own robot, cube, reward/
curriculum/PPO -- built on FrankaCubeLiftFarFixedEnvCfg, the confirmed-
necessary far-pose + REACH_STD=0.3 baseline -- and relocates just the robot
base + table to G1's exact scene geometry (standing height 0.76m, yawed -90
deg, table at G1's TABLE_POS). Stays entirely on Franka's own clean, already-
fixed-base dynamics: no mass hacks, no floating-base risk. Isolates SCENE
GEOMETRY (mount height, approach angle) as a variable, distinct from
EMBODIMENT (what env_cfg_g1_arm_isolated.py tests instead).

Table: reused directly from g1_lift_rl's own scene (a plain symmetric
CuboidCfg) rather than Franka's own SeattleLabTable mesh -- copying G1's
rot onto an asymmetric real mesh blind would risk facing it the wrong way;
a box looks the same regardless.

Cube: G1's own BLOCK_INIT_POS (x, y) reused as-is (same horizontal reach
point on the relocated table); z re-derived for FRANKA's own (smaller,
4.8cm, DexCube scale=0.8) cube -- TABLE_TOP_Z + half of ITS OWN size, not
G1's 6cm value. Keeping Franka's own cube (not swapping to G1's) matches
env_cfg_g1_franka_parity.py's own disclosed choice -- avoids a second
confound.

Arm starting pose: FAR_ARM_POSE, carried over UNCHANGED from the FarFixed
baseline -- NOT re-derived for this new geometry yet. Relocating the base
changes the EE's absolute world position, and relocating the cube changes
its position too; the resulting EE-to-cube distance under the same joint
angles is NOT guaranteed to still be ~0.555m (the value this whole ablation
chain has held constant throughout). MUST be measured against the live
scene (see measure_franka_g1_scene_reach.py in the scratchpad) before
trusting a training run here -- if it's drifted far from 0.555m, that
reintroduces the exact REACH_STD dead-zone this chain already fixed once,
as a confound on top of the new geometry variable rather than instead of it.
"""
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass

from .constants import BLOCK_INIT_POS, ROBOT_POS, ROBOT_ROT, TABLE_POS, TABLE_SIZE, TABLE_TOP_Z
from .env_cfg_franka_far import FrankaCubeLiftFarFixedEnvCfg, FrankaFarFixedPPORunnerCfg

# Measured (measure_franka_cube_size.py, earlier this session): DexCube base
# edge 0.06m * scale 0.8 = 0.048m.
FRANKA_CUBE_SIZE = 0.048
FRANKA_CUBE_POS_ON_G1_TABLE = (BLOCK_INIT_POS[0], BLOCK_INIT_POS[1], TABLE_TOP_Z + FRANKA_CUBE_SIZE / 2.0)


@configclass
class FrankaCubeLiftG1SceneEnvCfg(FrankaCubeLiftFarFixedEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Relocate Franka's base to G1's exact standing pose.
        self.scene.robot.init_state.pos = ROBOT_POS
        self.scene.robot.init_state.rot = ROBOT_ROT

        # Swap Franka's own SeattleLabTable mesh for G1's plain box table at
        # G1's position -- see module docstring for why (asymmetric-mesh risk).
        self.scene.table = AssetBaseCfg(
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

        # Keep Franka's own DexCube, repositioned onto G1's table at G1's
        # own (x, y) reach point.
        self.scene.object.init_state.pos = FRANKA_CUBE_POS_ON_G1_TABLE

        # NOTE: EventCfg.reset_object_position (reset_root_state_uniform)
        # samples an offset ADDED to this default init_state -- re-centers
        # automatically here, no separate change needed.


@configclass
class FrankaCubeLiftG1SceneEnvCfg_PLAY(FrankaCubeLiftG1SceneEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False


@configclass
class FrankaG1ScenePPORunnerCfg(FrankaFarFixedPPORunnerCfg):
    experiment_name = "franka_lift_g1_scene"
