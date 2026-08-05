# g1_lift_rl/env_cfg_franka_far.py
"""ABLATION EXPERIMENT (2026-07-27): the official Franka lift task, completely
unmodified except for ONE variable -- the robot's default/reset arm pose, moved
farther from the cube to match G1's own measured reach distance.

Registered here (piggybacking on g1_lift_rl's own gym-registration package,
which is already confirmed to auto-load via the standard `--task <id>`
invocation) purely for convenience -- this task has nothing to do with G1
mechanically, it's a controlled comparison for the g1_lift_combined
investigation (see [[g1_lift_ext]] / sim-to-sim-transfer.md's Franka section).

Why: training the real, unmodified Franka-lift reference converged cleanly
(lifting_object~13/15) while g1_lift_combined kept crashing at the same
curriculum-transition point. The two setups differ in more than one way at
once (env count, starting distance, robot embodiment, gripper). This isolates
JUST the starting-distance variable, keeping the 4096-env count, official
reward weights (REACH_STD=0.1 etc.), and everything else exactly as the
baseline Franka run -- to test whether reach distance alone reproduces
g1_lift_combined's instability.

Distance found via find_far_pose.py (mujoco_transfer/franka/), using the
already-validated MuJoCo Franka model for fast FK-only search (no Isaac Sim
boot needed). FIXED once already: the first search measured distance to the
hand BODY origin, not Isaac Lab's actual "ee_frame" (object_ee_distance's
target -- FrameTransformerCfg, panda_hand + 0.1034m local-Z offset); verified
in Isaac Lab that this gave a wrong distance (0.4719m measured vs 0.5653m
predicted). Re-measured with the offset applied correctly in MuJoCo too --
default Franka pose sits 0.3325m from the nominal cube center (0.5, 0, 0.055)
using the CORRECT ee_frame definition (matches Isaac Lab's own live-viewer
measurement earlier this session, ~0.39m against a randomized, not nominal,
cube position -- consistent). G1's own measured READY_ARM_POSE-to-cube
distance is ~0.555m. interp_0.80 = DEFAULT_POSE + 0.80*(tuck_1 - DEFAULT_POSE)
measured at 0.5572m -- within 2mm of target, used directly.
"""
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import FrankaCubeLiftEnvCfg
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.agents.rsl_rl_ppo_cfg import LiftCubePPORunnerCfg

# Measured via find_far_pose.py (with the ee_frame local-Z offset applied
# correctly): EE-to-nominal-cube-center = 0.5572m (target ~0.555m, matching
# G1's own measured distance).
FAR_ARM_POSE = {
    "panda_joint1": 0.0,
    "panda_joint2": -1.0738,
    "panda_joint3": 0.0,
    "panda_joint4": -2.482,
    "panda_joint5": 0.0,
    "panda_joint6": 2.0474,
    "panda_joint7": 0.741,
    "panda_finger_joint.*": 0.04,
}


@configclass
class FrankaCubeLiftFarEnvCfg(FrankaCubeLiftEnvCfg):
    """Identical to the official FrankaCubeLiftEnvCfg except the robot's
    init_state.joint_pos, which is overridden to FAR_ARM_POSE above -- the ONE
    isolated variable for this ablation. Scene, rewards, action space,
    observation space, num_envs (4096, inherited from LiftEnvCfg's own
    default), episode length, everything else stays exactly as the official
    reference task."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.init_state.joint_pos = dict(FAR_ARM_POSE)


@configclass
class FrankaCubeLiftFarEnvCfg_PLAY(FrankaCubeLiftFarEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False


# FIXED (2026-07-27 16:04): FrankaCubeLiftFarEnvCfg alone (far pose, REACH_STD
# still the reference's own 0.1) confirmed the prediction from g1_lift_combined's
# own REACH_STD investigation -- reaching_object stayed flat at ~0.0000-0.0033
# for the full 1050 iterations observed, action_std spiked to 1.79 (higher than
# even the baseline Franka run's own peak) then decayed without ever finding a
# real gradient. dist/std = 0.557/0.1 = 5.57 -- the same tanh-kernel dead zone
# already diagnosed and fixed for G1 (REACH_STD 0.1->0.3). Testing directly
# whether the SAME fix, at the SAME target distance (0.557m here vs G1's own
# 0.555m -- deliberately matched), also unblocks Franka once moved to G1's
# distance -- i.e. that this was never about G1's embodiment being harder, just
# REACH_STD vs. starting-distance geometry.
@configclass
class FrankaCubeLiftFarFixedEnvCfg(FrankaCubeLiftFarEnvCfg):
    """FrankaCubeLiftFarEnvCfg + REACH_STD widened 0.1->0.3, same fix and same
    target distance as g1_lift_combined's own REACH_STD correction. The ONLY
    difference from FrankaCubeLiftFarEnvCfg."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.reaching_object.params["std"] = 0.3


@configclass
class FrankaCubeLiftFarFixedEnvCfg_PLAY(FrankaCubeLiftFarFixedEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False


# NEW (2026-07-27 16:13): built on top of FrankaCubeLiftFarFixedEnvCfg (far
# pose + REACH_STD=0.3) -- next isolated variable is gripper CONTROL DYNAMICS,
# testing whether Dex1's much softer/less-damped actuator response (vs
# Franka's own tightly-controlled parallel gripper) contributes to the
# difficulty gap, independent of geometry/mesh.
#
# NOT unit-equivalent, flagged deliberately: Franka's finger joints are
# PRISMATIC (linear, meters, range 0-0.04) -- G1's Dex1 finger joints
# (g1_lift_rl/constants.py) are REVOLUTE (angular, radians, range -0.02 to
# +0.0245). Directly transplanting Dex1's raw stiffness=800/damping=3 onto
# Franka's linear joints isn't physically equivalent to Dex1's own real
# behavior (N/m vs N*m/rad). What IS being tested faithfully: the qualitative
# fact that Dex1's control is drastically softer and less damped than
# Franka's own stiffness=2000/damping=100 -- i.e. "what if the gripper's grip
# were much weaker/mushier," independent of the specific mechanism computing
# it. A full Dex1 mesh graft (real units, real geometry, real self-collision
# behavior) would be the faithful follow-up if this approximation alone
# already explains part of the gap.
@configclass
class FrankaCubeLiftFarFixedSoftGripperEnvCfg(FrankaCubeLiftFarFixedEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.actuators["panda_hand"].stiffness = 800.0
        self.scene.robot.actuators["panda_hand"].damping = 3.0


@configclass
class FrankaCubeLiftFarFixedSoftGripperEnvCfg_PLAY(FrankaCubeLiftFarFixedSoftGripperEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False


@configclass
class FrankaFarPPORunnerCfg(LiftCubePPORunnerCfg):
    """Identical to the official LiftCubePPORunnerCfg (same architecture,
    same algorithm hyperparameters) except experiment_name, so logs land in
    their own directory instead of colliding with the baseline Franka run."""

    experiment_name = "franka_lift_far"


@configclass
class FrankaFarSoftGripperPPORunnerCfg(LiftCubePPORunnerCfg):
    experiment_name = "franka_lift_far_soft_gripper"


@configclass
class FrankaFarFixedPPORunnerCfg(LiftCubePPORunnerCfg):
    experiment_name = "franka_lift_far_fixed"


# NEW (2026-07-27 16:46): built on top of FrankaCubeLiftFarFixedSoftGripperEnvCfg
# (far pose + REACH_STD=0.3 + Dex1-like soft gripper -- both already confirmed:
# distance/REACH_STD explains and fully fixes the reach problem; soft gripper
# alone doesn't degrade anything). Next isolated variable: num_envs 4096->2048,
# matching G1's own env count. The earlier attempt to test this on G1 itself
# was confounded (changed entropy_coef at the same time); this is the clean,
# single-variable version, tested on an already-proven-working setup instead
# of a struggling one -- if it still converges cleanly at 2048, that rules out
# exploration budget as an explanation for G1's remaining difficulty.
@configclass
class FrankaCubeLiftFarFixedSoftGripperSmallEnvCfg(FrankaCubeLiftFarFixedSoftGripperEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 2048


@configclass
class FrankaCubeLiftFarFixedSoftGripperSmallEnvCfg_PLAY(FrankaCubeLiftFarFixedSoftGripperSmallEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False


@configclass
class FrankaFarSoftGripperSmallPPORunnerCfg(LiftCubePPORunnerCfg):
    experiment_name = "franka_lift_far_soft_gripper_small"
