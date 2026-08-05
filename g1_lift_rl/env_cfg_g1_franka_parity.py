# g1_lift_rl/env_cfg_g1_franka_parity.py
"""ARM-SWAP ABLATION (2026-07-27): G1's real right arm + Dex1 gripper +
g1_lift_ext's own proven scene, wired up with the OFFICIAL Franka reward
weights/formulas and OFFICIAL Franka curriculum, UNMODIFIED -- the same
reward/curriculum treatment the Franka-Far/FarFixed/SoftGripper/Small chain
used throughout, none of which had g1_lift_combined's own later additions
(is_grasping gate, close_gradient/early_close, delayed/ramped curriculum).

Why: three variables have now been isolated and tested on Franka's own task,
pushed out to G1's exact reach distance --
  - REACH_STD/distance mismatch: CONFIRMED as the real cause of the original
    reach dead-zone, and fully fixes it once corrected.
  - Gripper actuator softness (Dex1-like stiffness=800/damping=3): RULED OUT,
    no measurable degradation.
  - num_envs (4096->2048, matching G1's own count): RULED OUT, converges
    cleanly just somewhat slower.
None of these reproduce g1_lift_combined's own persistent problems (still-zero
lifting_object even with REACH_STD fixed; deep negative reward crashes at the
curriculum transition that Franka's own task never showed even under the
identical 1000x jump -- Franka's own worst dip there was Mean reward 88.8->57.4,
recovering within ~30-40 iterations, never negative). This is the final,
most direct test: does G1's REAL arm+gripper, under the EXACT SAME (unmodified,
ungated) reward/curriculum Franka succeeds with, also succeed -- or does it
reproduce g1_lift_combined's failure mode, isolating embodiment as the cause.

What's reused unchanged from g1_lift_ext's own proven setup (env_cfg.py):
LiftSceneCfg (robot base pose, table, G1_DEX1_CFG's real actuator gains),
ActionsCfg, ObservationsCfg (39-dim, same as g1_lift_combined), EventCfg
(reset to READY_ARM_POSE, ~0.555m from the cube -- already matching the
distance this whole ablation chain has been testing at).

What's deliberately NOT matched to Franka (disclosed, not swept under the
rug): cube size stays G1's own 6cm (not Franka's 4.8cm DexCube) -- changing
geometry on top of embodiment would reintroduce a second variable. Goal
stays FIXED (INSPECT_POS, via object_to_inspect in the observation) rather
than Franka's own randomized UniformPoseCommandCfg -- building a randomized-
goal-command system for G1 is a separate feature, not essential to isolate
arm dynamics specifically.
"""
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

from . import mdp
from .env_cfg import ActionsCfg, EventCfg, LiftSceneCfg, ObservationsCfg, TerminationsCfg


@configclass
class RewardsCfg:
    """Franka's own literal reward weights (LiftEnvCfg.RewardsCfg), ungated
    versions -- the exact reward design the whole Far/FarFixed/SoftGripper/
    Small ablation chain trained under, unmodified."""

    reaching_object = RewTerm(func=mdp.reward_reach, weight=1.0)
    lifting_object = RewTerm(func=mdp.reward_object_lifted_ungated, weight=15.0)
    object_goal_tracking = RewTerm(func=mdp.reward_goal_tracking_coarse_ungated, weight=16.0)
    object_goal_tracking_fine_grained = RewTerm(func=mdp.reward_goal_tracking_fine_ungated, weight=5.0)
    action_rate = RewTerm(func=mdp.penalty_action_rate, weight=-1.0e-4)
    joint_vel = RewTerm(func=mdp.penalty_joint_vel, weight=-1.0e-4)


@configclass
class CurriculumCfg:
    """Franka's own literal curriculum (LiftEnvCfg.CurriculumCfg) -- instant
    1000x jump at 10,000 common-steps, NOT g1_lift_combined's delayed/ramped
    fix. Deliberately unmodified: the point of this experiment is to see
    whether G1's real arm reacts to this exact treatment the way Franka does
    (mild dip, quick recovery) or the way g1_lift_combined's earlier runs did
    (deep negative crash, slow recovery)."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000},
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000},
    )


@configclass
class G1FrankaParityEnvCfg(ManagerBasedRLEnvCfg):
    scene = LiftSceneCfg(num_envs=2048, env_spacing=2.5, replicate_physics=True)
    observations = ObservationsCfg()
    actions = ActionsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    events = EventCfg()
    commands = None
    curriculum = CurriculumCfg()

    def __post_init__(self):
        # Same sim/control settings as the rest of g1_lift_ext (100Hz control,
        # 8s episodes) -- deliberately NOT matched to Franka's own 50Hz/5s here,
        # since control-rate/episode-length is a still-untested, separate
        # variable, not the one this experiment isolates (embodiment).
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
class G1FrankaParityEnvCfg_PLAY(G1FrankaParityEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False


@configclass
class G1FrankaParityPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Franka's own literal PPO hyperparameters (LiftCubePPORunnerCfg,
    config/franka/agents/rsl_rl_ppo_cfg.py) -- same architecture, same
    algorithm settings, only experiment_name differs. Matches the whole
    Far/FarFixed/SoftGripper/Small chain, not g1_lift_combined's own PPO cfg
    (which uses G1's chain's own values: init_noise_std=0.6, entropy_coef=
    0.004, learning_rate=1e-3, gamma=0.99 -- all different from Franka's)."""

    num_steps_per_env = 24
    max_iterations = 1500
    resume = False
    save_interval = 50
    experiment_name = "g1_franka_parity"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
