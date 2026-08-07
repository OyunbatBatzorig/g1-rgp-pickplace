# g1_lift_rl/agents/rsl_rl_ppo_cfg_rgp.py
"""PPO runner configs for the RGP 4-policy chain."""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class G1RGPReachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    resume = False
    save_interval = 50
    experiment_name = "g1_rgp_reach"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.6,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.004,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1RGPGraspPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Policy 2 (grasp + lift), fresh training from scratch."""
    num_steps_per_env = 24
    max_iterations = 1500
    resume = False
    save_interval = 50
    experiment_name = "g1_rgp_grasp"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.6,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.004,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1RGPPlacePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Policy 3 (move to goal + place, gripper stays closed -- no release;
    release is a separate Policy 4), fresh training from scratch.

    entropy_coef/init_noise_std raised above Policy 1/2's own values (0.004 /
    0.6): a first run with those settings showed the cube moving 0.237m -> 0.15m
    from goal by step 49, then holding dead flat there for the rest of every
    episode (settle/place stuck at exactly 0.0, never once triggered) while
    action_std had already collapsed 0.6 -> 0.09 by iteration ~400 -- verified
    directly against the checkpoint (dist_to_goal, not just the training
    curve), not inferred from the curve alone. Reads as premature exploration
    collapse around a "close enough to dodge the carry/table penalties"
    resting point well outside SETTLE_NEAR_RADIUS_RGP (0.06m), rather than a
    reward-wiring bug (place/carry/table_clearance all read the correct
    values at that resting state)."""
    num_steps_per_env = 24
    max_iterations = 1500
    resume = False
    save_interval = 50
    experiment_name = "g1_rgp_place"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.012,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1RGPReleasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Policy 4 (release + return to READY_ARM_POSE), fresh training from
    scratch. Starts at Policy 1/2's own baseline entropy_coef/init_noise_std
    (0.004/0.6), not Policy 3's raised values (0.012/0.8) -- those were a
    response to a reward-shaping bug specific to Policy 3's own penalty
    gating (see mdp/rewards_rgp.py's CARRY_PENALTY_GATE_RADIUS_RGP comment),
    not a general property of this codebase's PPO settings. Revisit only if
    Policy 4 shows the same plateau-with-collapsing-action_std signature."""
    num_steps_per_env = 24
    max_iterations = 1500
    resume = False
    save_interval = 50
    experiment_name = "g1_rgp_release"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.6,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.004,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
