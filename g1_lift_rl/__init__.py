# g1_lift_rl/__init__.py
"""G1 + Dex1 lift (Policy 1) -- RL task registration.

Entry points are forward references (strings) -- env_cfg.G1LiftEnvCfg and
agents.rsl_rl_ppo_cfg.G1LiftPPORunnerCfg don't exist yet (Phase 1 / Phase 3), and
don't need to: gym.register() never resolves them until gym.make() is called.
"""

import gymnasium as gym

gym.register(
    id="Isaac-G1-Lift-Ext-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1LiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1LiftPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-G1-Lift-Ext-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1LiftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1LiftPPORunnerCfg",
    },
)

# Policy 2: grasp (verified) + carry to inspection.
gym.register(
    id="Isaac-G1-Policy2-Ext-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_policy2:G1Policy2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1Policy2PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-G1-Policy2-Ext-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_policy2:G1Policy2EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1Policy2PPORunnerCfg",
    },
)

# Policy 3: move to goal + place (release + return_to_ready split off into
# Policy 4 below -- see env_cfg_policy3.py's RewardsCfg docstring).
gym.register(
    id="Isaac-G1-Policy3-Ext-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_policy3:G1Policy3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1Policy3PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-G1-Policy3-Ext-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_policy3:G1Policy3EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1Policy3PPORunnerCfg",
    },
)

# Policy 4: release + return to ready.
gym.register(
    id="Isaac-G1-Policy4-Ext-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_policy4:G1Policy4EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1Policy4PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-G1-Policy4-Ext-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_policy4:G1Policy4EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1Policy4PPORunnerCfg",
    },
)

# Combined policy: single policy, reach+grasp+lift+carry-to-goal, Franka-lift-
# reference reward design -- comparison point against the 4-policy chain above.
gym.register(
    id="Isaac-G1-Lift-Combined-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_combined:G1LiftCombinedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1CombinedPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-G1-Lift-Combined-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_combined:G1LiftCombinedEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1CombinedPPORunnerCfg",
    },
)

# Franka ablation (2026-07-27): the OFFICIAL Franka lift task, unmodified except
# for the robot's starting arm pose (moved to match G1's own reach distance).
# Registered here purely for convenience (piggybacks on this package's already-
# working gym auto-discovery) -- see env_cfg_franka_far.py for the full rationale.
gym.register(
    id="Isaac-Lift-Cube-Franka-Far-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_franka_far:FrankaCubeLiftFarEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.env_cfg_franka_far:FrankaFarPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Lift-Cube-Franka-Far-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_franka_far:FrankaCubeLiftFarEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.env_cfg_franka_far:FrankaFarPPORunnerCfg",
    },
)

# Far pose + REACH_STD 0.1->0.3 (same fix as g1_lift_combined's own REACH_STD
# correction) -- tests whether that alone unblocks reaching at G1's distance.
gym.register(
    id="Isaac-Lift-Cube-Franka-FarFixed-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_franka_far:FrankaCubeLiftFarFixedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.env_cfg_franka_far:FrankaFarFixedPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Lift-Cube-Franka-FarFixed-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_franka_far:FrankaCubeLiftFarFixedEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.env_cfg_franka_far:FrankaFarFixedPPORunnerCfg",
    },
)

# Far pose + REACH_STD fix + Dex1-like soft gripper actuator (stiffness/damping
# only, not a mesh swap -- see env_cfg_franka_far.py for the unit-mismatch caveat).
gym.register(
    id="Isaac-Lift-Cube-Franka-FarFixed-SoftGripper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_franka_far:FrankaCubeLiftFarFixedSoftGripperEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.env_cfg_franka_far:FrankaFarSoftGripperPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Lift-Cube-Franka-FarFixed-SoftGripper-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_franka_far:FrankaCubeLiftFarFixedSoftGripperEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.env_cfg_franka_far:FrankaFarSoftGripperPPORunnerCfg",
    },
)

# Far pose + REACH_STD fix + soft gripper + num_envs 4096->2048 (matching G1's
# own env count) -- isolated exploration-budget test.
gym.register(
    id="Isaac-Lift-Cube-Franka-FarFixed-SoftGripper-Small-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_franka_far:FrankaCubeLiftFarFixedSoftGripperSmallEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.env_cfg_franka_far:FrankaFarSoftGripperSmallPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Lift-Cube-Franka-FarFixed-SoftGripper-Small-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_franka_far:FrankaCubeLiftFarFixedSoftGripperSmallEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.env_cfg_franka_far:FrankaFarSoftGripperSmallPPORunnerCfg",
    },
)

# Arm-swap ablation (2026-07-27): G1's real right arm + Dex1 gripper + proven
# scene, with FRANKA's own literal reward/curriculum/PPO hyperparameters
# (unmodified) -- see env_cfg_g1_franka_parity.py for full rationale.
gym.register(
    id="Isaac-G1-Franka-Parity-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_g1_franka_parity:G1FrankaParityEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.env_cfg_g1_franka_parity:G1FrankaParityPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-G1-Franka-Parity-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_g1_franka_parity:G1FrankaParityEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.env_cfg_g1_franka_parity:G1FrankaParityPPORunnerCfg",
    },
)

# Same as above, PLUS a startup mass/inertia scale-up (1000x) on every
# non-arm body (torso/pelvis/legs/head/left arm) -- approximates Franka's
# true fixed base, correcting the objection that G1FrankaParityEnvCfg reuses
# the whole (floating-base) body. See env_cfg_g1_arm_isolated.py.
gym.register(
    id="Isaac-G1-Arm-Isolated-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_g1_arm_isolated:G1ArmIsolatedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.env_cfg_g1_arm_isolated:G1ArmIsolatedPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-G1-Arm-Isolated-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_g1_arm_isolated:G1ArmIsolatedEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.env_cfg_g1_arm_isolated:G1ArmIsolatedPPORunnerCfg",
    },
)

# Alternative to the above: keeps FRANKA's own robot/cube/reward (built on the
# confirmed FarFixed baseline), relocates just the base+table to G1's exact
# scene geometry -- isolates scene/approach geometry, not embodiment, and
# stays on Franka's clean fixed-base dynamics (no mass hacks). See
# env_cfg_franka_g1_scene.py.
gym.register(
    id="Isaac-Lift-Cube-Franka-G1Scene-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_franka_g1_scene:FrankaCubeLiftG1SceneEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.env_cfg_franka_g1_scene:FrankaG1ScenePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Lift-Cube-Franka-G1Scene-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_franka_g1_scene:FrankaCubeLiftG1SceneEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.env_cfg_franka_g1_scene:FrankaG1ScenePPORunnerCfg",
    },
)

# True isolated-arm ablation (2026-07-28): G1's real right arm + Dex1
# gripper, extracted as a standalone FIXED-BASE USD via Isaac Sim GUI
# surgery (is_fixed_base=True, confirmed), mounted on Franka's own
# unmodified table/cube/reward/curriculum/PPO. See env_cfg_g1_arm_on_franka.py.
gym.register(
    id="Isaac-G1-RightArm-Dex1-OnFrankaTable-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_g1_arm_on_franka:G1RightArmDex1LiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.env_cfg_g1_arm_on_franka:G1RightArmDex1PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-G1-RightArm-Dex1-OnFrankaTable-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_g1_arm_on_franka:G1RightArmDex1LiftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.env_cfg_g1_arm_on_franka:G1RightArmDex1PPORunnerCfg",
    },
)

# Alternate-mount ablation (2026-07-30): same true-fixed-base arm/Franka
# table/cube as above, but mounted upright/natural (identity rotation) near
# Franka's own base position, arm left at its raw USD default joint pose
# (no jogged home pose) -- see env_cfg_g1_arm_on_franka.py's
# G1RightArmDex1UprightLiftEnvCfg docstring.
gym.register(
    id="Isaac-G1-RightArm-Dex1-OnFrankaTable-Upright-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_g1_arm_on_franka:G1RightArmDex1UprightLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.env_cfg_g1_arm_on_franka:G1RightArmDex1UprightPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-G1-RightArm-Dex1-OnFrankaTable-Upright-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_g1_arm_on_franka:G1RightArmDex1UprightLiftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.env_cfg_g1_arm_on_franka:G1RightArmDex1UprightPPORunnerCfg",
    },
)

# RGP chain: reach -> grasp+lift(7cm) -> place+release. Independent of the
# old 4-policy chain above (env_cfg.py / env_cfg_policy2/3/4.py), which stays
# untouched as reference. See README.md for the full chain layout.
gym.register(
    id="Isaac-G1-RGP-Reach-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_rgp_reach:G1RGPReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg_rgp:G1RGPReachPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-G1-RGP-Reach-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_rgp_reach:G1RGPReachEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg_rgp:G1RGPReachPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-G1-RGP-Grasp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_rgp_grasp:G1RGPGraspEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg_rgp:G1RGPGraspPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-G1-RGP-Grasp-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_rgp_grasp:G1RGPGraspEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg_rgp:G1RGPGraspPPORunnerCfg",
    },
)
