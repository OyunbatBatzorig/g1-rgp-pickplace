# g1_lift_rl/env_cfg_combined.py
"""COMBINED POLICY: reach + grasp + lift + carry to goal, as ONE policy.

Comparison point against the 4-policy chain (env_cfg.py / env_cfg_policy{2,3,4}.py):
same robot/scene/action-space/observation-space/starting-pose/termination logic as
Policy 1 (imported unchanged from env_cfg.py), but the reward design is ported from
IsaacLab's own official reference task (manipulation/lift/mdp/rewards.py, built for
Franka) instead of this project's custom multi-term ladder -- tanh-kernel reach,
threshold lift bonus, goal-tracking gated on lift success (coarse + fine-grained).

Starts every episode from READY_ARM_POSE (the neutral reset already baked into
env_cfg.py's _DEFAULT_JOINTS) -- the harder, more informative test of whether this
reward design can learn the full reach+grasp+lift+carry sequence end to end, from
the same starting point the chain's own Policy 1 already struggles from. Goal is the
fixed, already-verified INSPECT_POS (not a randomized command like the reference --
no need to generalize across goals for this deployment).

If this works, the plan is to extend THIS policy's reward set with more stages
(place at GOAL_POS, release) rather than training new policies -- this file is the
first version only.
"""
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from . import mdp
from .env_cfg import ActionsCfg, EventCfg, LiftSceneCfg, ObservationsCfg, TerminationsCfg

# ---------------------------------------------------------------------------
# Rewards: ported from IsaacLab's Franka lift reference task, literal weights
# (first pass, unvalidated for G1 -- expect these may need retuning once a
# training curve exists, same as every other constant in this project). See
# mdp/rewards.py's "Combined policy" section for the reward functions themselves
# and the LIFT_MINIMAL_HEIGHT re-basing note (the reference's minimal_height=0.04
# is absolute world z there; re-anchored onto this scene's own table height).
# ---------------------------------------------------------------------------
@configclass
class RewardsCfg:
    reaching_object = RewTerm(func=mdp.reward_reach, weight=1.0)
    lifting_object = RewTerm(func=mdp.reward_object_lifted, weight=15.0)
    object_goal_tracking = RewTerm(func=mdp.reward_goal_tracking_coarse, weight=16.0)
    object_goal_tracking_fine_grained = RewTerm(func=mdp.reward_goal_tracking_fine, weight=5.0)
    # FIXED (2026-07-27 11:51): added close_gradient + early_close. Neither the
    # reference nor our first-pass port had anything governing the GRIPPER's own
    # timing -- confirmed via live GUI + a deterministic single-episode replay
    # (diagnose_combined_checkpoint.py, checkpoint 2026-07-27_10-20-28/model_299)
    # that the policy kept the gripper closed the whole episode and used it to
    # bump/pry the cube from above rather than reaching open-handed and closing
    # only once aligned -- is_grasping()'s distance+closed check can still be
    # satisfied by a closed-fist graze, not just a real envelop grasp. Both terms
    # reused UNCHANGED (weights included) from Policy 2's own already-tuned
    # RewardsCfg (env_cfg_policy2.py), the only other place in this project that
    # actually rewards the grasp transition itself, not guessed fresh.
    close_gradient = RewTerm(func=mdp.reward_close_gradient, weight=1.0)
    early_close = RewTerm(func=mdp.penalty_early_close, weight=-0.5)
    action_rate = RewTerm(func=mdp.penalty_action_rate, weight=-1.0e-4)
    joint_vel = RewTerm(func=mdp.penalty_joint_vel, weight=-1.0e-4)


# ---------------------------------------------------------------------------
# Curriculum.
#
# FIXED (2026-07-27 14:31): was an instant 1000x step (-1e-4 -> -1e-1) at 10,000
# common-steps (~iteration 417, matching the Franka reference exactly). Three
# separate G1 runs all crashed hard at exactly this point (total reward -1.8,
# -5.2, then -11 in a run with MORE exploration budget, not less) -- root-caused
# by comparing against the Franka reference's own training curve: by iteration
# 417 Franka's lifting_object is already mature (surging since iteration ~150),
# so the penalty just has to polish an already-good policy. G1's reaching_object
# is, in every run, STILL climbing toward its own peak at iteration ~380-410 --
# the curriculum was yanking the smoothness penalty tight while G1's policy was
# still mid-discovery, not mid-polish. That more exploration made the crash
# WORSE (not better) is the key evidence: more entropy/envs means the policy is
# even less settled at any fixed iteration, so there's more in-flight
# exploratory behavior to disrupt when the penalty locks in -- this rules out
# "just needs more data" and points squarely at timing.
#
# Fix: delay the ramp until iteration ~900 (past the 600-870 window where real
# lift attempts have actually first appeared in this project's own runs) and
# spread it over 500 iterations (900->1400) instead of an instant jump, so
# whenever it does land, it's not a cliff. Uses mdp.modify_term_cfg (Isaac Lab's
# generic term-cfg curriculum, simplified "rewards.<term>.<attr>" addressing)
# instead of mdp.modify_reward_weight, since the latter only supports a hard
# step, no interpolation.
# ---------------------------------------------------------------------------
RAMP_START_STEP = 900 * 24   # iteration ~900 (num_steps_per_env=24)
RAMP_END_STEP = 1400 * 24    # iteration ~1400 -- full penalty only near the end
SMOOTHNESS_START_WEIGHT = -1.0e-4
SMOOTHNESS_END_WEIGHT = -1.0e-1


def _ramp_reward_weight(env, env_ids, data, start_step, end_step, start_weight, end_weight):
    """Linear ramp from start_weight to end_weight over [start_step, end_step] of
    env.common_step_counter -- replaces modify_reward_weight's instant jump."""
    step = env.common_step_counter
    if step <= start_step:
        return mdp.modify_term_cfg.NO_CHANGE
    if step >= end_step:
        return end_weight
    frac = (step - start_step) / (end_step - start_step)
    return start_weight + frac * (end_weight - start_weight)


@configclass
class CurriculumCfg:
    action_rate = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.action_rate.weight",
            "modify_fn": _ramp_reward_weight,
            "modify_params": {
                "start_step": RAMP_START_STEP, "end_step": RAMP_END_STEP,
                "start_weight": SMOOTHNESS_START_WEIGHT, "end_weight": SMOOTHNESS_END_WEIGHT,
            },
        },
    )
    joint_vel = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.joint_vel.weight",
            "modify_fn": _ramp_reward_weight,
            "modify_params": {
                "start_step": RAMP_START_STEP, "end_step": RAMP_END_STEP,
                "start_weight": SMOOTHNESS_START_WEIGHT, "end_weight": SMOOTHNESS_END_WEIGHT,
            },
        },
    )


@configclass
class G1LiftCombinedEnvCfg(ManagerBasedRLEnvCfg):
    # FIXED (2026-07-27 13:37), REVERTED (2026-07-27 14:31): tried 2048->4096
    # (matching Franka's own scale) to test whether more exploration data per
    # iteration would close the gap to Franka's clean convergence. It didn't --
    # a fresh run at 4096 envs + entropy_coef=0.006 produced the DEEPEST
    # curriculum-transition crash yet (-11 vs -1.8/-5.2 at 2048 envs) and never
    # recovered reach performance (parked at 31cm from the cube vs 9.2cm
    # before). That result is what pointed at the real root cause -- see
    # CurriculumCfg below -- so reverting this (and entropy_coef, in
    # agents/rsl_rl_ppo_cfg.py) back to the original values keeps the
    # curriculum-timing fix as the one isolated variable being tested next.
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
        # 8s episodes) -- no reason to switch to the reference's own 50Hz/5s.
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
class G1LiftCombinedEnvCfg_PLAY(G1LiftCombinedEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
