# g1_lift_rl/env_cfg_g1_arm_isolated.py
"""Arm-swap ablation, mass-override variant (2026-07-27): same as
env_cfg_g1_franka_parity.py (G1's real right arm + Dex1 gripper, Franka's
literal reward/curriculum/PPO hyperparameters), PLUS a startup-time mass/
inertia override on every body that is NOT the right arm/gripper, to correct
the physical objection raised when viewing G1FrankaParityEnvCfg's scene:
reusing G1's whole body means the right arm still has to react against a
real floating-base pelvis/torso/legs/other-arm mass it would never see as a
dedicated fixed-base arm (Franka's actual setup -- root welded to world via
a 0-DOF fixed joint, so reaction forces there produce exactly zero motion by
construction, not merely "held down by strong PD gains").

G1's pelvis has no such joint -- it's a genuine free 6-DOF rigid body, only
kept upright by the leg/waist actuators' PD springs (env_cfg.py's "body"/
"waist" actuator groups). No mass value can turn a floating joint into a
fixed one, but scaling the non-arm bodies' mass+inertia up by a large factor
(here 1000x) makes them a good practical approximation: given the force
magnitudes a 7-DOF arm can realistically generate, a 1000x-heavier torso
accelerates negligibly over an 8s episode, so the coupled multibody dynamics
should approach the fixed-base limit Franka gets structurally for free.

Deliberately the OPPOSITE of an earlier (wrong) proposal to zero out this
mass: a near-zero-mass body reacts explosively to any joint torque
(F=ma with m->0) and is the physically backwards direction for "removing a
body's ability to resist/absorb the arm's reaction forces" -- scaling mass
UP is what emulates a fixed anchor, not scaling it down.

If 1000x turns out to destabilize the sim (checked via the same zero-agent
sanity script used throughout this project), back off the factor -- this is
an approximation, not a structural fix, and may need retuning like every
other constant in this project.
"""
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
from .env_cfg import EventCfg
from .env_cfg_g1_franka_parity import (
    CurriculumCfg,
    G1FrankaParityEnvCfg,
    G1FrankaParityPPORunnerCfg,
    RewardsCfg,
)

# Everything except the right arm chain (ARM_JOINTS' link-side bodies) and the
# right Dex1 gripper (HAND_BASE_LINK / TABLE_CLEARANCE_LINKS in constants.py) --
# confirmed body names: leg/waist/torso/head/sensor links from the G1 URDF link
# list (unitree_ros/robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf),
# left-hand Dex1 links assumed symmetric with the right hand's names verified
# directly in this scene (check_hand_geometry_v2.py) -- confirm both resolve
# with zero leftover/unmatched names before trusting this list (see
# view_g1_arm_isolated_masses.py).
NON_ARM_BODY_NAMES = [
    "pelvis", "pelvis_contour_link",
    "left_hip_.*_link", "right_hip_.*_link",
    "left_knee_link", "right_knee_link",
    "left_ankle_.*_link", "right_ankle_.*_link",
    "waist_.*_link",
    "torso_link", "logo_link", "head_link",
    "imu_in_torso", "imu_in_pelvis", "d435_link", "mid360_link",
    "left_shoulder_.*_link", "left_elbow_link", "left_wrist_.*_link",
    "left_hand_base_link", "left_hand_Link.*",
]

MASS_SCALE_FACTOR = 1000.0


@configclass
class IsolatedEventCfg(EventCfg):
    """env_cfg.py's EventCfg (reset_robot, reset_object) plus a one-time
    startup mass/inertia scale-up on every non-arm body."""

    anchor_non_arm_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=NON_ARM_BODY_NAMES),
            "mass_distribution_params": (MASS_SCALE_FACTOR, MASS_SCALE_FACTOR),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )


@configclass
class G1ArmIsolatedEnvCfg(G1FrankaParityEnvCfg):
    events = IsolatedEventCfg()


@configclass
class G1ArmIsolatedEnvCfg_PLAY(G1ArmIsolatedEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False


@configclass
class G1ArmIsolatedPPORunnerCfg(G1FrankaParityPPORunnerCfg):
    experiment_name = "g1_arm_isolated"


# Re-exported so __init__.py's gym.register kwargs (which reference this
# module's own RewardsCfg/CurriculumCfg by name via string entry points) don't
# need to reach back into env_cfg_g1_franka_parity directly.
__all__ = [
    "RewardsCfg",
    "CurriculumCfg",
    "NON_ARM_BODY_NAMES",
    "MASS_SCALE_FACTOR",
    "IsolatedEventCfg",
    "G1ArmIsolatedEnvCfg",
    "G1ArmIsolatedEnvCfg_PLAY",
    "G1ArmIsolatedPPORunnerCfg",
]
