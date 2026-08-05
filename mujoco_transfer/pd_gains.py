"""PD gains and default joint positions for the MuJoCo G1+Dex1 replica, matching
Policy 1's actual trained actuator config exactly (env_cfg.py's G1_DEX1_CFG,
verified against Isaac Lab's own printed joint-info table).

Three actuator groups, by stiffness/damping:
  - "body" (most joints, incl. legs and arms): stiffness=150, damping=10
  - "waist" (yaw/roll/pitch): stiffness=10000, damping=10000 -- this session's
    fix for the torso-sag bug (see env_cfg.py's actuator comment)
  - "gripper" (all 4 hand joints): stiffness=800, damping=3

Default joint positions match _DEFAULT_JOINTS in env_cfg.py: legs/waist at
0.0, gripper at GRIPPER_OPEN, right arm at READY_ARM_POSE, left arm at
LEFT_ARM_STOW. This is the pose the robot rests at absent any policy action --
Policy 1 only ever commands the 7 right-arm joints + 1 gripper scalar; every
other joint just sits held at these defaults by its PD spring the whole
episode, exactly as during training.
"""
from joint_mapping import MUJOCO_JOINT_ORDER

GRIPPER_OPEN = -0.02  # matches constants.py

READY_ARM_POSE = {
    "right_shoulder_pitch_joint": 0.60,
    "right_shoulder_roll_joint": -0.20,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.80,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.40,
    "right_wrist_yaw_joint": 0.0,
}

LEFT_ARM_STOW = {
    "left_shoulder_pitch_joint": 0.3,
    "left_shoulder_roll_joint": 0.25,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.97,
    "left_wrist_roll_joint": 0.15,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
}

ARM_JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# Right hand uses the real Dex1 reconstruction's joint names (match Isaac Lab
# directly); left hand still uses the substitute URDF's hand.
_GRIPPER_MJC_NAMES = [
    "left_dex1_finger_joint_1", "left_dex1_finger_joint_2",
    "right_hand_Joint1_1", "right_hand_Joint2_1",
]
_WAIST_MJC_NAMES = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]


def _default_pos_for(name: str) -> float:
    if name in READY_ARM_POSE:
        return READY_ARM_POSE[name]
    if name in LEFT_ARM_STOW:
        return LEFT_ARM_STOW[name]
    if name in _GRIPPER_MJC_NAMES:
        return GRIPPER_OPEN
    # legs, waist: default 0.0
    return 0.0


def _stiffness_damping_for(name: str) -> tuple[float, float]:
    if name in _WAIST_MJC_NAMES:
        return 10000.0, 10000.0
    if name in _GRIPPER_MJC_NAMES:
        return 800.0, 3.0
    return 150.0, 10.0  # "body" group: legs + all arm joints


# All three arrays are in MUJOCO_JOINT_ORDER (index-aligned with m.qpos etc.).
DEFAULT_QPOS_MJC = [_default_pos_for(n) for n in MUJOCO_JOINT_ORDER]
STIFFNESS_MJC = [_stiffness_damping_for(n)[0] for n in MUJOCO_JOINT_ORDER]
DAMPING_MJC = [_stiffness_damping_for(n)[1] for n in MUJOCO_JOINT_ORDER]

if __name__ == "__main__":
    for i, n in enumerate(MUJOCO_JOINT_ORDER):
        print(f"  mjc[{i:2d}]={n:28s} default={DEFAULT_QPOS_MJC[i]:+.4f}"
              f"  Kp={STIFFNESS_MJC[i]:8.1f}  Kd={DAMPING_MJC[i]:7.1f}")
