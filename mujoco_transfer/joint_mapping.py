"""Isaac Lab <-> MuJoCo joint mapping for the G1+Dex1 sim2sim transfer.

Isaac Lab's order comes from how PhysX walks the USD's scene graph when
building the articulation -- not grouped by limb (interleaves left/right/waist
almost arbitrarily). MuJoCo's order comes from the URDF's kinematic tree,
walked depth-first from the root (pelvis) outward -- naturally grouped by limb.
Neither array is self-describing at runtime (both are just 33 raw floats), so
this mapping has to be built once by matching joint NAMES, then applied every
step to translate observations (MuJoCo -> Isaac Lab order) and actions (Isaac
Lab -> MuJoCo order).

Isaac Lab order captured directly from Isaac Sim's own internal log
(articulation.py's "Joint names:" printout, /tmp/isaaclab/logs/..., since the
listing script's own print() output was lost to a stdout-buffering bug --
simulation_app.close() exits before Python's normal flush-on-exit runs. Same
bug already hit once with chain_stage.py this session.).
"""

# Order exactly as printed by Isaac Lab's articulation.py at env creation time.
ISAAC_LAB_JOINT_ORDER = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
    "left_hand_Joint1_1", "left_hand_Joint2_1",
    "right_hand_Joint1_1", "right_hand_Joint2_1",
]
assert len(ISAAC_LAB_JOINT_ORDER) == 33

# Order as loaded from g1_dex1_patched.urdf into MuJoCo (verified via
# mujoco.mj_id2name over m.njnt).
MUJOCO_JOINT_ORDER = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint", "left_dex1_finger_joint_1", "left_dex1_finger_joint_2",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint", "right_hand_Joint1_1", "right_hand_Joint2_1",
]
assert len(MUJOCO_JOINT_ORDER) == 33

# RE-ENABLED real hand (build_scene.py's replace_right_hand_with_real_geometry):
# its joint names match Isaac Lab exactly (right_hand_Joint1_1/2_1), so no
# translation entry is needed for the right hand anymore. Left hand still uses
# the substitute URDF's hand (never observed/actuated by any of the 4
# policies -- always a static stowed pose), so it still needs one.
ISAAC_TO_MJC_NAME = {
    "left_hand_Joint1_1": "left_dex1_finger_joint_1",
    "left_hand_Joint2_1": "left_dex1_finger_joint_2",
}


def _mjc_name(isaac_name: str) -> str:
    return ISAAC_TO_MJC_NAME.get(isaac_name, isaac_name)


# isaac_to_mjc[i] = index into MUJOCO_JOINT_ORDER for ISAAC_LAB_JOINT_ORDER[i].
# mjc_to_isaac[j] = index into ISAAC_LAB_JOINT_ORDER for MUJOCO_JOINT_ORDER[j].
_mjc_index_of = {name: i for i, name in enumerate(MUJOCO_JOINT_ORDER)}
_isaac_index_of = {name: i for i, name in enumerate(ISAAC_LAB_JOINT_ORDER)}

ISAAC_TO_MJC = [_mjc_index_of[_mjc_name(name)] for name in ISAAC_LAB_JOINT_ORDER]
MJC_TO_ISAAC = [None] * 33
for isaac_i, mjc_i in enumerate(ISAAC_TO_MJC):
    MJC_TO_ISAAC[mjc_i] = isaac_i
assert all(x is not None for x in MJC_TO_ISAAC)
assert [ISAAC_TO_MJC[j] for j in MJC_TO_ISAAC] == list(range(33))  # sanity: round-trips

if __name__ == "__main__":
    print("ISAAC_TO_MJC:", ISAAC_TO_MJC)
    print("MJC_TO_ISAAC:", MJC_TO_ISAAC)
    for i, name in enumerate(ISAAC_LAB_JOINT_ORDER):
        print(f"  isaac[{i:2d}]={name:28s} -> mjc[{ISAAC_TO_MJC[i]:2d}]={MUJOCO_JOINT_ORDER[ISAAC_TO_MJC[i]]}")
