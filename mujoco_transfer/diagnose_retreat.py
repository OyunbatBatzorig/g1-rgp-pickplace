"""Fine-grained per-step diagnostic: why does the arm reach close to the cube
early (min dist 3.83cm) then retreat to a farther stable point (10.49cm) by
the end, instead of holding at the close point like it does in Isaac Lab?
Logs hand_base_to_object magnitude alongside ee_to_object distance, every
single step, to see if they correlate with the retreat.
"""
import sys
import mujoco
import numpy as np
import torch

sys.path.insert(0, ".")
from joint_mapping import MUJOCO_JOINT_ORDER
from pd_gains import DEFAULT_QPOS_MJC, ARM_JOINTS, GRIPPER_OPEN

GRIPPER_CLOSE = 0.0245
INSPECT_POS = np.array([-0.037, -0.222, 0.997])
ARM_MJC_IDS = [MUJOCO_JOINT_ORDER.index(n) for n in ARM_JOINTS]
EE_BODY_NAMES = ["right_dex1_finger_link_1", "right_dex1_finger_link_2"]
HAND_BASE_BODY_NAME = "right_wrist_yaw_link"
GRIPPER_MJC_NAMES = ["right_dex1_finger_joint_1", "right_dex1_finger_joint_2"]
GRIPPER_MJC_IDS = [MUJOCO_JOINT_ORDER.index(n) for n in GRIPPER_MJC_NAMES]

m = mujoco.MjModel.from_xml_path("g1_lift_scene.xml")
d = mujoco.MjData(m)
ee_body_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in EE_BODY_NAMES]
hand_base_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, HAND_BASE_BODY_NAME)
object_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")

ee_tip_local_offset = []
for bid in ee_body_ids:
    geom_ids = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid]
    tip_geom = max(geom_ids, key=lambda g: np.linalg.norm(m.geom_pos[g]))
    ee_tip_local_offset.append(m.geom_pos[tip_geom].copy())

default_qpos = np.array(DEFAULT_QPOS_MJC)
d.qpos[:33] = default_qpos
d.ctrl[:33] = default_qpos
mujoco.mj_forward(m, d)

policy = torch.jit.load(
    "/home/virtual-acc/projects/g1_lift_ext/logs/rsl_rl/g1_lift_policy1/2026-07-16_09-18-57/exported/policy.pt",
    map_location="cpu")
policy.eval()

last_action = np.zeros(8, dtype=np.float32)


def build_observation():
    arm_joint_pos_rel = d.qpos[ARM_MJC_IDS] - default_qpos[ARM_MJC_IDS]
    arm_joint_vel = d.qvel[ARM_MJC_IDS]
    gripper_joint_pos = d.qpos[GRIPPER_MJC_IDS]
    object_pos = d.xpos[object_body_id].copy()
    tip_world = [
        d.xpos[bid] + d.xmat[bid].reshape(3, 3) @ local_offset
        for bid, local_offset in zip(ee_body_ids, ee_tip_local_offset)
    ]
    ee_pos = np.mean(tip_world, axis=0)
    ee_to_object = object_pos - ee_pos
    object_to_inspect = INSPECT_POS - object_pos
    hand_base_pos = d.xpos[hand_base_body_id].copy()
    hand_base_to_object = object_pos - hand_base_pos
    obs = np.concatenate([
        arm_joint_pos_rel, arm_joint_vel, gripper_joint_pos,
        object_pos, ee_pos, ee_to_object, object_to_inspect, hand_base_to_object,
        last_action,
    ]).astype(np.float32)
    return obs, ee_pos, object_pos, hand_base_pos


print(f"{'step':>5} {'ee_dist':>8} {'hbase_dist':>10} {'hbase_to_obj_xyz':>28} {'action[:7]_norm':>16}")
for t in range(200):
    obs, ee_pos, object_pos, hand_base_pos = build_observation()
    with torch.no_grad():
        action = policy(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
    last_action[:] = action

    target_arm = default_qpos[ARM_MJC_IDS] + action[:7] * 0.5
    d.ctrl[ARM_MJC_IDS] = target_arm
    gripper_target = GRIPPER_OPEN if action[7] >= 0 else GRIPPER_CLOSE
    d.ctrl[GRIPPER_MJC_IDS] = gripper_target
    for _ in range(2):
        mujoco.mj_step(m, d)

    ee_dist = np.linalg.norm(ee_pos - object_pos)
    hbase_vec = object_pos - hand_base_pos
    hbase_dist = np.linalg.norm(hbase_vec)
    if t < 30 or t % 10 == 0:
        print(f"{t:5d} {ee_dist:8.4f} {hbase_dist:10.4f} "
              f"[{hbase_vec[0]:+.3f},{hbase_vec[1]:+.3f},{hbase_vec[2]:+.3f}]      "
              f"{np.linalg.norm(action[:7]):16.4f}")
