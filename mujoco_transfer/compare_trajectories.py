"""Compares Policy 1's Isaac Lab rollout (trajectory_isaac.npz, from
capture_isaac_trajectory.py) against its MuJoCo rollout (trajectory_mujoco_final.npz,
from run_policy1.py --save_trajectory), both fed the SAME exported policy.pt
and (as close as possible) the same cube spawn position -- isolating physics/
geometry reconstruction differences as the source of any behavioral gap.

obs layout (39,), identical index order on both sides (see build_observation()
in run_policy1.py / ObservationsCfg in env_cfg.py):
  0:7   arm_joint_pos_rel      7:14  arm_joint_vel        14:16 gripper_joint_pos
  16:19 object_pos             19:22 ee_pos                22:25 ee_to_object
  25:28 object_to_inspect      28:31 hand_base_to_object   31:39 last_action

Policy 1 never closes its gripper by design (grasping is Policy 2's job --
penalty_early_close exists specifically to stop Policy 1 from closing early),
so the diagnostic question here is NOT "does it grasp" but "does the arm
converge to PRE_GRASP_ARM_POSE, at the same EE-to-object distance, the way
Isaac Lab's does" -- i.e. is the reach-and-hold behavior itself preserved.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARM_JOINTS = [
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]
PRE_GRASP_ARM_POSE = np.array([0.1751, 0.0587, 0.1028, -0.3644, 0.1334, 0.5374, 0.1710])

isaac = np.load("trajectory_isaac.npz")
mjc = np.load("trajectory_mujoco_final.npz")

obs_i, act_i, arm_i, cube0_i = isaac["obs"], isaac["action"], isaac["arm_abs"], isaac["cube_pos0"]
obs_m, act_m, arm_m, cube0_m = mjc["obs"], mjc["action"], mjc["arm_abs"], mjc["cube_pos0"]

n = min(len(obs_i), len(obs_m))
obs_i, act_i, arm_i = obs_i[:n], act_i[:n], arm_i[:n]
obs_m, act_m, arm_m = obs_m[:n], act_m[:n], arm_m[:n]
t = np.arange(n) * 0.01  # seconds, decimation=2 * sim.dt=0.005

print(f"cube spawn -- isaac: {cube0_i}  mujoco: {cube0_m}  "
      f"(offset {np.linalg.norm(cube0_i - cube0_m)*100:.2f}cm)")

ee_to_object_i = np.linalg.norm(obs_i[:, 22:25], axis=1)
ee_to_object_m = np.linalg.norm(obs_m[:, 22:25], axis=1)
hand_base_to_object_i = np.linalg.norm(obs_i[:, 28:31], axis=1)
hand_base_to_object_m = np.linalg.norm(obs_m[:, 28:31], axis=1)
gripper_i = obs_i[:, 14:16].mean(axis=1)
gripper_m = obs_m[:, 14:16].mean(axis=1)
arm_action_norm_i = np.linalg.norm(act_i[:, :7], axis=1)
arm_action_norm_m = np.linalg.norm(act_m[:, :7], axis=1)
object_drift_i = np.linalg.norm(obs_i[:, 16:19] - obs_i[0, 16:19], axis=1)
object_drift_m = np.linalg.norm(obs_m[:, 16:19] - obs_m[0, 16:19], axis=1)
pregrasp_dist_i = np.linalg.norm(arm_i - PRE_GRASP_ARM_POSE, axis=1)
pregrasp_dist_m = np.linalg.norm(arm_m - PRE_GRASP_ARM_POSE, axis=1)

print(f"\n{'metric':32s} {'isaac final':>12s} {'mjc final':>12s} {'isaac min':>10s} {'mjc min':>10s}")
for name, i_arr, m_arr in [
    ("ee_to_object dist (m)", ee_to_object_i, ee_to_object_m),
    ("hand_base_to_object dist (m)", hand_base_to_object_i, hand_base_to_object_m),
    ("gripper_joint_pos (mean)", gripper_i, gripper_m),
    ("arm_action_norm", arm_action_norm_i, arm_action_norm_m),
    ("object drift from spawn (m)", object_drift_i, object_drift_m),
    ("dist to PRE_GRASP_ARM_POSE (rad)", pregrasp_dist_i, pregrasp_dist_m),
]:
    print(f"{name:32s} {i_arr[-1]:12.4f} {m_arr[-1]:12.4f} {i_arr.min():10.4f} {m_arr.min():10.4f}")

print(f"\nper-joint final position vs PRE_GRASP_ARM_POSE:")
print(f"{'joint':24s} {'target':>8s} {'isaac':>8s} {'mjc':>8s} {'isaac err':>10s} {'mjc err':>10s}")
for j, name in enumerate(ARM_JOINTS):
    tgt = PRE_GRASP_ARM_POSE[j]
    vi, vm = arm_i[-1, j], arm_m[-1, j]
    print(f"{name:24s} {tgt:8.4f} {vi:8.4f} {vm:8.4f} {abs(vi-tgt):10.4f} {abs(vm-tgt):10.4f}")

# ---- plots ----
fig, axes = plt.subplots(3, 2, figsize=(12, 11))

def plot_pair(ax, i_arr, m_arr, title, ylabel):
    ax.plot(t, i_arr, label="Isaac Lab", color="#2563eb", linewidth=1.5)
    ax.plot(t, m_arr, label="MuJoCo", color="#dc2626", linewidth=1.5)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plot_pair(axes[0, 0], ee_to_object_i, ee_to_object_m, "EE-to-object distance", "m")
plot_pair(axes[0, 1], hand_base_to_object_i, hand_base_to_object_m, "Hand-base-to-object distance", "m")
plot_pair(axes[1, 0], gripper_i, gripper_m, "Gripper joint pos (mean, Isaac-Lab-equivalent)", "rad")
plot_pair(axes[1, 1], pregrasp_dist_i, pregrasp_dist_m, "Joint-space distance to PRE_GRASP_ARM_POSE", "rad")
plot_pair(axes[2, 0], object_drift_i, object_drift_m, "Cube drift from spawn", "m")
plot_pair(axes[2, 1], arm_action_norm_i, arm_action_norm_m, "Arm action norm (raw policy output)", "-")

fig.tight_layout()
fig.savefig("trajectory_comparison.png", dpi=130)
print("\nsaved trajectory_comparison.png")

# per-joint arm position overlay, one subplot per joint
fig2, axes2 = plt.subplots(4, 2, figsize=(12, 14))
axes2 = axes2.flatten()
for j, name in enumerate(ARM_JOINTS):
    ax = axes2[j]
    ax.plot(t, arm_i[:, j], label="Isaac Lab", color="#2563eb", linewidth=1.2)
    ax.plot(t, arm_m[:, j], label="MuJoCo", color="#dc2626", linewidth=1.2)
    ax.axhline(PRE_GRASP_ARM_POSE[j], color="#16a34a", linestyle="--", linewidth=1, label="PRE_GRASP target")
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("rad")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
axes2[7].axis("off")
fig2.tight_layout()
fig2.savefig("trajectory_comparison_joints.png", dpi=130)
print("saved trajectory_comparison_joints.png")
