"""Compares RGP Policy 1's (reach) Isaac Lab rollout against its MuJoCo
rollout, both fed the SAME exported policy.pt and the SAME cube spawn
position -- isolating physics/geometry reconstruction differences as the
source of any behavioral gap.

Adapted from compare_trajectories.py (old chain, Policy 1) for RGP's 36-dim
obs layout (no object_to_inspect term, no PRE_GRASP_ARM_POSE-style fixed
joint target -- RGP Policy 1's reward deliberately doesn't have one, see
env_cfg_rgp_reach.py's module docstring):
  0:7   arm_joint_pos_rel      7:14  arm_joint_vel        14:16 gripper_joint_pos
  16:19 object_pos             19:22 ee_pos                22:25 ee_to_object
  25:28 hand_base_to_object    28:36 last_action
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARM_JOINTS = [
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]

isaac = np.load("trajectory_isaac_rgp_reach.npz")
mjc = np.load("trajectory_mujoco_rgp_reach.npz")

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
hand_base_to_object_i = np.linalg.norm(obs_i[:, 25:28], axis=1)
hand_base_to_object_m = np.linalg.norm(obs_m[:, 25:28], axis=1)
gripper_i = obs_i[:, 14:16].mean(axis=1)
gripper_m = obs_m[:, 14:16].mean(axis=1)
arm_action_norm_i = np.linalg.norm(act_i[:, :7], axis=1)
arm_action_norm_m = np.linalg.norm(act_m[:, :7], axis=1)
object_drift_i = np.linalg.norm(obs_i[:, 16:19] - obs_i[0, 16:19], axis=1)
object_drift_m = np.linalg.norm(obs_m[:, 16:19] - obs_m[0, 16:19], axis=1)
arm_joint_dist_im = np.linalg.norm(arm_i - arm_m, axis=1)  # direct Isaac-vs-MuJoCo joint gap, no fixed target needed

print(f"\n{'metric':32s} {'isaac final':>12s} {'mjc final':>12s} {'isaac min':>10s} {'mjc min':>10s}")
for name, i_arr, m_arr in [
    ("ee_to_object dist (m)", ee_to_object_i, ee_to_object_m),
    ("hand_base_to_object dist (m)", hand_base_to_object_i, hand_base_to_object_m),
    ("gripper_joint_pos (mean)", gripper_i, gripper_m),
    ("arm_action_norm", arm_action_norm_i, arm_action_norm_m),
    ("object drift from spawn (m)", object_drift_i, object_drift_m),
]:
    print(f"{name:32s} {i_arr[-1]:12.4f} {m_arr[-1]:12.4f} {i_arr.min():10.4f} {m_arr.min():10.4f}")

print(f"\ndirect Isaac-vs-MuJoCo arm joint-space gap (rad): final={arm_joint_dist_im[-1]:.4f} "
      f"mean(last 100 steps)={arm_joint_dist_im[-100:].mean():.4f} max={arm_joint_dist_im.max():.4f}")

print(f"\nper-joint final position, Isaac vs MuJoCo:")
print(f"{'joint':24s} {'isaac':>8s} {'mjc':>8s} {'gap':>8s}")
for j, name in enumerate(ARM_JOINTS):
    vi, vm = arm_i[-1, j], arm_m[-1, j]
    print(f"{name:24s} {vi:8.4f} {vm:8.4f} {abs(vi-vm):8.4f}")

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
axes[1, 1].plot(t, arm_joint_dist_im, color="#7c3aed", linewidth=1.5)
axes[1, 1].set_title("Isaac-vs-MuJoCo arm joint-space gap", fontsize=10)
axes[1, 1].set_xlabel("time (s)")
axes[1, 1].set_ylabel("rad (L2 norm, 7 joints)")
axes[1, 1].grid(alpha=0.3)
plot_pair(axes[2, 0], object_drift_i, object_drift_m, "Cube drift from spawn", "m")
plot_pair(axes[2, 1], arm_action_norm_i, arm_action_norm_m, "Arm action norm (raw policy output)", "-")

fig.tight_layout()
fig.savefig("trajectory_comparison_rgp_reach.png", dpi=130)
print("\nsaved trajectory_comparison_rgp_reach.png")

# per-joint arm position overlay, one subplot per joint
fig2, axes2 = plt.subplots(4, 2, figsize=(12, 14))
axes2 = axes2.flatten()
for j, name in enumerate(ARM_JOINTS):
    ax = axes2[j]
    ax.plot(t, arm_i[:, j], label="Isaac Lab", color="#2563eb", linewidth=1.2)
    ax.plot(t, arm_m[:, j], label="MuJoCo", color="#dc2626", linewidth=1.2)
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("rad")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
axes2[7].axis("off")
fig2.tight_layout()
fig2.savefig("trajectory_comparison_rgp_reach_joints.png", dpi=130)
print("saved trajectory_comparison_rgp_reach_joints.png")
