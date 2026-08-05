"""Renders the trajectory-comparison charts twice (light/dark) with a palette
matched to the HTML dashboard, and emits base64 <img> data URIs to
chart_data_uris.py for embedding -- keeps the artifact HTML self-contained
with no external image files."""
import base64
import io
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
obs_i, act_i, arm_i = isaac["obs"], isaac["action"], isaac["arm_abs"]
obs_m, act_m, arm_m = mjc["obs"], mjc["action"], mjc["arm_abs"]
n = min(len(obs_i), len(obs_m))
obs_i, arm_i = obs_i[:n], arm_i[:n]
obs_m, arm_m = obs_m[:n], arm_m[:n]
t = np.arange(n) * 0.01

ee_to_object_i = np.linalg.norm(obs_i[:, 22:25], axis=1)
ee_to_object_m = np.linalg.norm(obs_m[:, 22:25], axis=1)
hand_base_to_object_i = np.linalg.norm(obs_i[:, 28:31], axis=1)
hand_base_to_object_m = np.linalg.norm(obs_m[:, 28:31], axis=1)
object_drift_i = np.linalg.norm(obs_i[:, 16:19] - obs_i[0, 16:19], axis=1)
object_drift_m = np.linalg.norm(obs_m[:, 16:19] - obs_m[0, 16:19], axis=1)
pregrasp_dist_i = np.linalg.norm(arm_i - PRE_GRASP_ARM_POSE, axis=1)
pregrasp_dist_m = np.linalg.norm(arm_m - PRE_GRASP_ARM_POSE, axis=1)

CONTACT_START, CONTACT_END = 1.96, 2.19

THEMES = {
    "light": dict(fg="#1a1d21", grid="#1a1d2122", isaac="#3568c9", mujoco="#c85a28", target="#2f8f5b", alert="#c23b32", bg="none"),
    "dark":  dict(fg="#e6e8eb", grid="#e6e8eb22", isaac="#6fa3f5", mujoco="#f0895a", target="#5ecb92", alert="#e2695f", bg="none"),
}

mono = ["DejaVu Sans Mono", "monospace"]


def style(ax, th):
    ax.set_facecolor("none")
    ax.tick_params(colors=th["fg"], labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_color(th["fg"])
        spine.set_alpha(0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color=th["grid"], linewidth=0.7)
    ax.xaxis.label.set_color(th["fg"])
    ax.yaxis.label.set_color(th["fg"])
    ax.title.set_color(th["fg"])


def fig_to_datauri(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=185, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def mark_contact(ax, th):
    ax.axvspan(CONTACT_START, CONTACT_END, color=th["alert"], alpha=0.10, zorder=0)


results = {}
for theme_name, th in THEMES.items():
    plt.rcParams["font.family"] = mono
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.2))

    def plot_pair(ax, i_arr, m_arr, title, ylabel):
        mark_contact(ax, th)
        ax.plot(t, i_arr, label="Isaac Lab", color=th["isaac"], linewidth=1.8)
        ax.plot(t, m_arr, label="MuJoCo", color=th["mujoco"], linewidth=1.8)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("time (s)", fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=8.5)
        leg = ax.legend(fontsize=8, frameon=False, loc="best")
        for text in leg.get_texts():
            text.set_color(th["fg"])
        style(ax, th)

    plot_pair(axes[0, 0], ee_to_object_i, ee_to_object_m, "EE → object distance", "m")
    plot_pair(axes[0, 1], hand_base_to_object_i, hand_base_to_object_m, "Hand-base → object distance", "m")
    plot_pair(axes[1, 0], object_drift_i, object_drift_m, "Cube drift from spawn", "m")
    plot_pair(axes[1, 1], pregrasp_dist_i, pregrasp_dist_m, "Joint-space distance to PRE_GRASP_ARM_POSE", "rad")
    fig.tight_layout()
    results[f"overview_{theme_name}"] = fig_to_datauri(fig)

    fig2, axes2 = plt.subplots(4, 2, figsize=(9.2, 11.5))
    axes2 = axes2.flatten()
    for j, name in enumerate(ARM_JOINTS):
        ax = axes2[j]
        mark_contact(ax, th)
        ax.plot(t, arm_i[:, j], color=th["isaac"], linewidth=1.5, label="Isaac Lab")
        ax.plot(t, arm_m[:, j], color=th["mujoco"], linewidth=1.5, label="MuJoCo")
        ax.axhline(PRE_GRASP_ARM_POSE[j], color=th["target"], linestyle="--", linewidth=1.1, label="target")
        ax.set_title(name.replace("_", " "), fontsize=9.5, loc="left")
        ax.set_ylabel("rad", fontsize=8)
        leg = ax.legend(fontsize=7, frameon=False, loc="best")
        for text in leg.get_texts():
            text.set_color(th["fg"])
        style(ax, th)
    axes2[7].axis("off")
    fig2.tight_layout()
    results[f"joints_{theme_name}"] = fig_to_datauri(fig2)

with open("chart_data_uris.py", "w") as f:
    f.write("CHARTS = {\n")
    for k, v in results.items():
        f.write(f'    "{k}": "{v}",\n')
    f.write("}\n")

print("wrote chart_data_uris.py with keys:", list(results.keys()))
print("sizes (KB):", {k: len(v) // 1024 for k, v in results.items()})
