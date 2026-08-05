# g1_lift_ext — MuJoCo sim2sim transfer

Isaac Lab (PhysX) → MuJoCo replica of the G1+Dex1 lift task, built to verify Policy 1
(reach-and-hold at a pre-grasp point above a table cube) before any real-hardware attempt.
Extracted from the `g1_lift_ext` Isaac Lab package — trains there, gets verified here.

## Layout

- `build_scene.py` — generates `g1_lift_scene.xml` (robot + table + cube MJCF), matched
  to Isaac Lab's exact geometry, masses, and actuator gains.
- `joint_mapping.py`, `pd_gains.py` — Isaac Lab ↔ MuJoCo joint-name/order mapping and PD
  gain config, derived from Isaac Lab's own `G1_DEX1_CFG`.
- `run_policy1.py` — runs Policy 1's exported checkpoint in MuJoCo: reconstructs its exact
  39-dim observation, applies its 8-dim action, includes a runtime-calibrated
  (physically-anchored, not assumed-matching-range) gripper conversion.
- `capture_isaac_trajectory.py` / `run_policy1.py --save_trajectory` — log matched
  per-step trajectories (obs/action/joint positions) from both simulators for direct
  comparison.
- `compare_trajectories.py`, `render_dashboard_charts.py`, `build_dashboard.py` — analysis
  and HTML dashboard generation from the two trajectory logs.
- `query_hand_kinematics*.py`, `compute_hand_kinematics.py`, `query_hand_mass.py` —
  read-only Isaac Lab introspection used to derive the real Dex1 hand's exact geometry/mass
  for reconstruction in MuJoCo (see `build_scene.py`'s
  `replace_right_hand_with_real_geometry`, currently **enabled** — a simpler 2-body
  substitute hand is also available and was used earlier in the investigation below).
- `g1_dex1_patched.urdf` — `unitree_ros`'s G1+Dex1 URDF with a mesh-path fix for MuJoCo
  loading.

`meshes/` is a symlink to the sibling `unitree_ros` clone's mesh directory and isn't
included in this repo — point it at your own `unitree_ros/robots/g1_description/meshes`
checkout.

## Status

Policy 1 transfers and stays numerically stable (action-clipped, cube-velocity-capped),
but a step-for-step comparison against Isaac Lab (`trajectory_isaac.npz` /
`trajectory_mujoco_final.npz`, same checkpoint, same cube spawn) found the two
trajectories match closely for the first ~0.3s, then diverge after an apparent contact
event that doesn't occur in Isaac Lab (cube drift stays <1cm there vs tens of cm here).

Two physics-matching hypotheses were tested against this gap, in order:

1. **Contact stiffness.** MuJoCo's default `solref` (0.02s time constant, ~4 physics
   steps to resolve penetration) was untuned and much softer than Isaac Lab/PhysX's
   contact resolution. Tightening to `0.01` (2x the timestep) gave a real, confirmed
   improvement — cube drift 33cm → 20cm, joint-space error 0.99 → 0.35 rad. Pushing
   further to `0.005` (== the timestep itself) produced a **bit-identical** trajectory
   to `0.01` — contact resolution was already saturated; this lever is exhausted.
2. **Hand collision geometry.** Re-enabling the real 7-body Dex1 reconstruction (in
   place of the simpler substitute hand) on top of the tightened contact, delayed the
   contact event (t=1.15s → 1.96s) but barely moved the final outcome (drift 20cm →
   18cm). Confirmed the reconstruction itself is correct (gripper open/close
   separation now matches Isaac Lab's real hand almost exactly: 9.41cm vs 9.01cm open,
   0.51cm vs 0.12cm closed) — it just isn't the dominant factor.

**Conclusion:** two rounds of closer physics-matching each gave real, measured
improvement, then plateaued well short of Isaac Lab's behavior (best case: 18cm cube
drift / 0.35 rad joint error vs Isaac Lab's 0.36cm / 0.15 rad — still ~50x off on cube
drift). That pattern points away from "one more MuJoCo parameter to tune" and toward a
**policy-robustness explanation**: Policy 1's margin near the cube is apparently thin
enough that closed-loop control amplifies small, unavoidable simulator differences
(PhysX vs MuJoCo will never be bit-identical) into a real contact event. That's a
property of the policy, not a MuJoCo modeling gap — and it's exactly the kind of thing
sim2sim testing exists to surface before a real-hardware attempt. Suggested follow-up:
test Policy 1's sensitivity to small observation noise directly in Isaac Lab (no MuJoCo
involved) to confirm the margin is genuinely thin, rather than assuming it from the
transfer gap alone.

Run `python3 compare_trajectories.py` (after both trajectory captures) for the numeric
breakdown, or `python3 render_dashboard_charts.py && python3 build_dashboard.py` to
regenerate the HTML dashboard.
