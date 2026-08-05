# Installation & Usage Guide

Everything needed to go from a fresh clone to training, watching, and sim2sim-verifying
the RGP policies yourself. See [README.md](README.md) for the chain design/architecture —
this doc is the step-by-step path to running it.

Tested with: Isaac Lab 2.3.2, Isaac Sim 5.1.0, Python 3.11, Ubuntu 24.04. Isaac Lab's own
APIs move fast between releases — if a step below errors on a newer/older install, check
Isaac Lab's own changelog before assuming this guide is wrong.

## Step 1 — Install Isaac Lab

Follow NVIDIA's own guide:
https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html

This project's convention is to name the resulting conda env **`env_isaaclab`** — the
commands below assume it.

## Step 2 — Get this repository

`g1_lift_ext/` is an Isaac Lab *extension*, not a standalone install — it's expected to
sit as a **sibling** of your `IsaacLab/` checkout (both under the same parent directory).

```bash
cd ~/projects            # wherever IsaacLab/ already lives
git clone <this-repo-url> g1_lift_ext
```

## Step 3 — Get the robot model

The G1+Dex1 USD this task uses is **not vendored in this repo** — it comes from a
separate Unitree asset repo, plus one local patch:

1. **Clone the base robot description** (public repo, sibling of `g1_lift_ext/`):
   ```bash
   cd ~/projects
   git clone https://github.com/unitreerobotics/unitree_sim_isaaclab.git
   ```
   The unpatched USD ends up at
   `unitree_sim_isaaclab/assets/robots/g1-29dof-dex1-base-fix-usd/g1_29dof_with_dex1_base_fix1.usd`.

2. **Apply the finger-collision patch.** With `enabled_self_collisions=True`, the Dex1
   gripper's two finger chains clip each other well inside their own joint travel, so the
   gripper can't fully close. `patch_finger_collision.py` (repo root) fixes this with
   `UsdPhysics.FilteredPairsAPI` on just that one pair, and writes a new sibling USD file
   (`..._fingerfilter.usd`) rather than touching the original:
   ```bash
   conda activate env_isaaclab
   cd g1_lift_ext
   # edit SRC/DST at the top of the script to your own unitree_sim_isaaclab checkout path first
   python patch_finger_collision.py
   ```
3. **Point this repo at the result** — edit `ROBOT_USD` in
   [`g1_lift_rl/constants.py`](g1_lift_rl/constants.py) to the `..._fingerfilter.usd` path
   from step 2.

For the MuJoCo side (Step 8), the mesh files come from a second public repo:
```bash
cd ~/projects
git clone https://github.com/unitreerobotics/unitree_ros.git
```
`mujoco_transfer/meshes` and `mujoco_transfer/dex1_meshes` are symlinks expecting
`unitree_ros/robots/g1_description/meshes` and
`unitree_ros/robots/dexterous_hand_description/dex1_1/meshes` respectively — recreate them
pointing at your own checkout:
```bash
cd g1_lift_ext/mujoco_transfer
ln -s ~/projects/unitree_ros/robots/g1_description/meshes meshes
ln -s ~/projects/unitree_ros/robots/dexterous_hand_description/dex1_1/meshes dex1_meshes
```

## Step 4 — Install this package

```bash
conda activate env_isaaclab
cd g1_lift_ext
pip install -e .
```

Editable install — code changes take effect without reinstalling. This also registers
the `Isaac-G1-RGP-*` gym task IDs on import.

## Step 5 — Verify installation

```bash
python -c "import g1_lift_rl; print('g1_lift_rl import OK')"
```

## Step 6 — Quick sanity check

A short, small-scale run confirms the task loads and steps cleanly end-to-end (GPU,
physics, USD path, reward/observation wiring) before committing to a full training run:

```bash
cd ../IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-G1-RGP-Reach-v0 --num_envs 64 --max_iterations 20 --headless
```

You should see per-iteration `Episode_Reward/...` lines with no NaN/Inf and no crash.
This doesn't produce a usable checkpoint — it's a plumbing check.

## Step 7 — Train

Train **in order** — Policy 2's reset pose is measured from Policy 1's own trained
behavior (see README.md → "Chaining policies"), so Policy 1 needs a checkpoint first.

```bash
cd ../IsaacLab
conda activate env_isaaclab

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-G1-RGP-Reach-v0 --headless
# ... after it finishes ...
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-G1-RGP-Grasp-v0 --headless
```

Common flags: `--num_envs <N>` (defaults are set per-task, 2048 for Reach / 1024 for
Grasp — see the Gym IDs table in README.md), `--max_iterations <N>` (default 1500),
`--seed <N>`. Checkpoints land in
`IsaacLab/logs/rsl_rl/<experiment_name>/<timestamp>/model_<iter>.pt`
(`experiment_name` is `g1_rgp_reach` / `g1_rgp_grasp`, set in
[`agents/rsl_rl_ppo_cfg_rgp.py`](g1_lift_rl/agents/rsl_rl_ppo_cfg_rgp.py)).

## Step 8 — See the results

**Watch it live in Isaac Sim** — opens a GUI window and runs the latest checkpoint on a
small number of environments:
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-G1-RGP-Reach-Play-v0
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-G1-RGP-Grasp-Play-v0
```
Pass `--checkpoint <path/to/model_N.pt>` to pick a specific run instead of the latest.
`play.py` also exports a TorchScript policy to
`logs/rsl_rl/<experiment_name>/<run>/exported/policy.pt` — this is what the MuJoCo scripts
below load.

**Read the training curves:**
```bash
cd ../IsaacLab
tensorboard --logdir logs/rsl_rl/g1_rgp_reach   # or g1_rgp_grasp
```
Curve convergence alone isn't proof of a working policy in this project's own experience —
prefer watching it live (above) or a deterministic (no-exploration-noise) replay before
trusting a checkpoint enough to hand it to the next policy in the chain.

**Verify sim2sim in MuJoCo** — replays the same exported checkpoint outside Isaac Sim
entirely and compares the two trajectories step-by-step (needs Step 3's `unitree_ros`
meshes):
```bash
cd g1_lift_ext/mujoco_transfer
python3 build_scene_rgp.py
python3 capture_isaac_trajectory_rgp_reach.py --checkpoint <path/to/exported/policy.pt>
python3 run_rgp_reach.py --save_trajectory
python3 compare_trajectories_rgp_reach.py
```
See README.md → "MuJoCo sim2sim transfer" for what each script does and how to adapt the
`_rgp_reach` scripts to Policy 2.

## Troubleshooting

**`ModuleNotFoundError: No module named 'g1_lift_rl'`**
→ Run `pip install -e .` from `g1_lift_ext/` with `env_isaaclab` active.

**`gymnasium.error.NameNotFound: Environment Isaac-G1-RGP-... doesn't exist`**
→ `g1_lift_rl` registers its gym IDs on import; make sure whatever script you're running
actually does `import g1_lift_rl` (Isaac Lab's own `train.py`/`play.py` scan installed
extensions automatically, but a custom script needs the import explicitly).

**USD load error / `Could not open asset` for `ROBOT_USD`**
→ `constants.py`'s `ROBOT_USD` is an absolute path — it must point at your own
`unitree_sim_isaaclab` checkout's patched USD (Step 3), not the original author's machine.

**`ImportError` on `isaaclab.*` / `omni.*` at the top of a script**
→ Isaac Sim's modules only become importable after
`simulation_app = AppLauncher(args).app` has run. Every script in this repo parses CLI
args and launches the app *before* importing `isaaclab`/`omni`/`gymnasium` — if you're
writing a new script, follow that same order.

**MuJoCo scene build fails on missing mesh files**
→ Recreate the `mujoco_transfer/meshes` and `dex1_meshes` symlinks (Step 3) pointing at
your own `unitree_ros` checkout.

**CUDA out of memory**
→ Reduce `--num_envs` (e.g. 512 instead of the default 2048/1024).

**`plain python scripts/train.py` fails but `./isaaclab.sh -p scripts/train.py` works**
→ Use `isaaclab.sh -p` — it resolves Isaac Sim's own extension dependencies that bare
`python` doesn't pick up reliably, even with `env_isaaclab` active.
