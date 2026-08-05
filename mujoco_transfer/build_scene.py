"""Builds g1_lift_scene.xml: the native MJCF robot (g1_dex1_native.xml) wrapped
in a body at ROBOT_POS/ROBOT_ROT (matching Isaac Lab's G1_DEX1_CFG placement
exactly), plus table + cube geoms matching the training scene's geometry
(env_cfg.py's LiftSceneCfg / constants.py).

Size conversion: Isaac Lab's sim_utils.CuboidCfg(size=...) is FULL extents;
MuJoCo box geoms use HALF-extents -- every size value here is halved.
Positions and the cube's quaternion (w,x,y,z) match Isaac Lab's convention
directly, no conversion needed there.
"""
import xml.etree.ElementTree as ET
from pd_gains import STIFFNESS_MJC, DAMPING_MJC
from joint_mapping import MUJOCO_JOINT_ORDER

# ---- constants.py values (Policy 1) ----
ROBOT_POS = (0.0, 0.0, 0.76)
ROBOT_ROT = (0.7071, 0.0, 0.0, -0.7071)  # (w,x,y,z)

TABLE_TOP_Z = 0.81
TABLE_POS = (0.04, -0.33, TABLE_TOP_Z - 0.74 / 2.0)
TABLE_SIZE = (0.60, 0.50, 0.74)  # full extents (Isaac Lab convention)

BLOCK_SIZE = 0.06
BLOCK_INIT_POS = (-0.0585, -0.3341, TABLE_TOP_Z + BLOCK_SIZE / 2.0)
CUBE_ROT = (0.9945, 0.0, 0.0, 0.1045)  # (w,x,y,z)

tree = ET.parse("g1_dex1_native.xml")
root = tree.getroot()


def replace_right_hand_with_real_geometry(root):
    """Swaps the substitute hand (2 flat prismatic sliders, no mounting plate --
    structurally different from training's Dex1) for an accurate reconstruction
    of training's ACTUAL hand: a fixed mounting plate + two mirrored prismatic
    fingers, each a rigid 3-piece assembly. Derived from two live Isaac Lab
    snapshots (GRIPPER_OPEN/GRIPPER_CLOSE, query_hand_kinematics*.py +
    compute_hand_kinematics.py, read-only -- Isaac Lab/the USD/the checkpoint
    are never modified): Joint1_1 is a pure 1:1 slide along the wrist's local
    +Y (ratio 1.0000 measured), Joint2_1 along -Y (ratio 0.9977), and
    Link{1,2}_2/Link{1,2}_3 are rigidly fixed to Link{1,2}_1 (offsets identical
    to 4+ decimals across both snapshots -- confirmed not independently
    articulated, consistent with them never appearing in Isaac Lab's joint
    list at all).

    Collision geometry uses box approximations from each body's own measured
    local-frame bounding box (check_finger_cube_contact_v2.py's BBOXES,
    established earlier this session) -- not exact meshes, but a faithful
    stand-in for collision purposes, which is what actually matters for the
    contact-avoidance behavior this replacement targets.

    Real joint names (right_hand_Joint1_1/right_hand_Joint2_1) match Isaac Lab
    exactly now, unlike the substitute's dex1_finger_joint_* -- removes the
    need for joint_mapping.py's gripper name-translation entirely.
    """
    GRIPPER_STIFFNESS, GRIPPER_DAMPING = 800.0, 3.0  # matches pd_gains.py's gripper group

    # Real per-body masses, queried read-only from Isaac Lab (robot.root_physx_view.
    # get_masses(), query_hand_mass.py). MuJoCo's own auto-computed mass from these
    # box collision approximations at default density was ~4x too heavy overall
    # (0.778kg vs the real 0.193kg total) -- confirmed directly and worth fixing,
    # since a position-controlled joint's tracking dynamics (and, more importantly,
    # the arm's own gravity/reaction-torque coupling to a much-too-heavy hand) are
    # sensitive to this, not just the joint's final resting position.
    REAL_MASS = {
        "right_hand_base_link": 0.110000,
        "right_hand_Link1_1": 0.008440, "right_hand_Link1_2": 0.028354, "right_hand_Link1_3": 0.004583,
        "right_hand_Link2_1": 0.008440, "right_hand_Link2_2": 0.028354, "right_hand_Link2_3": 0.004583,
    }

    def box(name, bbox_min, bbox_max, body_name):
        c = [(mn + mx) / 2.0 for mn, mx in zip(bbox_min, bbox_max)]
        s = [(mx - mn) / 2.0 for mn, mx in zip(bbox_min, bbox_max)]
        mass = REAL_MASS[body_name]
        # alpha=0 -- collision-only proxy, invisible; the real appearance comes
        # from the visual mesh geom added alongside it (see mesh_visual() below).
        return ET.fromstring(
            f'<geom name="{name}" type="box" pos="{c[0]} {c[1]} {c[2]}" '
            f'size="{s[0]} {s[1]} {s[2]}" mass="{mass}" rgba="0.4 0.4 0.4 0"/>'
        )

    # Visual-only mesh geoms, using the OFFICIAL Dex1 reference's own STL files
    # (unitree_ros/robots/dexterous_hand_description/dex1_1/meshes/, symlinked
    # into dex1_meshes/) -- the box() geoms above exist purely for collision
    # (calibrated masses, tuned contact solref -- swapping them for mesh
    # collision would be far more expensive and would invalidate that tuning).
    # User feedback: the box-only version rendered as an unrecognizable flat
    # slab ("totally different" from Isaac Lab's actual gripper) -- this adds
    # the real geometry back for appearance without touching physics. Same
    # contype=0/conaffinity=0/group=1/density=0 convention already used for
    # the rest of the robot's own visual meshes (see e.g. the "pelvis" mesh
    # geom in g1_dex1_native.xml) -- density=0 since REAL_MASS is already
    # assigned via the box collision geom, not this one.
    asset = root.find("asset")
    _DEX1_MESH_FILES = {
        "right_hand_base_link": ("base_link", "0.79216 0.81961 0.93333 1"),
        "right_hand_Link1_1": ("Link1_1", "0.79216 0.81961 0.93333 1"),
        "right_hand_Link1_2": ("Link1_2", "0.89804 0.91765 0.92941 1"),
        "right_hand_Link1_3": ("Link1_3", "0.29804 0.29804 0.29804 1"),
        "right_hand_Link2_1": ("Link2_1", "0.79216 0.81961 0.93333 1"),
        "right_hand_Link2_2": ("Link2_2", "0.89804 0.91765 0.92941 1"),
        "right_hand_Link2_3": ("Link2_3", "0.29804 0.29804 0.29804 1"),
    }
    for body_name, (stl_stem, rgba) in _DEX1_MESH_FILES.items():
        mesh_name = f"dex1_{stl_stem}"
        ET.SubElement(asset, "mesh", {
            "name": mesh_name, "content_type": "model/stl", "file": f"dex1_meshes/{stl_stem}.STL",
        })

    def mesh_visual(body_name):
        stl_stem, rgba = _DEX1_MESH_FILES[body_name]
        return ET.fromstring(
            f'<geom type="mesh" contype="0" conaffinity="0" group="1" density="0" '
            f'rgba="{rgba}" mesh="dex1_{stl_stem}"/>'
        )

    wrist = None
    for body in root.iter("body"):
        if body.get("name") == "right_wrist_yaw_link":
            wrist = body
            break
    assert wrist is not None, "right_wrist_yaw_link not found"

    for child in list(wrist):
        if child.tag == "body" and child.get("name") in ("right_dex1_finger_link_1", "right_dex1_finger_link_2"):
            wrist.remove(child)

    # FIXED: the whole hand assembly (base_link + both fingers) is rotated
    # -90deg about its own local Z relative to the wrist -- confirmed
    # precisely (quat composition from the two Isaac Lab snapshots:
    # [0.7074, ~0, ~0, -0.7068], matching a clean -90deg-about-Z rotation to
    # within numerical noise). Position alone (already correct, measured
    # directly) isn't enough -- without this quat every hand body defaults to
    # identity orientation relative to the wrist, which is wrong and is
    # exactly why the rendered geometry looked skewed/incorrect. Cross-checked
    # against the official dedicated Dex1 reference (unitree_ros/robots/
    # dexterous_hand_description/dex1_1/dex1_1.urdf) -- Joint1_1's axis there
    # is -X in base_link's own frame, consistent with this same rotation
    # (base_link's local -X maps to the wrist's +Y, which is what was
    # actually measured as Joint1_1's real motion direction).
    _HAND_QUAT = "0.7071 0 0 -0.7071"

    # Mounting plate: fixed to the wrist, no joint.
    base = ET.SubElement(wrist, "body", {"name": "right_hand_base_link", "pos": "0.0415 -0.003 0.0", "quat": _HAND_QUAT})
    base.append(box("right_hand_base_link_col", (-0.035, 0.000, -0.035), (0.035, 0.0738, 0.089), "right_hand_base_link"))
    base.append(mesh_visual("right_hand_base_link"))

    # Finger 1: Link1_1 (prismatic, +Y) -> Link1_2 (fixed) -> Link1_3 (fixed).
    l1_1 = ET.SubElement(wrist, "body", {"name": "right_hand_Link1_1", "pos": "0.1008482 -0.04350275 0.0152", "quat": _HAND_QUAT})
    ET.SubElement(l1_1, "joint", {
        "name": "right_hand_Joint1_1", "type": "slide", "axis": "-1 0 0",  # re-expressed in the now-rotated local frame; matches the official Dex1 reference exactly
        "range": "-0.02 0.0245", "damping": str(GRIPPER_DAMPING),
    })
    l1_1.append(box("right_hand_Link1_1_col", (-0.080, -0.005, -0.008), (-0.000, 0.009, 0.0065), "right_hand_Link1_1"))
    l1_1.append(mesh_visual("right_hand_Link1_1"))
    l1_2 = ET.SubElement(l1_1, "body", {"name": "right_hand_Link1_2", "pos": "-0.0130 0.0130 0.0"})
    l1_2.append(box("right_hand_Link1_2_col", (-0.006, -0.0068, -0.0304), (0.013, 0.075, 0.000), "right_hand_Link1_2"))
    l1_2.append(mesh_visual("right_hand_Link1_2"))
    l1_3 = ET.SubElement(l1_2, "body", {"name": "right_hand_Link1_3", "pos": "-0.00252 0.02504 -0.001"})
    l1_3.append(box("right_hand_Link1_3_col", (-0.000, -0.002, -0.0284), (0.004, 0.0475, 0.000), "right_hand_Link1_3"))
    l1_3.append(mesh_visual("right_hand_Link1_3"))

    # Finger 2: mirror of finger 1, Joint2_1 slides along local +X (post-rotation).
    l2_1 = ET.SubElement(wrist, "body", {"name": "right_hand_Link2_1", "pos": "0.10075179 0.03759718 -0.01520001", "quat": _HAND_QUAT})
    ET.SubElement(l2_1, "joint", {
        "name": "right_hand_Joint2_1", "type": "slide", "axis": "1 0 0",
        "range": "-0.02 0.0245", "damping": str(GRIPPER_DAMPING),
    })
    l2_1.append(box("right_hand_Link2_1_col", (0.000, -0.005, -0.0065), (0.080, 0.009, 0.008), "right_hand_Link2_1"))
    l2_1.append(mesh_visual("right_hand_Link2_1"))
    l2_2 = ET.SubElement(l2_1, "body", {"name": "right_hand_Link2_2", "pos": "0.0130 0.0130 0.0"})
    l2_2.append(box("right_hand_Link2_2_col", (-0.013, -0.0068, 0.000), (0.006, 0.075, 0.0304), "right_hand_Link2_2"))
    l2_2.append(mesh_visual("right_hand_Link2_2"))
    l2_3 = ET.SubElement(l2_2, "body", {"name": "right_hand_Link2_3", "pos": "0.00252 0.02504 0.02940"})
    l2_3.append(box("right_hand_Link2_3_col", (-0.004, -0.002, -0.0284), (0.000, 0.0475, 0.000), "right_hand_Link2_3"))
    l2_3.append(mesh_visual("right_hand_Link2_3"))

    return GRIPPER_STIFFNESS


# RE-ENABLED: testing whether real hand geometry closes the remaining
# ~20cm trajectory gap (compare_trajectories.py) left after tightening
# contact solref. Validated earlier this session (matches the official Dex1
# reference exactly -- kinematics, mass, and the -90deg-about-Z orientation).
_real_hand_gripper_stiffness = replace_right_hand_with_real_geometry(root)

# FIXED: manually computing PD torque (tau = Kp*(target-qpos) + Kd*(0-qvel))
# and applying it via qfrc_applied, then stepping with the default explicit
# Euler integrator, is numerically UNSTABLE for stiff gains -- confirmed
# directly: with the waist's Kp=10000, qpos diverged to 377 rad and tau to
# -1.9e9 within 4 steps at the default timestep=0.002s, MuJoCo then reporting
# "Nan, Inf or huge value in QACC". Explicit integration of a stiff spring
# needs a much smaller timestep than this to stay stable (dt ~ 2/sqrt(k/m)).
# Fix: use MuJoCo's native <position> actuator (the P-term, solved implicitly
# by the engine -- the direct equivalent of Isaac Lab's ImplicitActuatorCfg)
# plus the joint's own damping= attribute (D-term, also always solved
# implicitly by MuJoCo regardless of integrator) instead of a manual
# qfrc_applied torque. implicitfast further improves stability for this kind
# of stiff-joint system.
option = ET.SubElement(root, "option")
option.set("integrator", "implicitfast")
# Matches Isaac Lab's env_cfg.py: self.sim.dt = 0.005 -- same physics
# granularity, not just the same control-period wall-clock coverage.
option.set("timestep", "0.005")
# must come before worldbody per MJCF schema ordering -- move it up.
root.remove(option)
root.insert(list(root).index(root.find("compiler")) + 1, option)

# EXPERIMENT: tighten default contact softness. MuJoCo's built-in default
# solref=[0.02, 1] resolves penetration over a 0.02s time constant -- 4 full
# physics steps at our 0.005s timestep -- deliberately soft/spring-like for
# solver stability. Isaac Lab/PhysX's contact resolution isn't a 1:1-comparable
# parameterization, but it's not this loose: penetration is corrected within
# the SAME step's 12-iteration position solve, plus enable_ccd=True as a
# fast-motion safety net MuJoCo has no equivalent of here. Every geom in this
# scene (robot, table, cube) has been running on MuJoCo's untouched default --
# never tuned to match training at all. Tightening toward timeconst ~= 2x the
# timestep (a common "as rigid as reasonably stable" MuJoCo recommendation) is
# a direct test of whether soft-contact timing (not hand geometry) explains
# the t=1.05-1.6s divergence found in compare_trajectories.py. Applied via a
# top-level <default><geom> so it covers every geom, not just the cube/table.
default_el = ET.SubElement(root, "default")
# SETTLED at 0.01 (2x timestep): confirmed helps over MuJoCo's 0.02 default
# (cube drift 33->20cm in compare_trajectories.py). Pushed further to 0.005
# (== the timestep itself) as a test -- produced a BIT-IDENTICAL trajectory to
# 0.01 (confirmed directly: obs/arm_abs arrays equal), meaning contact
# resolution was already saturated at 0.01 for this event -- tightening
# further buys nothing. Kept at 0.01, not 0.005, since there's no benefit to
# sitting exactly at the numerical floor. See mujoco_transfer/README.md's
# "Status" section for the full investigation and conclusion.
ET.SubElement(default_el, "geom", {"solref": "0.01 1"})
root.remove(default_el)
root.insert(list(root).index(root.find("option")) + 1, default_el)

worldbody = root.find("worldbody")

# Move every existing child of worldbody (the robot's geoms/bodies) under a
# new wrapper body placed at ROBOT_POS/ROBOT_ROT.
robot_children = list(worldbody)
for c in robot_children:
    worldbody.remove(c)

robot_wrapper = ET.SubElement(
    worldbody, "body", {
        "name": "robot_root",
        "pos": f"{ROBOT_POS[0]} {ROBOT_POS[1]} {ROBOT_POS[2]}",
        "quat": f"{ROBOT_ROT[0]} {ROBOT_ROT[1]} {ROBOT_ROT[2]} {ROBOT_ROT[3]}",
    }
)
for c in robot_children:
    robot_wrapper.append(c)

# Lighting -- the scene had no <light> elements at all, relying purely on
# MuJoCo's dim default camera headlight. A bright overhead directional light
# plus a softer fill light (and a ground plane, for visual context/shadow
# grounding) makes the viewer actually usable.
ET.SubElement(worldbody, "light", {
    "name": "sun", "directional": "true", "pos": "0 0 3", "dir": "0 0 -1",
    "diffuse": "0.8 0.8 0.8", "specular": "0.2 0.2 0.2", "castshadow": "true",
})
ET.SubElement(worldbody, "light", {
    "name": "fill", "directional": "true", "pos": "1 -1 2", "dir": "-1 1 -2",
    "diffuse": "0.4 0.4 0.4", "castshadow": "false",
})
# No ground plane -- the robot's legs hang freely (fixed pelvis, no floor
# contact anywhere in training), so a real collision geom at z=0 would risk
# the dangling feet colliding with it, a physics interaction training never
# had. Lighting only; MuJoCo's viewer already renders its own grid/horizon
# for visual reference without needing a physical geom.

# Set damping (D-term) on each robot joint, and build the actuator list
# (P-term via <position kp="...">) in the same pass.
damping_of = dict(zip(MUJOCO_JOINT_ORDER, DAMPING_MJC))
stiffness_of = dict(zip(MUJOCO_JOINT_ORDER, STIFFNESS_MJC))
actuator_specs = []  # (joint_name, kp)
for joint_el in robot_wrapper.iter("joint"):
    name = joint_el.get("name")
    if name not in damping_of:
        continue  # shouldn't happen -- every robot joint is in MUJOCO_JOINT_ORDER
    joint_el.set("damping", str(damping_of[name]))
    actuator_specs.append((name, stiffness_of[name]))
assert len(actuator_specs) == 33, f"expected 33 actuated joints, found {len(actuator_specs)}"

# Table -- kinematic/static in training (kinematic_enabled=True), so a fixed
# geom directly on worldbody (no joint) is the faithful equivalent.
half_table = tuple(s / 2.0 for s in TABLE_SIZE)
ET.SubElement(
    worldbody, "geom", {
        "name": "table",
        "type": "box",
        "pos": f"{TABLE_POS[0]} {TABLE_POS[1]} {TABLE_POS[2]}",
        "size": f"{half_table[0]} {half_table[1]} {half_table[2]}",
        "rgba": "0.6 0.55 0.5 1",
    }
)

# Cube -- a free body (dynamic, matches RigidObjectCfg) at BLOCK_INIT_POS/CUBE_ROT.
half_block = BLOCK_SIZE / 2.0
cube_body = ET.SubElement(
    worldbody, "body", {
        "name": "object",
        "pos": f"{BLOCK_INIT_POS[0]} {BLOCK_INIT_POS[1]} {BLOCK_INIT_POS[2]}",
        "quat": f"{CUBE_ROT[0]} {CUBE_ROT[1]} {CUBE_ROT[2]} {CUBE_ROT[3]}",
    }
)
ET.SubElement(cube_body, "freejoint", {"name": "object_freejoint"})
ET.SubElement(
    cube_body, "geom", {
        "name": "object_geom",
        "type": "box",
        "size": f"{half_block} {half_block} {half_block}",
        "rgba": "0.9 0.1 0.1 1",
        "mass": "0.1",
        "friction": "1.5 0.005 0.0001",
    }
)

actuator_el = ET.SubElement(root, "actuator")
for name, kp in actuator_specs:
    ET.SubElement(
        actuator_el, "position", {
            "name": f"act_{name}",
            "joint": name,
            "kp": str(kp),
        }
    )

tree.write("g1_lift_scene.xml", xml_declaration=False)
print(f"wrote g1_lift_scene.xml with {len(actuator_specs)} position actuators")
