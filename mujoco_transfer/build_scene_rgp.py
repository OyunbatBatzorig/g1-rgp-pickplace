"""Builds g1_rgp_reach_scene.xml for the NEW RGP chain's Policy 1 (reach).

Identical to build_scene.py in every respect EXCEPT the cube block size/
position -- the RGP chain uses a smaller 3cm cube (RGP_BLOCK_SIZE in
env_cfg_rgp_scene.py) instead of the old chain's 6cm BLOCK_SIZE. Robot
placement (ROBOT_POS/ROT), table geometry, the full real-hand-geometry
reconstruction (mass-calibrated box collision + STL visual mesh, tuned
contact solref, implicitfast integrator) are all shared physical facts
about the SAME robot USD/asset and SAME table -- reused verbatim, not
re-derived, since none of that changed for the RGP chain. See
build_scene.py's own docstring/comments for the full history and
justification of each of those choices; not re-documented here.

Size conversion: Isaac Lab's sim_utils.CuboidCfg(size=...) is FULL extents;
MuJoCo box geoms use HALF-extents -- every size value here is halved.
"""
import xml.etree.ElementTree as ET
from pd_gains import STIFFNESS_MJC, DAMPING_MJC
from joint_mapping import MUJOCO_JOINT_ORDER

# ---- constants.py / env_cfg_rgp_scene.py values (RGP Policy 1) ----
ROBOT_POS = (0.0, 0.0, 0.76)
ROBOT_ROT = (0.7071, 0.0, 0.0, -0.7071)  # (w,x,y,z)

TABLE_TOP_Z = 0.81
TABLE_POS = (0.04, -0.33, TABLE_TOP_Z - 0.74 / 2.0)
TABLE_SIZE = (0.60, 0.50, 0.74)  # full extents (Isaac Lab convention)

# RGP-specific: 3cm cube (env_cfg_rgp_scene.py's RGP_BLOCK_SIZE), NOT the old
# chain's 6cm BLOCK_SIZE. x/y match constants.py's BLOCK_INIT_POS unchanged
# (RGP_BLOCK_INIT_POS only recomputes z for the new half-size).
BLOCK_SIZE = 0.03
BLOCK_INIT_POS = (-0.065, -0.349, TABLE_TOP_Z + BLOCK_SIZE / 2.0)
CUBE_ROT = (0.9945, 0.0, 0.0, 0.1045)  # (w,x,y,z) -- same as old chain, unchanged

# Policy 3/4's goal marker (env_cfg_rgp_scene.py's RGP_GOAL_POS xy, sitting
# flush on the table like the Isaac Lab disc -- TABLE_TOP_Z, not the cube's
# own resting-centre height). Visual-only: no collision, no joint.
GOAL_MARKER_POS = (0.070, -0.389, TABLE_TOP_Z)

tree = ET.parse("g1_dex1_native.xml")
root = tree.getroot()


def replace_right_hand_with_real_geometry(root):
    """Swaps the substitute hand (2 flat prismatic sliders, no mounting plate --
    structurally different from training's Dex1) for an accurate reconstruction
    of training's ACTUAL hand: a fixed mounting plate + two mirrored prismatic
    fingers, each a rigid 3-piece assembly. See build_scene.py's own docstring
    for the full derivation history -- unchanged here, same robot asset."""
    GRIPPER_STIFFNESS, GRIPPER_DAMPING = 800.0, 3.0  # matches pd_gains.py's gripper group

    REAL_MASS = {
        "right_hand_base_link": 0.110000,
        "right_hand_Link1_1": 0.008440, "right_hand_Link1_2": 0.028354, "right_hand_Link1_3": 0.004583,
        "right_hand_Link2_1": 0.008440, "right_hand_Link2_2": 0.028354, "right_hand_Link2_3": 0.004583,
    }

    def box(name, bbox_min, bbox_max, body_name):
        c = [(mn + mx) / 2.0 for mn, mx in zip(bbox_min, bbox_max)]
        s = [(mx - mn) / 2.0 for mn, mx in zip(bbox_min, bbox_max)]
        mass = REAL_MASS[body_name]
        return ET.fromstring(
            f'<geom name="{name}" type="box" pos="{c[0]} {c[1]} {c[2]}" '
            f'size="{s[0]} {s[1]} {s[2]}" mass="{mass}" rgba="0.4 0.4 0.4 0"/>'
        )

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

    _HAND_QUAT = "0.7071 0 0 -0.7071"

    base = ET.SubElement(wrist, "body", {"name": "right_hand_base_link", "pos": "0.0415 -0.003 0.0", "quat": _HAND_QUAT})
    base.append(box("right_hand_base_link_col", (-0.035, 0.000, -0.035), (0.035, 0.0738, 0.089), "right_hand_base_link"))
    base.append(mesh_visual("right_hand_base_link"))

    l1_1 = ET.SubElement(wrist, "body", {"name": "right_hand_Link1_1", "pos": "0.1008482 -0.04350275 0.0152", "quat": _HAND_QUAT})
    ET.SubElement(l1_1, "joint", {
        "name": "right_hand_Joint1_1", "type": "slide", "axis": "-1 0 0",
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


_real_hand_gripper_stiffness = replace_right_hand_with_real_geometry(root)

option = ET.SubElement(root, "option")
option.set("integrator", "implicitfast")
option.set("timestep", "0.005")
root.remove(option)
root.insert(list(root).index(root.find("compiler")) + 1, option)

default_el = ET.SubElement(root, "default")
ET.SubElement(default_el, "geom", {"solref": "0.01 1"})
root.remove(default_el)
root.insert(list(root).index(root.find("option")) + 1, default_el)

worldbody = root.find("worldbody")

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

ET.SubElement(worldbody, "light", {
    "name": "sun", "directional": "true", "pos": "0 0 3", "dir": "0 0 -1",
    "diffuse": "0.8 0.8 0.8", "specular": "0.2 0.2 0.2", "castshadow": "true",
})
ET.SubElement(worldbody, "light", {
    "name": "fill", "directional": "true", "pos": "1 -1 2", "dir": "-1 1 -2",
    "diffuse": "0.4 0.4 0.4", "castshadow": "false",
})

damping_of = dict(zip(MUJOCO_JOINT_ORDER, DAMPING_MJC))
stiffness_of = dict(zip(MUJOCO_JOINT_ORDER, STIFFNESS_MJC))
actuator_specs = []
for joint_el in robot_wrapper.iter("joint"):
    name = joint_el.get("name")
    if name not in damping_of:
        continue
    joint_el.set("damping", str(damping_of[name]))
    actuator_specs.append((name, stiffness_of[name]))
assert len(actuator_specs) == 33, f"expected 33 actuated joints, found {len(actuator_specs)}"

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

# Goal marker (yellow disc, matching Isaac Lab's RGPSceneCfg.goal exactly --
# same position, same 5cm radius/2mm-thick cylinder). Non-colliding
# (contype/conaffinity 0) so it can never physically interact with anything.
ET.SubElement(
    worldbody, "geom", {
        "name": "goal_marker",
        "type": "cylinder",
        "pos": f"{GOAL_MARKER_POS[0]} {GOAL_MARKER_POS[1]} {GOAL_MARKER_POS[2]}",
        "size": "0.05 0.001",
        "rgba": "1.0 1.0 0.0 1",
        "contype": "0",
        "conaffinity": "0",
    }
)

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

tree.write("g1_rgp_reach_scene.xml", xml_declaration=False)
print(f"wrote g1_rgp_reach_scene.xml with {len(actuator_specs)} position actuators")
