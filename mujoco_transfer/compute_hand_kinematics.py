"""Pure math, no simulation needed: derives the real Dex1 hand's kinematic
structure from the two Isaac Lab snapshots (GRIPPER_OPEN / GRIPPER_CLOSE)
already captured. Computes, in each body's own PARENT-LOCAL frame (what MJCF
needs): the fixed relative transform of each rigid sub-segment, and Joint1_1/
Joint2_1's local slide axis + scale (value -> displacement along that axis).
"""
import numpy as np


def quat_to_R(q):
    """wxyz quaternion -> 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-w*z),     2*(x*z+w*y)],
        [2*(x*y+w*z),     1 - 2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),     2*(y*z+w*x),     1 - 2*(x*x+y*y)],
    ])


def local_offset(parent_pos, parent_quat, child_pos):
    """Child position expressed in the parent body's own local frame."""
    R = quat_to_R(parent_quat)
    return R.T @ (np.array(child_pos) - np.array(parent_pos))


# snapshot 1: GRIPPER_OPEN (-0.02)
S1 = {
    "right_wrist_yaw_link": ([-0.21686159074306488, 0.09515005350112915, 0.7168461084365845], [-0.36983323097229004, -0.5145476460456848, -0.5839201211929321, 0.5074459314346313]),
    "right_hand_base_link": ([-0.22796286642551422, 0.10464471578598022, 0.6778863668441772], [0.0970599502325058, 0.04874454811215401, -0.7767535448074341, 0.6203686594963074]),
    "right_hand_Link1_1": ([-0.30008864402770996, 0.10593269020318985, 0.6308752298355103], [0.0970599427819252, 0.04874454066157341, -0.7767534852027893, 0.6203685998916626]),
    "right_hand_Link1_2": ([-0.2899453341960907, 0.10828351229429245, 0.6157231330871582], [0.0970599427819252, 0.048744551837444305, -0.7767534852027893, 0.6203685998916626]),
    "right_hand_Link1_3": ([-0.29231008887290955, 0.11479225754737854, 0.5915048122406006], [0.0970599353313446, 0.048744555562734604, -0.7767535448074341, 0.6203685998916626]),
    "right_hand_Link2_1": ([-0.17910057306289673, 0.13010498881340027, 0.6117192506790161], [0.09705996513366699, 0.04874454811215401, -0.7767534852027893, 0.6203686594963074]),
    "right_hand_Link2_2": ([-0.19434382021427155, 0.13361802697181702, 0.6020599603652954], [0.0970599502325058, 0.04874454066157341, -0.7767535448074341, 0.6203685998916626]),
    "right_hand_Link2_3": ([-0.20436808466911316, 0.11076620221138, 0.5724769830703735], [0.0970599576830864, 0.04874454438686371, -0.7767535448074341, 0.6203686594963074]),
}
# snapshot 2: GRIPPER_CLOSE (0.0245)
S2 = {
    "right_wrist_yaw_link": ([-0.2143624722957611, 0.0889599621295929, 0.714716911315918], [-0.3779735863208771, -0.5130064487457275, -0.5791797637939453, 0.5084396004676819]),
    "right_hand_base_link": ([-0.22509697079658508, 0.097800612449646, 0.6755013465881348], [0.09200391173362732, 0.04648413136601448, -0.7723109126091003, 0.6268255114555359]),
    "right_hand_Link1_1": ([-0.2531777024269104, 0.09609723091125488, 0.6188117861747742], [0.09200391918420792, 0.04648413509130478, -0.7723109722137451, 0.6268255710601807]),
    "right_hand_Link1_2": ([-0.24288681149482727, 0.0982593521475792, 0.6037312746047974], [0.09200390428304672, 0.04648413509130478, -0.7723109126091003, 0.6268255710601807]),
    "right_hand_Link1_3": ([-0.24502654373645782, 0.10438202321529388, 0.579391598701477], [0.09200391173362732, 0.04648413509130478, -0.7723109722137451, 0.6268255710601807]),
    "right_hand_Link2_1": ([-0.21911242604255676, 0.12438872456550598, 0.6183555722236633], [0.09200391173362732, 0.04648413136601448, -0.7723108530044556, 0.6268254518508911]),
    "right_hand_Link2_2": ([-0.23426900804042816, 0.1276828944683075, 0.6084851026535034], [0.09200391918420792, 0.04648413509130478, -0.7723109722137451, 0.6268255114555359]),
    "right_hand_Link2_3": ([-0.2438833862543106, 0.10433115065097809, 0.5791576504707336], [0.09200390428304672, 0.04648413509130478, -0.7723109722137451, 0.6268254518508911]),
}
GRIPPER_OPEN, GRIPPER_CLOSE_VAL = -0.02, 0.024499982595443726  # actual reached value, joint1

for label, S in [("SNAPSHOT 1 (GRIPPER_OPEN)", S1), ("SNAPSHOT 2 (GRIPPER_CLOSE)", S2)]:
    print(f"=== {label} ===")
    wrist_pos, wrist_quat = S["right_wrist_yaw_link"]
    for name in ["right_hand_base_link", "right_hand_Link1_1", "right_hand_Link2_1"]:
        off = local_offset(wrist_pos, wrist_quat, S[name][0])
        print(f"  {name:24s} local-to-wrist offset = {off}")
    # fixed sub-segment offsets, relative to their OWN chain's Link_1 (constant regardless of joint value if truly rigid)
    for chain in ["1", "2"]:
        l1_pos, l1_quat = S[f"right_hand_Link{chain}_1"]
        for seg in ["2", "3"]:
            child = S[f"right_hand_Link{chain}_{seg}"]
            off = local_offset(l1_pos, l1_quat, child[0])
            print(f"  Link{chain}_{seg} local-to-Link{chain}_1 offset = {off}")
    print()

print("=== Joint1_1 / Joint2_1 axis + scale (from wrist-local displacement between snapshots) ===")
for chain in ["1", "2"]:
    wrist_pos1, wrist_quat1 = S1["right_wrist_yaw_link"]
    wrist_pos2, wrist_quat2 = S2["right_wrist_yaw_link"]
    off1 = local_offset(wrist_pos1, wrist_quat1, S1[f"right_hand_Link{chain}_1"][0])
    off2 = local_offset(wrist_pos2, wrist_quat2, S2[f"right_hand_Link{chain}_1"][0])
    delta = off2 - off1
    dist = np.linalg.norm(delta)
    joint_delta_value = GRIPPER_CLOSE_VAL - GRIPPER_OPEN
    print(f"  Link{chain}_1: wrist-local displacement = {delta}  magnitude={dist:.5f}m"
          f"  (joint value delta={joint_delta_value:.5f})  ratio={dist/joint_delta_value:.4f}")
    axis = delta / dist
    print(f"    -> normalized axis (in WRIST-local frame) = {axis}")
