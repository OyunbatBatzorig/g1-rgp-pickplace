"""Standalone viewer: just the Dex1 gripper (base + both finger chains),
extracted from the full G1 scene, floating in space. Cycles open<->closed
slowly so both extremes are visible."""
import time
import numpy as np
import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path("gripper_only_scene.xml")
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)

GRIPPER_OPEN = -0.02
GRIPPER_CLOSE = 0.0245
PERIOD_S = 3.0  # seconds per open<->close half-cycle

viewer = mujoco.viewer.launch_passive(m, d)
print("viewer open, is_running=", viewer.is_running())

start = time.time()
try:
    while viewer.is_running():
        t = (time.time() - start) % (2 * PERIOD_S)
        frac = t / PERIOD_S if t < PERIOD_S else 2 - t / PERIOD_S  # 0->1->0
        target = GRIPPER_OPEN + frac * (GRIPPER_CLOSE - GRIPPER_OPEN)
        d.ctrl[0] = target
        d.ctrl[1] = target
        mujoco.mj_step(m, d)
        viewer.sync()
        time.sleep(0.005)
except KeyboardInterrupt:
    pass
