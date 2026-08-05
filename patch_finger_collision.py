#!/usr/bin/env python3
"""Patch the G1+Dex1 USD so the two right-hand finger chains never collide with
each other or with the hand base, while leaving self-collision ON everywhere else.

Why: with enabled_self_collisions=True, the Dex1 fingers' simplified collision
hulls clip each other well inside the joint's own travel range (+0.0007 reached
instead of +0.0245 commanded). That block point is *inside* the joint limit, not
at it, so it's a collision-mesh artifact, not a real mechanical interaction.
UsdPhysics.FilteredPairsAPI removes just that one spurious pair; finger-vs-cube
contact and all other self-collision in the robot are untouched.

The original asset under unitree_sim_isaaclab is never written to -- this writes
a new sibling file in the SAME directory (so any relative sublayer references in
the source USD keep resolving correctly), and g1_lift_rl/constants.py's ROBOT_USD
is what gets repointed at it.

Run (see INSTALL.md Step 3 -- edit SRC/DST below to your own unitree_sim_isaaclab
checkout path first):
    cd ~/projects/g1_lift_ext
    python patch_finger_collision.py
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Usd, UsdPhysics

SRC = (
    "/home/virtual-acc/projects/unitree_sim_isaaclab/assets/robots/"
    "g1-29dof-dex1-base-fix-usd/g1_29dof_with_dex1_base_fix1.usd"
)
DST = (
    "/home/virtual-acc/projects/unitree_sim_isaaclab/assets/robots/"
    "g1-29dof-dex1-base-fix-usd/g1_29dof_with_dex1_base_fix1_fingerfilter.usd"
)

CHAIN1 = ["right_hand_Link1_1", "right_hand_Link1_2", "right_hand_Link1_3"]
CHAIN2 = ["right_hand_Link2_1", "right_hand_Link2_2", "right_hand_Link2_3"]
BASE = "right_hand_base_link"


def main():
    stage = Usd.Stage.Open(SRC)
    root = stage.GetDefaultPrim().GetPath()

    def p(name):
        return root.AppendChild(name)

    # symmetric adjacency: every chain1 link <-> every chain2 link, and every
    # finger link (both chains) <-> the hand base. NOT chain1-vs-chain1 or
    # chain2-vs-chain2 (adjacent links within one finger are already excluded
    # from self-collision by the articulation's own joint-connectivity rule).
    adjacency: dict[str, set[str]] = {}
    pairs = [(a, b) for a in CHAIN1 for b in CHAIN2] + [(a, BASE) for a in CHAIN1 + CHAIN2]
    for a, b in pairs:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    for name, targets in adjacency.items():
        prim = stage.GetPrimAtPath(p(name))
        if not prim.IsValid():
            raise RuntimeError(f"prim not found: {p(name)}")
        api = UsdPhysics.FilteredPairsAPI.Apply(prim)
        rel = api.CreateFilteredPairsRel()
        for t in sorted(targets):
            rel.AddTarget(p(t))
        print(f"[patch] {name:22s} -> filtered against {sorted(targets)}")

    stage.GetRootLayer().Export(DST)
    print(f"\n[patch] exported -> {DST}")

    # sanity: re-open the exported file fresh and confirm the relationships stuck
    check = Usd.Stage.Open(DST)
    bad = []
    for name, targets in adjacency.items():
        prim = check.GetPrimAtPath(p(name))
        api = UsdPhysics.FilteredPairsAPI(prim)
        got = {t.name for t in api.GetFilteredPairsRel().GetTargets()}
        if got != targets:
            bad.append((name, targets, got))
    if bad:
        print("\n[patch] VERIFY FAILED for:")
        for name, want, got in bad:
            print(f"   {name}: want={sorted(want)} got={sorted(got)}")
    else:
        print("[patch] VERIFY OK: all filtered-pair relationships round-tripped correctly.")

    simulation_app.close()


if __name__ == "__main__":
    main()
