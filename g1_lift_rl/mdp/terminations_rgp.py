# g1_lift_rl/mdp/terminations_rgp.py
"""Termination functions for the RGP chain. Policy 1 (reach) only."""
from __future__ import annotations

import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv

from ..constants import TABLE_TOP_Z

_DROP_MARGIN_RGP = 0.10    # m below table top counts as dropped/knocked off
_LAUNCH_MARGIN_RGP = 0.30  # m above table top counts as launched (a contact event, not a real lift)


def object_dropped_rgp(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Cube has fallen well below the table top."""
    obj: RigidObject = env.scene["object"]
    return obj.data.root_pos_w[:, 2] < (TABLE_TOP_Z - _DROP_MARGIN_RGP)


def object_launched_rgp(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Cube rocketed far above any sane task height."""
    obj: RigidObject = env.scene["object"]
    return obj.data.root_pos_w[:, 2] > (TABLE_TOP_Z + _LAUNCH_MARGIN_RGP)
