# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`omni.isaac.orbit.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import Articulation, RigidObject
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.sensors import ContactSensor

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv
    from omni.isaac.orbit.managers.command_manager import CommandTerm


def base_height_bounded(env: RLTaskEnv, height_bound: tuple[float, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate when the asset's height is outside the provided height bound.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.

    Args :
        height_bound: [min, max] height bound to consider for the termination
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return (height_bound[0] > asset.data.root_pos_w[:, 2]) + (asset.data.root_pos_w[:, 2] < height_bound[1])
