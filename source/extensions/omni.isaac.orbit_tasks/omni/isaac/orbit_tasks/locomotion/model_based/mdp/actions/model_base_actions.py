# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import carb

import omni.isaac.orbit.utils.string as string_utils
from omni.isaac.orbit.assets.articulation import Articulation
from omni.isaac.orbit.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import BaseEnv

    from . import actions_cfg

import jax
from jax import random

print("ALO alo ALO")
key = random.key(0)
x = random.normal(key, (10,))
print(x)

class ModelBaseAction(ActionTerm):
    """Base class for model base actions.

    The action term is responsible for processing the raw actions sent to the environment
    and applying them to the asset managed by the term. The action term is comprised of two
    operations:

    * Processing of actions: This operation is performed once per **environment step** and
      is responsible for pre-processing the raw actions sent to the environment.
    * Applying actions: This operation is performed once per **simulation step** and is
      responsible for applying the processed actions to the asset managed by the term.
    """

    cfg: actions_cfg.ModelBaseActionCfg

    def __init__(self, cfg: actions_cfg.ModelBaseActionCfg, env: BaseEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)


    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        raise NotImplementedError

    @property
    def raw_actions(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def processed_actions(self) -> torch.Tensor:
        raise NotImplementedError
    


    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        """Processes the actions sent to the environment.
        Note: This function is called once per environment step by the manager.
        Args: actions: The actions to process.
        """
        raise NotImplementedError
    
    def apply_actions(self):
        """Applies the actions to the asset managed by the term.
        Note: This is called at every simulation step by the manager.
        """
        raise NotImplementedError
