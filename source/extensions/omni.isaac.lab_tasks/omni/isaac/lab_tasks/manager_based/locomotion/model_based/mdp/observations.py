# Copyright (c) 2022-2024, The lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Model Base specific functions that can be used to create observation terms.

The functions can be passed to the :class:`omni.isaac.lab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from omni.isaac.lab_tasks.manager_based.locomotion.model_based.mdp.actions import ModelBaseAction


def leg_phase(env: ManagerBasedEnv, action_name: str) -> torch.Tensor:
    """ The last leg phase used by the model based controller

    Returns :
        - leg_phase : the internal leg phase used by the model base controller of shape(batch_size, num_legs)
    """
    return env.action_manager.get_term(action_name).controller.phase


def leg_contact(env: ManagerBasedEnv, action_name: str) -> torch.Tensor:
    """ The last contact used by the model based controller (computed by doing phase < d)
    c==0 : Leg in swing
    c==1 : Leg in stance

    Returns :
        - leg_contact : the internal leg phase used by the model base controller of shape(batch_size, num_legs)
    """
    return env.action_manager.get_term(action_name).c0_star


def last_model_base_action(env: ManagerBasedEnv, action_name: str | None = 'None') -> torch.Tensor:
    """The last input action to the environment.

    The name of the action term for which the action is required. If None, the
    entire action tensor is returned.
    """

    modelBaseAction: ModelBaseAction = env.action_manager.get_term(action_name)

    return modelBaseAction.RL_applied_actions