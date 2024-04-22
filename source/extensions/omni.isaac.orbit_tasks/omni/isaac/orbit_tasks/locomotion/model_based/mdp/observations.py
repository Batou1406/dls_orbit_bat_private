# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Model Base specific functions that can be used to create observation terms.

The functions can be passed to the :class:`omni.isaac.orbit.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import BaseEnv, RLTaskEnv


def leg_phase(env: BaseEnv, action_name: str) -> torch.Tensor:
    """ The last leg phase used by the model based controller

    Returns :
        - leg_phase : the internal leg phase used by the model base controller of shape(batch_size, num_legs)
    """
    return env.action_manager.get_term(action_name).controller.phase

# def last_action(env: BaseEnv, action_name: str | None = None) -> torch.Tensor:
#     """The last input action to the environment.

#     The name of the action term for which the action is required. If None, the
#     entire action tensor is returned.
#     """
#     if action_name is None:
#         return env.action_manager.action
#     else:
#         return env.action_manager.get_term(action_name).raw_actions


# def generated_commands(env: RLTaskEnv, command_name: str) -> torch.Tensor:
#     """The generated command from command term in the command manager with the given name."""
#     return env.command_manager.get_command(command_name)