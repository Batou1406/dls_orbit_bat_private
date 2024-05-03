# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.sensors import ContactSensor

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv

# TODO : This function needs a rework
def feet_air_time(env: RLTaskEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def penalize_leg_frequency(env: RLTaskEnv, action_name: str, bound: tuple[float, float]) -> torch.Tensor:
    """ Penalize leg frequency that are outside boundaries in ]-inf, 0]
    Penalize linearly with frequency violation

    Returns :
        - penalty (torch.Tensor): penalty term in ]-inf, 0] for leg frequency outside bound of shape(batch_size)
    """
    f:torch.Tensor = env.action_manager.get_term(action_name).f

    penalty = -torch.sum(torch.abs(f-f.clamp(bound[0], bound[1])), dim=1)

    return penalty


def penalize_leg_duty_cycle(env: RLTaskEnv, action_name: str, bound: tuple[float, float]) -> torch.Tensor:
    """ Penalize leg duty cycle that are outside boundaries in ]-inf, 0]
    Penalize linearly with duty cycle violation

    Returns :
        - penalty (torch.Tensor): penalty term in ]-inf, 0] for leg duty cycle outside bound of shape(batch_size)
    """
    d:torch.Tensor = env.action_manager.get_term(action_name).d

    penalty = -torch.sum(torch.abs(d-d.clamp(bound[0], bound[1])), dim=1)

    return penalty

def penalize_big_steps(env: RLTaskEnv, action_name: str, bound_x: tuple[float, float], bound_y: tuple[float, float]) -> torch.Tensor:
    """ Penalize leg duty cycle that are outside boundaries in ]-inf, 0]
    Penalize linearly with duty cycle violation

    Returns :
        - penalty (torch.Tensor): penalty term in ]-inf, 0] for leg duty cycle outside bound of shape(batch_size)
    """
    d:torch.Tensor = env.action_manager.get_term(action_name).d

    penalty = -torch.sum(torch.abs(d-d.clamp(bound[0], bound[1])), dim=1)

    return penalty