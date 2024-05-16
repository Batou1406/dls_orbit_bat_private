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


def penalize_large_leg_frequency_L1(env: RLTaskEnv, action_name: str, bound: tuple[float, float]) -> torch.Tensor:
    """ Penalize leg frequency that are outside boundaries, penalty in ]-inf, 0]
    Penalize linearly with frequency violation

    Args :
        - bound   (float, float): Boundary in which the leg frequency isn't penalize

    Returns :
        - penalty (torch.Tensor): penalty term in ]-inf, 0] for leg frequency outside bound of shape(batch_size)
    """
    f:torch.Tensor = env.action_manager.get_term(action_name).f

    penalty = -torch.sum(torch.abs(f-f.clamp(bound[0], bound[1])), dim=1)

    return penalty


def penalize_large_leg_duty_cycle_L1(env: RLTaskEnv, action_name: str, bound: tuple[float, float]) -> torch.Tensor:
    """ Penalize leg duty cycle that are outside boundaries, penalty in ]-inf, 0]
    Penalize linearly with duty cycle violation

    Args :
        - bound   (float, float): Boundary in which the leg duty cycle isn't penalize

    Returns :
        - penalty (torch.Tensor): penalty term in ]-inf, 0] for leg duty cycle outside bound of shape(batch_size)
    """
    d:torch.Tensor = env.action_manager.get_term(action_name).d

    penalty = -torch.sum(torch.abs(d-d.clamp(bound[0], bound[1])), dim=1)

    return penalty


def penalize_large_steps_L1(env: RLTaskEnv, action_name: str, bound_x: tuple[float, float], bound_y: tuple[float, float], bound_z: tuple[float, float]) -> torch.Tensor:
    """ Penalize steps that are outside boundaries, penalty in ]-inf, 0]
    Penalize linearly with steps size violation

    Args :
        - bound_x (float, float): Boundary in which the step size in x direction isn't penalize
        - bound_y (float, float): Boundary in which the step size in y direction isn't penalize
        - bound_z (float, float): Boundary in which the step size in z direction isn't penalize

    Returns :
        - penalty (torch.Tensor): penalty term in ]-inf, 0] for leg step size outside bound of shape(batch_size)
    """
    # Shape (batch_size, num_legs, 3, number_predict step) -> (batch_size, num_legs, 3)
    p:torch.Tensor = env.action_manager.get_term(action_name).p_norm[...,0]

    penalty_x = -torch.sum(torch.abs(p[:,:,0]-p[:,:,0].clamp(bound_x[0], bound_x[1])), dim=1)
    penalty_y = -torch.sum(torch.abs(p[:,:,1]-p[:,:,1].clamp(bound_y[0], bound_y[1])), dim=1)
    penalty_z = 0

    # if we optimize also for the step height, p is 3 dimensional (x,y,z)
    if env.action_manager.get_term(action_name).cfg.optimize_step_height :
        penalty_z = -torch.sum(torch.abs(p[:,:,2]-p[:,:,2].clamp(bound_z[0], bound_z[1])), dim=1)

    penalty = penalty_x + penalty_y + penalty_z
    return penalty


def penalize_large_Forces_L1(env: RLTaskEnv, action_name: str, bound: tuple[float, float]) -> torch.Tensor:
    """ Penalize Forces that are outside boundaries, penalty in ]-inf, 0]
    Penalize linearly with Forces violation

    Args :
        - bound   (float, float): Boundary in which the force isn't penalize

    Returns :
        - penalty (torch.Tensor): penalty term in ]-inf, 0] for Forces outside bound of shape(batch_size)
    """
    #shape (batch_size, num_legs, 3, time_horizon) ->(batch_size, num_legs, 3)
    F:torch.Tensor = env.action_manager.get_term(action_name).F_lw[...,0]

    # Compute the norm -> shape(batch_size, num_legs)
    F = torch.linalg.vector_norm(F, dim=2)

    penalty = -torch.sum(torch.abs(F-F.clamp(bound[0], bound[1])), dim=1)

    return penalty


def penalize_frequency_variation_L2(env: RLTaskEnv, action_name: str) -> torch.Tensor:
    """ Penalize leg frequency variation quadraticaly with L2 kernel (penalty term in ]-inf, 0])

    Returns :
        - penalty (torch.Tensor): penalty term in ]-inf, 0] for leg frequency variation of shape(batch_size)
    """
    # Shape (batch_size, num_legs, 3, number_predict step) -> (batch_size, num_legs * 3 * number_predict_step)
    f:     torch.Tensor = env.action_manager.get_term(action_name).f.flatten(1,-1)
    f_prev:torch.Tensor = env.action_manager.get_term(action_name).f_prev.flatten(1,-1)

    penalty = -torch.sum(torch.square(f-f_prev), dim=1)

    return penalty


def penalize_duty_cycle_variation_L2(env: RLTaskEnv, action_name: str) -> torch.Tensor:
    """ Penalize leg duty cycle variation quadraticaly with L2 kernel (penalty term in ]-inf, 0])

    Returns :
        - penalty (torch.Tensor): penalty term in ]-inf, 0] for leg duty cycle variation of shape(batch_size)
    """
    # Shape (batch_size, num_legs, 3, number_predict step) -> (batch_size, num_legs * 3 * number_predict_step)
    d:     torch.Tensor = env.action_manager.get_term(action_name).d.flatten(1,-1)
    d_prev:torch.Tensor = env.action_manager.get_term(action_name).d_prev.flatten(1,-1)

    penalty = -torch.sum(torch.square(d-d_prev), dim=1)

    return penalty


def penalize_steps_variation_L2(env: RLTaskEnv, action_name: str) -> torch.Tensor:
    """ Penalize steps variation quadraticaly with L2 kernel (penalty term in ]-inf, 0])

    Returns :
        - penalty (torch.Tensor): penalty term in ]-inf, 0] for step variation of shape(batch_size)
    """
    # Shape (batch_size, num_legs, 3, number_predict step) -> (batch_size, num_legs * 3 * number_predict_step)
    p:     torch.Tensor = env.action_manager.get_term(action_name).p_norm.flatten(1,-1)
    p_prev:torch.Tensor = env.action_manager.get_term(action_name).p_norm_prev.flatten(1,-1)

    penalty = -torch.sum(torch.square(p-p_prev), dim=1)

    return penalty


def penalize_Forces_variation_L2(env: RLTaskEnv, action_name: str) -> torch.Tensor:
    """ Penalize Ground Reaction Forces variation quadraticaly with L2 kernel (penalty term in ]-inf, 0])

    Returns :
        - penalty (torch.Tensor): penalty term in ]-inf, 0] for Forces (GRF) variation of shape(batch_size)
    """
    # Shape (batch_size, num_legs, 3, prediction_horizon) -> (batch_size, num_legs * 3 * prediction_horizon)
    F:     torch.Tensor = env.action_manager.get_term(action_name).F_norm.flatten(1,-1)
    F_prev:torch.Tensor = env.action_manager.get_term(action_name).F_norm_prev.flatten(1,-1)

    penalty = -torch.sum(torch.square(F-F_prev), dim=1)

    return penalty


def friction_constraint(env: RLTaskEnv, sensor_cfg: SceneEntityCfg, mu: float = 0.8) -> torch.Tensor:
    """Penalize contact forces out of the friction cone
 
    Args:
        sensor_cfg: The contact sensor configuration
        mu: the friction coefficient
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w
 
    # ||F_xy|| - (mu*F_z) : if greater than 0 -> friction cone violation : shape(batch_size, num_legs)
    residuals = torch.norm(net_contact_forces[:, sensor_cfg.body_ids, :2], dim=-1) - (mu * net_contact_forces[:, sensor_cfg.body_ids, 2])
 
    # sum along each robot to get the total violation cost : shape(batch_size)
    costs = torch.sum(residuals.clamp(min=0), dim=1)