# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.sensors import ContactSensor
from omni.isaac.orbit_tasks.locomotion.model_based.mdp.actions import ModelBaseAction
from omni.isaac.orbit.assets.articulation import Articulation

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
    # extract the used quantities (to enable type-hinting)
    action : ModelBaseAction = env.action_manager.get_term(action_name)

    # Shape (batch_size, num_legs, 3, number_predict step) -> (batch_size, num_legs * 3 * number_predict_step)
    f      = action.f.flatten(1,-1)
    f_prev = action.f_prev.flatten(1,-1)

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
    # extract the used quantities (to enable type-hinting)
    action : ModelBaseAction = env.action_manager.get_term(action_name)

    # Shape (batch_size, num_legs, 3, prediction_horizon) -> (batch_size, num_legs * 3 * prediction_horizon)
    F:     torch.Tensor = action.F_norm.flatten(1,-1)
    F_prev:torch.Tensor = action.F_norm_prev.flatten(1,-1)

    penalty = -torch.sum(torch.square(F-F_prev), dim=1)

    return penalty

def penalize_Forces_variation_orientation_L2(env: RLTaskEnv, action_name: str) -> torch.Tensor:
    """ Penalize Ground Reaction Forces variation quadraticaly with L2 kernel (penalty term in ]-inf, 0])

    Returns :
        - penalty (torch.Tensor): penalty term in ]-inf, 0] for Forces (GRF) variation of shape(batch_size)
    """
    # extract the used quantities (to enable type-hinting)
    action : ModelBaseAction = env.action_manager.get_term(action_name)

    # Shape (batch_size, num_legs, 3, prediction_horizon) -> (batch_size, num_legs * 3 * prediction_horizon)
    F:     torch.Tensor = action.F_norm.flatten(1,-1)
    F_prev:torch.Tensor = action.F_norm_prev.flatten(1,-1)

    penalty = -torch.sum(torch.square(F-F_prev), dim=1)

    return penalty


def friction_constraint(env: RLTaskEnv, sensor_cfg: SceneEntityCfg, mu: float = 0.8) -> torch.Tensor:
    """Penalize contact forces out of the friction cone
 
    Args:
        sensor_cfg: The contact sensor configuration
        mu: the friction coefficient

    Returns :
        - penalty (torch.Tensor): penalty term in [0, +inf[ for friction cone violation
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w
 
    # ||F_xy|| - (mu*F_z) : if greater than 0 -> friction cone violation : shape(batch_size, num_legs)
    residuals = torch.norm(net_contact_forces[:, sensor_cfg.body_ids, :2], dim=-1) - (mu * net_contact_forces[:, sensor_cfg.body_ids, 2])
 
    # sum along each robot to get the total violation cost : shape(batch_size)
    penalty = torch.sum(residuals.clamp(min=0), dim=1)

    return penalty


def penalize_foot_in_contact_displacement_l2(env: RLTaskEnv, actionName: str="model_base_variable", assetName: str="robot") -> torch.Tensor:
    """ Penalize foot spliping by penalizing quadratically the distance the foot moved while in contact

    Args : 
        - actionName (str): Action term name. The actionTerm must be of type 'ModelBaseAction'.
        - assetName  (str): Asset name of the robot in the simulation
    
    Returns :
        - penalty (torch.Tensor): Penalty term i [0, +inf[ for foot displacement while in contact"""
    
    # extract the used quantities (to enable type-hinting)
    action: ModelBaseAction = env.action_manager.get_term(actionName)
    robot: Articulation = env.scene[assetName]

    # Retrieve the foot linear velocity -> proportionnal to the foot displacement : shape(batch, legs)
    p_dot_w = torch.norm(robot.data.body_lin_vel_w[:, action.foot_idx,:], dim=2)

    # Retrieve which foot is in contact : True if foot in contact, False in in swing, shape (batch_size, num_legs)
    in_contact = action.c_star[:,:,0] == 1

    # Compute the penalty only for leg in contact
    penalty = torch.sum(p_dot_w * in_contact)

    return penalty

def reward_terrain_progress(env: RLTaskEnv, assetName: str="robot") -> torch.Tensor:
    """ Reward for progress made in the terrain
    
    Returns :
        - reward (torch.Tensor): """
    
    # extract the used quantities (to enable type-hinting)
    robot: Articulation = env.scene[assetName]

    # Reward linearly for average speed in terrain progression
    reward = (torch.norm(robot.data.root_pos_w - env.scene.env_origins, dim=1).clamp(min=0, max=0.7)) / (env.episode_length_buf * env.step_dt)

    return reward

def penalize_cost_of_transport(env: RLTaskEnv, assetName: str="robot", alpha: float=0.3) -> torch.Tensor:
    """ Penalize for cost of transport : CoT = P/(m*g*v)
    Which is a dimensionless unit that measure the energy efficiency of the displacement
    The energy consumption of the robot is not accessible, and thus is estimated from the motor torque.
    However, the mechanical power P=T*q_dot, neglect stalk talk (for gravity compensation eg.), assume 
    perfect conversion efficiency and perfect regenerative breaking. This is why we reformulate this formula to 
    account for stalk torque and assume no regenerative breaking.
    Moreover, CoT is not defined for 0 speed and tends toward infinity with low speed. Thus speed is clamped to a
    minimum of 0.1 ([m/s])
    
    Note :
        The CoT is computed as presented in 'Fast and Efficient Locomotion via Learned Gait Transitions'
        https://arxiv.org/pdf/2104.04644

    Args :
        alpha (float): Coefficient to compute the stalk torque. For Unitree A1 = 0.3 (see source)

    Returns:
        penalty (torch.Tensor): The CoT in [0, +inf] of shape (batch_size)
    """
    
    # extract the used quantities (to enable type-hinting)
    robot: Articulation = env.scene[assetName]
    
    # Retrieve the joint torques and joint angular velocity of shape(num_envs, num_joints)
    torque = robot.data.applied_torque
    q_dot = robot.data.joint_vel

    # Retrieve the robot speed, clamp to minimum 0.1 to avoid division by 0 and CoT that tends to infinity with low speed
    speed = torch.clamp_min(torch.norm(robot.data.root_vel_b[:,:3], dim=1) , min=0.1) 

    # Compute the Robot's power [W] : T*q_dot : mechanical power, alpha*T^2 : stalk torque power dissipation, no negative value : no regenerative breaking
    power = torch.sum(torch.max(torque*q_dot + alpha*torque*torque, 0), dim=1)

    # Compute the Cost of Transport : P/(m*g*v)
    CoT = (power) / (9.81 * 20 * speed)

    penalty = CoT

    return penalty


def soft_track_lin_vel_xy_exp(
    env: RLTaskEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using modified exponential kernel for soft tracking.
    Maximal reward is awarded for a plateau ranging from [v_cmd - epislon*difficulty, v_cmd]
    With difficulty, the terrain difficulty, thus enabling more flexibility in the speed tracking for challenging terrain"""

    # Retrieve used quantities
    v_cmd = env.command_manager.get_command(command_name)[:, :2]    # Commanded speed in xy plane   : shape(batch_size, 2)
    v_rob = env.scene[asset_cfg.name].data.root_lin_vel_b[:, :2]    # Robot's speed in xy plane     : shape(batch_size, 2)
    difficulty = env.scene.terrain.difficulty.float()               # Terrain difficulty            : shape(batch_size,)

    # Compute the dot product : shape(batch_size)
    dot_product = torch.sum(v_cmd * v_rob, dim=1) #torch.dot(v_cmd, v_rob)

    # Compute the magnitudes (norms) : shape(batch_size)
    v_cmd_norm = torch.norm(v_cmd)
    v_rob_norm = torch.norm(v_rob)

    # Compute the cosine of the angle : shape(batch_size)
    cos_theta = dot_product / (v_cmd_norm * v_rob_norm)

    # Compute the angle in radians between commanded speed and robot speed in xy plane : shape(batch_size)
    theta = torch.acos(cos_theta)

    # Project the robot's speed on the commanded speed and compute the error as parrallel (forward) and perpendicular (lateral) component
    forward_speed_error = torch.norm(v_cmd) - (torch.norm(v_rob)*torch.cos(theta)) # shape(batch_size)
    lateral_speed_error = torch.norm(v_rob)*torch.sin(theta)                       # shape(batch_size)

    # Give some tolerance on the forward error (difficulty in [0,10]. Give 50% of the commanded speed * difficulty in % as tolerance)
    tol = 0.05 * v_cmd_norm * difficulty
    forward_speed_error[forward_speed_error <= -tol] += tol[forward_speed_error <= -tol]
    forward_speed_error[forward_speed_error <=   0 ]  = 0

    # compute the error
    error = torch.square(forward_speed_error) + torch.square(lateral_speed_error) # shape(batch_size)

    return torch.exp(-error / std**2)