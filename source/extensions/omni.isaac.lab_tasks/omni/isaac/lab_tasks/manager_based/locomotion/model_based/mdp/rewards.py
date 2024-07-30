# Copyright (c) 2022-2024, The LAB Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING

from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor
from omni.isaac.lab_tasks.manager_based.locomotion.model_based.mdp.actions import ModelBaseAction
from omni.isaac.lab.assets.articulation import Articulation

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

# TODO : This function needs a rework
def feet_air_time(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
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


def penalize_large_leg_frequency_L1(env: ManagerBasedRLEnv, action_name: str, bound: tuple[float, float]) -> torch.Tensor:
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


def penalize_large_leg_duty_cycle_L1(env: ManagerBasedRLEnv, action_name: str, bound: tuple[float, float]) -> torch.Tensor:
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


def penalize_large_steps_L1(env: ManagerBasedRLEnv, action_name: str, bound_x: tuple[float, float], bound_y: tuple[float, float]) -> torch.Tensor:
    """ Penalize steps that are outside boundaries, penalty in ]-inf, 0]
    Penalize linearly with steps size violation

    Args :
        - bound_x (float, float): Boundary in which the step size in x direction isn't penalize
        - bound_y (float, float): Boundary in which the step size in y direction isn't penalize

    Returns :
        - penalty (torch.Tensor): penalty term in ]-inf, 0] for leg step size outside bound of shape(batch_size)
    """
    # Shape (batch_size, num_legs, 3, number_predict step) -> (batch_size, num_legs, 2)
    p:torch.Tensor = env.action_manager.get_term(action_name).delta_p_h[...,0]

    penalty_x = -torch.sum(torch.abs(p[:,:,0]-p[:,:,0].clamp(bound_x[0], bound_x[1])), dim=1)
    penalty_y = -torch.sum(torch.abs(p[:,:,1]-p[:,:,1].clamp(bound_y[0], bound_y[1])), dim=1)

    penalty = penalty_x + penalty_y
    return penalty


def penalize_large_Forces_L1(env: ManagerBasedRLEnv, action_name: str, bound: tuple[float, float]) -> torch.Tensor:
    """ Penalize Forces that are outside boundaries, penalty in ]-inf, 0]
    Penalize linearly with Forces violation

    Args :
        - bound   (float, float): Boundary in which the force isn't penalize

    Returns :
        - penalty (torch.Tensor): penalty term in ]-inf, 0] for Forces outside bound of shape(batch_size)
    """
    #shape (batch_size, num_legs, 3, time_horizon) ->(batch_size, num_legs, 3)
    F:torch.Tensor = env.action_manager.get_term(action_name).F0_star_lw

    # Compute the norm -> shape(batch_size, num_legs)
    F = torch.linalg.vector_norm(F, dim=2)

    penalty = -torch.sum(torch.abs(F-F.clamp(bound[0], bound[1])), dim=1)

    return penalty


def penalize_frequency_variation_L2(env: ManagerBasedRLEnv, action_name: str) -> torch.Tensor:
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


def penalize_duty_cycle_variation_L2(env: ManagerBasedRLEnv, action_name: str) -> torch.Tensor:
    """ Penalize leg duty cycle variation quadraticaly with L2 kernel (penalty term in ]-inf, 0])

    Returns :
        - penalty (torch.Tensor): penalty term in ]-inf, 0] for leg duty cycle variation of shape(batch_size)
    """
    # Shape (batch_size, num_legs, 3, number_predict step) -> (batch_size, num_legs * 3 * number_predict_step)
    d:     torch.Tensor = env.action_manager.get_term(action_name).d.flatten(1,-1)
    d_prev:torch.Tensor = env.action_manager.get_term(action_name).d_prev.flatten(1,-1)

    penalty = -torch.sum(torch.square(d-d_prev), dim=1)

    return penalty


def penalize_steps_variation_L2(env: ManagerBasedRLEnv, action_name: str) -> torch.Tensor:
    """ Penalize steps variation quadraticaly with L2 kernel (penalty term in ]-inf, 0])

    Returns :
        - penalty (torch.Tensor): penalty term in ]-inf, 0] for step variation of shape(batch_size)
    """
    # Shape (batch_size, num_legs, 3, number_predict step) -> (batch_size, num_legs * 3 * number_predict_step)
    p:     torch.Tensor = env.action_manager.get_term(action_name).delta_p_h.flatten(1,-1)
    p_prev:torch.Tensor = env.action_manager.get_term(action_name).delta_p_h_prev.flatten(1,-1)

    penalty = -torch.sum(torch.square(p-p_prev), dim=1)

    return penalty


def penalize_Forces_variation_L2(env: ManagerBasedRLEnv, action_name: str) -> torch.Tensor:
    """ Penalize Ground Reaction Forces variation quadraticaly with L2 kernel (penalty term in ]-inf, 0])

    Returns :
        - penalty (torch.Tensor): penalty term in ]-inf, 0] for Forces (GRF) variation of shape(batch_size)
    """
    # extract the used quantities (to enable type-hinting)
    action : ModelBaseAction = env.action_manager.get_term(action_name)

    # Shape (batch_size, num_legs, 3, prediction_horizon) -> (batch_size, num_legs * 3 * prediction_horizon)
    F:     torch.Tensor = action.delta_F_h.flatten(1,-1)
    F_prev:torch.Tensor = action.delta_F_h_prev.flatten(1,-1)

    penalty = -torch.sum(torch.square(F-F_prev), dim=1)

    return penalty


def friction_constraint(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, mu: float = 0.8) -> torch.Tensor:
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


def penalize_foot_in_contact_displacement_l2(env: ManagerBasedRLEnv, actionName: str="model_base_variable", assetName: str="robot") -> torch.Tensor:
    """ Penalize foot spliping by penalizing quadratically the distance the foot moved while in contact

    Args : 
        - actionName (str): Action term name. The actionTerm must be of type 'ModelBaseAction'.
        - assetName  (str): Asset name of the robot in the simulation
    
    Returns :
        - penalty (torch.Tensor): Penalty term i [0, +inf[ for foot displacement while in contact of shape(num_envs)"""
    
    # extract the used quantities (to enable type-hinting)
    action: ModelBaseAction = env.action_manager.get_term(actionName)
    robot: Articulation = env.scene[assetName]

    # Retrieve the foot linear velocity -> proportionnal to the foot displacement : shape(batch, legs)
    p_dot_w = torch.norm(robot.data.body_lin_vel_w[:, action.foot_idx,:], dim=2)

    # Retrieve which foot is in contact : True if foot in contact, False in in swing, shape (batch_size, num_legs)
    in_contact = action.c0_star == 1

    # Sum the foot velocity for only for legs in contact : shape(batch_size)
    penalty = torch.sum(p_dot_w * in_contact, dim=1)

    return penalty


def reward_terrain_progress(env: ManagerBasedRLEnv, assetName: str="robot") -> torch.Tensor:
    """ Reward for progress made in the terrain
    
    Returns :
        - reward (torch.Tensor): """
    
    # extract the used quantities (to enable type-hinting)
    robot: Articulation = env.scene[assetName]

    # Reward linearly for average speed in terrain progression
    reward = (torch.norm(robot.data.root_pos_w - env.scene.env_origins, dim=1).clamp(min=0, max=0.7)) / (env.episode_length_buf * env.step_dt)

    return reward


def penalize_cost_of_transport(env: ManagerBasedRLEnv, assetName: str="robot", alpha: float=0.3) -> torch.Tensor:
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
    speed = torch.clamp_min(torch.norm(robot.data.root_lin_vel_b, dim=1) , min=0.1) 
    # Compute the Robot's power [W] : T*q_dot : mechanical power, alpha*T^2 : stalk torque power dissipation, no negative value : no regenerative breaking
    power = torch.sum((torque*q_dot + alpha*torque*torque).clamp(min=0.0), dim=1)

    # Compute the Cost of Transport : P/(m*g*v)
    CoT = (power) / (9.81 * 20 * speed)

    penalty = CoT

    return penalty


def soft_track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using modified exponential kernel for soft tracking.
    Maximal reward is awarded for a plateau ranging from [v_cmd - epislon*difficulty, v_cmd]
    With difficulty, the terrain difficulty, thus enabling more flexibility in the speed tracking for challenging terrain"""

    # Retrieve used quantities
    v_cmd = env.command_manager.get_command(command_name)[:, :2]    # Commanded speed in xy plane                  : shape(batch_size, 2)
    v_rob = env.scene[asset_cfg.name].data.root_lin_vel_b[:, :2]    # Robot's speed in xy plane                    : shape(batch_size, 2)
    difficulty = env.scene.terrain.terrain_levels.float()           # Terrain's actual difficulty (sampled)        : shape(batch_size,)
    # difficulty = env.scene.terrain.difficulty.float()             # Terrain maximum difficulty  (sampling range) : shape(batch_size,)
          
    # Compute the dot product : shape(batch_size)
    dot_product = torch.sum(v_cmd * v_rob, dim=1) #torch.dot(v_cmd, v_rob)

    # Compute the magnitudes (norms) : shape(batch_size)
    v_cmd_norm = torch.norm(v_cmd, dim=1)
    v_rob_norm = torch.norm(v_rob, dim=1)

    # Compute the cosine of the angle : shape(batch_size)
    cos_theta = dot_product / (v_cmd_norm * v_rob_norm + 1e-7)

    # Compute the angle in radians between commanded speed and robot speed in xy plane : shape(batch_size)
    theta = torch.acos(cos_theta)

    # Project the robot's speed on the commanded speed and compute the error as parrallel (forward) and perpendicular (lateral) component
    # forward_speed_error = (torch.norm(v_rob)*torch.cos(theta)) - torch.norm(v_cmd) # shape(batch_size)
    # lateral_speed_error = torch.norm(v_rob)*torch.sin(theta)                       # shape(batch_size)
    forward_speed_error = (v_rob_norm*cos_theta) - v_cmd_norm # shape(batch_size)
    lateral_speed_error = v_rob_norm*torch.sin(theta)                       # shape(batch_size)


    # Give some tolerance on the forward error (difficulty in [0,10]. Give 50% of the commanded speed * difficulty in % as tolerance)
    tol = 0.05 * v_cmd_norm * difficulty

    # Apply the tolerance : continous function, with error = 0 if error in [-tol, 0]
    forward_speed_error = torch.where(
        forward_speed_error > 0, forward_speed_error, # if error > 0 -> error=error
        torch.where(
            forward_speed_error < -tol, forward_speed_error + tol, # if error < -tol -> error=error+tol
            0 # if error inside tolerance -> error=0
        ))
    # forward_speed_error[forward_speed_error <= -tol] += tol[forward_speed_error <= -tol] # Same results but slower implementation
    # forward_speed_error[forward_speed_error <=   0 ]  = 0

    # compute the error
    error = torch.square(forward_speed_error) + torch.square(lateral_speed_error) # shape(batch_size)

    return torch.exp(-error / std**2)


def track_proprioceptive_height_exp(env: ManagerBasedRLEnv, target_height: float, height_bound: tuple[float,float]|None=None, std: float=math.sqrt(0.25), assetName: str="robot", footName: str=".*foot", method: str="Action", actionName: str="model_base_variable", sensorCfg:SceneEntityCfg|None=None) -> torch.Tensor:
    """Penalize asset height from its target using exponential-kernel. The height is the proprioceptive height, which is the
    height distance between the CoM and the average position (z only) of the feet in contact.

    Two methods are proposed to determine the feet in contact :
        - Using the contact variable c from the ModelBasedAction term (method=='Action')
        - Using the force sensor on the feet  (method=='Sensor') (eg. SceneEntityCfg("contact_forces", body_names=".*foot"))


    Args :
        target_height       (float): Target height to track
        height_bound (float, float): lower and upper bound  arround the target height before the robot is penalized (eg. (-0.1,+0.1))
        std                 (float): Standard variation for the exponential kernel
        assetName             (str): Name of the 'robot' in the scene manager
        actionName            (str): Name of the action term (ModelBasedAction) in the action manager
        footName              (str): Regex Epression to retrieve the foot indexes in the robot

    Return :
        penalty      (torch.Tensor): Penalty term using L2 kernel in [0, +inf] 
    """
    # extract the used quantities (to enable type-hinting)
    robot: Articulation = env.scene[assetName]

    # Retrieve the foot idx
    foot_idx = robot.find_bodies(footName)[0]

    # Retrieve the feet height : shape(batch_size, num_legs)
    foot_height_w = robot.data.body_pos_w[:,foot_idx,2]

    # Retrieve the CoM height : shape(batch_size) 
    CoM_height_w = robot.data.root_pos_w[:,2]

    # Retrieve the number of feet in contact (set as a minimum of 1 to avoid division by zero)
    if method == "Action":
        action: ModelBaseAction = env.action_manager.get_term(actionName)
        feet_in_contact = action.c0_star
        num_feet_in_contact = (torch.sum(feet_in_contact, dim=1)).clamp(min=1)
    elif method == "Sensor":
        sensor: ContactSensor = env.scene.sensors[sensorCfg.name]
        contact_forces = sensor.data.net_forces_w_history[:,0,sensorCfg.body_ids,2] # 0=last measurement, 2=z dim, shape(batch_size, num_legs)
        feet_in_contact = contact_forces > 1 # set contact threshold to 1 [N]
        num_feet_in_contact = (torch.sum(feet_in_contact, dim=1)).clamp(min=1) 
    else :
        raise NotImplementedError("Provided method for proprioceptive height not in {Action, Sensor}")

    # Compute the proprioceptive robot height
    robot_height_prop = CoM_height_w - ((torch.sum(foot_height_w * feet_in_contact, dim=1)) / num_feet_in_contact)

    # Compute the tracking error
    tracking_error = robot_height_prop - target_height

    # If tolerance bound are provided adjusted error with bounds
    if height_bound is not None :
        tracking_error = torch.where(
            robot_height_prop > target_height + height_bound[1],    
            tracking_error - height_bound[1],
            torch.where(
                robot_height_prop < target_height + height_bound[0],
                tracking_error - height_bound[0],
                torch.zeros_like(tracking_error)                    # 0 if the error is inside boundaries
            )
        )

    reward =  torch.exp(-torch.square(tracking_error) / std**2)
    
    return reward


def penalize_close_feet(env: ManagerBasedRLEnv, assetName: str="robot", threshold: float=0.05, footName: str=".*foot", kernel: str='constant'):
    """Penalize feet that are too close to each other (in the xy plane), with the XX kernel
    
    Args :
        assetName   (str): Name of the 'robot' in the scene manage
        threshold (float): Distance threshold where the feet are considered too close to each other and would be penalize
        footName    (str): Regex Epression to retrieve the foot indexes in the robot 'articulation'
        kernel      (str): Kernel type for the penalty in 'constant', 'linear', 'quadratic'

    Returns :
        penalty (torch.Tensor): penalty term in [0, +inf]
    """
    # extract the used quantities (to enable type-hinting)
    robot: Articulation = env.scene[assetName]

    # Retrieve the foot idx
    foot_idx = robot.find_bodies(footName)[0]

   # Retrieve foot position in world frame
    foot_xy_pos_w = robot.data.body_pos_w[:,foot_idx,:2]

    # Augment the dimension to  efficiently compute the difference
    foot_xy_pos1_w = foot_xy_pos_w.unsqueeze(2) # shape(batch_size, num_legs,        1, 2)
    foot_xy_pos2_w = foot_xy_pos_w.unsqueeze(1) # shape(batch_size,        1, num_legs, 2)

    # Compute the pair wise difference of the foot positions (it compute 16 difference while only 6 are need but it's a fast and optimized operation for GPU)
    foot_pos_diff = foot_xy_pos1_w - foot_xy_pos2_w # shape(batch_size, num_legs, num_legs, 2) 

    # compute the norm of the position difference to obtain the distance between the feet
    foot_distances = torch.norm(foot_pos_diff, dim=3) # shape(batch_size, num_legs, num_legs)

    # Retrieve only the distances of interest, ie. the upper diagonal of foot_distances (diagonal=0 : diff between same foot, lowerDiag=upperDiag ||f1-f2|| = ||f2-f1||)
    foot_distances = foot_distances.triu(diagonal=1) 
    upper_triangle_indices = torch.triu_indices(4, 4, offset=1)
    foot_distances_diag = foot_distances[:, upper_triangle_indices[0], upper_triangle_indices[1]]# shape(batch_size, 6)

    # Compute the penalty
    if kernel == 'quadratic':
        # Compute the L2 penalty : with f(x)=ax²+bx+x, with f(x)=1, f(+/-threshold)=0
        penalty = (-1/(threshold**2))*(foot_distances_diag**2) + 1
        penalty= torch.sum(torch.where(penalty < 0, 0, penalty), dim=1)
    elif kernel == 'linear':
        # Compute the linear penalty : f(0)=1, f(+/-threshold)=0
        penalty = 1-(1/threshold)*torch.abs(foot_distances_diag)
        penalty= torch.sum(torch.where(penalty < 0, 0, penalty), dim=1)
    elif kernel == 'constant' :
        penalty = torch.sum(foot_distances_diag < threshold,dim=1).float()

    return penalty


def penalize_foot_trajectory_tracking_error(env: ManagerBasedRLEnv, assetName: str="robot", actionName: str="model_base_variable", footName: str=".*foot"): 
    """ Penalize for error between the desired foot touch down position and the actual foot touch down position (in the xy plane only)

    Args :
        assetName        (str): Name of the 'robot' in the scene manage
        actionName       (str): Name of the action term (ModelBasedAction) in the action manager
        footName         (str): Regex Epression to retrieve the foot indexes in the robot 'articulation'

    Returns :
        penalty (torch.Tensor): penalty term in [0, +inf]
    """
    # extract the used quantities (to enable type-hinting)
    robot: Articulation = env.scene[assetName]
    action: ModelBaseAction = env.action_manager.get_term(actionName)

    # Retrieve the foot idx
    foot_idx = robot.find_bodies(footName)[0]

   # Retrieve foot position in local world frame
    foot_xy_pos_lw = robot.data.body_pos_w[:,foot_idx,:2] - env.scene.env_origins[:,:2].unsqueeze(1)  # shape (num_envs, num_foot, 2)

    # Retrieve foot desired trajectory in local world frame
    pt0_star_xy_lw = action.pt_star_lw[:,:,:2,0] # shape (num_envs, num_foot, 2)

    # Compute the penalty for the leg in swing
    penalty = torch.sum((~action.c0_star) * torch.sum(torch.square(foot_xy_pos_lw - pt0_star_xy_lw), dim=-1), dim=-1) # shape (num_envs, num_foot, 2) -> (num_envs, num_foot) -> (num_envs)

    return penalty









