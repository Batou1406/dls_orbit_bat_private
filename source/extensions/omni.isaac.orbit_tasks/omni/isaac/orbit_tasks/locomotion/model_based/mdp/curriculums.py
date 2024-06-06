# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`omni.isaac.orbit.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
Copied from locomotion.mdp module
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import Articulation
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.terrains import TerrainImporter, TerrainImporterUniformDifficulty
from omni.isaac.orbit_tasks.locomotion.model_based.mdp import CurriculumUniformVelocityCommand

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv


def terrain_levels_vel(
    env: RLTaskEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`omni.isaac.orbit.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level)
    return torch.mean(terrain.terrain_levels.float())


def improved_terrain_levels_vel(
    env: RLTaskEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`omni.isaac.orbit.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    speed_command = env.command_manager.get_command("base_velocity")

    # compute the distance the robot walked : not only in the xy plane but in 3D space ! Usefull for very big steps 
    distance = torch.norm(asset.data.root_pos_w[env_ids, :] - env.scene.env_origins[env_ids, :], dim=1)

    # robots that reached 80% of the distance the the border progress to harder terrains + must be time_out reset. Ie. can't be a fall or other early termination condition
    move_up = (distance > terrain.cfg.terrain_generator.size[0] / 2) #* env.reset_time_outs[env_ids,] : last term don't work at init...

    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(speed_command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5

    # Robots that move down can't move up
    move_up *= ~move_down

    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)

    # return the mean difficulty
    if isinstance(terrain, TerrainImporterUniformDifficulty) :
        return torch.mean(terrain.difficulty.float())

    # else is instance of TerrainImporter and return the mean terrain level    
    return torch.mean(terrain.terrain_levels.float())
 


def speed_command_levels_walked_distance(env: RLTaskEnv, env_ids: Sequence[int], commandTermName: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """ Curriculum based on the distance the robot walken when commanded to move at a desired velocity.
    This curriculum term is called after an episodic reset, before all the other managers (eg. before the command update)
    
    This term is used to progressively increase the difficulty of tracking a speed command as the robot becomes better. 
    - When the robot walks > 80% of the required distance -> increase the difficulty
    - when the robot walsk < 50% of the required distance -> decrease the difficulty

    Args :
        env       : The RL environment
        env_ids   : The list of environment IDs to update. If None, all the environments are updated. Defaults to None.
        asset_cfg : The configuration of the robot
    
    Returns :
        The mean maximum commanded velocity"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    speed_command = env.command_manager.get_command("base_velocity")

    # compute the distance the robot walked
    walked_distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)

    # compute the commanded distance
    required_distance = torch.norm(speed_command[env_ids, :2], dim=1) * env.max_episode_length_s

    # Compute the number of environment that progress or regress in the difficulty (ie. maximal velocity command sampling range)
    increase_difficulty = torch.sum( walked_distance > (0.8 * required_distance) )
    decrease_difficulty = torch.sum( walked_distance < (0.5 * required_distance) )

    difficulty_progress = (increase_difficulty - decrease_difficulty) / env.num_envs

    new_difficulty = env.command_manager.get_term(commandTermName).update_difficulty(difficulty_progress)

    return new_difficulty


def speed_command_levels_tracking_rewards(env: RLTaskEnv, env_ids: Sequence[int], commandTermName: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """ Curriculum based on the tracking reward achieved by the robot when commanded to move at a desired velocity.
    This curriculum term is called after an episodic reset, before all the other managers (eg. before the command update)
    
    This term is used to progressively increase the difficulty of tracking a speed command as the robot becomes better. 
    - When the robot achieve > 90% of maximum tracking reward -> increase the difficulty
    - when the robot achieve < 75% of maximum tracking reward -> decrease the difficulty

    Args :
        env       : The RL environment
        env_ids   : The list of environment IDs to update. If None, all the environments are updated. Defaults to None.
        asset_cfg : The configuration of the robot
    
    Returns :
        The mean maximum commanded velocity"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command("base_velocity")

    # Compute the speed tracking reward
    lin_velocity_tracking = (env.reward_manager._episode_sums['track_lin_vel_xy_exp'][env_ids]) / env.max_episode_length_s
    ang_velocity_tracking = (env.reward_manager._episode_sums['track_ang_vel_z_exp'][env_ids]) / env.max_episode_length_s

    # Compute the tracking quality (bounded between 0 and 1)
    tracking_quality = (lin_velocity_tracking + ang_velocity_tracking) / (env.reward_manager.get_term_cfg('track_lin_vel_xy_exp').weight + env.reward_manager.get_term_cfg('track_ang_vel_z_exp').weight)

    # Compute the number of environment that progress or regress in the difficulty (ie. maximal velocity command sampling range)
    increase_difficulty = torch.sum( tracking_quality > 0.90 )
    decrease_difficulty = torch.sum( tracking_quality < 0.75 )

    difficulty_progress = (increase_difficulty - decrease_difficulty) / env.num_envs

    new_difficulty = env.command_manager.get_term(commandTermName).update_difficulty(difficulty_progress)

    return new_difficulty


def speed_command_levels_fast_walked_distance(env: RLTaskEnv, env_ids: Sequence[int], commandTermName: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) :
    """ Curriculum based on the distance the robot walken when commanded to move at a desired velocity.
    This curriculum term is called after an episodic reset, before all the other managers (eg. before the command update)
    
    This term is used to progressively increase the difficulty of tracking a speed command as the robot becomes better. 
    - When the robot walks > 80% of the required distance -> increase the difficulty
    - when the robot walsk < 50% of the required distance -> decrease the difficulty

    Args :
        env       : The RL environment
        env_ids   : The list of environment IDs to update. If None, all the environments are updated. Defaults to None.
        asset_cfg : The configuration of the robot
    
    Returns :
        The mean maximum commanded velocity"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    speed_command = env.command_manager.get_command("base_velocity")
    speed_commandTerm: CurriculumUniformVelocityCommand = env.command_manager.get_term(commandTermName)

    # compute the distance the robot walked
    # walked_distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1) # Neglect the radius due to angular speed
    walked_distance = speed_commandTerm.metrics['cumulative_distance'][env_ids]

    # compute the commanded distance
    required_distance = torch.norm(speed_command[env_ids, :2], dim=1) * env.max_episode_length_s

    # Update difficulty only for Robots that needed to travel faster than 90% of maximum available speed
    fast = speed_command[env_ids, 0] > (0.9 * speed_commandTerm.cfg.ranges.for_vel_b[1] * speed_commandTerm.difficulty)

    # Compute the number of environment that progress or regress in the difficulty (ie. maximal velocity command sampling range)
    increase_difficulty = torch.sum( walked_distance[fast] > (0.9 * required_distance[fast]) )
    decrease_difficulty = torch.sum( walked_distance[fast] < (0.75 * required_distance[fast]) )

    difficulty_progress = (increase_difficulty - decrease_difficulty) / env.num_envs

    new_difficulty = speed_commandTerm.update_difficulty(difficulty_progress)

    return new_difficulty