# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from omni.isaac.orbit.utils import configclass

from omni.isaac.orbit_tasks.locomotion.model_based.model_based_env_cfg import LocomotionModelBasedEnvCfg

##
# Pre-defined configs
##
from omni.isaac.orbit_assets.unitree import UNITREE_ALIENGO_CFG, UNITREE_GO2_CFG, UNITREE_ALIENGO_TORQUE_CONTROL_CFG  # isort: skip
from omni.isaac.orbit_assets.anymal import ANYMAL_C_CFG  # isort: skip

from omni.isaac.orbit.terrains.config.climb import STAIRS_TERRAINS_CFG
from omni.isaac.orbit.terrains import TerrainImporterUniformDifficulty


@configclass
class UnitreeAliengoClimbEnvCfg(LocomotionModelBasedEnvCfg):
    def __post_init__(self):

        # --- Select the Climb terrain -> Must be done before super().__post_init__() otherwise it won't load the terrain properly
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_CFG
        self.scene.terrain.class_type = TerrainImporterUniformDifficulty

        # post init of parent
        super().__post_init__()

        """ ----- Scene Settings ----- """
        # --- Select the robot : Unitree Aliengo
        # self.scene.robot = UNITREE_ALIENGO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # self.scene.robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot = UNITREE_ALIENGO_TORQUE_CONTROL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")                 

        # --- Select the prime path of the height sensor
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"                                               # Unnecessary : already default 

        # # --- Select the Climb terrain
        # self.scene.terrain.terrain_generator = STAIRS_TERRAINS_CFG


        """ ----- Commands ----- """
        self.commands.base_velocity.ranges.for_vel_b = ( 0.2, 0.5)
        self.commands.base_velocity.ranges.lat_vel_b = (-0.1, 0.1)
        self.commands.base_velocity.ranges.ang_vel_b = (0.0, 0.0)
        self.commands.base_velocity.ranges.initial_heading_err = (0.0, 0.0)


        """ ----- Observation ----- """
        # To add or not noise on the observations
        self.observations.policy.enable_corruption = False


        """ ----- Curriculum ----- """
        Terrain_curriculum = True
        Speed_curriculum = False

        if Terrain_curriculum : 
            pass
        else :
            self.curriculum.terrain_levels = None                                                                       # By default activated

        if Speed_curriculum :
            self.commands.base_velocity.initial_difficulty = 0.2
            self.commands.base_velocity.minmum_difficulty = 0.2
            self.commands.base_velocity.difficulty_scaling = 0.2
        else :
            self.curriculum.speed_levels = None


        """ ----- Event randomization ----- """
        Event = {'Base Mass'        : False, 
                 'External Torque'  : False,
                 'External Force'   : False,
                 'Random joint pos' : False,
                 'Push Robot'       : False}

        # --- startup
        if Event['Base Mass'] : 
            self.events.add_base_mass.params["mass_range"] = (-3.0, 3.0) #(0.0, 0.0)                                     # Default was 0

        # --- Reset
        if Event['External Force'] :
            self.events.base_external_force_torque.params["force_range"]  = (-10.0, 10.0) # (0.0, 0.0)                  # Default was 0
        if Event['External Torque'] :
            self.events.base_external_force_torque.params["torque_range"] = (-1.0, 1.0) # (0.0, 0.0)                    # Default was 0

        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},                                   # Some randomization improve training speed
            # "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {                                                                                         # Default was ±0.5
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        if Event["Random joint pos"] :
            self.events.reset_robot_joints.params["position_range"] = (0.8, 1.2)                                        # default was (1.0, 1.0)
        
        # --- Interval
        if not Event['Push Robot'] :
            self.events.push_robot = None                                                                               # Default was activated


        """ ----- rewards ----- """
        # -- task
        self.rewards.track_lin_vel_xy_exp.weight         = 1.5
        self.rewards.track_soft_vel_xy_exp.weight        = 1.5
        self.rewards.track_ang_vel_z_exp.weight          = 0.75
        self.rewards.track_robot_height                  = None  # Needs a negative weight
        self.rewards.reward_terrain_progress             = None

        # -- Additionnal penalties : Need a negative weight
        self.rewards.penalty_lin_vel_z_l2.weight         = -2.0
        self.rewards.penalty_ang_vel_xy_l2.weight        = -0.05
        self.rewards.penalty_dof_torques_l2              = None
        self.rewards.penalty_dof_acc_l2                  = None
        self.rewards.penalty_action_rate_l2              = None
        self.rewards.undesired_contacts.weight           = -1.0
        self.rewards.flat_orientation_l2                 = None
        self.rewards.dof_pos_limits                      = None
        self.rewards.penalty_friction                    = None #-0.25
        self.rewards.penalty_stance_foot_vel.weight      = -1.0
        self.rewards.penalty_CoT                         = None

        # -- Model based penalty : Positive weight -> penalty is already negative
        self.rewards.penalty_leg_frequency               = None
        self.rewards.penalty_leg_duty_cycle              = None
        self.rewards.penalty_large_force.weight          = 0.1
        self.rewards.penalty_large_step                  = None
        self.rewards.penalty_frequency_variation.weight  = 1.0
        self.rewards.penatly_duty_cycle_variation.weight = 2.5
        self.rewards.penalty_step_variation.weight       = 2.5
        self.rewards.penatly_force_variation.weight      = 1e-4 #4e-5

        # -- Additionnal Reward : Need a positive weight
        self.rewards.reward_is_alive                     = None


        """ ----- terminations ----- """

 