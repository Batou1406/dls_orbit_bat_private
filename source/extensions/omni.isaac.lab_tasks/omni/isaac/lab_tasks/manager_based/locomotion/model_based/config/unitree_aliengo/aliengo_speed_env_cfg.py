# Copyright (c) 2022-2024, The lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.manager_based.locomotion.model_based.model_based_env_cfg import LocomotionModelBasedEnvCfg

##
# Pre-defined configs
##
from omni.isaac.lab_assets.unitree import UNITREE_ALIENGO_CFG, UNITREE_GO2_CFG, UNITREE_ALIENGO_TORQUE_CONTROL_CFG, UNITREE_ALIENGO_SELF_COLLISION_TORQUE_CONTROL_CFG  # isort: skip
from omni.isaac.lab_assets.anymal import ANYMAL_C_CFG  # isort: skip

from omni.isaac.lab.terrains.config.speed import SPEED_TERRAINS_CFG
from omni.isaac.lab_tasks.manager_based.locomotion.model_based.mdp import CurriculumNormalVelocityCommandCfg, modify_reward_weight

from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm

@configclass
class UnitreeAliengoSpeedEnvCfg(LocomotionModelBasedEnvCfg):
    def __post_init__(self):

        # --- Select the speed terrain -> Must be done before super().__post_init__() otherwise it won't load the terrain properly
        self.scene.terrain.terrain_generator = SPEED_TERRAINS_CFG

        # --- Initialsie the large step
        Large_step_curriculum = False 
        if Large_step_curriculum :
            self.curriculum.penalty_large_step_curr = CurrTerm(func=modify_reward_weight, params={"term_name": "penalty_large_step", "weight": 1.0, "num_steps": (400*24)})


        """ ----- Reward and Event Curriculum ----- """


        """ ----- Scene Settings ----- """
        # --- Select the robot : Unitree Aliengo
        # self.scene.robot = UNITREE_ALIENGO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # self.scene.robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # self.scene.robot = UNITREE_ALIENGO_TORQUE_CONTROL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")   
        self.scene.robot = UNITREE_ALIENGO_SELF_COLLISION_TORQUE_CONTROL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")                 

        # --- Select the prime path of the height sensor
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"                                               # Unnecessary : already default 

        # --- Select the speed terrain
        # self.scene.terrain.terrain_generator = SPEED_TERRAINS_CFG


        """ ----- Commands ----- """
        self.commands.base_velocity.ranges.for_vel_b = ( 0.0, 2.1)
        self.commands.base_velocity.ranges.lat_vel_b = (-0.2, 0.2)
        self.commands.base_velocity.ranges.ang_vel_b = (-0.5, 0.5)
        self.commands.base_velocity.ranges.initial_heading_err = (-0.5, 0.5)


        """ ----- Observation ----- """
        # To add or not noise on the observations
        self.observations.policy.enable_corruption = False
        # self.observations.policy.enable_corruption = True # 23/09


        """ ----- Curriculum ----- """
        Terrain_curriculum = False
        Speed_curriculum = True
        # Speed_curriculum = False # -> For Dagger
        # Large_step_curriculum = True : 'Must be declared before post init'

        if Terrain_curriculum : 
            pass
        else :
            self.curriculum.terrain_levels = None                                                                       # By default activated

        if Speed_curriculum :
            self.commands.base_velocity.initial_difficulty = 0.2
            self.commands.base_velocity.minmum_difficulty = 0.2
            self.commands.base_velocity.difficulty_scaling = 0.1
        else :
            self.curriculum.speed_levels = None


        """ ----- Event randomization ----- """
        Event = {'Base Mass'        : False, #True 23/09
                 'External Torque'  : False,
                 'External Force'   : False,
                 'Random joint pos' : True, #True 23/09
                 'Push Robot'       : False,
                 'Friction'         : False,}

        # --- startup
        if Event['Base Mass'] : 
            self.events.add_base_mass.params["mass_distribution_params"] = (-1.5, 1.5) #(0.0, 0.0)                                     # Default was 0

        # --- Reset
        if Event['External Force'] :
            self.events.base_external_force_torque.params["force_range"]  = (-5.0, 5.0) # (0.0, 0.0)                  # Default was 0
        if Event['External Torque'] :
            self.events.base_external_force_torque.params["torque_range"] = (-0.5, 0.5) # (0.0, 0.0)                    # Default was 0

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

        if Event["Friction"]:
            self.events.physics_material.params["static_friction_range"]  = (0.6, 0.8)                                  # default was 0.8
            self.events.physics_material.params["dynamic_friction_range"] = (0.4, 0.6)                                  # default was 0.6

        # --- Interval
        if Event['Push Robot']:
            self.events.push_robot.params["velocity_range"] = {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}                      # default was (-0.2, 0.2)

        if not Event['Push Robot'] :
            self.events.push_robot = None                                                                               # Default was activated



        """ ----- rewards ----- """
        training = 'normal' # 'normal' or 'with_sampling' or 'with_sampling_and_normal or 'play_eval'

        if training == 'normal' :
            # -- task
            self.rewards.track_lin_vel_xy_exp.weight         = 1.7 # 24/09 1.5
            self.rewards.track_lin_vel_xy_exp.params['std']  = math.sqrt(0.16)        
            self.rewards.track_soft_vel_xy_exp               = None
            self.rewards.track_ang_vel_z_exp.weight          = 0.75
            self.rewards.track_robot_height_exp              = None
            # self.rewards.track_robot_height_exp.weight       = 0.2 #0.1
            # self.rewards.track_robot_height_exp.params['height_bound'] = (-0.015,0.03) # 21/09 0.0.15
            # self.rewards.track_robot_height_exp.params['target_height'] = 0.35 # 21/09 0.40

            # -- Additionnal penalties : Need a negative weight
            self.rewards.penalty_lin_vel_z_l2.weight         = -1.0  # -2.5 21/09
            self.rewards.penalty_ang_vel_xy_l2.weight        = -0.1 #-0.2 # 21/09 -0.15
            self.rewards.penalty_dof_torques_l2              = None  #-0.00005 #-0.0001
            self.rewards.penalty_dof_acc_l2                  = None  #-1.0e-07
            self.rewards.penalty_action_rate_l2              = None
            self.rewards.undesired_contacts                  = None     #-1.0
            self.rewards.flat_orientation_l2                 = None     #-2.0
            self.rewards.dof_pos_limits                      = None     #-2.0
            self.rewards.penalty_friction.weight             = -0.3
            self.rewards.penalty_stance_foot_vel             = None     #-1.0
            self.rewards.penalty_CoT.weight                  = -0.8    #-0.4#-0.002 # 21/09 -0.5
            self.rewards.penalty_close_feet                  = None     #-0.01
            self.rewards.penalize_foot_trac_err              = None
            self.rewards.penalty_constraint_violation        = None

            # -- Model based penalty : Positive weight -> penalty is already negative
            self.rewards.penalty_leg_frequency               = None
            self.rewards.penalty_leg_duty_cycle              = None
            self.rewards.penalty_large_force                 = None
            self.rewards.penalty_large_step                  = None
            self.rewards.penalty_frequency_variation.weight  = 0.0
            self.rewards.penatly_duty_cycle_variation.weight = 0.0
            self.rewards.penalty_step_variation.weight       = 0.0
            self.rewards.penatly_force_variation.weight      = 0.0

            # -- Additionnal Reward : Need a positive weight
            self.rewards.reward_is_alive                     = None
            self.rewards.penalty_failed                      = None

            # Deactivated for DAgger
            num_iter_activate = 800
            self.curriculum.penalty_frequency_variation_curr    = CurrTerm(func=modify_reward_weight, params={"term_name": "penalty_frequency_variation",  "weight": 0.2,  "num_steps": (num_iter_activate*24)})
            self.curriculum.penatly_duty_cycle_variation_curr   = CurrTerm(func=modify_reward_weight, params={"term_name": "penatly_duty_cycle_variation", "weight": 0.7,  "num_steps": (num_iter_activate*24)}) # 24/09 1.0
            self.curriculum.penalty_step_variation_curr         = CurrTerm(func=modify_reward_weight, params={"term_name": "penalty_step_variation",       "weight": 1.0,  "num_steps": (num_iter_activate*24)})
            self.curriculum.penatly_force_variation_curr        = CurrTerm(func=modify_reward_weight, params={"term_name": "penatly_force_variation",      "weight": 3e-4, "num_steps": (num_iter_activate*24)}) # 21/09 2e-5


        if training == 'with_sampling' :
            # -- task
            self.rewards.track_lin_vel_xy_exp                = None
            self.rewards.track_soft_vel_xy_exp               = None
            self.rewards.track_ang_vel_z_exp                 = None
            self.rewards.track_robot_height_exp              = None

            # -- Additionnal penalties : Need a negative weight
            self.rewards.penalty_lin_vel_z_l2                = None
            self.rewards.penalty_ang_vel_xy_l2               = None
            self.rewards.penalty_dof_torques_l2              = None
            self.rewards.penalty_dof_acc_l2                  = None
            self.rewards.penalty_action_rate_l2              = None
            self.rewards.undesired_contacts                  = None
            self.rewards.flat_orientation_l2                 = None
            self.rewards.dof_pos_limits.weight               = -3.0
            self.rewards.penalty_friction                    = None
            self.rewards.penalty_stance_foot_vel             = None
            self.rewards.penalty_CoT.weight                  = -0.05
            self.rewards.penalty_close_feet                  = None
            self.rewards.penalize_foot_trac_err              = None
            self.rewards.penalty_constraint_violation        = None

            # -- Model based penalty : Positive weight -> penalty is already negative
            self.rewards.penalty_leg_frequency.weight        = 0.0
            self.rewards.penalty_leg_duty_cycle.weight       = 0.0
            self.rewards.penalty_large_force                 = None
            self.rewards.penalty_large_step                  = None
            self.rewards.penalty_frequency_variation.weight  = 0.2 #1.0
            self.rewards.penatly_duty_cycle_variation.weight = 1.0 #2.5
            self.rewards.penalty_step_variation.weight       = 0.2 #2.5
            self.rewards.penatly_force_variation.weight      = 1e-5 #1e-4
            self.rewards.penalty_leg_frequency.params  = {"action_name": "model_base_variable", "bound": (0.6,2.0)}
            self.curriculum.penalty_leg_frequency_curr = CurrTerm(func=modify_reward_weight, params={"term_name": "penalty_leg_frequency", "weight": 0.1, "num_steps": (400*24)})
            self.rewards.penalty_leg_duty_cycle.params  = params={"action_name": "model_base_variable", "bound": (0.35,0.7)}
            self.curriculum.penalty_leg_duty_cycle_curr = CurrTerm(func=modify_reward_weight, params={"term_name": "penalty_leg_duty_cycle", "weight": 0.1, "num_steps": (400*24)})



            self.rewards.penalty_sampling_rollout.weight     = -0.1
            # self.curriculum.penalty_sampling_rollout_curr = CurrTerm(func=modify_reward_weight, params={"term_name": "penalty_sampling_rollout", "weight": -2.5e-2, "num_steps": (1000*24)})

            # -- Additionnal Reward : Need a positive weight
            self.rewards.reward_is_alive.weight              = 2.5 #0.25
            self.rewards.penalty_failed                      = None

        if training == 'with_sampling_and_normal' :
            # -- task
            self.rewards.track_lin_vel_xy_exp.weight         = 1.5
            self.rewards.track_lin_vel_xy_exp.params['std']  = math.sqrt(0.16)        
            self.rewards.track_soft_vel_xy_exp               = None
            self.rewards.track_ang_vel_z_exp.weight          = 0.75
            self.rewards.track_robot_height_exp.weight       = 0.2 #0.1
            self.rewards.track_robot_height_exp.params['height_bound'] = (-0.015,0.015)
            self.rewards.track_robot_height_exp.params['target_height'] = 0.40

            # -- Additionnal penalties : Need a negative weight
            self.rewards.penalty_lin_vel_z_l2.weight         = -2.5
            self.rewards.penalty_ang_vel_xy_l2.weight        = -0.15 #-0.2
            self.rewards.penalty_dof_torques_l2              = None  #-0.00005 #-0.0001
            self.rewards.penalty_dof_acc_l2                  = None  #-1.0e-07
            self.rewards.penalty_action_rate_l2              = None
            self.rewards.undesired_contacts                  = None     #-1.0
            self.rewards.flat_orientation_l2                 = None     #-2.0
            self.rewards.dof_pos_limits                      = None     #-2.0
            self.rewards.penalty_friction.weight             = -0.3
            self.rewards.penalty_stance_foot_vel             = None     #-1.0
            self.rewards.penalty_CoT.weight                  = -0.05    #-0.4#-0.002
            self.rewards.penalty_close_feet                  = None     #-0.01
            self.rewards.penalize_foot_trac_err              = None
            self.rewards.penalty_constraint_violation        = None

            # -- Model based penalty : Positive weight -> penalty is already negative
            self.rewards.penalty_leg_frequency.weight        = 0.0
            self.rewards.penalty_leg_duty_cycle.weight       = 0.0
            self.rewards.penalty_large_force                 = None
            self.rewards.penalty_large_step                  = None
            self.rewards.penalty_frequency_variation.weight  = 0.0
            self.rewards.penatly_duty_cycle_variation.weight = 0.0
            self.rewards.penalty_step_variation.weight       = 0.0
            self.rewards.penatly_force_variation.weight      = 0.0

            # -- Additionnal Reward : Need a positive weight
            self.rewards.reward_is_alive.weight              = 1
            self.rewards.penalty_failed                      = None

            self.rewards.penalty_leg_frequency.params  = {"action_name": "model_base_variable", "bound": (0.6,2.0)}
            self.curriculum.penalty_leg_frequency_curr = CurrTerm(func=modify_reward_weight, params={"term_name": "penalty_leg_frequency", "weight": 1.0, "num_steps": (400*24)})
            self.rewards.penalty_leg_duty_cycle.params  = params={"action_name": "model_base_variable", "bound": (0.35,0.7)}
            self.curriculum.penalty_leg_duty_cycle_curr = CurrTerm(func=modify_reward_weight, params={"term_name": "penalty_leg_duty_cycle", "weight": 1.0, "num_steps": (400*24)})

            self.rewards.penalty_sampling_rollout.weight     = -1.0

            num_iter_activate = 800
            self.curriculum.penalty_frequency_variation_curr    = CurrTerm(func=modify_reward_weight, params={"term_name": "penalty_frequency_variation",  "weight": 0.2,  "num_steps": (num_iter_activate*24)})
            self.curriculum.penatly_duty_cycle_variation_curr   = CurrTerm(func=modify_reward_weight, params={"term_name": "penatly_duty_cycle_variation", "weight": 1.0,  "num_steps": (num_iter_activate*24)})
            self.curriculum.penalty_step_variation_curr         = CurrTerm(func=modify_reward_weight, params={"term_name": "penalty_step_variation",       "weight": 1.0,  "num_steps": (num_iter_activate*24)})
            self.curriculum.penatly_force_variation_curr        = CurrTerm(func=modify_reward_weight, params={"term_name": "penatly_force_variation",      "weight": 1e-5, "num_steps": (num_iter_activate*24)})

        """ ----- terminations ----- """


        # post init of parent
        super().__post_init__()
 