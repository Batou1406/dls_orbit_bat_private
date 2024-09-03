# Copyright (c) 2022-2024, The LAB Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass
from omni.isaac.lab_tasks.manager_based.locomotion.model_based.model_based_env_cfg import LocomotionModelBasedEnvCfg
from omni.isaac.lab_assets.unitree import UNITREE_ALIENGO_SELF_COLLISION_TORQUE_CONTROL_CFG  # isort: skip
from omni.isaac.lab.terrains.config.niceFlat import COBBLESTONE_ROAD_CFG, COBBLESTONE_FLAT_CFG
from omni.isaac.lab.terrains.config.climb import STAIRS_TERRAINS_CFG
from omni.isaac.lab.terrains.config.speed import SPEED_TERRAINS_CFG
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG
from omni.isaac.lab_tasks.manager_based.locomotion.model_based.mdp import modify_reward_weight
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.terrains import randomTerrainImporter
import omni.isaac.lab_tasks.manager_based.locomotion.model_based.mdp as mdp


@configclass
class ActionsCfg:
    """Action specifications for the MDP.
    - Robot joint position - dim=12
    """
    model_base_variable = mdp.ModelBaseActionCfg(
        asset_name="robot",
        joint_names=[".*"], 
        controller=mdp.samplingController,
        optimizerCfg=mdp.ModelBaseActionCfg.OptimizerCfg(
            multipolicy=1,
            prevision_horizon=5,
            discretization_time=0.02,
            parametrization_p='first',
            parametrization_F='cubic_spline'
            ),
        )
    


@configclass
class UnitreeAliengoTestEnvCfg(LocomotionModelBasedEnvCfg):
    actions = ActionsCfg()
    
    def __post_init__(self):

        """ ----- Scene Settings ----- """
        self.scene.robot = UNITREE_ALIENGO_SELF_COLLISION_TORQUE_CONTROL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"  

        self.scene.terrain.terrain_generator = COBBLESTONE_FLAT_CFG # very Flat
        # self.scene.terrain.terrain_generator = COBBLESTONE_ROAD_CFG # Flat
        # self.scene.terrain.terrain_generator = STAIRS_TERRAINS_CFG
        # self.scene.terrain.terrain_generator = SPEED_TERRAINS_CFG
        # self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG    
         
        self.scene.terrain.class_type = randomTerrainImporter      

        """ ----- Commands ----- """
        self.commands.base_velocity.ranges.for_vel_b = (-0.0, 0.0)
        self.commands.base_velocity.ranges.lat_vel_b = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_b = (-0.0, 0.0)
        self.commands.base_velocity.ranges.initial_heading_err = (-0.0, 0.0)    


        """ ----- Observation ----- """
        # To add or not noise on the observations
        self.observations.policy.enable_corruption = False   


        """ ----- Event randomization ----- """
        Event = {'Base Mass'        : False, 
                 'External Torque'  : False,
                 'External Force'   : False,
                 'Random joint pos' : False,
                 'Push Robot'       : False}

        # --- startup
        if Event['Base Mass'] : 
            self.events.add_base_mass.params["mass_distribution_params"] = (-3.0, 3.0) #(0.0, 0.0)                                    # Default was 0

        # --- Reset
        if Event['External Force'] :
            self.events.base_external_force_torque.params["force_range"]  = (-10.0, 10.0) # (0.0, 0.0)                  # Default was 0
        if Event['External Torque'] :
            self.events.base_external_force_torque.params["torque_range"] = (-1.0, 1.0) # (0.0, 0.0)                    # Default was 0

        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},                                   # Some randomization improve training speed
            # "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (-0.0, 0.0)}, 
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
        self.rewards.track_lin_vel_xy_exp.weight         = 1.5
        self.rewards.track_soft_vel_xy_exp               = None
        self.rewards.track_ang_vel_z_exp.weight          = 0.75
        self.rewards.track_robot_height_exp.weight       = 0.2

        self.rewards.penalty_lin_vel_z_l2.weight         = -2.0
        self.rewards.penalty_ang_vel_xy_l2.weight        = -0.05
        self.rewards.penalty_dof_torques_l2              = None
        self.rewards.penalty_dof_acc_l2                  = None
        self.rewards.penalty_action_rate_l2              = None
        self.rewards.undesired_contacts                  = None
        self.rewards.flat_orientation_l2.weight          = -1.0
        self.rewards.dof_pos_limits.weight               = -3.0
        self.rewards.penalty_friction                    = None
        self.rewards.penalty_stance_foot_vel             = None
        self.rewards.penalty_CoT.weight                  = -0.04
        self.rewards.penalty_close_feet                  = None
        self.rewards.penalize_foot_trac_err              = None
        self.rewards.penalty_constraint_violation        = None

        self.rewards.penalty_leg_frequency               = None
        self.rewards.penalty_leg_duty_cycle              = None
        self.rewards.penalty_large_force                 = None
        self.rewards.penalty_large_step                  = None
        self.rewards.penalty_frequency_variation         = None
        self.rewards.penatly_duty_cycle_variation        = None
        self.rewards.penalty_step_variation              = None
        self.rewards.penatly_force_variation             = None
        self.rewards.penalty_sampling_rollout            = None
        self.rewards.reward_is_alive                     = None 
        self.rewards.penalty_failed                      = None


        # post init of parent
        super().__post_init__()

        self.decimation = 2 #2


 