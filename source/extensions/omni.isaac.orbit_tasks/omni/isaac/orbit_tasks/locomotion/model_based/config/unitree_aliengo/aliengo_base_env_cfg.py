# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.orbit.utils import configclass

from omni.isaac.orbit_tasks.locomotion.model_based.model_based_env_cfg import LocomotionModelBasedEnvCfg

##
# Pre-defined configs
##
from omni.isaac.orbit_assets.unitree import UNITREE_ALIENGO_CFG, UNITREE_GO2_CFG  # isort: skip
from omni.isaac.orbit_assets.anymal import ANYMAL_C_CFG  # isort: skip


@configclass
class UnitreeAliengoBaseEnvCfg(LocomotionModelBasedEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # # ----- Select the robot : Unitree Aliengo -----
        self.scene.robot = UNITREE_ALIENGO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # self.scene.robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


        # ----- Select the prime path of the height sensor : already default setting -----
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"                                               # Unnecessary : already default 


        # ----- Set the terrain curriculum -----
        # self.curriculum.terrain_levels = None                                                                           # By default activated


        # ----- scale down the terrains because the robot is small -----
        # self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        # self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        # self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01


        # ----- reduce action scale : TODO Why ? -----
        # self.actions.joint_pos.scale = 0.25


        # ----- Event randomization -----
        # -- startup
        # self.events.add_base_mass.params["mass_range"] = (0.0, 0.0)                                                     # Default was ±5
        self.events.add_base_mass.params["mass_range"] = (-1.0, 1.0)                                                     # Default was ±5
        # -- Reset
        self.events.base_external_force_torque.params["force_range"] = (0.0, 0.0)                                       # Unnecessary : already default
        self.events.base_external_force_torque.params["torque_range"] = (0.0, 0.0)                                      # Unnecessary : already default
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
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
        # self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_robot_joints.params["position_range"] = (0.8, 1.2)
        # -- Interval
        self.events.push_robot = None                                                                                   # Default was activated
        

        # ----- rewards -----
        # self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"                                          # Changed regex expression
        # self.rewards.feet_air_time.weight = 0.01                                                                        # default was 0.125
        # self.rewards.undesired_contacts = None                                                                          # default was activated
        # self.rewards.dof_torques_l2.weight = -0.0002                                                                    # default was 0.00001
        # self.rewards.track_lin_vel_xy_exp.weight = 1.5                                                                  # default was 1
        # self.rewards.track_ang_vel_z_exp.weight = 0.75                                                                  # default was 0.5
        # self.rewards.dof_acc_l2.weight = -2.5e-7                                                                        # Unnecessary : already default 

        # ----- terminations -----
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"                                         # Unnecessary : already default 