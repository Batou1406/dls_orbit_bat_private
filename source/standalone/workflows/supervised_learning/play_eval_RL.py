# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

""" Parse Arguments And Launch App"""
if True:
    import argparse
    from omni.isaac.lab.app import AppLauncher
    import cli_args  # isort: skip
    
    parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
    parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
    parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
    parser.add_argument("--seed",                 type=int, default=None, help="Seed used for the environment")
    parser.add_argument("--num_envs",             type=int, default=1, help="Number of environments to simulate.")
    parser.add_argument("--multipolicies_folder", type=str, default=None, help="Path to folder that contains the different policies in model/multipolicies_folder")
    parser.add_argument("--result_folder",        type=str, default=None, help="Where to save the results in ./eval/result_folder")
    parser.add_argument("--num_trajectory",       type=int, default=None, help="Number of step to generate the data")
    parser.add_argument("--eval_task",            type=str, default=None, help="Task to try the controller")
    parser.add_argument("--model_name",            type=str, default=None, help="Name of the model for naming the results")

    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

""" Import """
if True :
    import gymnasium as gym
    import os
    import torch
    import json
    import numpy as np
    from rsl_rl.runners import OnPolicyRunner
    import omni.isaac.lab_tasks  # noqa: F401
    from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
    from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
    from train import Model # Where to get the model architecture for MLP, may want to change that

    from rsl_rl.modules import ActorCritic

    from omni.isaac.lab_tasks.manager_based.locomotion.model_based.config.unitree_aliengo import agents
    from omni.isaac.lab.utils import configclass
    from omni.isaac.lab_tasks.manager_based.locomotion.model_based.model_based_env_cfg import LocomotionModelBasedEnvCfg
    from omni.isaac.lab_tasks.manager_based.locomotion.model_based.config.unitree_aliengo.aliengo_speed_env_cfg import UnitreeAliengoSpeedEnvCfg
    from omni.isaac.lab_tasks.manager_based.locomotion.model_based.config.unitree_aliengo.aliengo_rough_env_cfg import UnitreeAliengoRoughEnvCfg
    from omni.isaac.lab_tasks.manager_based.locomotion.model_based.config.unitree_aliengo.aliengo_climb_env_cfg import UnitreeAliengoClimbEnvCfg
    from omni.isaac.lab_tasks.manager_based.locomotion.model_based.config.unitree_aliengo.aliengo_base_env_cfg import UnitreeAliengoBaseEnvCfg
    from omni.isaac.lab_assets.unitree import UNITREE_ALIENGO_SELF_COLLISION_TORQUE_CONTROL_CFG  # isort: skip
    from omni.isaac.lab.terrains.config.niceFlat import COBBLESTONE_ROAD_CFG, COBBLESTONE_FLAT_CFG
    from omni.isaac.lab.terrains.config.climb import STAIRS_TERRAINS_CFG
    from omni.isaac.lab.terrains.config.speed import SPEED_TERRAINS_CFG
    from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG
    from omni.isaac.lab_tasks.manager_based.locomotion.model_based.mdp import modify_reward_weight
    from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
    from omni.isaac.lab.terrains import randomTerrainImporter
    import omni.isaac.lab_tasks.manager_based.locomotion.model_based.mdp as mdp
    from omni.isaac.lab.terrains import TerrainImporterUniformDifficulty

    import pandas as pd
    import time


eval_task       = args_cli.eval_task
result_folder   = args_cli.result_folder
num_trajectory  = args_cli.num_trajectory
model_name      = args_cli.model_name

task_name               = f"model-{model_name}-eval-{eval_task}"
result_folder_path      = f'eval/{args_cli.result_folder}'
full_result_folder_path = f'eval/{args_cli.result_folder}/{task_name}'
platform_width          = 2.5
step_width              = 0.3
max_stairs              = 13

""" Create Full result directory """
if True :
    # Open result directory and load info_dict, or create the dir and dict if necessary
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)
    if os.path.isfile(result_folder_path +'/info.json'):
        with open(result_folder_path +'/info.json', 'r') as json_file:
                info_dict = json.load(json_file)  # Load JSON data into a Python dictionary
    else :
        info_dict = {}

    # Create full result dir and result_dict
    if not os.path.exists(full_result_folder_path):
        os.makedirs(full_result_folder_path)
    else :
        raise KeyError('There is already an experiment setup in this directory, Please provide another folder_name')
    if os.path.isfile(full_result_folder_path +'/result.json'):
        with open(full_result_folder_path +'/result.json', 'r') as json_file:
                result_dict = json.load(json_file)  # Load JSON data into a Python dictionary
    else :
        result_dict = {}

""" Infer optimization type from folder name"""
if True :
    if 'RL' in args_cli.multipolicies_folder :
        controller = mdp.modelBaseController
        optimizerCfg=None

    elif 'IL' in args_cli.multipolicies_folder:
        controller = mdp.samplingController
        optimizerCfg=mdp.ModelBaseActionCfg.OptimizerCfg(
            multipolicy=1,
            prevision_horizon=10,
            discretization_time=0.02,
            parametrization_p='first',
            parametrization_F='cubic_spline'
            ),

    elif 'NO_WS' in args_cli.multipolicies_folder:
        controller = mdp.samplingController
        optimizerCfg=mdp.ModelBaseActionCfg.OptimizerCfg(
            multipolicy=1,
            prevision_horizon=10,
            discretization_time=0.02,
            parametrization_p='first',
            parametrization_F='cubic_spline'
            ),
    
    else :
        controller = mdp.modelBaseController
        optimizerCfg=None
        

@configclass
class ActionsCfg:
    """Action specifications for the MDP.
    - Robot joint position - dim=12
    """
    model_base_variable = mdp.ModelBaseActionCfg(
        asset_name="robot",
        joint_names=[".*"], 
        controller=controller,
        optimizerCfg=optimizerCfg
        )


@configclass
class env_cfg(LocomotionModelBasedEnvCfg):
    actions = ActionsCfg()
    
    def __post_init__(self):

        if eval_task == 'base_test' :
            """ ----- Scene Settings ----- """
            self.scene.robot = UNITREE_ALIENGO_SELF_COLLISION_TORQUE_CONTROL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"  

            self.scene.terrain.terrain_generator = COBBLESTONE_FLAT_CFG # very Flat
            self.scene.terrain.class_type = randomTerrainImporter   

            """ ----- Commands ----- """
            self.commands.base_velocity.ranges.for_vel_b = (0.3, 0.6)
            self.commands.base_velocity.ranges.lat_vel_b = (-0.2, 0.2)
            self.commands.base_velocity.ranges.ang_vel_b = (-0.5, 0.5)
            self.commands.base_velocity.ranges.initial_heading_err = (-0.0, 0.0)      
            self.commands.base_velocity.resampling_time_range = (10000.0,10000.0)

            """ ----- Observation ----- """
            self.observations.policy.enable_corruption = False

            """ ----- Curriculum ----- """
            Terrain_curriculum = False
            Speed_curriculum = False

            if not Terrain_curriculum : 
                self.curriculum.terrain_levels = None                                                                  

            if not Speed_curriculum :
                self.curriculum.speed_levels = None

            """ ----- Event randomization ----- """
            Event = {'Base Mass'        : False, 
                    'External Torque'  : False,
                    'External Force'   : False,
                    'Random joint pos' : False,
                    'Push Robot'       : True}

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

        if eval_task == 'speed_test' :
            """ ----- Scene Settings ----- """
            self.scene.robot = UNITREE_ALIENGO_SELF_COLLISION_TORQUE_CONTROL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"  

            self.scene.terrain.terrain_generator = COBBLESTONE_FLAT_CFG # very Flat
            self.scene.terrain.class_type = randomTerrainImporter   

            """ ----- Commands ----- """
            self.commands.base_velocity.ranges.for_vel_b = ( 0.0, 2.1)
            self.commands.base_velocity.ranges.lat_vel_b = (-0.2, 0.2)
            self.commands.base_velocity.ranges.ang_vel_b = (-0.5, 0.5)
            self.commands.base_velocity.ranges.initial_heading_err = (-0.5, 0.5)
            self.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)

            """ ----- Observation ----- """
            self.observations.policy.enable_corruption = False

            """ ----- Curriculum ----- """
            Terrain_curriculum = False
            Speed_curriculum = False

            if not Terrain_curriculum : 
                self.curriculum.terrain_levels = None                                                                  

            if not Speed_curriculum :
                self.curriculum.speed_levels = None

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

        if eval_task == 'stair_test' : 
            """ ----- Scene Settings ----- """
            self.scene.robot = UNITREE_ALIENGO_SELF_COLLISION_TORQUE_CONTROL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"  

            self.scene.terrain.terrain_generator = STAIRS_TERRAINS_CFG
            self.scene.terrain.class_type = TerrainImporterUniformDifficulty  

            """ ----- Commands ----- """
            self.commands.base_velocity.ranges.for_vel_b = ( 0.3, 0.5)
            self.commands.base_velocity.ranges.lat_vel_b = (-0.1, 0.1)
            self.commands.base_velocity.ranges.ang_vel_b = (-0.0, 0.0)
            self.commands.base_velocity.ranges.initial_heading_err = (0.0, 0.0)  
            self.commands.base_velocity.resampling_time_range = (10000.0, 10000.0)

            """ ----- Observation ----- """
            self.observations.policy.enable_corruption = False

            """ ----- Curriculum ----- """
            Terrain_curriculum = True
            Speed_curriculum = False

            if Terrain_curriculum : 
                self.scene.terrain.max_init_terrain_level = 9
            else :
                self.curriculum.terrain_levels = None                                                                  

            if not Speed_curriculum :
                self.curriculum.speed_levels = None

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

        if eval_task == 'survival_test' : 
            """ ----- Scene Settings ----- """
            self.scene.robot = UNITREE_ALIENGO_SELF_COLLISION_TORQUE_CONTROL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"  

            self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG # very Flat
            # self.scene.terrain.class_type = randomTerrainImporter   

            """ ----- Commands ----- """
            self.commands.base_velocity.ranges.for_vel_b = (0.3, 0.6)
            self.commands.base_velocity.ranges.lat_vel_b = (-0.2, 0.2)
            self.commands.base_velocity.ranges.ang_vel_b = (-0.5, 0.5)
            self.commands.base_velocity.ranges.initial_heading_err = (-0.0, 0.0)     
            self.commands.base_velocity.resampling_time_range = (10000.0, 10000.0)

            """ ----- Observation ----- """
            self.observations.policy.enable_corruption = True

            """ ----- Curriculum ----- """
            Terrain_curriculum = False
            Speed_curriculum = False

            if Terrain_curriculum : 
                self.scene.terrain.max_init_terrain_level = 9
            else :
                self.curriculum.terrain_levels = None                                                                     

            if not Speed_curriculum :
                self.curriculum.speed_levels = None

            """ ----- Event randomization ----- """
            Event = {'Base Mass'        : True, 
                    'External Torque'  : True,
                    'External Force'   : True,
                    'Random joint pos' : True,
                    'Push Robot'       : True,
                    'Friction'         : True,}

            # --- startup
            if Event['Base Mass'] : 
                self.events.add_base_mass.params["mass_distribution_params"] = (-3.0, 3.0) #(0.0, 0.0)                                     # Default was 0

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

            if Event["Friction"]:
                self.events.physics_material.params["static_friction_range"]  = (0.6, 0.8)                                  # default was 0.8
                self.events.physics_material.params["dynamic_friction_range"] = (0.4, 0.6)                                  # default was 0.6
            
            # --- Interval
            if not Event['Push Robot'] :
                self.events.push_robot = None    


        """ Rewards - Same for every task """
        if True : 
            self.rewards.track_lin_vel_xy_exp.weight         = 1.5
            self.rewards.track_soft_vel_xy_exp               = None
            self.rewards.track_ang_vel_z_exp.weight          = 0.75
            self.rewards.track_robot_height_exp.weight       = 0.2

            # -- Additionnal penalties : Need a negative weight
            self.rewards.penalty_lin_vel_z_l2.weight         = -2.0
            self.rewards.penalty_ang_vel_xy_l2.weight        = -0.05
            self.rewards.penalty_dof_torques_l2              = None
            self.rewards.penalty_dof_acc_l2                  = None
            self.rewards.penalty_action_rate_l2              = None
            self.rewards.undesired_contacts                  = None
            self.rewards.flat_orientation_l2                 = None
            self.rewards.dof_pos_limits.weight               = -3.0
            self.rewards.penalty_friction                    = None
            self.rewards.penalty_stance_foot_vel             = None
            self.rewards.penalty_CoT.weight                  = -0.04
            self.rewards.penalty_close_feet                  = None
            self.rewards.penalize_foot_trac_err              = None
            self.rewards.penalty_constraint_violation        = None

            # -- Model based penalty : Positive weight -> penalty is already negative
            self.rewards.penalty_leg_frequency               = None
            self.rewards.penalty_leg_duty_cycle              = None
            self.rewards.penalty_large_force                 = None
            self.rewards.penalty_large_step                  = None
            self.rewards.penalty_frequency_variation.weight  = 0.5 #1.0
            self.rewards.penatly_duty_cycle_variation.weight = 1.0 #2.5
            self.rewards.penalty_step_variation.weight       = 0.2 #2.5
            self.rewards.penatly_force_variation.weight      = 1e-5 #1e-4

            self.rewards.penalty_sampling_rollout            = None

            # -- Additionnal Reward : Need a positive weight
            self.rewards.reward_is_alive                     = None #0.25
            self.rewards.penalty_failed                      = None

        """ general simulatin settings """
        if True :
            self.decimation = 4 #2
            self.episode_length_s = 15.0
            self.sim.dt = 0.005
            self.sim.disable_contact_processing = True
            self.sim.physics_material = self.scene.terrain.physics_material
            self.sim.render_interval = 4 #50Hz render
            # update sensor update periods
            if self.scene.height_scanner is not None:
                self.scene.height_scanner.update_period = self.decimation * self.sim.dt
            if self.scene.contact_forces is not None:
                self.scene.contact_forces.update_period = self.sim.dt

            # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
            if getattr(self.curriculum, "terrain_levels", None) is not None:
                if self.scene.terrain.terrain_generator is not None:
                    self.scene.terrain.terrain_generator.curriculum = True
            else:
                if self.scene.terrain.terrain_generator is not None:
                    self.scene.terrain.terrain_generator.curriculum = False

            self.viewer.eye             = (1.5, 1.5, 0.9)
            self.viewer.lookat          = (0.0, 0.0, 0.0)
            self.viewer.cam_prim_path   = "/OmniverseKit_Persp"
            self.viewer.resolution      = (1024, 1024)      # Square
            self.viewer.origin_type     = "asset_root"
            self.viewer.env_index       = 0
            self.viewer.asset_name      = "robot"


gym.register(
    id=task_name,
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True, #True
    kwargs={
        "env_cfg_entry_point": env_cfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.UnitreeAliengoBasePPORunnerCfg,
    },
)


def infer_input_output_sizes(state_dict):
    # Find the first layer's weight (input size)
    first_layer_key = next(iter(state_dict.keys()))
    input_size = state_dict[first_layer_key].shape[1]
    
    # Find the last layer's weight (output size)
    last_layer_key = list(state_dict.keys())[-2]  # Assuming the last layer is a Linear layer with weights and biases
    output_size = state_dict[last_layer_key].shape[0]
    
    return input_size, output_size


def load_rsl_rl_policy(path, device="cpu", num_actions=28):

    loaded_dict = torch.load(path)

    actor_critic = ActorCritic(
        num_actor_obs=259,
        num_critic_obs=259,
        num_actions=num_actions,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
    )
    actor_critic.load_state_dict(loaded_dict["model_state_dict"])
    actor_critic.to(device)
    actor_critic.eval()

    policy = actor_critic.act_inference

    return policy


def main():

    """ LOAD THE ENVIRONMENT"""
    if True:
        # parse configuration
        env_cfg = parse_env_cfg(task_name, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(task_name, args_cli)

        # create isaac environment and wrap around environment for rsl-rl
        env = gym.make(task_name, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env)

    """ Load the policies """
    if True :
        multipolicy_folder_path = f"model/{args_cli.multipolicies_folder}"
        policy_path_list = [os.path.join(multipolicy_folder_path, file) for file in os.listdir(multipolicy_folder_path) if os.path.isfile(os.path.join(multipolicy_folder_path, file))]

        policies = []
        for policy_path in policy_path_list : 
            print('Policy : ',policy_path)

            if '.pt' in os.path.basename(policy_path):

                # Is a RSL RL policy with a Actor-Critic architecture
                if 'actor_critic' in os.path.basename(policy_path):
                    policy = load_rsl_rl_policy(path=policy_path, device=agent_cfg.device)

                # Is a Imitation Learning Policy with a simple MLP architecture
                elif 'MLP' in os.path.basename(policy_path):
                    # Load the state dictionary and retrieve input and output size from the network
                    model_as_state_dict = torch.load(policy_path)
                    input_size, output_size = infer_input_output_sizes(model_as_state_dict)

                    info_dict['Action size'] = output_size

                    # Load the model
                    policy = Model(input_size, output_size)
                    policy.load_state_dict(torch.load(policy_path))
                    policy = policy.to(env.device)

                # Invalid policy name
                else :
                    # raise NameError(F"Invalid policy name or network type ('actor_critic' or 'MLP') for {policy_path}")
                    # Load the state dictionary and retrieve input and output size from the network
                    model_as_state_dict = torch.load(policy_path)
                    input_size, output_size = infer_input_output_sizes(model_as_state_dict)

                    info_dict['Action size'] = output_size

                    # Load the model
                    policy = Model(input_size, output_size)
                    policy.load_state_dict(torch.load(policy_path))
                    policy = policy.to(env.device)
                
                # Append the loaded policy to the list of policies.
                policies.append(policy)

    """ Run the Simulation and collect Data """
    if True :
        cumulated_rewards   = torch.zeros((args_cli.num_envs), device=env.device)
        trajectories_length = torch.zeros((args_cli.num_envs), device=env.device)
        cumulated_distances = torch.zeros((args_cli.num_envs), device=env.device)
        terrains_difficulty = torch.zeros((args_cli.num_envs), device=env.device)
        cost_of_transports  = torch.zeros((args_cli.num_envs), device=env.device) 
        velocity_commands_b = torch.zeros((args_cli.num_envs, 3), device=env.device)
        robots_pos_lw       = torch.zeros((args_cli.num_envs, 3), device=env.device)
        last_time = time.time()

        result_df = pd.DataFrame(columns=['cumulated_reward', 'trajectory_length', 'survived', 'commanded_speed_for', 'commanded_speed_lat', 'commanded_speed_ang', 'average_speed', 'cumulated_distance', 'CoT', 'stairs_cleared', 'terrain_difficulty'])

        iteration = 0 # Reward manager is accessible only after a first step

        # reset environment
        obs, _ = env.get_observations()

        # simulate environment
        while len(result_df) < num_trajectory:
            i = len(result_df)
            iteration +=1

            if time.time() > last_time+5:
                last_time = time.time()
                print(f"Iteration : {100*i/num_trajectory} %")

            with torch.inference_mode():
                # agent stepping
                action_list = []
                for policy in policies :
                    action_list.append(policy(obs)) #shape (num_envs, action_shape)

                # Reshape the actions
                actions = torch.cat(action_list, dim=1)

                # env stepping
                obs, rew, dones, extras = env.step(actions) 

                cumulated_rewards   += rew
                trajectories_length += env.unwrapped.step_dt

                # CoT_cfg = env.unwrapped.get_term_cfg('penalty_CoT')
                # CoT = CoT_cfg.func(env.unwrapped, **CoT_cfg.params)

                # One of the trajectory terminated
                if dones.any():
                    env_terminated_idx = torch.nonzero(dones)

                    cumulated_reward    = cumulated_rewards[env_terminated_idx].squeeze()
                    survived            = extras['time_outs'][env_terminated_idx].int().squeeze() # shape(env_terminated)
                    cumulated_distance  = cumulated_distances[env_terminated_idx].squeeze()
                    trajectory_length   = trajectories_length[env_terminated_idx].squeeze()
                    commanded_speed_for = velocity_commands_b[env_terminated_idx][..., 0].squeeze()
                    commanded_speed_lat = velocity_commands_b[env_terminated_idx][..., 1].squeeze()
                    commanded_speed_ang = velocity_commands_b[env_terminated_idx][..., 2].squeeze()
                    average_speed       = cumulated_distance / trajectory_length
                    cost_of_transport   = (cost_of_transports / trajectories_length)[env_terminated_idx].squeeze()
                    stairs_cleared      = (((torch.max(robots_pos_lw[env_terminated_idx][...,:2], dim=-1).values - platform_width )/ step_width).int()).squeeze()
                    terrain_difficulty  = terrains_difficulty[env_terminated_idx].squeeze()

                    # Append the new result to the existing DataFrame
                    tensor_list = [cumulated_reward, trajectory_length, survived, commanded_speed_for, commanded_speed_lat, commanded_speed_ang, average_speed, cumulated_distance, cost_of_transport, stairs_cleared, terrain_difficulty]
                    new_result_df = pd.DataFrame([tensor.cpu().numpy() for tensor in tensor_list]).T
                    result_df = pd.concat([result_df, new_result_df.set_axis(result_df.columns, axis=1)], ignore_index=True)

                    #reset cumulated variable
                    cumulated_rewards[env_terminated_idx] = 0.0
                    trajectories_length[env_terminated_idx] = 0.0

                # Value reseted by env, must be kept
                cumulated_distances = env.unwrapped.command_manager.get_term('base_velocity').metrics['cumulative_distance'].clone().detach()
                velocity_commands_b = env.unwrapped.command_manager.get_term('base_velocity').vel_command_b.clone().detach()
                robots_pos_lw       = env.unwrapped.scene['robot'].data.root_pos_w - env.unwrapped.scene.env_origins
                terrains_difficulty = env.unwrapped.scene.terrain.difficulty.clone().detach()
                cost_of_transports  = env.unwrapped.reward_manager._episode_sums['penalty_CoT']

                # if iteration > 1 : cost_of_transports  = env.unwrapped.reward_manager._episode_sums['penalty_CoT']

        # close the simulator
        env.close()

    """ Process Data"""
    if True :
        # Process Result Dict
        result_dict['eval_name'] = task_name
        result_dict['number_eval_steps'] = num_trajectory

        result_dict['cumulated_reward']   = result_df['cumulated_reward'].mean(skipna=True)
        result_dict['trajectory_length']  = result_df['trajectory_length'].mean(skipna=True)
        result_dict['survived']           = result_df['survived'].mean(skipna=True)
        result_dict['average_speed']      = result_df['average_speed'].mean(skipna=True)
        result_dict['cumulated_distance'] = result_df['cumulated_distance'].mean(skipna=True)
        result_dict['stairs_cleared']     = result_df['stairs_cleared'].median(skipna=True)
        result_dict['terrain_difficulty'] = result_df['terrain_difficulty'].median(skipna=True)


        # Add result dict into info dict
        info_dict[task_name] = result_dict

    """ Save data """
    if True :
        # Save info dict
        with open(result_folder_path +'/info.json', 'w') as json_file:
            json.dump(info_dict, json_file, indent=4)

        # Save result dict
        with open(full_result_folder_path +'/result.json', 'w') as json_file:
            json.dump(info_dict, json_file, indent=4)

        # Save the result_df to a pickle file
        result_df.to_pickle(full_result_folder_path + '/result_df.pkl')




if __name__ == "__main__":
    # To automatically close all the unecessary windows
    import omni.ui
    windows = omni.ui.Workspace.get_windows()   
    for window in windows: 
        name = window.title
        if name=="Property" or name=="Content" or name=="Layer" or name=="Semantics Schema Editor" or name=="Stage" or name=="Render Settings" or name=="Console" or name=="Simulation Settings":
            omni.ui.Workspace.show_window(str(window), False)
            
    # run the main function
    main()
    # close sim app
    simulation_app.close()
