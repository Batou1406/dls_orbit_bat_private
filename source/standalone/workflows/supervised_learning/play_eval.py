# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""
import argparse
from omni.isaac.lab.app import AppLauncher
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--multipolicies_folder", type=str, default=None, help="Path to folder that contains the different policies in model/multipolicies_folder")
parser.add_argument("--experiment_folder", type=str, default=None, help="Where to save the results in ./eval/experiment_folder")
parser.add_argument("--num_steps", type=int, default=None, help="Number of step to generate the data")
parser.add_argument("--num_samples", type=int, default=None, help="Number of samples for the sampling controller")
parser.add_argument("--controller", type=str, default=None, help="Type of controller to use")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""
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
from omni.isaac.lab_assets.unitree import UNITREE_ALIENGO_SELF_COLLISION_TORQUE_CONTROL_CFG  # isort: skip
from omni.isaac.lab.terrains.config.niceFlat import COBBLESTONE_ROAD_CFG, COBBLESTONE_FLAT_CFG
from omni.isaac.lab.terrains.config.climb import STAIRS_TERRAINS_CFG
from omni.isaac.lab.terrains.config.speed import SPEED_TERRAINS_CFG
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG
from omni.isaac.lab_tasks.manager_based.locomotion.model_based.mdp import modify_reward_weight
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.terrains import randomTerrainImporter
import omni.isaac.lab_tasks.manager_based.locomotion.model_based.mdp as mdp


json_info_path = f"./model/{args_cli.multipolicies_folder}/info.json"
if os.path.isfile(json_info_path):
    with open(json_info_path, 'r') as json_file:
        info_dict = json.load(json_file)  # Load JSON data into a Python dictionary


num_samples = args_cli.num_samples
controller_name = args_cli.controller

if controller_name == 'samplingController':
    propotion_previous_solution = 0.0
    debug_apply_action = None
    warm_start='-'
elif controller_name == '-samplingController_no_warm_start':
    propotion_previous_solution = 1.0
    debug_apply_action = 'trot'
    warm_start = 'no_warm_start'

# task_name = f"{info_dict['p_typeAction']}-{info_dict['F_typeAction']}-H{info_dict['prediction_horizon_step']}-dt{info_dict['prediction_horizon_time'][2:4]}-{info_dict['tot_epoch']}"
# task_name = f"{info_dict['p_typeAction']}-{info_dict['F_typeAction']}-H{info_dict['prediction_horizon_step']}-dt{info_dict['prediction_horizon_time'][2:4]}"
task_name = f"{info_dict['p_typeAction']}-{info_dict['F_typeAction']}-H{info_dict['prediction_horizon_step']}-dt{info_dict['prediction_horizon_time'][2:4]}-samples{num_samples}-{warm_start}"

if info_dict['F_typeAction'] == 'spline' :
    info_dict['F_typeAction'] = 'cubic_spline' 
if info_dict['p_typeAction'] == 'spline' :
    info_dict['p_typeAction'] = 'cubic_spline' 

@configclass
class ActionsCfg:
    """Action specifications for the MDP.
    - Robot joint position - dim=12
    """
    model_base_variable = mdp.ModelBaseActionCfg(
        asset_name="robot",
        joint_names=[".*"], 
        controller=mdp.samplingController,
        # controller=mdp.samplingTrainer,
        optimizerCfg=mdp.ModelBaseActionCfg.OptimizerCfg(
            multipolicy=1,
            prevision_horizon=info_dict['prediction_horizon_step'],
            discretization_time=float(info_dict['prediction_horizon_time'][0:4]),
            parametrization_p=info_dict['p_typeAction'],
            parametrization_F=info_dict['F_typeAction'],

            propotion_previous_solution= propotion_previous_solution,
            debug_apply_action = debug_apply_action,
            num_samples=num_samples,
            ),
        )

@configclass
class env_cfg(LocomotionModelBasedEnvCfg):
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
        self.commands.base_velocity.ranges.for_vel_b = (0.3, 0.6)
        self.commands.base_velocity.ranges.lat_vel_b = (-0.2, 0.2)
        self.commands.base_velocity.ranges.ang_vel_b = (-0.5, 0.5)
        self.commands.base_velocity.ranges.initial_heading_err = (-0.0, 0.0)  
        # self.commands.base_velocity.ranges.for_vel_b = (0.0, 0.0)
        # self.commands.base_velocity.ranges.lat_vel_b = (-0.0, -0.0)
        # self.commands.base_velocity.ranges.ang_vel_b = (-0., 0.)
        # self.commands.base_velocity.ranges.initial_heading_err = (-0.0, 0.0)    
        self.commands.base_velocity.resampling_time_range = (7.5, 7.5)

        """ ----- Observation ----- """
        # To add or not noise on the observations
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


        # post init of parent
        super().__post_init__()

        self.decimation = 2 #2

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


def load_rsl_rl_policy(path, device="cpu", num_actions=108):

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


def compute_metrics(tensor):
    metrics = {
        'mean_dim0': tensor.mean(dim=0).tolist(),
        'mean_dim1': tensor.mean(dim=1).tolist(),
        'mean_all': tensor.mean().item(),

        'median_dim0': tensor.median(dim=0).values.tolist(),
        'median_dim1': tensor.median(dim=1).values.tolist(),
        'median_all': tensor.median().item(),

        'std_dim0': tensor.std(dim=0).tolist(),
        'std_dim1': tensor.std(dim=1).tolist(),
        'std_all': tensor.std().item(),

        'var_dim0': tensor.var(dim=0).tolist(),
        'var_dim1': tensor.var(dim=1).tolist(),
        'var_all': tensor.var().item(),

        'max_dim0': tensor.max(dim=0).values.tolist(),
        'min_dim0': tensor.min(dim=0).values.tolist(),
        'max_all': tensor.max().item(),
        'min_all': tensor.min().item()
    }

    return metrics

def main():

    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(task_name, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(task_name, args_cli)

    # create isaac environment and wrap around environment for rsl-rl
    env = gym.make(task_name, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # Create logging directory if necessary
    # logging_directory = f'eval/{args_cli.experiment_folder}/{args_cli.experiment}'
    logging_directory = f'eval/{args_cli.experiment_folder}/{task_name}'

    if not os.path.exists(logging_directory):
        os.makedirs(logging_directory)
    else :
        raise KeyError('There is already an experiment setup in this directory, Please provide another folder_name')


    # Load the policies 
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

    rewards = torch.empty((args_cli.num_envs,args_cli.num_steps), device=env.device)
    sampling_cost = torch.empty((args_cli.num_envs,args_cli.num_steps), device=env.device)
    sampling_step_cost = torch.empty((args_cli.num_envs,args_cli.num_steps, info_dict['prediction_horizon_step']), device=env.device)
    initial_cost = torch.empty((args_cli.num_envs,args_cli.num_steps), device=env.device)

    info_dict['eval_name'] = task_name
    info_dict['number_eval_steps'] = args_cli.num_steps

    # reset environment
    obs, _ = env.get_observations()

    # simulate environment
    for i in range(args_cli.num_steps) :
        # run everything in inference mode

        if i % 200 == 0:
            print(f"Iteration : {100*i/args_cli.num_steps} %")

        with torch.inference_mode():
            # agent stepping
            action_list = []
            for policy in policies :
                action_list.append(policy(obs)) #shape (num_envs, action_shape)

            # Reshape the actions
            actions = torch.cat(action_list, dim=1)

            # env stepping
            obs, rew, dones, extras = env.step(actions) 

            rewards[:,i] = rew  #shape(num_envs, num_steps)->(num_envs)
            sampling_cost[:,i] = env.unwrapped.action_manager.get_term('model_base_variable').controller.batched_cost
            sampling_step_cost[:,i,:] = env.unwrapped.action_manager.get_term('model_base_variable').controller.samplingOptimizer.step_cost
            initial_cost[:,i] = env.unwrapped.action_manager.get_term('model_base_variable').controller.samplingOptimizer.initial_cost

    # close the simulator
    env.close()

    # Save the Results
    np.savetxt(f'{logging_directory}/rewards.csv', rewards.cpu().numpy(), delimiter=',', fmt='%.6f')
    np.savetxt(f'{logging_directory}/sampling_cost.csv', sampling_cost.cpu().numpy(), delimiter=',', fmt='%.6f')
    # np.savetxt(f'{logging_directory}/sampling_step_cost.csv', sampling_step_cost.cpu().numpy(), delimiter=',', fmt='%.6f')

    rewards_metrics = compute_metrics(rewards)
    sampling_metrics = compute_metrics(sampling_cost)

    # Filter NaN
    filtered_sampling_step_cost = (sampling_step_cost.flatten(0, 1))[~(torch.any(sampling_step_cost.flatten(0, 1).isnan(),dim=1))] #shape (num_envs, num_steps, horizon) -> (num_envs*num_iter, horizon)
    filtered_initial_cost =  (initial_cost)[~(torch.any(initial_cost.isnan(),dim=1))]

    step_cost_mean_values = filtered_sampling_step_cost.mean(dim=0)
    step_cost_median_values = filtered_sampling_step_cost.median(dim=0).values
    step_cost_std_values = filtered_sampling_step_cost.std(dim=0)
    step_costq1_values = torch.quantile(filtered_sampling_step_cost, 0.25, dim=0)
    step_costq3_values = torch.quantile(filtered_sampling_step_cost, 0.75, dim=0)


    initial_cost_median_values = filtered_initial_cost.median(dim=1).values
    initial_costq1_values = torch.quantile(filtered_initial_cost, 0.25)
    initial_costq3_values = torch.quantile(filtered_initial_cost, 0.75)

    # Store the results in a dictionary
    step_cost_results = {
        'step_cost_mean': step_cost_mean_values.cpu().numpy().tolist(),
        'step_cost_median': step_cost_median_values.cpu().numpy().tolist(),
        'step_coststd': step_cost_std_values.cpu().numpy().tolist(),
        'step_costq1_values': step_costq1_values.cpu().numpy().tolist(),
        'step_costq3_values': step_costq3_values.cpu().numpy().tolist(),

        'initial_cost_median': initial_cost_median_values.cpu().numpy().tolist(),
        'initial_costq1_values': initial_costq1_values.cpu().numpy().tolist(),
        'initial_costq3_values': initial_costq3_values.cpu().numpy().tolist()
    }

    with open(f'{logging_directory}/rewards_metrics.json', 'w') as json_file:
        json.dump(rewards_metrics, json_file, indent=4)
    with open(f'{logging_directory}/sampling_metrics.json', 'w') as json_file:
        json.dump(sampling_metrics, json_file, indent=4)

    info_dict['rewards_median_all'] = rewards.median().item()
    info_dict['sampling_cost_median_all'] = sampling_cost.median().item()
    info_dict['step_cost_results'] = step_cost_results
    with open(f'{logging_directory}/info.json', 'w') as json_file:
        json.dump(info_dict, json_file, indent=4)


    result_log_dir = f'eval/{args_cli.experiment_folder}'
    json_info_res_path = f"{result_log_dir}/info.json"
    if os.path.isfile(json_info_res_path):
        with open(json_info_res_path, 'r') as json_file:
            info_res_dict = json.load(json_file)  # Load JSON data into a Python dictionary
    else :
        info_res_dict = {}

    res_dict = {}
    res_dict['rewards_median_all'] = rewards.median().item()
    res_dict['sampling_cost_median_all'] = sampling_cost.median().item()
    res_dict['step_cost_results'] = step_cost_results
    info_res_dict[task_name] = res_dict

    with open(f'{result_log_dir}/info.json', 'w') as json_file:
        json.dump(info_res_dict, json_file, indent=4)





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
