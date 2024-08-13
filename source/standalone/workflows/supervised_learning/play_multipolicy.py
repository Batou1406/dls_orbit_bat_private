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
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--multipolicies_folder", type=str, default=None, help="Path to folder that contains the different policies in model/multipolicies_folder")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""
import gymnasium as gym
import os
import torch
from rsl_rl.runners import OnPolicyRunner
import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (RslRlOnPolicyRunnerCfg,
                                                        RslRlVecEnvWrapper,
                                                        export_policy_as_jit,
                                                        export_policy_as_onnx,)
from train import Model # Where to get the model architecture for MLP, may want to change that

from rsl_rl.modules import ActorCritic

def infer_input_output_sizes(state_dict):
    # Find the first layer's weight (input size)
    first_layer_key = next(iter(state_dict.keys()))
    input_size = state_dict[first_layer_key].shape[1]
    
    # Find the last layer's weight (output size)
    last_layer_key = list(state_dict.keys())[-2]  # Assuming the last layer is a Linear layer with weights and biases
    output_size = state_dict[last_layer_key].shape[0]
    
    return input_size, output_size


def load_rsl_rl_policy(path, device="cpu"):

    loaded_dict = torch.load(path)

    actor_critic = ActorCritic(
        num_actor_obs=259,
        num_critic_obs=259,
        num_actions=28,
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

    fake_task = 'Isaac-Model-Based-Fake-Aliengo-v0'

    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # create isaac environment and wrap around environment for rsl-rl
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)


    # Load the policies 
    multipolicy_folder_path = f"model/{args_cli.multipolicies_folder}"
    policy_path_list = [os.path.join(multipolicy_folder_path, file) for file in os.listdir(multipolicy_folder_path) if os.path.isfile(os.path.join(multipolicy_folder_path, file))]

    policies = []
    for policy_path in policy_path_list : 
        print('Policy : ',policy_path)

        # Is a RSL RL policy with a Actor-Critic architecture
        if 'actor_critic' in os.path.basename(policy_path):
            policy = load_rsl_rl_policy(path=policy_path, device=agent_cfg.device)

        # Is a Imitation Learning Policy with a simple MLP architecture
        elif 'MLP' in os.path.basename(policy_path):
            # Load the state dictionary and retrieve input and output size from the network
            model_as_state_dict = torch.load(policy_path)
            input_size, output_size = infer_input_output_sizes(model_as_state_dict)

            # Load the model
            policy = Model(input_size, output_size)
            policy.load_state_dict(torch.load(policy_path))
            policy = policy.to(env.device)

        # Invalid policy name
        else :
            raise NameError(F"Invalid policy name or network type ('actor_critic' or 'MLP') for {policy_path}")
        
        # Append the loaded policy to the list of policies.
        policies.append(policy)



    # reset environment
    obs, _ = env.get_observations()

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            action_list = []
            for policy in policies :
                action_list.append(policy(obs)) #shape (num_envs, action_shape)

            # Reshape the actions
            actions = torch.cat(action_list, dim=1)

            # env stepping
            obs, _, _, _ = env.step(actions) 

    # close the simulator
    env.close()



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
