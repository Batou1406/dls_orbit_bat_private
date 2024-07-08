# Copyright (c) 2022-2024, The LAB Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to generate a dataset for supervised learning with RL agent from RSL-RL. 
The action space consists of multiple actions of the original environment"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""
import argparse
from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--cpu", action="store_true",   default=False,                                  help="Use CPU pipeline.")
parser.add_argument("--disable_fabric", action="store_true", default=False,                         help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int,         default=1,                                      help="Number of environments to simulate.")
parser.add_argument("--task", type=str,             default='Isaac-Model-Based-Base-Aliengo-v0',    help="Name of the task.")
parser.add_argument("--seed", type=int,             default=None,                                   help="Seed used for the environment")
parser.add_argument("--controller_name", type=str,  default='aliengo_model_based_base',             help="Name of the controller")
# parser.add_argument("--model_name", type=str,       default='baseTaskNoise5ActGood1/model1',   help="Name of the model to load (in /model/controller/)")
parser.add_argument("--model_name", type=str,       default='baseGiulio2/model1',   help="Name of the model to load (in /model/controller/)")

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
import torch.distributions.constraints

# Jax prealoccate memory
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from rsl_rl.runners import OnPolicyRunner

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

from train import Model

import matplotlib.pyplot as plt


def infer_input_output_sizes(state_dict):
    # Find the first layer's weight (input size)
    first_layer_key = next(iter(state_dict.keys()))
    input_size = state_dict[first_layer_key].shape[1]
    
    # Find the last layer's weight (output size)
    last_layer_key = list(state_dict.keys())[-2]  # Assuming the last layer is a Linear layer with weights and biases
    output_size = state_dict[last_layer_key].shape[0]
    
    return input_size, output_size

""" --- Main --- """
def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    # agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # wrap around environment for rsl-rl -> usefull to get the properties and method 
    env = RslRlVecEnvWrapper(env) 

    # Construct the model path
    model_path = 'model/' + args_cli.controller_name + '/' + args_cli.model_name + '.pt'

    # Load the state dictionary and retrieve input and output size from the network
    model_as_state_dict = torch.load(model_path)
    input_size, output_size = infer_input_output_sizes(model_as_state_dict)

    # Load the model
    policy = Model(input_size, output_size)
    policy.load_state_dict(torch.load(model_path))
    policy = policy.to(env.device)

    # From model output and env actions, retrieve the buffer size
    # buffer_size = output_size // env.num_actions 
    buffer_size = (output_size - 8) // 20

    # Print
    print('\nModel : ',args_cli.model_name)
    print('Task  : ',args_cli.controller_name)
    print('Path  : ', model_path,'\n')
    print('Model Input  size :',input_size)
    print('Model Output size :',output_size)
    print('   Buffer    size :',buffer_size)
    print(' Env  Action size :',env.num_actions)
    print(' Env   Obs   size :',env.num_obs,'\n')

    # reset environment
    obs, _ = env.get_observations()

    abs_list = []
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)

            # abs_list.append(torch.sum(actions[0,256].abs()).cpu())
            # if(len(abs_list)>80):
            #     plt.plot(abs_list)
            #     plt.yscale("log")
            #     plt.show()
            #     return 

            # env stepping
            obs, _, _, _ = env.step(actions) 

            if not (torch.isfinite(obs).all()):
                print('Problem with NaN value in observation')
                obs, _ = env.reset()




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
