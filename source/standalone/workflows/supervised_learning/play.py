# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to generate a dataset for supervised learning with RL agent from RSL-RL. 
The action space consists of multiple actions of the original environment"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""
import argparse
from omni.isaac.orbit.app import AppLauncher

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
parser.add_argument("--model_name", type=str,       default='baseTaskMultActGood1/model1',          help="Name of the model to load (in /model/controller/)")

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

import omni.isaac.contrib_tasks  # noqa: F401
import omni.isaac.orbit_tasks  # noqa: F401
from omni.isaac.orbit_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.orbit_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

from train import Model


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

    # Retrieve the Model parameters
    buffer_size = 5
    input_size = env.num_obs  
    output_size = buffer_size*env.num_actions  

    model_path = 'model/' + args_cli.controller_name + '/' + args_cli.model_name + '.pt'
    policy = Model(input_size, output_size)
    policy.load_state_dict(torch.load(model_path))
    policy = policy.to(env.device)

    # reset environment
    obs, _ = env.get_observations()

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)

            # Select only actions at next step
            actions = actions.view(-1, buffer_size, env.num_actions) # reshape /!\ buffer_size comes before num_actions otherwise it don't work /!\
            actions = actions[:,0,:] # select first action

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
