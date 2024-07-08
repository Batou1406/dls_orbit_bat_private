# Copyright (c) 2022-2024, The LAB Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to generate a dataset for supervised learning with RL agent from RSL-RL."""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--dataset_name", type=str, default=None, help="Folder where to log the generated dataset (in /dataset/task/)")
parser.add_argument("--chunk_size", type=int, default=100, help="Number of data to save at the same time. Smaller chunk avoid memory overflow")

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
import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_onnx,
)


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_onnx(ppo_runner.alg.actor_critic, export_model_dir, filename="policy.onnx")

    # Create logging directory 
    logging_directory = f'dataset/{agent_cfg.experiment_name}/{args_cli.dataset_name}'
    if not os.path.exists(logging_directory):
        os.makedirs(logging_directory)

    # Variable for datalogging
    chunk_counter = 0
    chunk_size = args_cli.chunk_size
    observations_list = []
    actions_list = []
    iteration = 0

    file_prefix = 'training_data'
    num_samples = 1000

    # reset environment
    obs, _ = env.get_observations()

    # simulate environment
    while simulation_app.is_running() and iteration < num_samples:
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)

            # env stepping
            obs, _, _, _ = env.step(actions)

            # Datalogging for Dataset generation
            observations_list.append(obs.cpu())
            actions_list.append(actions.cpu())

            # Save the chunk
            if (iteration + 1) % chunk_size == 0:
                # Concatenate all observations and actions
                # observations_tensor = torch.stack(observations_list)                      # shape(len_list, num_envs, obs_dim)
                # actions_tensor      = torch.stack(actions_list)                           # shape(len_list, num_envs, act_dim)
                observations_tensor = torch.cat(observations_list).view(-1, obs.shape[-1])  # shape(len_list*num_envs, obs_dim)
                actions_tensor      = torch.cat(actions_list).view(-1, actions.shape[-1])   # shape(len_list*num_envs, act_dim)

                # Save the generated dataset chunk
                data = {
                    'observations': observations_tensor,
                    'actions': actions_tensor
                }
                torch.save(data, f'{logging_directory}/{file_prefix}_chunk_{chunk_counter}.pt')

                # Reset lists for the next chunk
                observations_list = []
                actions_list = []
                chunk_counter += 1
            
            # Update the iteration counter
            iteration +=1

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
