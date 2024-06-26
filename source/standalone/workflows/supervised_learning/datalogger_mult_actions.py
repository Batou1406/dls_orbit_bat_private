# Copyright (c) 2022-2024, The LAB Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ./lab.sh -p source/standalone/workflows/supervised_learning/datalogger_mult_actions.py --task Isaac-Model-Based-Base-Aliengo-v0  --num_envs 256 --load_run test --checkpoint model_14999.pt --dataset_name baseTask15Act25HzGood1 --buffer_size 15 --seed 456

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
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs",     type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--num_step",     type=int, default=1000, help="Number of simulation step : the number of datapoints would be : num_step*num_envs")
parser.add_argument("--task",         type=str, default=None, help="Name of the task.")
parser.add_argument("--seed",         type=int, default=None, help="Seed used for the environment")
parser.add_argument("--dataset_name", type=str, default=None, help="Folder where to log the generated dataset (in /dataset/task/)")
parser.add_argument("--buffer_size",  type=int, default=5,    help="Number of prediction steps")
parser.add_argument("--testing_flag", action="store_true",default=False,help="Flag to generate testing data, default is training data")

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
import time


""" --- Main --- """
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
    observations_list = []
    actions_list = []

    # Temporary variable to create the rolling buffer
    buffer_obs = []
    buffer_act = []

    # Set the data to be testing or training data
    file_prefix = 'training_data'
    if args_cli.testing_flag:
        file_prefix = 'testing_data'

    num_samples = args_cli.num_step
    buffer_size =  args_cli.buffer_size
    t = time.time()

    # reset environment
    obs, _ = env.get_observations()
    actions = policy(obs) #just to get the shape for printing

    print('\nobservation shape:', obs.shape[-1])
    print('     action shape:', actions.shape[-1])
    print('   Buffer   size :', buffer_size,'\n')

    count = 0

    # simulate environment
    while simulation_app.is_running() and len(observations_list) < (num_samples + buffer_size):

        count +=1
        # print(len(observations_list))
        # print(count)
        # print()

        # Printing
        if len(observations_list) % 100 == 0:
            progress = 100 * float(len(observations_list)) / num_samples
            iteration = len(observations_list)
            time_remaining = (time.time() - t) * ((num_samples - len(observations_list)) / 100)
            print(f'\rProgression {progress:6.2f}%, Iteration: {iteration:6d}, Time remaining: {time_remaining:6.0f}s', end='\n')
            print('\033[F', end='')
            t = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)

            # Log data into the rolling buffer Log

            # To save at 25Hz and not 50Hz
            if count%2 :
                buffer_obs.append(obs)
                buffer_act.append(actions)         

            # env stepping
            obs, _, _, _ = env.step(actions)


            # Skip the buffer_size first iteration until we have enough data in the buffer to start logging
            if len(buffer_obs) < buffer_size:
                continue

            # To save at 25Hz and not 50Hz
            if count%2 :
                # Datalogging for Dataset generation - save observation at time i and action at time i, i+1,...,i+buffer_size
                observations_list.append(buffer_obs[0].cpu())                                  # shape(batch_size, obs_dim)
                actions_list.append(torch.stack(buffer_act).permute(1,0,2).flatten(1,2).cpu()) # shape(buffer_size, batch_size, act_dim) -> (batch_size, buffer_size*act_dim) 

                # Remove the oldest observation and action
                buffer_obs.pop(0)
                buffer_act.pop(0)


    # Concatenate all observations and actions

    observations_tensor = torch.cat(observations_list).view(-1, obs.shape[-1])              # shape(len_list*num_envs, obs_dim)
    actions_tensor      = torch.cat(actions_list).view(-1, buffer_size*actions.shape[-1])   # shape(len_list*num_envs, buffer_size*act_dim)

    print('\n\n\nobservations_tensor shape ', observations_tensor.shape[-1])
    print('   actions_tensor   shape ', actions_tensor.shape[-1])

    # Save the Generated dataset
    data = {
        'observations': observations_tensor,
        'actions': actions_tensor
    }
    torch.save(data, f'{logging_directory}/{file_prefix}.pt') 

    print('\nData succesfully saved as ', file_prefix)
    print('Saved at :',f'{logging_directory}/{file_prefix}.pt')
    print('\nDataset of ',observations_tensor.shape[0],'datapoints')
    print('Input  size :', observations_tensor.shape[-1])
    print('Output size :', actions_tensor.shape[-1],'\n')

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

    print('Everything went well, closing\n')

    # close sim app
    simulation_app.close()
