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
parser.add_argument("--freq_reduction",type=int,default=2,    help="Factor of reduction of the recording frequency compare to playing frequency")

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

# import omni.isaac.contrib_tasks  # noqa: F401
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

    # specify directory for loading experiments
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_onnx(ppo_runner.alg.actor_critic, export_model_dir, filename="policy.onnx")

    # Create logging directory to save the recorded dataset
    logging_directory = f'dataset/{agent_cfg.experiment_name}/{args_cli.dataset_name}'
    if not os.path.exists(logging_directory):
        os.makedirs(logging_directory)

    # Set the data to be testing or training data
    file_prefix = 'training_data'
    if args_cli.testing_flag:
        file_prefix = 'testing_data'

    # Type of action recorded
    typeAction = 'discrete' # 'discrete', 'spline' # TODO implement spline

    # Variable for datalogging
    observations_list = []
    actions_list = []

    # Temporary variable to create the rolling buffer
    buffer_obs = []
    buffer_act = []


    # oder helper variables
    num_samples = args_cli.num_step
    buffer_size =  args_cli.buffer_size
    t = time.time()
    printing_freq = 10
    count = 0
    frequency_reduction = args_cli.freq_reduction
    f_len, d_len, p_len, F_len = 4, 4, 8, 12

    # reset environment
    obs, _ = env.get_observations()
    actions = policy(obs) #just to get the shape for printing

    # Printing
    print('\n---------------------------------------------------------------------')
    print('\n----- Datalogger Configuration -----\n')

    print(f"\nLoading experiment from directory: {log_root_path}")
    print(f"Loading model checkpoint from: {resume_path}")

    print(f"\nNumber of envs: {args_cli.num_envs}")
    print(f"Number of samples:  {args_cli.num_step}")
    print(f"Recorded dataset will consists of {args_cli.num_envs*args_cli.num_step} datapoints")

    print(f"\nDataset will be recorded as {file_prefix}")
    print('and saved at :',f'{logging_directory}/{file_prefix}.pt')

    print('\nInitial observation shape:', obs.shape[-1])
    print('Initial   action    shape:', actions.shape[-1])

    print(f"\nSimulation runs with time step {env.unwrapped.physics_dt} [s], at frequency {1/env.unwrapped.physics_dt} [Hz]")
    print(f"Policy runs with time step {env.unwrapped.step_dt} [s], at frequency {1/env.unwrapped.step_dt} [Hz]")
    print(f"Dataset will be recorded with time step {env.unwrapped.step_dt*frequency_reduction} [s], at frequency {1/(frequency_reduction*env.unwrapped.step_dt)} [Hz]")
    print(f"Which will correspond in a prediction horizon of {buffer_size*env.unwrapped.step_dt*frequency_reduction} [s]")

    print(f"\nType of action recorded: {typeAction}")
    print(f"with N = {buffer_size} prediction horizon")

    print('\nDataset Input  size :', obs.shape[-1])
    print('Dataset Output size :', 8 + buffer_size*(12+8),'\n')

    # Check if configuration is correct    
    while True:
        is_input_valid = input("Proceed with these parameters [Yes/No] ?")
        if is_input_valid == "No":
            return
        elif is_input_valid == "Yes":
            break
    print('\n----- Simulation -----')

    # simulate environment
    while simulation_app.is_running() and len(observations_list) < (num_samples):

        count +=1
        # print(len(observations_list))
        # print(count)
        # print()

        # Printing
        if len(observations_list) % printing_freq == 0:
            progress = 100 * float(len(observations_list)) / num_samples
            iteration = len(observations_list)
            time_remaining = (time.time() - t) * ((num_samples - len(observations_list))) / printing_freq
            print(f'\rProgression {progress:6.2f}%, Iteration: {iteration:6d}, Time remaining: {time_remaining:6.0f}s', end='\n')
            print('\033[F', end='')
            t = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)

            # Log data into the rolling buffer Log

            # To save at variable frequency and not policy frequency
            if count%frequency_reduction == 0 :
                buffer_obs.append(obs)
                buffer_act.append(actions)         

            # env stepping
            obs, _, _, _ = env.step(actions)

            # Skip the buffer_size first iteration until we have enough data in the buffer to start logging
            if len(buffer_obs) < buffer_size:
                continue

            # To save at variable frequency and not policy frequency
            if count%frequency_reduction == 0:
                # Datalogging for Dataset generation - save observation at time i and action at time i, i+1,...,i+buffer_size
                observations_list.append(buffer_obs[0].cpu())                                   # shape(batch_size, obs_dim)

                raw_actions = torch.stack(buffer_act).permute(1,2,0)                            # shape (batch_size, act_dim, buffer_size)
                f = raw_actions[:, 0                 : f_len                   , 0]             # shape (batch_size, f_len)
                d = raw_actions[:, f_len             :(f_len+d_len)            , 0]             # shape (batch_size, d_len)
                p = raw_actions[:,(f_len+d_len)      :(f_len+d_len+p_len)      , :].flatten(1,2)# shape (batch_size, buffer_size*p_len) /!\ Transpose to store the data with the right format
                F = raw_actions[:,(f_len+d_len+p_len):(f_len+d_len+p_len+F_len), :].flatten(1,2)# shape (batch_size, buffer_size*F_len)
                action_to_store = torch.cat((f,d,p,F),dim=1)                                    # shape (batch_size, f_len+d_len+buffer_size*(p_len+F_len))
                actions_list.append(action_to_store.cpu())                                      # shape (batch_size, f_len+d_len+buffer_size*(p_len+F_len))

                # Remove the oldest observation and action
                buffer_obs.pop(0)
                buffer_act.pop(0)


    # Concatenate all observations and actions
    observations_tensor = torch.cat(observations_list).view(-1, obs.shape[-1])                  # shape(len_list*num_envs, obs_dim)
    actions_tensor      = torch.cat(actions_list).view(-1,f_len+d_len+buffer_size*(p_len+F_len))# shape(len_list*num_envs, f_len+d_len+buffer_size*(p_len+F_len))

    print('\n\n\n----- Saving -----')
    print('\nobservations_tensor shape ', observations_tensor.shape[-1])
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
