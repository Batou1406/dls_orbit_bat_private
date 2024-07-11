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
parser.add_argument("--disable_fabric", action="store_true", default=False,  help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs",     type=int,   default=256,               help="Number of environments to simulate.")
parser.add_argument("--task",         type=str,   default=None,              help="Name of the task.")
parser.add_argument("--seed",         type=int,   default=None,              help="Seed used for the environment")
parser.add_argument("--buffer_size",  type=int,   default=5,                 help="Number of prediction steps")
parser.add_argument('--epochs',       type=int,   default=25,  metavar='N',  help='number of epochs to train (default: 14)')
parser.add_argument('--batch-size',   type=int,   default=64,  metavar='N',  help='input batch size for training (default: 64)')
parser.add_argument('--lr',           type=float, default=1.0, metavar='LR', help='learning rate (default: 1.0)')
parser.add_argument('--gamma',        type=float, default=0.7, metavar='M',  help='Learning rate step gamma (default: 0.7)')
parser.add_argument("--model-name",   type=str,   default='dagger50hzSpline',  help="Name of the model to be saved")
parser.add_argument('--folder-name',  type=str,   default='goodPolicy',      help="Name of the folder to save the trained model in 'model/task/folder-name'")
# parser.add_argument("--freq_reduction",type=int,default=2,    help="Factor of reduction of the recording frequency compare to playing frequency")

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

from torch.utils.data import Dataset, DataLoader
# from dataloader import ObservationActionDataset, ChunkedObservationActionDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt


""" --- Model Definition --- """    
class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.lin1 = nn.Linear(input_size, 512)
        self.lin2 = nn.Linear(512, 256)
        self.lin3 = nn.Linear(256, 128)
        self.lin4 = nn.Linear(128, output_size)

    def forward(self, x):
        x = self.lin1(x)
        x = F.elu(x)
        x = self.lin2(x)
        x = F.elu(x)
        x = self.lin3(x)
        x = F.elu(x)
        x = self.lin4(x)
        return x


""" --- Training and Testing Function --- """
def train(model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if (batch_idx % 1 == 0) :
            # print('Train Epoch: {} [{}/{} ({:.0f}%)] batch cum. Loss: {:.6f}'.format(
            #     epoch, batch_idx * len(data), len(train_loader.dataset),
            #     100. * batch_idx /  len(train_loader), loss.item()), end='\r', flush=True)
            print('Training [{:7d}/{:7d} ({:.0f}%)] batch cum. Loss: {:.6f}'.format(
                batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx /  len(train_loader), loss.item()), end='\r', flush=True)

    train_loss_avg = train_loss / len(train_loader.dataset)
    # print(f"\nEpoch {epoch} : Average Train Loss {train_loss_avg}")
    print(f"\nAverage Train Loss {train_loss_avg:.4f}")
    return train_loss_avg


""" --- Dataset Class --- """
class ObservationActionDataset(Dataset):
    """Cutsom dataloader to load generated dataset
    Args :
        file_path : The file path to the dataset"""
    def __init__(self, obs, act):
        self.observations = obs #.clone().detach()
        self.actions = act #.clone().detach()

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]


""" --- Decay Function --- """
def alpha(epoch, type='indicator'):
    """ Decay function for the dagger Alogirthm"""

    # Indicator Function
    if type == 'indicator':
        if epoch in [0,1,2]:
            alpha = 1
        else :
            alpha = 0

    # eponential decay
    if type == 'exp':
        alpha = 0.6**(epoch)

    return alpha 


""" --- Cubic Spline Fitting function"""
def fit_cubic_with_constraint(y: torch.tensor, x: torch.tensor | None = None) -> torch.tensor:
    """ Minimize the sum of squared error between datapoints y_i and a cubic function parametrized with a,b,c,d 
    ie. -> Fit parameters a,b,c,d  that minimize : min(a,b,c,d) : sum( (y_i - (ax_i^3 + bx_i^2 + cx_i + d) )^2 )
    This problem is solved in closed form with exact solution
    
    Moreover, there is a constraint that d=y_0 -> which correspond to theta_0 = y_0 ie. the spline is exact for the first datapoint

    Then compute theta_-1, theta_0, theta_1, theta_2 parameters that are the cubic Hermite spline parameters 
    (Predictive Sampling : Real-time Behaviour Synthesis with MuJoCo - https://arxiv.org/pdf/2212.00541)

    Time has been imposed to be between [0, 1], with first x_0 = 0

    Args :
        x     (torch.tensor): time of the datapoints (if None linear in [0 1])           of shape (horizon)
        y     (torch.tensor): value of the datapoints           of shape (batch, num_legs, dim_3D, horizon)

    Returns :
        theta (torch.tensor): cubic Hermite spline parameters   of shape (batch, num_legs, 3, 4)
    """
    batch_size, num_legs, dim_3D, horizon = y.shape

    if x is None :
        x = torch.linspace(0, 1, steps=horizon, device=y.device)  # shape (horizon)
    
    # Extract the constraint value and set d directly
    theta_0 = y[..., 0] # (batch_size, num_legs, dim_3D, horizon) -> (batch_size, num_legs, dim_3D)

    # Construct the design matrix X for the remaining points
    X = torch.stack([x**3, x**2, x], dim=-1)  # shape: (horizon, num_param)
    num_param = X.shape[-1] # = 3 ie. a,b,c
    
    # Remove the first row corresponding to the constraint
    X_rest = X[1:]  # shape: (horizon-1, num_param)
    
    # Construct the target matrix Y for the remaining points and subtract the constraint value from them
    Y_rest = y[..., 1:]  # shape: (batch_size, num_legs, dim_3D, horizon-1)

    # Adjust targets by subtracting the constraint value
    Y_rest = Y_rest - theta_0.unsqueeze(-1)  # shape: (batch_size, num_legs, dim_3D, horizon-1)

    # Compute the normal equations components for the remaining points
    XtX_rest = torch.einsum('nk,nm->km', X_rest, X_rest)      # (horizon-1, num_param) x (horizon-1, num_param) -> shape: (num_param, num_param)
    XtY_rest = torch.einsum('nk,bijn->bijk', X_rest, Y_rest)  # (horizon-1, num_param) x (batch_size, num_legs, dim_3D, horizon-1) -> shape: (batch_size, num_legs, dim_3D, num_param)

    # Expand XtX to match the dimension of XtY for the solver
    XtX_rest.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(batch_size, num_legs, dim_3D, num_param, num_param) # shape (batch_size, num_legs, dim_3D, num_param, num_param)

    # Solve for the remaining coefficients a, b, c
    beta = torch.linalg.solve(XtX_rest, XtY_rest.transpose(-1,-2)).transpose(-1,-2) # shape: (batch_size, num_legs, dim_3D, num_param)

    # Retrieve coefficients a, b, c, d for each batch, legs, dim_3D
    a = beta[..., 0]    # shape (batch_size, num_legs, dim_3D)
    b = beta[..., 1]    # shape (batch_size, num_legs, dim_3D)
    c = beta[..., 2]    # shape (batch_size, num_legs, dim_3D)
    d = theta_0         # shape (batch_size, num_legs, dim_3D)

    # Find the missing theta parameters
    theta_n1 =   a +   b -   c + d  # shape (batch_size, num_legs, dim_3D)
    theta_1  =   a +   b +   c + d  # shape (batch_size, num_legs, dim_3D)
    theta_2  = 6*a + 4*b + 2*c + d  # shape (batch_size, num_legs, dim_3D)

    # Concatenate the coefficient
    theta = torch.cat((theta_n1.unsqueeze(-1), theta_0.unsqueeze(-1),theta_1.unsqueeze(-1), theta_2.unsqueeze(-1)), dim=-1) # shape (batch_size, num_legs, dim_3D, 4)
    
    return theta  # Coefficients a, b, c, d for each batch, legs, dim_3D


""" --- Main --- """
def main():
    """Play with RSL-RL agent."""

    # --- Step 1 : Load the expert Policy
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
    expert_policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)


    # --- Step 2 : Define usefull variables
    #  Set the data to be testing or training data
    file_prefix = 'training_data'

    # Type of action recorded
    typeAction = 'spline' # 'discrete', 'spline' # TODO implement spline

    # Temporary variable to create the rolling buffer
    buffer_obs = []
    buffer_act = []

    # Buffer size : number of prediction horizon for the student policy
    buffer_size =  args_cli.buffer_size

    # Trajectory length that are recorded between epoch
    trajectory_length_s = 3 # [s]

    # Number of epoch
    tot_epoch = args_cli.epochs

    # Dataset maximum size before clipping
    dataset_max_size = 300000 # [datapoints]

    # oder helper variables
    trajectory_length_iter = int(trajectory_length_s / (buffer_size*env.unwrapped.step_dt))
    last_time_outloop = time.time()
    last_time = time.time()
    # frequency_reduction = args_cli.freq_reduction
    f_len, d_len, p_len, F_len = 4, 4, 8, 12
    pF_param = buffer_size
    if typeAction == 'spline':
        pF_param = 4
    p_shape = (env.num_envs, 4, 2, pF_param)
    F_shape = (env.num_envs, 4, 3, pF_param)


    # Create logging directory if necessary
    logging_directory = f'model/{args_cli.task}/{args_cli.folder_name}'
    if not os.path.exists(logging_directory):
        os.makedirs(logging_directory)
    #increment logging dir number if already exists
    else :
        i = 1
        new_logging_directory = f"{logging_directory}{i}"
        while os.path.exists(new_logging_directory):
            i += 1
            new_logging_directory = f"{logging_directory}{i}"
        os.makedirs(new_logging_directory)
        logging_directory = new_logging_directory


    # --- Step 3 : Reset environment
    obs, _ = env.get_observations()
    actions = expert_policy(obs) #just to get the shape for printing


    # --- Step 4 : Define variable for model training
    use_cuda = not args_cli.cpu and torch.cuda.is_available()

    # Set seeds
    # torch.manual_seed(args_cli.seed)

    # Set device
    if use_cuda:  device = torch.device("cuda")
    else:         device = torch.device("cpu")

    # Set training and testing arguments
    train_kwargs = {'batch_size': args_cli.batch_size}

    #  Define Model criteria : model, optimizer and loss criterion and scheduler
    input_size = obs.shape[-1]
    output_size = 8 + pF_param*(p_len+F_len)
    student_policy  = Model(input_size, output_size).to(device)
    optimizer       = optim.Adadelta(student_policy.parameters(), lr=args_cli.lr)
    train_criterion = nn.MSELoss() 
    scheduler       = StepLR(optimizer, step_size=1, gamma=args_cli.gamma)
    epoch_avg_train_loss_list = []
    avg_epoch_reward_list = []

    # Printing
    if True : 
        print('\n---------------------------------------------------------------------')
        print('\n----- Datalogger Configuration -----\n')

        print(f"\nLoading experiment from directory: {log_root_path}")
        print(f"Loading model checkpoint from: {resume_path}")

        print(f"\nNumber of envs: {args_cli.num_envs}")
        # print(f"Number of samples:  {args_cli.num_step}")
        # print(f"Recorded dataset will consists of {args_cli.num_envs*args_cli.num_step} datapoints")

        print(f"\nDataset will be recorded as {file_prefix}")
        print('and saved at :',f'{logging_directory}/{file_prefix}.pt')

        print('\nInitial observation shape:', obs.shape[-1])
        print('Initial   action    shape:', actions.shape[-1])

        print(f"\nSimulation runs with time step {env.unwrapped.physics_dt} [s], at frequency {1/env.unwrapped.physics_dt} [Hz]")
        print(f"Policy runs with time step {env.unwrapped.step_dt} [s], at frequency {1/env.unwrapped.step_dt} [Hz]")
        # print(f"Dataset will be recorded with time step {env.unwrapped.step_dt*frequency_reduction} [s], at frequency {1/(frequency_reduction*env.unwrapped.step_dt)} [Hz]")
        # print(f"Which will correspond in a prediction horizon of {buffer_size*env.unwrapped.step_dt*frequency_reduction} [s]")

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

    observations_data = torch.empty(0,device=device)
    actions_data = torch.empty(0, device=device)

    for epoch in range(tot_epoch):
        print(f'\n----- Epoch {epoch+1} / {tot_epoch} ----- Total Remaining Time {(tot_epoch-epoch)*(time.time()-last_time_outloop):4.1f}[s]')
        last_time_outloop = time.time()

        # --- Step 1 : Get ID for student actions
        n = min(int(alpha(epoch)*args_cli.num_envs), args_cli.num_envs)
        expert_idx = torch.randperm(args_cli.num_envs)[:n]


        with torch.inference_mode(): # step 2 and 3 (and 4 but not required) in inference mode to avoid gradient computations
            epoch_reward = 0.0
            # Roll the simulation for trajectory_length_s time
            for j in range(trajectory_length_iter):
                # Printing
                print('Recording data : {:2.1f}% - time remaning : {:4.1f}[s]'.format(100*j/trajectory_length_iter, ((time.time()-last_time)*(trajectory_length_iter-j))), end='\r', flush=True)
                last_time = time.time()

                # --- Step 2 : Sample 'Observation' Trajectory with actions from mixture policy (Expert + Student)
                buffer_obs = []

                # Roll the simulation for buffer_size time to generate a single datapoint
                for i in range(buffer_size): 

                    buffer_obs.append(obs)

                    expert_actions  = expert_policy(obs)    # shape (num_envs, 4 + 4 + 8 + 12)
                    student_actions = student_policy(obs)   # shape (num_envs, 4 + 4 + buffer_size*(8 + 12))

                    # extract first action from student policy
                    f = student_actions[:, 0                             : f_len                           ]                  # shape (num_envs, f_len)
                    d = student_actions[:, f_len                         :(f_len+d_len)                    ]                  # shape (num_envs, d_len)
                    p = student_actions[:,(f_len+d_len)                  :(f_len+d_len+pF_param*p_len)     ].reshape(p_shape) # shape (num_envs, 4, 2, buffer_size)
                    F = student_actions[:,(f_len+d_len+pF_param*p_len):(f_len+d_len+pF_param*(p_len+F_len))].reshape(F_shape) # shape (num_envs, 4, 3, buffer_size)
                    student_first_action  = torch.cat((f,d,p[:,:,:,0].flatten(1,-1),F[:,:,:,0].flatten(1,-1)),dim=1) 

                    # if spline parametrisation : first action (ie. theta_0 is at index 1)
                    if typeAction == 'spline':
                        student_first_action  = torch.cat((f,d,p[:,:,:,1].flatten(1,-1),F[:,:,:,1].flatten(1,-1)),dim=1) 

                    aggregate_actions = student_first_action #.clone().detach()
                    aggregate_actions[expert_idx] = expert_actions[expert_idx]

                    obs, rew, dones, extras = env.step(aggregate_actions)

                    epoch_reward += float(torch.sum(rew) / env.num_envs)


                # --- Step 3 : Re-roll these trajectory and query expert 'Action'
                buffer_act = []

                for i in range(len(buffer_obs)):

                    expert_actions  = expert_policy(buffer_obs[i])
                    buffer_act.append(expert_actions) 


                # --- Step 4 : Aggreagate the new expert demonstration to the dataset
                raw_actions = torch.stack(buffer_act).permute(1,2,0)                            # shape (num_envs, act_dim, buffer_size)
                f = raw_actions[:, 0                 : f_len                   , 0]             # shape (num_envs, f_len)
                d = raw_actions[:, f_len             :(f_len+d_len)            , 0]             # shape (num_envs, d_len)
                p = raw_actions[:,(f_len+d_len)      :(f_len+d_len+p_len)      , :].flatten(1,2)# shape (num_envs, buffer_size*p_len) /!\ Transpose to store the data with the right format
                F = raw_actions[:,(f_len+d_len+p_len):(f_len+d_len+p_len+F_len), :].flatten(1,2)# shape (num_envs, buffer_size*F_len)
                
                
                # --- Step 4.5 : If type of action is 'spline' find the spline parameters that correspond to the action
                if typeAction == 'spline':
                    # extract the p and F action with the right parameters
                    p = raw_actions[:,(f_len+d_len)      :(f_len+d_len+p_len)      , :].unsqueeze(2).reshape(args_cli.num_envs, 4, 2, buffer_size) # shape (num_envs, num_legs, 2, buffer_size)
                    F = raw_actions[:,(f_len+d_len+p_len):(f_len+d_len+p_len+F_len), :].unsqueeze(2).reshape(args_cli.num_envs, 4, 3, buffer_size) # shape (num_envs, num_legs, 3, buffer_size)

                    # Fit a cubic spline interpolation these data and retrieve the interpolation parameters
                    p = fit_cubic_with_constraint(y=p).flatten(1,3) # shape (num_envs, num_legs, 2, pF_param) -> ()
                    F = fit_cubic_with_constraint(y=F).flatten(1,3) # shape (num_envs, num_legs, 3, pF_param) -> ()


                process_actions = torch.cat((f,d,p,F),dim=1)                                    # shape (num_envs, f_len+d_len+buffer_size*(p_len+F_len))

                # Concatenate all observations and actions
                observations_data = torch.cat((observations_data, buffer_obs[0]), dim=0)    # shape(num_data, obs_dim)
                actions_data      = torch.cat((actions_data, process_actions), dim=0)       # shape(num_data, f_len+d_len+buffer_size*(p_len+F_len))

                # If the Dataset becomes too large : downsample randomly.
                if observations_data.size(0) > dataset_max_size:
                    indices = torch.randperm(observations_data.size(0))[:dataset_max_size]
                    observations_data = observations_data[indices]
                    actions_data = actions_data[indices]


        # --- Step 5 : Train the student policy on the new dataset, for one iteration
        # Dataset has been updated -> reload it
        train_dataset = ObservationActionDataset(observations_data, actions_data)
        train_loader = DataLoader(train_dataset,**train_kwargs)
        
        # Train the network and update the scheduler
        avg_train_loss = train(student_policy, device, train_loader, optimizer, epoch, train_criterion)
        scheduler.step()

        # Save the training metrics
        epoch_avg_train_loss_list.append(avg_train_loss)
        avg_epoch_reward_list.append(epoch_reward / (trajectory_length_iter*buffer_size) )
        # print('Average Epoch %d Reward : %.2f \n' % (epoch, avg_epoch_reward_list[-1]))
        print('Average Epoch Reward : %.2f' % (avg_epoch_reward_list[-1]))


    # Save the trained model
    torch.save(student_policy.state_dict(),logging_directory + '/' + args_cli.model_name + '.pt')
    print('\nModel saved as : ',logging_directory + '/' + args_cli.model_name + '.pt\n')

    print('\n\n\n----- Saving -----')
    print('\nobservations_data shape ', observations_data.shape[-1])
    print('   actions_data   shape ', actions_data.shape[-1])

    # Save the Generated dataset
    data = {
        'observations': observations_data,
        'actions': actions_data
    }
    torch.save(data, f'{logging_directory}/{file_prefix}.pt') 

    print('\nData succesfully saved as ', file_prefix)
    print('Saved at :',f'{logging_directory}/{file_prefix}.pt')
    print('\nDataset of ',observations_data.shape[0],'datapoints')
    print('Input  size :', observations_data.shape[-1])
    print('Output size :', actions_data.shape[-1],'\n')

    # close the simulator
    env.close()

    plt.figure(1)
    plt.plot(epoch_avg_train_loss_list)
    plt.title('Average Training Loss')
    plt.xlabel('iterations')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(logging_directory, 'average_training_loss.png'))

    plt.figure(2)
    plt.plot(avg_epoch_reward_list)
    plt.title('Average Epoch Reward')
    plt.xlabel('iterations')
    plt.ylabel('Reward')
    plt.savefig(os.path.join(logging_directory, 'average_epoch_reward.png'))

    plt.show()


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
