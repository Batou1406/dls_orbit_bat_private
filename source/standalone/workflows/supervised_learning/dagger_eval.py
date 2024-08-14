# Copyright (c) 2022-2024, The LAB Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""
import argparse
from omni.isaac.lab.app import AppLauncher
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--disable_fabric", action="store_true", default=False,  help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs",     type=int,   default=256,               help="Number of environments to simulate.")
parser.add_argument("--task",         type=str,   default=None,              help="Name of the task.")
parser.add_argument("--seed",         type=int,   default=None,              help="Seed used for the environment")
parser.add_argument('--epochs',       type=int,   default=30,  metavar='N',  help='number of epochs to train (default: 14)')
parser.add_argument('--batch-size',   type=int,   default=2048,  metavar='N',  help='input batch size for training (default: 64)')
parser.add_argument('--lr',           type=float, default=1.0, metavar='LR', help='learning rate (default: 1.0)')
parser.add_argument('--gamma',        type=float, default=0.7, metavar='M',  help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--folder-name',  type=str,   default='DAggerEvaluation',help="Name of the folder to save the trained model in 'model/task/folder-name'")

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
)
import time

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import matplotlib
# matplotlib.use('GTK4Agg')
matplotlib.use('Agg') # not interactive, used to be run in headless
import matplotlib.pyplot as plt
import json
import numpy as np

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
    

def model_to_dict(model):
    model_dict = {
        "model_name": model.__class__.__name__,
        "layers": []
    }

    for name, module in model.named_modules():
        if name == '':
            continue
        layer_info = {
            "name": name,
            "type": module.__class__.__name__,
            "parameters": {}
        }
        for param_name, param in module.named_parameters(recurse=False):
            layer_info["parameters"][param_name] = param.size()
        model_dict["layers"].append(layer_info)

    return model_dict

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


def test(model, device, logging_directory, buffer_size, max_buffer_size, frequency_reduction, p_typeAction, F_typeAction, p_param, F_param):
    f_len, d_len, p_len, F_len = 4, 4, 8, 12

    model.eval()
    with torch.no_grad():
        test_dataset  = DatasetFromFile(f'{logging_directory}/testing_data.pt')
        test_observations, test_actions = test_dataset()

        # --- Step 1 : extract the test expert actions
        f_test = test_actions[:, 0                                   : f_len                                                       ]                                    # shape (test_points, f_len)
        d_test = test_actions[:, f_len                               :(f_len+d_len)                                                ]                                    # shape (test_points, d_len)
        p_test = test_actions[:,(f_len+d_len)                        :(f_len+d_len+(max_buffer_size*p_len))                        ].reshape(-1, 4, 2, max_buffer_size) # shape (test_points, 4, 2, buffer_size)
        F_test = test_actions[:,(f_len+d_len+(max_buffer_size*p_len)):(f_len+d_len+(max_buffer_size*p_len)+(max_buffer_size*F_len))].reshape(-1, 4, 3, max_buffer_size) # shape (test_points, 4, 3, buffer_size)

        # reshape the actions to match the current frequency_reduction
        p_test_discrete = p_test[:,:,:,::frequency_reduction][:,:,:,:buffer_size] # Discrete exact actions
        F_test_discrete = F_test[:,:,:,::frequency_reduction][:,:,:,:buffer_size]

        f_test_encoded = f_test
        d_test_encoded = d_test

        # reconstruct the encoded actions
        if p_typeAction == 'discrete':
            p_test_encoded = p_test_discrete
        if p_typeAction == 'spline':
            p_test_encoded = fit_cubic(y=p_test_discrete) # shape (test_points, num_legs, 2, 4)
        if p_typeAction == 'first':
            p_test_encoded = p_test_discrete[:,:,:,0].unsqueeze(-1)

        # double not possible, because c is not available
        if F_typeAction == 'discrete':
            F_test_encoded = F_test_discrete
        if F_typeAction == 'spline':
            F_test_encoded = fit_cubic(y=F_test_discrete) # shape (test_points, num_legs, 3, 4)

        test_encoded_action = torch.cat((f_test_encoded, d_test_encoded, p_test_encoded.flatten(1,-1), F_test_encoded.flatten(1,-1)),dim=1)


        # --- Step 2 : Extract the student actions
        stud_encoded_action = model(test_observations)

        f_stud_encoded = stud_encoded_action[:, 0                           : f_len                                       ]                                    # shape (test_points, f_len)
        d_stud_encoded = stud_encoded_action[:, f_len                       :(f_len+d_len)                                ]                                    # shape (test_points, d_len)
        p_stud_encoded = stud_encoded_action[:,(f_len+d_len)                :(f_len+d_len+(p_param*p_len))                ].reshape(-1, 4, 2, p_param) # shape (test_points, 4, 2, p_param)
        F_stud_encoded = stud_encoded_action[:,(f_len+d_len+(p_param*p_len)):(f_len+d_len+(p_param*p_len)+(F_param*F_len))].reshape(-1, 4, 3, F_param) # shape (test_points, 4, 3, F_param)


        # --- Step 3 : Compute MSE
        mse_f   = float(torch.mean(torch.square(f_test_encoded - f_stud_encoded)))
        mse_d   = float(torch.mean(torch.square(d_test_encoded - d_stud_encoded)))
        mse_p   = float(torch.mean(torch.square(p_test_encoded - p_stud_encoded)))
        mse_F   = float(torch.mean(torch.square(F_test_encoded - F_stud_encoded)))
        mse_tot = float(torch.mean(torch.square(test_encoded_action - stud_encoded_action)))

        return mse_f, mse_d, mse_p, mse_F, mse_tot


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
    
class DatasetFromFile(Dataset):
    """Cutsom dataloader to load generated dataset
    Args :
        file_path : The file path to the dataset"""
    def __init__(self, file_path):
        data = torch.load(file_path)
        self.observations = data['observations']
        self.actions = data['actions']

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]
    
    def __call__(self) : 
        return self.observations, self.actions


""" --- Decay Function --- """
def alpha(epoch, type='indicator', param=[0,1,2]):
    """ Decay function for the dagger Alogirthm"""

    # Indicator Function
    if type == 'indicator':
        if epoch in param:
            alpha = 1
        else :
            alpha = 0

    # eponential decay
    if type == 'exp':
        alpha = param**(epoch)

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
    theta = torch.cat((theta_n1.unsqueeze(-1)/10.0, theta_0.unsqueeze(-1),theta_1.unsqueeze(-1), theta_2.unsqueeze(-1)/10.0), dim=-1) # shape (batch_size, num_legs, dim_3D, 4)
    
    return theta  # Coefficients a, b, c, d for each batch, legs, dim_3D


def fit_cubic(y: torch.tensor, x: torch.tensor | None = None) -> torch.tensor:
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

    # Construct the design matrix X for the remaining points

    # This formulation gives a,b,c,d
    # X = torch.stack([x**3, x**2, x, torch.ones_like(x)], dim=-1)  # shape: (horizon, num_param)

    # This formulation directly gives theta, not a,b,c,d
    X = 0.5*torch.stack([-(x**3) + 2*(x**2) - x, 3*(x**3) - 5*(x**2) + 2, -3*(x**3) + 4*(x**2) + x, (x**3) - (x**2)], dim=-1)  # shape: (horizon, num_param
    num_param = X.shape[-1] # = 4 ie. a,b,c, d
    
    # Construct the target matrix Y 
    Y = y  # shape: (batch_size, num_legs, dim_3D, horizon)


    # Compute the normal equations components for the remaining points
    XtX = torch.einsum('nk,nm->km', X, X)      # (horizon-1, num_param) x (horizon, num_param) -> shape: (num_param, num_param)
    XtY = torch.einsum('nk,bijn->bijk', X, Y)  # (horizon-1, num_param) x (batch_size, num_legs, dim_3D, horizon) -> shape: (batch_size, num_legs, dim_3D, num_param)

    # Expand XtX to match the dimension of XtY for the solver
    XtX.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(batch_size, num_legs, dim_3D, num_param, num_param) # shape (batch_size, num_legs, dim_3D, num_param, num_param)

    # Solve for the remaining coefficients a, b, c
    beta = torch.linalg.solve(XtX, XtY.transpose(-1,-2)).transpose(-1,-2) # shape: (batch_size, num_legs, dim_3D, num_param)

    # Retrieve coefficients a, b, c, d for each batch, legs, dim_3D
    a = beta[..., 0]/10.0    # shape (batch_size, num_legs, dim_3D)
    b = beta[..., 1]    # shape (batch_size, num_legs, dim_3D)
    c = beta[..., 2]    # shape (batch_size, num_legs, dim_3D)
    d = beta[..., 3]/10.0    # shape (batch_size, num_legs, dim_3D)

    # Find the missing theta parameters
    # theta_n1 =   a +   b -   c + d  # shape (batch_size, num_legs, dim_3D)
    # theta_0  =                   d  # shape (batch_size, num_legs, dim_3D)
    # theta_1  =   a +   b +   c + d  # shape (batch_size, num_legs, dim_3D)
    # theta_2  = 6*a + 4*b + 2*c + d  # shape (batch_size, num_legs, dim_3D)

    # t = torch.arange(0, 101, device=y.device)
    # F = torch.empty((y.shape[0], y.shape[1], y.shape[2], 101), device=y.device)
    # for i in range(101):
    #     F[:,:,:,i] = compute_cubic_spline(parameters=torch.cat((a.unsqueeze(-1), b.unsqueeze(-1),c.unsqueeze(-1), d.unsqueeze(-1)), dim=-1), step=int(t[i]), horizon=100)

    # plt.plot(t.cpu().numpy()/100,F[102,1,-1,:].cpu().numpy())
    # plt.scatter(x=x.cpu().numpy(), y=y[102,1,-1,:].cpu().numpy(), c='red')
    # plt.scatter(x=torch.tensor([[-1, 0, 1, 2]]), y=torch.cat((a.unsqueeze(-1), b.unsqueeze(-1),c.unsqueeze(-1), d.unsqueeze(-1)), dim=-1)[102,1,-1,:].cpu().numpy())

    # param = fit_cubic_with_constraint(y=y)
    # plt.scatter(x=torch.tensor([[-1, 0, 1, 2]]), y=param[102,1,-1,:].cpu().numpy(), c='green')

    # plt.show()

    # Concatenate the coefficient
    # theta = torch.cat((theta_n1.unsqueeze(-1), theta_0.unsqueeze(-1),theta_1.unsqueeze(-1), theta_2.unsqueeze(-1)), dim=-1) # shape (batch_size, num_legs, dim_3D, 4)
    theta = torch.cat((a.unsqueeze(-1), b.unsqueeze(-1),c.unsqueeze(-1), d.unsqueeze(-1)), dim=-1) # shape (batch_size, num_legs, dim_3D, 4)

    return theta  # Coefficients a, b, c, d for each batch, legs, dim_3D


@torch.jit.script
def compute_cubic_spline(parameters: torch.Tensor, step: int, horizon: int):
    """ Given a set of spline parameters, and the point in the trajectory return the function value 
    
    Args :
        parameters (Tensor): Spline action parameter      of shape(batch, num_legs, 3, spline_param)              
        step          (int): The point in the curve in [0, horizon]
        horizon       (int): The length of the curve
        
    Returns : 
        actions    (Tensor): Discrete action              of shape(batch, num_legs, 3)
    """
    # Find the point in the curve q in [0,1]
    tau = step/(horizon)        
    q = (tau - 0.0)/(1.0-0.0)
    
    # Compute the spline interpolation parameters
    a =  2*q*q*q - 3*q*q     + 1
    b =    q*q*q - 2*q*q + q
    c = -2*q*q*q + 3*q*q
    d =    q*q*q -   q*q

    # Compute intermediary parameters 
    phi_1 = 0.5*(parameters[...,2]  - (10*parameters[...,0])) # shape (batch, num_legs, 3)
    phi_2 = 0.5*((10*parameters[...,3])  - parameters[...,1]) # shape (batch, num_legs, 3)

    # Compute the spline
    actions = a*parameters[...,1] + b*phi_1 + c*parameters[...,2]  + d*phi_2 # shape (batch, num_legs, 3)

    return actions


""" --- Return the first two touch down in p --- """
def get_touchdowns(c, p):
    n, h = c.shape
    
    # Initialize p2 with the last values of p as default
    p2 = p[:, -1:].repeat(1, 2)
    
    # Find transitions from 0 to 1 in c
    transitions = (c[:, 1:] == 1) & (c[:, :-1] == 0)
    
    for i in range(n):
        indices = torch.where(transitions[i])[0]
        if len(indices) > 0:
            if len(indices) == 1:
                p2[i, 0] = p[i, indices[0]]
            else:
                p2[i, 0] = p[i, indices[0]]
                p2[i, 1] = p[i, indices[1]]
    
    return p2


""" --- DAgger Trainer --- """
def DAgger_Train(env, expert_policy, student_policy, scheduler, optimizer, train_criterion, device, tot_epoch, logging_directory, experiment_directory, datapoints_generated_per_iter, buffer_size, max_buffer_size,  frequency_reduction,
                 p_typeAction, F_typeAction, p_param, F_param, p_shape, F_shape, dataset_max_size, modelDict, test_iter, activation_fuction, mini_batch_size):

    observations_data = torch.empty(0,device=device)
    actions_data = torch.empty(0, device=device)
    debug_counter=10
    f_len, d_len, p_len, F_len = 4, 4, 8, 12
    epoch_avg_train_loss_list = []
    avg_epoch_reward_list = []
    epoch_mse_test_loss = []
    last_time_outloop = time.time()
    last_time = time.time()
    results = {}

    with torch.inference_mode():
        obs, _ = env.get_observations()



    # Before training, evaluate the uninitalised policy
    print('\n---- Evaluating the policy before training ----')
    epoch_reward_test = 0.0
    epoch_reward_stud = 0.0
    with torch.inference_mode():
        obs, _ = env.reset()
        for i in range(test_iter):
            print(f'Runing : {100*i/test_iter}%', end='\r', flush=True)

            expert_actions  = expert_policy(obs)    # shape (num_envs, 4 + 4 + 8 + 12)
            student_actions = student_policy(obs)   # shape (num_envs, 4 + 4 + buffer_size*(8 + 12))

            # extract first action from student policy
            f = student_actions[:, 0                           : f_len                           ]                  # shape (num_envs, f_len)
            d = student_actions[:, f_len                       :(f_len+d_len)                    ]                  # shape (num_envs, d_len)
            p = student_actions[:,(f_len+d_len)                :(f_len+d_len+(p_param*p_len))     ].reshape(p_shape) # shape (num_envs, 4, 2, buffer_size)
            F = student_actions[:,(f_len+d_len+(p_param*p_len)):(f_len+d_len+(p_param*p_len)+(F_param*F_len))].reshape(F_shape) # shape (num_envs, 4, 3, buffer_size)
            
            if p_typeAction == 'discrete': p = p[:,:,:,0].flatten(1,-1)
            if p_typeAction == 'first':    p = p[:,:,:,0].flatten(1,-1)
            if p_typeAction == 'spline':   p = compute_cubic_spline(parameters=p, step=0, horizon=1).flatten(1,-1)
            if p_typeAction == 'double' :  p = p[:,:,:,0].flatten(1,-1)

            if F_typeAction == 'discrete': F = F[:,:,:,0].flatten(1,-1)
            if F_typeAction == 'spline':   F = compute_cubic_spline(parameters=F, step=0, horizon=1).flatten(1,-1)

            student_first_action  = torch.cat((f,d,p,F),dim=1) 
            aggregate_actions = student_first_action
            aggregate_actions[int(env.num_envs/2):] = expert_actions[int(env.num_envs/2):]

            obs, rew, dones, extras = env.step(aggregate_actions)
            
            epoch_reward_stud += float(torch.sum(rew[:int(env.num_envs/2)] / (env.num_envs/2)))
            epoch_reward_test += float(torch.sum(rew[int(env.num_envs/2):] / (env.num_envs/2)))

    Epoch_Reward = {'epoch_reward_untrained_stud': epoch_reward_stud, 'epoch_reward_test': epoch_reward_test}
    results['untrained_stud_Epoch_Reward'] = Epoch_Reward


    # Number of simulations iteration required to have the desired trajectory length
    # trajectory_length_iter = int(trajectory_length_s / (frequency_reduction*buffer_size*env.unwrapped.step_dt))
    num_iter_to_gen_required_datapoints = int(datapoints_generated_per_iter / args_cli.num_envs)

    for epoch in range(tot_epoch):
        print(f'\n----- Epoch {epoch+1} / {tot_epoch} ----- Total Remaining Time {(tot_epoch-epoch)*(time.time()-last_time_outloop):4.1f}[s]')
        last_time_outloop = time.time()

        # --- Step 1 : Get ID for student actions
        n = min(int(alpha(epoch, type=activation_fuction['type'], param=activation_fuction['param'])*args_cli.num_envs), args_cli.num_envs)
        random_indx = torch.randperm(args_cli.num_envs)
        # expert_idx = torch.randperm(args_cli.num_envs)[:n]
        expert_idx = random_indx[:n]
        student_idx = random_indx[n:]

        # If we want to plot splines
        # if (epoch == 0) or (epoch == 20) or (epoch == 10) or (epoch == 30) :
        #     debug_counter=0


        with torch.inference_mode(): # step 2 and 3 (and 4 but not required) in inference mode to avoid gradient computations
            epoch_reward = 0.0
            # Roll the simulation for trajectory_length_s time
            # for j in range(trajectory_length_iter):
            for j in range(num_iter_to_gen_required_datapoints):
                # Printing
                print('Recording data : {:2.1f}% - time remaning : {:4.1f}[s]'.format(100*j/num_iter_to_gen_required_datapoints, ((time.time()-last_time)*(num_iter_to_gen_required_datapoints-j))), end='\r', flush=True)
                last_time = time.time()

                # --- Step 2 : Sample 'Observation' Trajectory with actions from mixture policy (Expert + Student)
                buffer_obs = []
                buffer_c = []

                # Step the simulation buffer_size*frequency_reduction time to generate a single (num_envs) datapoint (obs_0, act_0, ..., act_buffer_size)
                for i in range(frequency_reduction*buffer_size): 
                    
                    if i%frequency_reduction == 0 : 
                        buffer_obs.append(obs)

                    expert_actions  = expert_policy(obs)    # shape (num_envs, 4 + 4 + 8 + 12)
                    student_actions = student_policy(obs)   # shape (num_envs, 4 + 4 + buffer_size*(8 + 12))

                    # extract first action from student policy
                    f = student_actions[:, 0                             : f_len                           ]                  # shape (num_envs, f_len)
                    d = student_actions[:, f_len                         :(f_len+d_len)                    ]                  # shape (num_envs, d_len)
                    p = student_actions[:,(f_len+d_len)                  :(f_len+d_len+(p_param*p_len))     ].reshape(p_shape) # shape (num_envs, 4, 2, buffer_size)
                    F = student_actions[:,(f_len+d_len+(p_param*p_len)):(f_len+d_len+(p_param*p_len)+(F_param*F_len))].reshape(F_shape) # shape (num_envs, 4, 3, buffer_size)
                    
                    if p_typeAction == 'discrete':
                        p = p[:,:,:,0].flatten(1,-1)
                    if p_typeAction == 'spline':
                        p = compute_cubic_spline(parameters=p, step=0, horizon=1).flatten(1,-1)
                    if p_typeAction == 'double' : 
                        p = p[:,:,:,0].flatten(1,-1)
                    if p_typeAction == 'first' : 
                        p = p[:,:,:,0].flatten(1,-1)

                    if F_typeAction == 'discrete':
                        F = F[:,:,:,0].flatten(1,-1)
                    if F_typeAction == 'spline':
                        F = compute_cubic_spline(parameters=F, step=0, horizon=1).flatten(1,-1)

                    student_first_action  = torch.cat((f,d,p,F),dim=1) 

                    aggregate_actions = student_first_action #.clone().detach()
                    aggregate_actions[expert_idx] = expert_actions[expert_idx]

                    obs, rew, dones, extras = env.step(aggregate_actions)

                    # epoch_reward += float(torch.sum(rew) / env.num_envs)
                    epoch_reward += float(torch.sum(rew[student_idx]) / min(len(student_idx), 1))

                    # If necessary, retrieve the contact sequence
                    if (p_typeAction) == 'double' and (i%frequency_reduction == 0):
                        buffer_c.append(env.unwrapped.action_manager.get_term('model_base_variable').c0_star.unsqueeze(-1).expand(args_cli.num_envs, 4, 2))    #(batch_size, num_legs, 1->2)


                # --- Step 3 : Re-roll these trajectory and query expert 'Action'
                buffer_act = []

                for i in range(len(buffer_obs)):

                    expert_actions  = expert_policy(buffer_obs[i])
                    buffer_act.append(expert_actions) 


                # --- Step 4 : Aggreagate the new expert demonstration to the dataset
                raw_actions = torch.stack(buffer_act,dim=2)                            # shape (num_envs, act_dim, buffer_size)
                f = raw_actions[:, 0                 : f_len                   , 0]             # shape (num_envs, f_len)
                d = raw_actions[:, f_len             :(f_len+d_len)            , 0]             # shape (num_envs, d_len)
                p = raw_actions[:,(f_len+d_len)      :(f_len+d_len+p_len)      , :].flatten(1,2)# shape (num_envs, buffer_size*p_len) /!\ Transpose to store the data with the right format
                F = raw_actions[:,(f_len+d_len+p_len):(f_len+d_len+p_len+F_len), :].flatten(1,2)# shape (num_envs, buffer_size*F_len)
                
                
                # --- Step 4.5 : If type of action is 'spline' find the spline parameters that correspond to the action
                if p_typeAction == 'spline':
                    # extract the p and F action with the right parameters
                    p = raw_actions[:,(f_len+d_len)      :(f_len+d_len+p_len)      , :].unsqueeze(2).reshape(args_cli.num_envs, 4, 2, buffer_size) # shape (num_envs, num_legs, 2, buffer_size)
                    # Fit a cubic spline interpolation these data and retrieve the interpolation parameters
                    p = fit_cubic(y=p).flatten(1,3) # shape (num_envs, num_legs, 2, p_param) -> ()

                if p_typeAction == 'double':
                    c = torch.stack(buffer_c, dim=3)                                                                                        # shape (num_envs, num_legs, 2, buffer_size)
                    p_raw = raw_actions[:,(f_len+d_len):(f_len+d_len+p_len), :].unsqueeze(2).reshape(args_cli.num_envs, 4, 2, buffer_size)  # shape (num_envs, num_legs, 2, buffer_size)
                    p = get_touchdowns(c=c.reshape(-1, buffer_size), p=p_raw.reshape(-1, buffer_size)).reshape(args_cli.num_envs, 4, 2, 2).flatten(1,3) # shape (num_envs*num_legs*2, buffer_size)
                    
                if p_typeAction == 'first':
                    p = raw_actions[:,(f_len+d_len)      :(f_len+d_len+p_len)      , 0]# shape (num_envs, p_len)

                if F_typeAction == 'spline':
                    # extract the p and F action with the right parameters
                    F = raw_actions[:,(f_len+d_len+p_len):(f_len+d_len+p_len+F_len), :].unsqueeze(2).reshape(args_cli.num_envs, 4, 3, buffer_size) # shape (num_envs, num_legs, 3, buffer_size)
                    F_expert_raw = F.clone().detach()
                    # Fit a cubic spline interpolation these data and retrieve the interpolation parameters
                    F = fit_cubic(y=F).flatten(1,3) # shape (num_envs, num_legs, 3, F_param) -> ()


                process_actions = torch.cat((f,d,p,F),dim=1)                                    # shape (num_envs, f_len+d_len+buffer_size*(p_len+F_len))

                # Concatenate all observations and actions
                observations_data = torch.cat((observations_data, buffer_obs[0]), dim=0)    # shape(num_data, obs_dim)
                actions_data      = torch.cat((actions_data, process_actions), dim=0)       # shape(num_data, f_len+d_len+buffer_size*(p_len+F_len))

                # Plot Spline
                if debug_counter < 5:
                    debug_counter+=1

                    env_idx = 10
                    leg_idx = 1
                    x_discrete = torch.linspace(0, 1, steps=buffer_size, device=F.device) # Discrete x = [0, 0.25, 0.5, 0.75, 1.0]
                    x_param = torch.tensor([[-1, 0, 1, 2]])
                    t = torch.arange(0, 101, device=F.device)
                    
                    # plot expert action
                    plt.scatter(x=x_discrete.cpu().numpy(), y=F_expert_raw[env_idx,leg_idx,-1,:].cpu().numpy(), c='red')

                    # plot expert param
                    F_expert_param = fit_cubic_with_constraint(y=F_expert_raw)
                    plt.scatter(x=x_param.cpu().numpy(), y=F_expert_param[env_idx,leg_idx,-1,:].cpu().numpy(), c='red')

                    # plot expert trajectory
                    F_expert_traj = torch.empty((F_expert_param.shape[0], F_expert_param.shape[1], F_expert_param.shape[2], 101), device=F_expert_param.device)
                    for i in range(101):
                        F_expert_traj[:,:,:,i] = compute_cubic_spline(parameters=F_expert_param, step=int(t[i]), horizon=100)
                    plt.plot(t.cpu().numpy()/100,F_expert_traj[env_idx,leg_idx,-1,:].cpu().numpy(), c='red')

                    # plot student param
                    student_actions = student_policy(buffer_obs[0]) # shape (num_envs, 4 + 4 + buffer_size*(8 + 12))
                    F_student_param = student_actions[:,(f_len+d_len+(p_param*p_len)):(f_len+d_len+(p_param*p_len)+(F_param*F_len))].reshape(F_shape) # shape (num_envs, 4, 3, buffer_size)
                    plt.scatter(x=x_param.cpu().numpy(), y=F_student_param[env_idx,leg_idx,-1,:].cpu().numpy(), c='blue')

                    # plot student trajectory
                    F_student_traj = torch.empty((F_student_param.shape[0], F_student_param.shape[1], F_student_param.shape[2], 101), device=F_student_param.device)
                    for i in range(101):
                        F_student_traj[:,:,:,i] = compute_cubic_spline(parameters=F_student_param, step=int(t[i]), horizon=100)
                    plt.plot(t.cpu().numpy()/100,F_student_traj[env_idx,leg_idx,-1,:].cpu().numpy(), c='blue')

                    plt.show()


                # If the Dataset becomes too large : downsample randomly.
                if observations_data.size(0) > dataset_max_size:
                    indices = torch.randperm(observations_data.size(0))[:dataset_max_size]
                    observations_data = observations_data[indices]
                    actions_data = actions_data[indices]


        # --- Step 5 : Train the student policy on the new dataset, for one iteration
        # Set training and testing arguments
        train_kwargs = {'batch_size': mini_batch_size}

        # Dataset has been updated -> reload it
        train_dataset = ObservationActionDataset(observations_data, actions_data)
        train_loader = DataLoader(train_dataset,**train_kwargs)
        
        # Train the network and update the scheduler
        avg_train_loss = train(student_policy, device, train_loader, optimizer, epoch, train_criterion)

        # Test the network and the test set (return mse_f, mse_d, mse_p, mse_F and mse_tot)
        mse_test_loss = test(student_policy, device, logging_directory, buffer_size, max_buffer_size, frequency_reduction, p_typeAction, F_typeAction, p_param, F_param)

        # Step the trainer parameters
        scheduler.step()

        # Save the training metrics
        epoch_avg_train_loss_list.append(float(avg_train_loss))
        epoch_mse_test_loss.append(mse_test_loss)
        avg_epoch_reward_list.append(float(epoch_reward / (num_iter_to_gen_required_datapoints*buffer_size) ))
        print(f"Average Test Loss {mse_test_loss[-1]:.4f}")
        print('Average Epoch Reward : %.2f' % (avg_epoch_reward_list[-1]))


    # Save the trained model
    torch.save(student_policy.state_dict(),experiment_directory + '/' + 'model.pt')
    print('\nModel saved as : ',experiment_directory + '/' + 'model.pt\n')
    print('\n\n\n----- Saving -----')
    print('\nobservations_data shape ', observations_data.shape[-1])
    print('   actions_data   shape ', actions_data.shape[-1])

    print('\nDataset of ',observations_data.shape[0],'datapoints')
    print('Input  size :', observations_data.shape[-1])
    print('Output size :', actions_data.shape[-1],'\n')

    # Save the model info as a JSON file
    json_data = {}
    json_data['p_typeAction'] = p_typeAction
    json_data['F_typeAction'] = F_typeAction
    json_data['num_datapoints'] = observations_data.shape[0]
    json_data['Input_size'] = observations_data.shape[-1]
    json_data['Output_size'] = actions_data.shape[-1]
    json_data['prediction_horizon_step'] = buffer_size
    json_data['prediction_horizon_time'] = f"{env.unwrapped.step_dt*frequency_reduction}[s]"
    json_data['num_envs'] = env.num_envs
    json_data['Activation function'] = activation_fuction
    json_data['minibatach size'] = mini_batch_size
    # json_data['trajectory_length_s'] = trajectory_length_s
    json_data['datapoints_generated_per_iter'] = datapoints_generated_per_iter
    json_data['tot_epoch'] = tot_epoch
    json_data['dataset_max_size'] = dataset_max_size
    json_data['p_typeAction'] = p_typeAction
    json_data['p_param'] = p_param
    json_data['F_typeAction'] = F_typeAction
    json_data['F_param'] = F_param
    json_data['model'] = modelDict
    with open(f'{experiment_directory}/info.json', 'w') as file:
        json.dump(json_data, file, indent=4)
    

    # Plot the training results
    if True :
        plt.figure(1, figsize=(15, 10)).clf()
        plt.plot(epoch_avg_train_loss_list)
        plt.title('Average Training Loss')
        plt.xlabel('iterations')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(experiment_directory, 'average_training_loss.pdf'), bbox_inches='tight')
        plt.figure(2, figsize=(15, 10)).clf()
        plt.plot(avg_epoch_reward_list)
        plt.title('Average Epoch Reward')
        plt.xlabel('iterations')
        plt.ylabel('Reward')
        plt.savefig(os.path.join(experiment_directory, 'average_epoch_reward.pdf'), bbox_inches='tight')
        plt.figure(3, figsize=(15, 10)).clf()
        plt.plot([mse_tot[-1] for mse_tot in epoch_mse_test_loss])
        plt.title('Average Testing Loss')
        plt.xlabel('iterations')
        plt.ylabel('Reward')
        plt.savefig(os.path.join(experiment_directory, 'average_testing_loss.pdf'), bbox_inches='tight')

        # These figure aren't reseted, thus cumulative line would be drawn
        label = f'H{buffer_size} dt{env.unwrapped.step_dt*frequency_reduction}[s] - ({p_typeAction},{F_typeAction})'
        plt.figure(4, figsize=(15, 10))
        plt.plot(epoch_avg_train_loss_list, label=label)
        plt.title('Average Training Loss')
        plt.xlabel('iterations')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.savefig(os.path.join(logging_directory, 'average_training_loss.pdf'), bbox_inches='tight')
        plt.figure(5, figsize=(15, 10))
        plt.plot(avg_epoch_reward_list, label=label)
        plt.title('Average Epoch Reward')
        plt.xlabel('iterations')
        plt.ylabel('Reward')
        plt.legend()
        plt.savefig(os.path.join(logging_directory, 'average_epoch_reward.pdf'), bbox_inches='tight')
        plt.figure(6, figsize=(15, 10))
        plt.plot([mse_tot[-1] for mse_tot in epoch_mse_test_loss], label=label)
        plt.title('Average Testing Loss')
        plt.xlabel('iterations')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.savefig(os.path.join(logging_directory, 'average_testing_loss.pdf'), bbox_inches='tight')

    # plt.show()
    np.savetxt(f'{experiment_directory}/average_training_loss.csv', epoch_avg_train_loss_list, delimiter=',', fmt='%.6f')
    np.savetxt(f'{experiment_directory}/average_epoch_reward.csv', avg_epoch_reward_list, delimiter=',', fmt='%.6f')
    np.savetxt(f'{experiment_directory}/epoch_mse_test_loss.csv', epoch_mse_test_loss, delimiter=',', fmt='%.6f')



    # Run the evaluation : 
    # 1. Reward on a trajectory : Expert vs student
    # 2. MSE : On reconstructed actions : expert vs student
    # 3. MSE : On dataset feating : Same as 2. for not encoded actions
    # 4. MSE : On first action 

    # Run a trajectory
    epoch_reward_test = 0.0
    epoch_reward_stud = 0.0
    print('\n---- Evaluating the policy After training ----')
    with torch.inference_mode():
        obs, _ = env.reset()

        for i in range(test_iter):
            print(f'Runing : {100*i/test_iter}%', end='\r', flush=True)
            expert_actions  = expert_policy(obs)    # shape (num_envs, 4 + 4 + 8 + 12)
            student_actions = student_policy(obs)   # shape (num_envs, 4 + 4 + buffer_size*(8 + 12))

            # extract first action from student policy
            f = student_actions[:, 0                           : f_len                           ]                  # shape (num_envs, f_len)
            d = student_actions[:, f_len                       :(f_len+d_len)                    ]                  # shape (num_envs, d_len)
            p = student_actions[:,(f_len+d_len)                :(f_len+d_len+(p_param*p_len))     ].reshape(p_shape) # shape (num_envs, 4, 2, buffer_size)
            F = student_actions[:,(f_len+d_len+(p_param*p_len)):(f_len+d_len+(p_param*p_len)+(F_param*F_len))].reshape(F_shape) # shape (num_envs, 4, 3, buffer_size)
            
            if p_typeAction == 'discrete':
                p = p[:,:,:,0].flatten(1,-1)
            if p_typeAction == 'first':
                p = p[:,:,:,0].flatten(1,-1)
            if p_typeAction == 'spline':
                p = compute_cubic_spline(parameters=p, step=0, horizon=1).flatten(1,-1)
            if p_typeAction == 'double' : 
                p = p[:,:,:,0].flatten(1,-1)

            if F_typeAction == 'discrete':
                F = F[:,:,:,0].flatten(1,-1)
            if F_typeAction == 'spline':
                F = compute_cubic_spline(parameters=F, step=0, horizon=1).flatten(1,-1)

            student_first_action  = torch.cat((f,d,p,F),dim=1) 

            aggregate_actions = student_first_action
            aggregate_actions[int(env.num_envs/2):] = expert_actions[int(env.num_envs/2):]

            obs, rew, dones, extras = env.step(aggregate_actions)
            
            epoch_reward_stud += float(torch.sum(rew[:int(env.num_envs/2)] / (env.num_envs/2)))
            epoch_reward_test += float(torch.sum(rew[int(env.num_envs/2):] / (env.num_envs/2)))

    Epoch_Reward = {'epoch_reward_stud': epoch_reward_stud, 'epoch_reward_test': epoch_reward_test}
    results['Epoch_Reward'] = Epoch_Reward

    # --- Step 1 : Load the test set
    test_dataset  = DatasetFromFile(f'{logging_directory}/testing_data.pt')

    # --- Step 2 : Reconstruct the test expert actions
    test_observations, test_actions = test_dataset()

    # extract the test expert actions
    f_test = test_actions[:, 0                                   : f_len                                                       ]                                    # shape (test_points, f_len)
    d_test = test_actions[:, f_len                               :(f_len+d_len)                                                ]                                    # shape (test_points, d_len)
    p_test = test_actions[:,(f_len+d_len)                        :(f_len+d_len+(max_buffer_size*p_len))                        ].reshape(-1, 4, 2, max_buffer_size) # shape (test_points, 4, 2, buffer_size)
    F_test = test_actions[:,(f_len+d_len+(max_buffer_size*p_len)):(f_len+d_len+(max_buffer_size*p_len)+(max_buffer_size*F_len))].reshape(-1, 4, 3, max_buffer_size) # shape (test_points, 4, 3, buffer_size)

    # reshape the actions to match the current frequency_reduction
    p_test_discrete = p_test[:,:,:,::frequency_reduction][:,:,:,:buffer_size] # Discrete exact actions
    F_test_discrete = F_test[:,:,:,::frequency_reduction][:,:,:,:buffer_size]

    # reconstruct the encoded actions
    p_test_encoded = p_test_discrete
    F_test_encoded = F_test_discrete
    if p_typeAction == 'spline':
        p_test_encoded = fit_cubic(y=p_test_discrete) # shape (test_points, num_legs, 2, 4)
    if p_typeAction == 'first':
        p_test_encoded = p_test_encoded[:,:,:,0].unsqueeze(-1)

    # double not possible, because c is not available
        
    if F_typeAction == 'spline':
        F_test_encoded = fit_cubic(y=F_test_discrete) # shape (test_points, num_legs, 3, 4)
    
    # --- Step 3 : Compute MSE
    student_actions = student_policy(test_observations)

    f_stud = student_actions[:, 0                           : f_len                                       ]                                    # shape (test_points, f_len)
    d_stud = student_actions[:, f_len                       :(f_len+d_len)                                ]                                    # shape (test_points, d_len)
    p_stud = student_actions[:,(f_len+d_len)                :(f_len+d_len+(p_param*p_len))                ].reshape(-1, 4, 2, p_param) # shape (test_points, 4, 2, p_param)
    F_stud = student_actions[:,(f_len+d_len+(p_param*p_len)):(f_len+d_len+(p_param*p_len)+(F_param*F_len))].reshape(-1, 4, 3, F_param) # shape (test_points, 4, 3, F_param)

    p_stud_encoded = p_stud
    F_stud_encoded = F_stud

    # Reconstruct the discrete actions
    if p_typeAction == 'discrete':
        p_stud_discrete = p_stud_encoded
    if p_typeAction == 'first':
        p_stud_discrete = p_stud_encoded.expand_as(p_test_discrete)
    if p_typeAction == 'spline':
        p_stud_discrete = torch.empty_like(p_test_discrete)
        for i in range(buffer_size):
            p_stud_discrete[:,:,:,i] = compute_cubic_spline(parameters=p_stud_encoded, step=i, horizon=buffer_size) # shape (test_points, num_legs, 2, buffer_size)

    # double not possible, because c is not available
        
    if F_typeAction == 'discrete':
        F_stud_discrete = F_stud_encoded
    if F_typeAction == 'spline':
        F_stud_discrete = torch.empty_like(F_test_discrete)
        for i in range(buffer_size):
            F_stud_discrete[:,:,:,i] = compute_cubic_spline(parameters=F_stud_encoded, step=i, horizon=buffer_size) # shape (test_points, num_legs, 2, buffer_size)



    # --- Step 4 : Compute MSE

    # 1. MSE : On first action 
    mse_first_action = {}
    mse_first_action['f'] = float(torch.mean(torch.square(f_test - f_stud)))
    mse_first_action['d'] = float(torch.mean(torch.square(d_test - d_stud)))
    mse_first_action['p'] = float(torch.mean(torch.square(p_test_discrete[:,:,:,0] - p_stud_discrete[:,:,:,0])))
    mse_first_action['F'] = float(torch.mean(torch.square(F_test_discrete[:,:,:,0] - F_stud_discrete[:,:,:,0])))
    
    test_first_action = torch.cat((f_test, d_test, p_test_discrete[:,:,:,0].flatten(1,-1), F_test_discrete[:,:,:,0].flatten(1,-1)),dim=1)
    stud_first_action = torch.cat((f_stud, d_stud, p_stud_discrete[:,:,:,0].flatten(1,-1), F_stud_discrete[:,:,:,0].flatten(1,-1)),dim=1)
    mse_first_action['Total'] = float(torch.mean(torch.square(test_first_action - stud_first_action)))

    results['mse_first_action'] = mse_first_action


    # 2. MSE : On all discrete action
    mse_discrete_action = {}
    mse_discrete_action['f'] = float(torch.mean(torch.square(f_test - f_stud)))
    mse_discrete_action['d'] = float(torch.mean(torch.square(d_test - d_stud)))
    mse_discrete_action['p'] = float(torch.mean(torch.square(p_test_discrete - p_stud_discrete)))
    mse_discrete_action['F'] = float(torch.mean(torch.square(F_test_discrete - F_stud_discrete)))
    
    test_discrete_action = torch.cat((f_test, d_test, p_test_discrete.flatten(1,-1), F_test_discrete.flatten(1,-1)),dim=1)
    stud_discrete_action = torch.cat((f_stud, d_stud, p_stud_discrete.flatten(1,-1), F_stud_discrete.flatten(1,-1)),dim=1)
    mse_discrete_action['Total'] = float(torch.mean(torch.square(test_discrete_action - stud_discrete_action)))

    results['mse_discrete_action'] = mse_discrete_action

    # 3. MSE : On encoded actions
    mse_encoded_action = {}
    mse_encoded_action['f'] = float(torch.mean(torch.square(f_test - f_stud)))
    mse_encoded_action['d'] = float(torch.mean(torch.square(d_test - d_stud)))
    mse_encoded_action['p'] = float(torch.mean(torch.square(p_test_encoded - p_stud_encoded)))
    mse_encoded_action['F'] = float(torch.mean(torch.square(F_test_encoded - F_stud_encoded)))
    
    test_encoded_action = torch.cat((f_test, d_test, p_test_encoded.flatten(1,-1), F_test_encoded.flatten(1,-1)),dim=1)
    stud_encoded_action = torch.cat((f_stud, d_stud, p_stud_encoded.flatten(1,-1), F_stud_encoded.flatten(1,-1)),dim=1)
    mse_encoded_action['Total'] = float(torch.mean(torch.square(test_encoded_action - stud_encoded_action)))

    results['mse_encoded_action'] = mse_encoded_action

    with open(f'{experiment_directory}/results.json', 'w') as file:
        json.dump(results, file, indent=4)
    
    return


""" --- Record a Test Set --- """
def Record_test_set(env, expert_policy, device, test_set_size, max_buffer_size, logging_directory):


    observations_data = torch.empty(0,device=device)
    actions_data = torch.empty(0, device=device)
    f_len, d_len, p_len, F_len = 4, 4, 8, 12

    print('\n---- Recording Test Set ----')


    with torch.inference_mode():
        obs, _ = env.reset()

        while (len(observations_data) < test_set_size):
            print(f'Recording : {100*len(observations_data)/test_set_size}%', end='\r', flush=True)

            buffer_obs = []
            buffer_act = []

            # Roll the simultaion to query enough data points
            for i in range(max_buffer_size):
                # Record the observation
                buffer_obs.append(obs)

                # Query the expert for the action
                expert_actions  = expert_policy(obs)

                # Step the envirionment
                obs, rew, dones, extras = env.step(expert_actions)

                # Save the expert action
                buffer_act.append(expert_actions) 
            
            # Reconstruct the actions correctly
            raw_actions = torch.stack(buffer_act,dim=2)                            # shape (num_envs, act_dim, buffer_size)
            f = raw_actions[:, 0                 : f_len                   , 0]             # shape (num_envs, f_len)
            d = raw_actions[:, f_len             :(f_len+d_len)            , 0]             # shape (num_envs, d_len)
            p = raw_actions[:,(f_len+d_len)      :(f_len+d_len+p_len)      , :].flatten(1,2)# shape (num_envs, buffer_size*p_len) /!\ Transpose to store the data with the right format
            F = raw_actions[:,(f_len+d_len+p_len):(f_len+d_len+p_len+F_len), :].flatten(1,2)# shape (num_envs, buffer_size*F_len)
            process_actions = torch.cat((f,d,p,F),dim=1)                                    # shape (num_envs, f_len+d_len+buffer_size*(p_len+F_len))

            # Concatenate all observations and actions
            observations_data = torch.cat((observations_data, buffer_obs[0]), dim=0)    # shape(num_data, obs_dim)
            actions_data      = torch.cat((actions_data, process_actions), dim=0)       # shape(num_data, f_len+d_len+buffer_size*(p_len+F_len))


        # Downsample to keep the correct dataset size
        if observations_data.size(0) > test_set_size:
            indices = torch.randperm(observations_data.size(0))[:test_set_size]
            observations_data = observations_data[indices]
            actions_data = actions_data[indices]

    # Save the Generated dataset
    data = {
        'observations': observations_data,
        'actions': actions_data
    }
    torch.save(data, f'{logging_directory}/testing_data.pt') 


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



    # --- Step 2 : Define training Variables
    # Mini batch size for the gradient descent step 
    # mini_batch_size_list = [64, 128, 256, 512, 1024]
    mini_batch_size_list = [64]

    # Activation function for Dagger
    # activation_function_list = [{'type':'indicator', 'param':[0]}, {'type':'indicator', 'param':[0,1,2]}, {'type':'exp', 'param':0.6}, {'type':'exp', 'param':0.9}, {'type':'exp', 'param':0.4}]
    activation_function_list = [{'type':'exp', 'param':0.6}]#,  {'type':'indicator', 'param':[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}]

    # Buffer size : number of prediction horizon for the student policy
    buffer_size_list = [5, 10, 15]
    # buffer_size_list = [5]

    # Factor of the simulation frequency at which the dataset will be recorded
    frequency_reduction_list = [1, 2]
    # frequency_reduction_list = [1]

    # The encoding of the actions
    action_encoding_list = [('discrete', 'discrete'), ('discrete', 'spline'), ('spline', 'discrete'), ('spline', 'spline'), ('first', 'discrete'), ('first', 'spline')] 
    # action_encoding_list = [('first', 'spline'), ('first', 'discrete')]#, ('discrete', 'discrete'), ('discrete', 'spline')] 
    # action_encoding_list = [('discrete', 'discrete'), ('first', 'spline')]


    # Trajectory length that are recorded between epoch
    # trajectory_length_s = 10 # [s]
    # trajectory_length_s = 5 # [s]

    # Number of epoch
    tot_epoch = args_cli.epochs

    # Dataset maximum size before clipping
    dataset_max_size =  600000 # 300000 # [datapoints] 800000 too much for GPU and horizon=15

    datapoints_generated_per_iter = int(0.10 * dataset_max_size)
    # datapoints_generated_per_iter = 4*args_cli.num_envs

    test_set_size = 50000 # [datapoints]

    test_iter = 50*15



    # --- Step 3 : Define Helper variables    
    f_len, d_len, p_len, F_len = 4, 4, 8, 12

    # Set seeds
    # torch.manual_seed(args_cli.seed)

    # Set device
    use_cuda = not args_cli.cpu and torch.cuda.is_available()
    if use_cuda:  device = torch.device("cuda")
    else:         device = torch.device("cpu")

    # Create logging directory if necessary
    logging_directory = f'model/{args_cli.task}/{args_cli.folder_name}'
    if not os.path.exists(logging_directory):
        os.makedirs(logging_directory)
    else :
        raise KeyError('There is already an experiment setup in this directory, Please provide another folder_name')


    # Save the model info as a JSON file
    json_data = {}
    json_data['num_envs'] = env.num_envs
    # json_data['trajectory_length_s'] = trajectory_length_s
    json_data['datapoints_generated_per_iter'] = datapoints_generated_per_iter
    json_data['tot_epoch'] = tot_epoch
    json_data['dataset_max_size'] = dataset_max_size
    with open(f'{logging_directory}/info.json', 'w') as file:
        json.dump(json_data, file, indent=4)


    # --- Step 4 : Record a test set
    max_buffer_size = max(buffer_size_list)*max(frequency_reduction_list)

    Record_test_set(env, expert_policy, device, test_set_size, max_buffer_size, logging_directory)


    # --- Step 5 : Create the config for the Experiment
    experiment_idx = 0
    for mini_batch_size in mini_batch_size_list : 
        for activation_fuction in activation_function_list :
            for frequency_reduction in frequency_reduction_list :
                for buffer_size in buffer_size_list :
                    for p_typeAction, F_typeAction in action_encoding_list : 

                        experiment_idx += 1

                        # Type of action recorded
                        if F_typeAction == 'spline':
                            F_param = 4
                        if F_typeAction == 'discrete':
                            F_param = buffer_size
                        if p_typeAction == 'spline':
                            p_param = 4
                        if p_typeAction == 'discrete':
                            p_param = buffer_size
                        if p_typeAction == 'double':
                            p_param = 2
                        if p_typeAction == 'first':
                            p_param = 1
                        p_shape = (env.num_envs, 4, 2, p_param)
                        F_shape = (env.num_envs, 4, 3, F_param)

                        experiment_directory = f'model/{args_cli.task}/{args_cli.folder_name}/experiment{experiment_idx}'
                        if not os.path.exists(experiment_directory):
                            os.makedirs(experiment_directory)

                        # Get New Observations and reset the Environment
                        with torch.inference_mode():
                            obs, _ = env.reset()


                        #  Define Model criteria : model, optimizer and loss criterion and scheduler
                        input_size = obs.shape[-1]
                        output_size = 8 + (p_param*p_len) + (F_param*F_len)

                        student_policy  = Model(input_size, output_size).to(device) #Model(input_size, output_size).to(device)
                        modelDict = model_to_dict(student_policy)

                        optimizer       = optim.Adadelta(student_policy.parameters(), lr=args_cli.lr)
                        train_criterion = nn.MSELoss() 
                        scheduler       = StepLR(optimizer, step_size=1, gamma=args_cli.gamma) # for adadelta

                            # Printing
                        if True : 
                            print('\n---------------------------------------------------------------------')
                            print('\n----- Datalogger Configuration -----\n')

                            print(f"\nSimulation runs with time step {env.unwrapped.physics_dt} [s], at frequency {1/env.unwrapped.physics_dt} [Hz]")
                            print(f"Policy runs with time step {env.unwrapped.step_dt} [s], at frequency {1/env.unwrapped.step_dt} [Hz]")
                            print(f"Dataset will be recorded with time step {env.unwrapped.step_dt*frequency_reduction} [s], at frequency {1/(frequency_reduction*env.unwrapped.step_dt)} [Hz]")
                            print(f"Which will correspond in a prediction horizon of {buffer_size*env.unwrapped.step_dt*frequency_reduction} [s]")

                            print(f"\nType of p action recorded: {p_typeAction}")
                            print(f"Type of F action recorded: {F_typeAction}")
                            print(f"with N = {buffer_size} prediction horizon")

                            print(f"\nMini Batch size {mini_batch_size}")
                            print(f"Policy selector function {activation_fuction['type']}, with parameters {activation_fuction['param']}")

                            print('\nModel Input  size :', obs.shape[-1])
                            print('Model Output size :', output_size,'\n')

                            print('\n----- Simulation -----')

                            experiment_dict = {}
                            experiment_dict['buffer_size'] = buffer_size
                            experiment_dict['frequency'] = f'{50/frequency_reduction} [Hz]'
                            experiment_dict['p_typeAction'] = p_typeAction
                            experiment_dict['F_typeAction'] = F_typeAction
                            experiment_dict['mini batch size'] = mini_batch_size
                            experiment_dict['activation function'] = activation_fuction
                            json_data[f'experiment{experiment_idx}'] = experiment_dict
                            with open(f'{logging_directory}/info.json', 'w') as file:
                                json.dump(json_data, file, indent=4)


                        # --- Step 6 : Run the experiment
                        # Train a policy
                        DAgger_Train(env,
                                    expert_policy,
                                    student_policy, 
                                    scheduler, 
                                    optimizer, 
                                    train_criterion,
                                    device, 
                                    tot_epoch, 
                                    logging_directory,
                                    experiment_directory, 
                                    #  trajectory_length_s, 
                                    datapoints_generated_per_iter, 
                                    buffer_size,
                                    max_buffer_size, 
                                    frequency_reduction,
                                    p_typeAction, F_typeAction, 
                                    p_param, F_param, 
                                    p_shape, F_shape, 
                                    dataset_max_size,  
                                    modelDict,
                                    test_iter,
                                    activation_fuction,
                                    mini_batch_size)
    

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
