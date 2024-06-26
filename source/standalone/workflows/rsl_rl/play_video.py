# Copyright (c) 2022-2024, The lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
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
import random

from rsl_rl.runners import OnPolicyRunner

# import omni.isaac.contrib_tasks  # noqa: F401
import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_onnx,
)

from omni.isaac.lab.envs import ManagerBasedRLEnv
import math
def move_camera(env: ManagerBasedRLEnv, w: float):
    """ Update default cam eye to rotate arround the target"""

    (pos_x, pos_y, pos_z) = env.viewport_camera_controller.default_cam_eye 

    hypothenuse = (pos_x**2 + pos_y**2)**0.5

    alpha = math.atan2(pos_y, pos_x)

    alpha = alpha + (w*env.step_dt)

    pos_x = hypothenuse*math.cos(alpha)
    pos_y = hypothenuse*math.sin(alpha)

    env.viewport_camera_controller.default_cam_eye = (pos_x, pos_y, pos_z)

def change_camera_target(env: ManagerBasedRLEnv):
    """ Change default cam target and keep the angle between the robot and the camera constant with the update"""

    # Retrieve the robot index
    robot_index = env.viewport_camera_controller.cfg.env_index

    # Retrieve the angle made by the camera and the origin [rad]
    (pos_x, pos_y, pos_z) = env.viewport_camera_controller.default_cam_eye 
    hypothenuse = (pos_x**2 + pos_y**2)**0.5
    alpha_camera = math.atan2(pos_y, pos_x)
    
    # Retrieve the angle made by the robot and the origin [rad]
    alpha_robot = env.scene["robot"].data.heading_w[robot_index]

    # Compute the relative angle
    alpha_relative = (alpha_camera - alpha_robot) % (2*math.pi)

    # sample new robot index
    new_robot_index = random.randint(0, env.num_envs-1)

    # Retrieve the new robot orientation with the origin [rad]
    alpha_new_robot = env.scene["robot"].data.heading_w[new_robot_index]

    # Compute the new angle required by the camera (relative angle + angle new robot with origin) [rad]
    alpha = (alpha_relative + alpha_new_robot) % (2*math.pi)

    # update the camera with the new angle 
    pos_x = hypothenuse*math.cos(alpha)
    pos_y = hypothenuse*math.sin(alpha)
    env.viewport_camera_controller.default_cam_eye = (pos_x, pos_y, pos_z)
    
    # Update the target index of the camera
    env.viewport_camera_controller.cfg.env_index = new_robot_index

def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    video_kwargs = {
        "video_folder": "videos",
        "name_prefix" : args_cli.task,
        "step_trigger": lambda step: step == 81,
        "video_length": 600,
    }
    env = gym.wrappers.RecordVideo(env, **video_kwargs)

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

    # variable for moving target
    update_target = 0
    w = 0.5 # 1.0 # angular speed in [rad/s]

    # reset environment
    obs, _ = env.get_observations()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)

            # Update the camera tracking, camera angular speed in radian per second
            move_camera(env=env.unwrapped, w=w)
            
            update_target += 1
            if update_target == 80:
                update_target = 0
                change_camera_target(env=env.unwrapped)


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
