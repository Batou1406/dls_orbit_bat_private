# Master Thesis - Specific Documentation


This lists the different implementations done during the course of the project. The different implementations are essentially in two folders :
- [source/extension/.../locomotion/model_based](/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/locomotion/model_based) : contains the controller implementation, with the RL part, the environment, the sampling controller, etc. 
- [source/standalone/.../dls_lib](source/standalone/workflows/dls_lib) : contains the executing files that can train/play/evaluate a policy
In the following, the implementations done will be exhaustively listed

## How to run
#### Train a policy
Policies are trained using the library [RSL_RL](https://github.com/leggedrobotics/rsl_rl), which can be shipped with the IsaacLab installation. A policy can be trained with the following line
```
python source/standalone/workflows/rsl_rl/train.py --task Isaac-Model-Based-Speed-Aliengo-v0  --num_envs 4096 --headless
```
the flag `--help` display the different arguments available. The `--task` argument refers to the task for which a policy will be trained for. A task is a gymnasium registered environment and is composed of an environment (with the observation, action, terrain configuration, reward function, etc.) and a learning algorithm (PPO) with its parameters. 

The tasks are registered [here](source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/locomotion/model_based/config/unitree_aliengo/__init__.py) and the following tasks are available :
- `Isaac-Model-Based-Base-Aliengo-v0` associated with the environment [aliengo_base_env_cfg.py](source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/locomotion/model_based/config/unitree_aliengo/aliengo_base_env_cfg.py) and the algorithm [UnitreeAliengoBasePPORunnerCfg](/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/locomotion/model_based/config/unitree_aliengo/agents/rsl_rl_cfg.py#L15C7-L15C37)
- `Isaac-Model-Based-Rough-Aliengo-v0` associated with the environment [aliengo_rough_env_cfg.py](source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/locomotion/model_based/config/unitree_aliengo/aliengo_rough_env_cfg.py) and the algorithm [UnitreeAliengoRoughPPORunnerCfg](source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/locomotion/model_based/config/unitree_aliengo/agents/rsl_rl_cfg.py#L45C7-L45C38) 
- `Isaac-Model-Based-Speed-Aliengo-v0` associated with the environment [aliengo_speed_env_cfg.py](source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/locomotion/model_based/config/unitree_aliengo/aliengo_speed_env_cfg.py) and the algorithm [UnitreeAliengoSpeedPPORunnerCfg](/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/locomotion/model_based/config/unitree_aliengo/agents/rsl_rl_cfg.py#L75) 
- `Isaac-Model-Based-Climb-Aliengo-v0` associated with the environment [aliengo_climb_env_cfg.py](source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/locomotion/model_based/config/unitree_aliengo/aliengo_climb_env_cfg.py) and the algorithm [UnitreeAliengoClimbPPORunnerCfg](/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/locomotion/model_based/config/unitree_aliengo/agents/rsl_rl_cfg.py#L105) 

#### Play a policy 
If the policy has been trained with RSL_RL and is made of an actor-critic network. The policy must be located in `logs/rls_rl/task_name/folder/policy.pt`
```
python source/standalone/workflows/rsl_rl/play.py --task Isaac-Model-Based-Speed-Aliengo-v0  --num_envs 8 --load_run folder --checkpoint policy
```
There are four  baselines available policies for the four tasks
- `python source/standalone/workflows/rsl_rl/play.py --task Isaac-Model-Based-Base-Aliengo-v0  --num_envs 8 --load_run base_RL --checkpoint base_baseline_actor_critic`
- `python source/standalone/workflows/rsl_rl/play.py --task Isaac-Model-Based-Rough-Aliengo-v0  --num_envs 8 --load_run rough_RL --checkpoint rough_baseline_actor_critic`
- `python source/standalone/workflows/rsl_rl/play.py --task Isaac-Model-Based-Speed-Aliengo-v0  --num_envs 8 --load_run speed_RL --checkpoint speed_baseline_actor_critic`
- `python source/standalone/workflows/rsl_rl/play.py --task Isaac-Model-Based-Climb-Aliengo-v0  --num_envs 8 --load_run climb_RL --checkpoint climb_baseline_actor_critic`

#### Run evaluations
A wide series of evaluations can be configured. The evaluation_RL script calls play_eval_RL, which runs the evaluations. Many different policies can be evaluated in a single script
```bash
python3 source/standalone/workflows/dls_lib/evaluation_RL.py
```
#### Train an imitation policy with DAgger
```bash
python3 source/standalone/workflows/dls_lib/train_dagger.py --task Isaac-Model-Based-Climb-Aliengo-v0 --load_run climb_RL --checkpoint climb_baseline_actor_critic.pt --num_envs 4096  --folder-name IL_climb  --epochs 30 --headless
```
#### Live plot
```
python3 live_plot/live_plot_all.py
```
When the sampling controller is running and the [live_plot flag](/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/locomotion/model_based/mdp/actions/model_base_controller.py#L798) is set ( :warning: one flag per controller :warning: ), variables of interest are saved into a [CSV file](/live_variable) and can be displayed live by running these scripts. 
Live plot available are : `live_plot_all.py`, `live_plot_cost.py`, `live_plot_height.py`, `live_plot_GRF.py`
#### Record a video
To record a video trained with RSL_RL (single action - single observation, actor-critic network) (Policy must be in `logs/rsl_rl/folder/policy.pt`)
```bash
python3 source/standalone/workflows/dls_lib/play.py --task Isaac-Model-Based-Rough-Aliengo-v0 --num_envs 64 --load_run base_RL  --checkpoint base_baseline_actor_critic.pt

```
To record a video of other network (DAgger, etc.) (Can be any type of action, just remind to adapt the action space by [specifying the action term]([url](/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/locomotion/model_based/model_based_env_cfg.py#L178)) and [action config](/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/locomotion/model_based/mdp/actions/actions_cfg.py#L27)) (Policy must be in model/folder/policy.pt)
```bash
python3 source/standalone/workflows/dls_lib/play_adaptative_video.py
```

The rotation and camera change can be deactivated by commenting the corresponding lines.

## Folder architecure
[MDP](/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/locomotion/model_based/mdp)
- [actions](/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/locomotion/model_based/mdp/actions) : Folder that contains the action term with f,d,p,F actions
   - [actions_cfg.py](/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/locomotion/model_based/mdp/actions/actions_cfg.py) : Contains the config of the action term and controllers
      - `ModelBaseActionCfg`
      - `OptimizerCfg`
      - `FootTrajectoryCfg`
      - `SwingControllerCfg`
      - `HeightScanCfg`
      - `ActionNormalizationCfg` : parameters for the scaling/normalization of the RL actions (f,d,p,F)
   - [helper.py](/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/locomotion/model_based/mdp/actions/helper.py) : Contains JITted function
      - `inverse_conjugate_euler_xyz_rate_matrix`
      - `rotation_matrix_from_w_to_b`
      - `gait_generator` : compute the phase and contact sequence given `f` and `d`
      - `compute_cubic_spline` : Given spline parameters, compute the action
      - `fit_cubic` : Given a sequence of actions, fit the (least square) spline parameters
      - `compute_discrete` : Given a sequence of discrete actions, return the discrete action
      - `compute_unique` : Given a unique action, return the unique action 
      - `normal_sampling`
      - `uniform_sampling`
      - `enforce_friction_cone_constraints_torch`
      - `from_zero_twopi_to_minuspi_pluspi`
   - [model_based_actions.py](/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/locomotion/model_based/mdp/actions/model_based_actions.py) : Action term - high level control
      - ModelBaseAction
   - [model_based_controller.py](/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/locomotion/model_based/mdp/actions/model_based_controller.py) : Sampling Controller, low-level controllers
      - `baseController` : Class for inheritance - define the basis for any controllers
      - `modelBaseController` : controller for single obs - single act. Used for RL training
      - `samplingController`
      - `samplingTrainer` : Compute just one rollout - To use with RL to have the cost in the reward
      - `SamplingOptimizer` : sampling MPC
      - `SamplingBatchedTrainer` : vectorized sampling MPC - used only with sampling trainer to compute the rollout cost
- [rewards.py](/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/locomotion/model_based/mdp/rewards.py)
   - `penalize_large_leg_frequency_L1`
   - `penalize_large_leg_duty_cycle_L1`
   - `penalize_large_steps_L1`
   - `penalize_large_Forces_L1`
   - `penalize_frequency_variation_L2`
   - `penalize_duty_cycle_variation_L2`
   - `penalize_steps_variation_L2`
   - `penalize_Forces_variation_L2`
   - `friction_constraint`
   - `penalize_foot_in_contact_displacement_l2`
   - `reward_terrain_progress`
   - `penalize_cost_of_transport`
   - `soft_track_lin_vel_xy_exp`
   - `track_proprioceptive_height_exp`
   - `penalize_close_feet`
   - `penalize_foot_trajectory_tracking_error`
   - `penalize_constraint_violation`
   - `penalize_sampling_controller_cost`
- [observations.py](/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/locomotion/model_based/mdp/observations.py)
   - `leg_phase`
   - `leg_contact`
   - `last_model_base_action`
- [terminations.py](/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/locomotion/model_based/mdp/terminations.py)
   - `base_height_bounded`
- [curriculums.py](/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/locomotion/model_based/mdp/curriculums.py)
   - `terrain_levels_vel`
   - `improved_terrain_levels_vel`
   - `climb_terrain_curriculum`
   - `speed_command_levels_walked_distance`
   - `speed_command_levels_tracking_rewards`
   - `speed_command_levels_fast_walked_distance`


---
![Isaac Lab](docs/source/_static/isaaclab.jpg)

---

# Isaac Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Docs status](https://img.shields.io/badge/docs-passing-brightgreen.svg)](https://isaac-sim.github.io/IsaacLab)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)


**Isaac Lab** is a unified and modular framework for robot learning that aims to simplify common workflows
in robotics research (such as RL, learning from demonstrations, and motion planning). It is built upon
[NVIDIA Isaac Sim](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html) to leverage the latest
simulation capabilities for photo-realistic scenes and fast and accurate simulation.

Please refer to our [documentation page](https://isaac-sim.github.io/IsaacLab) to learn more about the
installation steps, features, tutorials, and how to set up your project with Isaac Lab.

## Contributing to Isaac Lab

We wholeheartedly welcome contributions from the community to make this framework mature and useful for everyone.
These may happen as bug reports, feature requests, or code contributions. For details, please check our
[contribution guidelines](https://isaac-sim.github.io/IsaacLab/source/refs/contributing.html).

## Troubleshooting

Please see the [troubleshooting](https://isaac-sim.github.io/IsaacLab/source/refs/troubleshooting.html) section for
common fixes or [submit an issue](https://github.com/isaac-sim/IsaacLab/issues).

For issues related to Isaac Sim, we recommend checking its [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html)
or opening a question on its [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

* Please use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for discussing ideas, asking questions, and requests for new features.
* Github [Issues](https://github.com/isaac-sim/IsaacLab/issues) should only be used to track executable pieces of work with a definite scope and a clear deliverable. These can be fixing bugs, documentation issues, new features, or general updates.

## License

The Isaac Lab framework is released under [BSD-3 License](LICENSE). The license files of its dependencies and assets are present in the [`docs/licenses`](docs/licenses) directory.

## Acknowledgement

Isaac Lab development initiated from the [Orbit](https://isaac-orbit.github.io/) framework. We would appreciate if you would cite it in academic publications as well:

```
@article{mittal2023orbit,
   author={Mittal, Mayank and Yu, Calvin and Yu, Qinxi and Liu, Jingzhou and Rudin, Nikita and Hoeller, David and Yuan, Jia Lin and Singh, Ritvik and Guo, Yunrong and Mazhar, Hammad and Mandlekar, Ajay and Babich, Buck and State, Gavriel and Hutter, Marco and Garg, Animesh},
   journal={IEEE Robotics and Automation Letters},
   title={Orbit: A Unified Simulation Framework for Interactive Robot Learning Environments},
   year={2023},
   volume={8},
   number={6},
   pages={3740-3747},
   doi={10.1109/LRA.2023.3270034}
}
```
