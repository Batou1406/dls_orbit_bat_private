# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from .base_env_window import BaseEnvWindow

import torch

import asyncio
import os
import weakref

import omni.kit.app
import omni.kit.commands
import omni.usd

if TYPE_CHECKING:
    from omni.isaac.lab_tasks.manager_based.locomotion.model_based.mdp.actions.model_base_actions import ModelBaseAction
    from omni.isaac.lab_tasks.manager_based.locomotion.model_based.mdp.actions.model_base_controller import modelBaseController, samplingController, SamplingOptimizer

    import omni.ui

    from ..manager_based_rl_env import ManagerBasedRLEnv


class ManagerBasedRLEnvWindow(BaseEnvWindow):
    """Window manager for the RL environment.

    On top of the basic environment window, this class adds controls for the RL environment.
    This includes visualization of the command manager.
    """

    def __init__(self, env: ManagerBasedRLEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)

        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("commands", self.env.command_manager)


class BatManagerBasedRLEnvWindow(BaseEnvWindow):
    """Window manager for the RL environment.

    On top of the basic environment window, this class adds controls for the RL environment.
    This includes visualization of the command manager.
    """

    def __init__(self, env: ManagerBasedRLEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)

        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:

            # create collapsable frame for Sampling controller
            try : 
                self.modelBaseAction: ModelBaseAction = env.action_manager.get_term('model_base_variable')
                self.velocityCommand = env.command_manager.get_term('base_velocity')
                try:          
                    self.modelBaseAction.controller.samplingOptimizer   
                    self._build_sampler_frame()
                    # self._build_plotting_frame()
                except :
                    print('No \'samplingOptimizer\'provided : sampling controller frame not created ;')
            except : 
                print('No \'model_base_variable\' provided : sampling controller frame not created')


            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:

                    # add command manager visualization
                    self._create_debug_vis_ui_element("commands", self.env.command_manager)

                    # add action debugger
                    self._create_debug_vis_ui_element("actions", self.env.action_manager.get_term("model_base_variable"))



    def _build_sampler_frame(self):
        """Build the Sampling controller related frame for the UI."""
        from omni.kit.window.extensions import SimpleCheckBox

        # create collapsable frame for viewer
        self.ui_window_elements["sampler_frame"] = omni.ui.CollapsableFrame(
            title="Sampling Controller Settings",
            width=omni.ui.Fraction(1),
            height=0,
            collapsed=False,
            style=omni.isaac.ui.ui_utils.get_style(),
            horizontal_scrollbar_policy=omni.ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            vertical_scrollbar_policy=omni.ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
        )
        with self.ui_window_elements["sampler_frame"]:
            # create stack for controls
            self.ui_window_elements["sampler_vstack"] = omni.ui.VStack(spacing=5, height=0)
            with self.ui_window_elements["sampler_vstack"]:

                # create a slider to change the number of samples
                num_samples_cfg = {
                    "label": "number of samples",
                    "type": "button",
                    "default_val": self.modelBaseAction.cfg.optimizerCfg.num_samples,
                    "min": 2,
                    "max": 20000,
                    "tooltip": "Number of samples used by the sampling optimizer",
                }
                self.ui_window_elements["num samples"] = omni.isaac.ui.ui_utils.int_builder(**num_samples_cfg)
                self.ui_window_elements["num samples"].add_value_changed_fn(self._update_num_samples)

                # Create a slider to change the proportion of the samples sampled from previous best sample
                prop_best_cfg = {
                    "label": "Prop. of best solution",
                    "type": "button",
                    "default_val": self.modelBaseAction.cfg.optimizerCfg.propotion_previous_solution,
                    "min": 0.0,
                    "max": 1.0,
                    "tooltip": "Proportion of the samples sampled from previous best sample",
                }
                self.ui_window_elements["proportion Best"] = omni.isaac.ui.ui_utils.float_builder(**prop_best_cfg)
                self.ui_window_elements["proportion Best"].add_value_changed_fn(self._update_proportion_best)

                # create a slider to change the number iteration of the optimizer
                num_iter_cfg = {
                    "label": "number of iterations",
                    "type": "button",
                    "default_val": self.modelBaseAction.cfg.optimizerCfg.num_optimizer_iterations,
                    "min": 1,
                    "max": 5,
                    "tooltip": "Number of time the sampling optiizer will iterate",
                }
                self.ui_window_elements["num iter"] = omni.isaac.ui.ui_utils.int_builder(**num_iter_cfg)
                self.ui_window_elements["num iter"].add_value_changed_fn(self._update_num_iter)

                # Create a slider to change the robot mass
                robot_mass_cfg = {
                    "label": "Robot Mass",
                    "type": "button",
                    "default_val": self.modelBaseAction.robot_mass,
                    "min": 10.0,
                    "max": 30.0,
                    "tooltip": "Proportion of the samples sampled from previous best sample",
                }
                self.ui_window_elements["robot mass"] = omni.isaac.ui.ui_utils.float_builder(**robot_mass_cfg)
                self.ui_window_elements["robot mass"].add_value_changed_fn(self._update_probot_mass)

                # Create a button to enable or not the optimizer
                with omni.ui.HStack():
                    omni.ui.Label(
                        "Enable Optimization",
                        width=omni.isaac.ui.ui_utils.LABEL_WIDTH - 12,
                        alignment=omni.ui.Alignment.LEFT_CENTER,
                        tooltip="Wether to enable or not the optimization - If not enabled, the actions are directly the actions provided by the network",
                    )
                    self.ui_window_elements["Enable Optimization"] = SimpleCheckBox(
                        model=omni.ui.SimpleBoolModel(),
                        enabled=True,
                        checked=True,
                        on_checked_fn=self._toggle_opt,
                    )
                    omni.isaac.ui.ui_utils.add_line_rect_flourish()

                # create a Dropdown menu to select a debug gait or not
                debug_gait_follow_cfg = {
                    "label": "Debug Gait",
                    "type": "dropdown",
                    "default_val": 0,
                    "items": [name.replace("_", " ").title() for name in ['None', 'full_stance', 'trot']], # TODO hardcoded for now... Do better
                    "tooltip": "Disable RL action (f,d,p) and apply a static gait",
                    "on_clicked_fn": self._set_debug_gait,
                }
                self.ui_window_elements["debug_gait"] = omni.isaac.ui.ui_utils.dropdown_builder(**debug_gait_follow_cfg)

                # add viewer default eye and lookat locations
                self.ui_window_elements["velocity_setter"] = omni.isaac.ui.ui_utils.xyz_builder(
                    label="Velocity Command",
                    tooltip="Modify the Forward, lateral and angular velocity command.",
                    step=0.01,
                    min=-1.0,
                    max=1.0,
                    on_value_changed_fn=[self._set_velocity_fn] * 3,
                )

                # Create a button to enable leg frequency optimization
                with omni.ui.HStack():
                    omni.ui.Label(
                        "Frequency Optimization",
                        width=omni.isaac.ui.ui_utils.LABEL_WIDTH - 12,
                        alignment=omni.ui.Alignment.LEFT_CENTER,
                        tooltip="Wether to enable the leg frequency optimization or not",
                    )
                    self.ui_window_elements["Frequency Optimization"] = SimpleCheckBox(
                        model=omni.ui.SimpleBoolModel(),
                        enabled=True,
                        checked=self.modelBaseAction.cfg.optimizerCfg.optimize_f,
                        on_checked_fn=self._toggle_f_opt,
                    )
                    omni.isaac.ui.ui_utils.add_line_rect_flourish()

                # Create a button to enable leg duty cycle optimization
                with omni.ui.HStack():
                    omni.ui.Label(
                        "Duty Cyle Optimization",
                        width=omni.isaac.ui.ui_utils.LABEL_WIDTH - 12,
                        alignment=omni.ui.Alignment.LEFT_CENTER,
                        tooltip="Wether to enable the leg duty cycle optimization or not",
                    )
                    self.ui_window_elements["Duty Cylce Optimization"] = SimpleCheckBox(
                        model=omni.ui.SimpleBoolModel(),
                        enabled=True,
                        checked=self.modelBaseAction.cfg.optimizerCfg.optimize_d,
                        on_checked_fn=self._toggle_d_opt,
                    )
                    omni.isaac.ui.ui_utils.add_line_rect_flourish()

                # Create a button to enable foot touch down optimization
                with omni.ui.HStack():
                    omni.ui.Label(
                        "Foot step Optimization",
                        width=omni.isaac.ui.ui_utils.LABEL_WIDTH - 12,
                        alignment=omni.ui.Alignment.LEFT_CENTER,
                        tooltip="Wether to enable the foot touch down optimization or not",
                    )
                    self.ui_window_elements["Foot step Optimization"] = SimpleCheckBox(
                        model=omni.ui.SimpleBoolModel(),
                        enabled=True,
                        checked=self.modelBaseAction.cfg.optimizerCfg.optimize_p,
                        on_checked_fn=self._toggle_p_opt,
                    )
                    omni.isaac.ui.ui_utils.add_line_rect_flourish()

                # Create a button to enable ground reaction forces optimization
                with omni.ui.HStack():
                    omni.ui.Label(
                        "Force Optimization",
                        width=omni.isaac.ui.ui_utils.LABEL_WIDTH - 12,
                        alignment=omni.ui.Alignment.LEFT_CENTER,
                        tooltip="Wether to enable the ground reaction forces optimization or not",
                    )
                    self.ui_window_elements["Force Optimization"] = SimpleCheckBox(
                        model=omni.ui.SimpleBoolModel(),
                        enabled=True,
                        checked=self.modelBaseAction.cfg.optimizerCfg.optimize_F,
                        on_checked_fn=self._toggle_F_opt,
                    )
                    omni.isaac.ui.ui_utils.add_line_rect_flourish()

                # create a Dropdown menu to select the sampling law
                sampling_law_cfg = {
                    "label": "Sampling Law",
                    "type": "dropdown",
                    "default_val": {'normal':0, 'uniform':1}[self.modelBaseAction.cfg.optimizerCfg.sampling_law],
                    "items": [name.replace("_", " ").title() for name in ['normal', 'uniform']], # TODO hardcoded for now... Do better
                    "tooltip": "The sampling to generate samples from",
                    "on_clicked_fn": self._set_sampling_law,
                }
                self.ui_window_elements["sampling_law"] = omni.isaac.ui.ui_utils.dropdown_builder(**sampling_law_cfg)

                # Create a button to enable or not sample clipping
                with omni.ui.HStack():
                    omni.ui.Label(
                        "Sample Clipping",
                        width=omni.isaac.ui.ui_utils.LABEL_WIDTH - 12,
                        alignment=omni.ui.Alignment.LEFT_CENTER,
                        tooltip="Wether to clip the sample to 2*std when the sampling is with an unbounded law",
                    )
                    self.ui_window_elements["Sample Clipping"] = SimpleCheckBox(
                        model=omni.ui.SimpleBoolModel(),
                        enabled=True,
                        checked=self.modelBaseAction.cfg.optimizerCfg.clip_sample,
                        on_checked_fn=self._toggle_clip_sample,
                    )
                    omni.isaac.ui.ui_utils.add_line_rect_flourish()                

                # Create a slider to chnage the leg frequency standard variation in the sampling law
                f_std_cfg = {
                    "label": "f std [Hz]",
                    "type": "button",
                    "default_val": float(self.modelBaseAction.controller.samplingOptimizer.std_f),
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Set the standard deviation of the leg frequency in the sampling law",
                }
                self.ui_window_elements["f std"] = omni.isaac.ui.ui_utils.float_builder(**f_std_cfg)
                self.ui_window_elements["f std"].add_value_changed_fn(self._update_f_std)

                # Create a slider to chnage the leg duty cycle standard variation in the sampling law
                d_std_cfg = {
                    "label": "d std",
                    "type": "button",
                    "default_val": float(self.modelBaseAction.controller.samplingOptimizer.std_d),
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Set the standard deviation of the leg duty cycle in the sampling law",
                }
                self.ui_window_elements["d std"] = omni.isaac.ui.ui_utils.float_builder(**d_std_cfg)
                self.ui_window_elements["d std"].add_value_changed_fn(self._update_d_std)

                # Create a slider to chnage the foot touch down position standard variation in the sampling law
                p_std_cfg = {
                    "label": "p std [m]",
                    "type": "button",
                    "default_val": float(self.modelBaseAction.controller.samplingOptimizer.std_p),
                    "min": 0.0,
                    "max": 0.1,
                    "step": 0.005,
                    "tooltip": "Set the standard deviation of the foot step in the sampling law",
                }
                self.ui_window_elements["p std"] = omni.isaac.ui.ui_utils.float_builder(**p_std_cfg)
                self.ui_window_elements["p std"].add_value_changed_fn(self._update_p_std)

                # Create a slider to chnage the ground reaction forces standard variation in the sampling law
                F_std_cfg = {
                    "label": "F std [N]",
                    "type": "button",
                    "default_val": float(self.modelBaseAction.controller.samplingOptimizer.std_F),
                    "min": 0.0,
                    "max": 30.0,
                    "tooltip": "Set the standard deviation of the GRF in the sampling law",
                }
                self.ui_window_elements["F std"] = omni.isaac.ui.ui_utils.float_builder(**F_std_cfg)
                self.ui_window_elements["F std"].add_value_changed_fn(self._update_F_std)


    def _build_plotting_frame(self):
        """Build the plotting controller related frame for the UI."""

        # create collapsable frame for viewer
        self.ui_window_elements["plotting_frame"] = omni.ui.CollapsableFrame(
            title="Sampling Controller Settings",
            width=omni.ui.Fraction(1),
            height=0,
            collapsed=False,
            style=omni.isaac.ui.ui_utils.get_style(),
            horizontal_scrollbar_policy=omni.ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            vertical_scrollbar_policy=omni.ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
        )
        with self.ui_window_elements["plotting_frame"]:
            # create stack for controls
            self.ui_window_elements["plotting_vstack"] = omni.ui.VStack(spacing=5, height=0)
            with self.ui_window_elements["plotting_vstack"]:

                plot_cfg = {
                    "label": "frequency",
                    "data": self.modelBaseAction.controller.samplingOptimizer.best_frequency_list,
                    "min": 0.0,
                    "max": 3.0,
                }
                self.ui_window_elements["plot"] = omni.isaac.ui.ui_utils.plot_builder(**plot_cfg)


    def _toggle_f_opt(self, value: bool):
        self.modelBaseAction.controller.samplingOptimizer.optimize_f = value

    def _toggle_d_opt(self, value: bool):
        self.modelBaseAction.controller.samplingOptimizer.optimize_d = value

    def _toggle_p_opt(self, value: bool):
        self.modelBaseAction.controller.samplingOptimizer.optimize_p = value

    def _toggle_F_opt(self, value: bool):
        self.modelBaseAction.controller.samplingOptimizer.optimize_F = value

    def _set_sampling_law(self, value: str):
        # value is modified and comes with a capital letter first
        if   value == 'Normal' : self.modelBaseAction.controller.samplingOptimizer.sampling_law = self.modelBaseAction.controller.samplingOptimizer.normal_sampling
        elif value == 'Uniform': self.modelBaseAction.controller.samplingOptimizer.sampling_law = self.modelBaseAction.controller.samplingOptimizer.uniform_sampling

    def _toggle_clip_sample(self, value: bool):
        self.modelBaseAction.controller.samplingOptimizer.clip_sample = value

    def _toggle_opt(self, value: bool):
        from omni.kit.window.extensions import SimpleCheckBox

        self.modelBaseAction.controller.optimizer_active = value

        # with self.ui_window_elements["sampler_frame"]:
        #     with self.ui_window_elements["sampler_vstack"]:
        #         with omni.ui.HStack():
        #             # self.ui_window_elements["Frequency Optimization"].enabled  = value
        #             # self.ui_window_elements["Duty Cylce Optimization"].enabled = value
        #             # self.ui_window_elements["Foot step Optimization"].enabled  = value
        #             # self.ui_window_elements["Force Optimization"].enabled      = value
        #             # self.ui_window_elements["Frequency Optimization"]  = SimpleCheckBox(model=omni.ui.SimpleBoolModel(),enabled=value,checked=self.modelBaseAction.cfg.optimizerCfg.optimize_f,on_checked_fn=self._toggle_f_opt,)
        #             # self.ui_window_elements["Duty Cylce Optimization"] = SimpleCheckBox(model=omni.ui.SimpleBoolModel(),enabled=value,checked=self.modelBaseAction.cfg.optimizerCfg.optimize_d,on_checked_fn=self._toggle_d_opt,)
        #             # self.ui_window_elements["Foot step Optimization"]  = SimpleCheckBox(model=omni.ui.SimpleBoolModel(),enabled=value,checked=self.modelBaseAction.cfg.optimizerCfg.optimize_p,on_checked_fn=self._toggle_p_opt,)
        #             # self.ui_window_elements["Force Optimization"]      = SimpleCheckBox(model=omni.ui.SimpleBoolModel(),enabled=value,checked=self.modelBaseAction.cfg.optimizerCfg.optimize_F,on_checked_fn=self._toggle_F_opt,)

    def _update_num_samples(self, model: omni.ui.SimpleIntModel):
        self.modelBaseAction.controller.samplingOptimizer.num_samples = model.as_int

    def _update_proportion_best(self, model: omni.ui.SimpleFloatModel):
        self.modelBaseAction.controller.samplingOptimizer.propotion_previous_solution = model.as_float

    def _update_probot_mass(self, model: omni.ui.SimpleFloatModel):
        self.modelBaseAction.controller.samplingOptimizer.robot_mass = model.as_float

    def _update_f_std(self, model: omni.ui.SimpleFloatModel):
        self.modelBaseAction.controller.samplingOptimizer.std_f = (model.as_float)*torch.ones_like(self.modelBaseAction.controller.samplingOptimizer.std_f)

    def _update_d_std(self, model: omni.ui.SimpleFloatModel):
        self.modelBaseAction.controller.samplingOptimizer.std_d = (model.as_float)*torch.ones_like(self.modelBaseAction.controller.samplingOptimizer.std_d)

    def _update_p_std(self, model: omni.ui.SimpleFloatModel):
        self.modelBaseAction.controller.samplingOptimizer.std_p = (model.as_float)*torch.ones_like(self.modelBaseAction.controller.samplingOptimizer.std_p)

    def _update_F_std(self, model: omni.ui.SimpleFloatModel):
        self.modelBaseAction.controller.samplingOptimizer.std_F = (model.as_float)*torch.ones_like(self.modelBaseAction.controller.samplingOptimizer.std_F)

    def _update_num_iter(self, model: omni.ui.SimpleIntModel):
        self.modelBaseAction.controller.samplingOptimizer.num_optimizer_iterations = model.as_int

    def _set_debug_gait(self, value: str): 
        # value is modified and comes with a capital letter first
        if   value == 'None'        : self.modelBaseAction.debug_apply_action_status = None
        if   value == 'Full Stance' : self.modelBaseAction.debug_apply_action_status = 'full_stance'
        if   value == 'Trot'        : self.modelBaseAction.debug_apply_action_status = 'trot'

    def _set_velocity_fn(self, model: omni.ui.SimpleFloatModel):
        # Set the velocity sampling ranges
        self.velocityCommand.cfg.ranges.for_vel_b = [self.ui_window_elements["velocity_setter"][0].get_value_as_float(), self.ui_window_elements["velocity_setter"][0].get_value_as_float()]
        self.velocityCommand.cfg.ranges.lat_vel_b = [self.ui_window_elements["velocity_setter"][1].get_value_as_float(), self.ui_window_elements["velocity_setter"][1].get_value_as_float()]
        self.velocityCommand.cfg.ranges.ang_vel_b = [self.ui_window_elements["velocity_setter"][2].get_value_as_float(), self.ui_window_elements["velocity_setter"][2].get_value_as_float()]
        self.velocityCommand.cfg.ranges.initial_heading_err = [0.0, 0.0]

        # Reset to resample a new command
        self.velocityCommand.reset(env_ids=range(self.env.num_envs))

        
