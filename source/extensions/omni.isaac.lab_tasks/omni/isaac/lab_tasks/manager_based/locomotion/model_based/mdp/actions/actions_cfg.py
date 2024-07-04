# Copyright (c) 2022-2024, The lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

from omni.isaac.lab.managers.action_manager import ActionTerm, ActionTermCfg
from omni.isaac.lab.utils import configclass

import torch

from . import model_base_actions

from . import model_base_controller


##
# Model-base Latent space actions.
##

@configclass
class ModelBaseActionCfg(ActionTermCfg):
    """Configuration for the base model base action term.

    See :class:`ModelBaseAction` for more details.
    """

    class_type: type[ActionTerm] = model_base_actions.ModelBaseAction

    debug_vis: bool = False

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""

    height_scan_available: bool = True

    controller: type[model_base_controller.modelBaseController] = MISSING
    """Model base controller that compute u: output torques from z: latent variable
    can be of type 'modelBaseController' or 'samplingController' """ 

    @configclass
    class OptimizerCfg:
        """ Config class for the optimizer """
        
        optimizerType:str = 'sampling'
        """ Different type of optimizer. For now, only 'sampling' is implemented """

        prevision_horizon: int = 5 # 15
        """ Prevision horizon for predictive optimization (in number of time steps) """

        discretization_time: float = 0.02 # 0.04
        """ Duration of a time step in seconds for the predicitve optimization """

        num_samples: int = 10000
        """ Number of samples used if the optimizerType is 'sampling' """

        parametrization_F: str = 'discrete'
        """ Define how F, Ground Reaction Forces, are encoded : can be 'discrete' or 'cubic spline', this modify F_param """

        parametrization_p: str = 'discrete'
        """ Define how p, foot touch down position, are encoded : can be 'discrete' or 'cubic spline', this modify p_param  """

        height_ref: float = 0.38
        """ Height reference for the optimization, defined as mean distance between legs in contact and base """

        optimize_f: bool = False
        """ If enabled, leg frequency will be optimized"""

        optimize_d: bool = False
        """ If enabled, duty cycle will be optimized"""

        optimize_p: bool = False
        """ If enabled, Foot step will be optimized"""

        optimize_F: bool = True
        """ If enabled, Ground Reaction Forces will be optimized"""

        propotion_previous_solution: float = 0.2
        """ Proportion of the previous solution that will be used to generate samples"""

    optimizerCfg: OptimizerCfg | None = None
    """ Must be provided if a controller with optimizer is selected (eg. 'samplingController')"""

    @configclass
    class FootTrajectoryCfg:
        """ Config class for foot trajectory generator hyperparameters
        """
        step_height: float = 0.05
        """ Default step height : used by the swing trajectory generator to determine the apex height in the middle of the trajectory  """

        foot_offset: float = 0.015
        """ Offset between the foot position (as returned in body view by the simulator) and the ground when in contact. """

    footTrajectoryCfg: FootTrajectoryCfg = FootTrajectoryCfg()
    """ Hyperparameter of the foot trajectory generator"""

    @configclass
    class SwingControllerCfg:
        """ Config class for swing foot trajectory controller hyperparameters
        """
        swing_ctrl_pos_gain_fb: float = 5000.0
        """ Position gain feedback for swing trajectory tracking in [0, +inf] """

        swing_ctrl_vel_gain_fb: float = 100.0
        """ Velocity gain feedback for swing trajectory tracking in [0, +inf] """

    swingControllerCfg: SwingControllerCfg = SwingControllerCfg()
    """ Hyperparameter of the swing foot controller"""

    @configclass
    class HeightScanCfg:
        """ Config class for height scan parameter and intermediary variable that speed up computation
        """
        resolution: float = MISSING
        """ Spatial resolution of the height scan in [m]"""

        size: tuple[float, float] = MISSING
        """ Grid size of the height scan (length in x dir. [m], width in y dir. [m]) """

        hip_offset: torch.Tensor = MISSING
        """ XY offset of the robot's hip wrt to the robot's base of shape (1, num_legs, 2=xy)"""

        scale_y: int = MISSING
        """ Index scaling in the flattened grid list for a shift in the y dir. in the xy grid. """

        max_x: int = MISSING
        """ Number of tile in the x dir. of the grid"""

        max_y: int = MISSING
        """ Number of tile in the y dir. of the grid """


    @configclass
    class ActionNormalizationCfg:
        """  Set of parameters for scaling and shifting raw action

        Raw actions are distributed (initially) with mean=0 and std=1 
        """

        # Frequency f : mean=(std_n+std_p)/2, std=(std_p-std_n)/2     : clipped to (min, max)
        std_p_f = 1.7        # [Hz]
        std_n_f = 1.3        # [Hz]
        max_f = 3            # [Hz]
        min_f = 0            # [Hz]

        # Duty Cycle d : mean=(std_n+std_p)/2, std=(std_p-std_n)/2     : clipped to (min, max)
        std_p_d = 0.63       # [2pi Rad]
        std_n_d = 0.57       # [2pi Rad]
        max_d = 1.0          # [2pi Rad]
        min_d = 0.0          # [2pi Rad]

        # Foot touch down position : mean=(std_n+std_p)/2, std=(std_p-std_n)/2     : clipped to (min, max)
        std_p_x_p = +0.03    # [m]
        std_n_x_p = -0.01    # [m] 
        std_p_y_p = +0.01    # [m]
        std_n_y_p = -0.01    # [m]
        max_x_p = +0.36      # [m]
        min_x_p = -0.24      # [m]
        max_y_p = +0.20      # [m]
        min_y_p = -0.20      # [m]

        # Ground Reaction Forces : clipped to (min, max), not clipped if set to None
        std_xy_F = (10 / 2)  # [N]
        max_xy_F = None      # [N]
        min_xy_F = None      # [N]

        mean_z_F = (200 / 2) # [N] : 200/2 ~= 20[kg_aliengo] * 9.81 [m/s²] / 2 [leg in contact]
        std_z_F = mean_z_F/5   # [N]
        max_z_F = None       # [N]
        min_z_F = 0          # [N]


    actionNormalizationCfg = ActionNormalizationCfg()
    """ Hyperparameter for normalizing raw actions (shift and scale) """



