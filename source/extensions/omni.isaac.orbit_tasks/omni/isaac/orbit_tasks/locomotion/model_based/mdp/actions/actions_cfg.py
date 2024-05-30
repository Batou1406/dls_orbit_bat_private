# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

from omni.isaac.orbit.managers.action_manager import ActionTerm, ActionTermCfg
from omni.isaac.orbit.utils import configclass

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

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""

    scale: float | dict[str, float] = 1.0
    """Scale factor for the action (float or dict of regex expressions). Defaults to 1.0."""
    
    offset: float | dict[str, float] = 0.0
    """Offset factor for the action (float or dict of regex expressions). Defaults to 0.0."""

    prevision_horizon: int = 10
    """Prediction time horizon for the Model Base controller (runs at outer loop frequecy)"""

    number_predict_step: int = 3
    """number of predicted touch down position (used by sampling controller, prior by RL)"""

    optimize_step_height: bool = False

    height_scan_available: bool = True

    # controller: model_base_controller.modelBaseController = MISSING
    controller: type[model_base_controller.samplingController] = MISSING
    """Model base controller that compute u: output torques from z: latent variable""" 

    @configclass
    class FootTrajectoryCfg:
        """ Config class for foot trajectory generator hyperparameters
        """
        step_height: float = 0.05
        """ Default step height : used by the swing trajectory generator to determine the apex height in the middle of the trajectory  """

        foot_offset: float = 0.1
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





