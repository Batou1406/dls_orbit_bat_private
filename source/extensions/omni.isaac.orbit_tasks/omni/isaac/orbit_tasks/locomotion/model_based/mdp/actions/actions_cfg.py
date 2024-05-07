# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

from omni.isaac.orbit.managers.action_manager import ActionTerm, ActionTermCfg
from omni.isaac.orbit.utils import configclass

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

    # controller: model_base_controller.modelBaseController = MISSING
    controller: model_base_controller.samplingController = MISSING
    """Model base controller that compute u: output torques from z: latent variable""" 

    @configclass
    class FootTrajectoryCfg:
        """ Config class for foot trajectory generator hyperparameters
        """
        step_height: float = 0.05
        """ Default step height : used by the swing trajectory generator to determine the apex height in the middle of the trajectory  """

        foot_offset: float = 0.03
        """ Offset between the foot position (as returned in body view by the simulator) and the ground when in contact. """

    footTrajectoryCfg: FootTrajectoryCfg = FootTrajectoryCfg()
    """ Hyperparameter of the foot trajectory generator"""

    @configclass
    class SwingControllerCfg:
        """ Config class for swing foot trajectory controller hyperparameters
        """
        swing_ctrl_pos_gain_fb: float = 10000.0
        """ Position gain feedback for swing trajectory tracking in [0, +inf] """

        swing_ctrl_vel_gain_fb: float = 0.0
        """ Velocity gain feedback for swing trajectory tracking in [0, +inf] """

    swingControllerCfg: SwingControllerCfg = SwingControllerCfg()
    """ Hyperparameter of the swing foot controller"""





