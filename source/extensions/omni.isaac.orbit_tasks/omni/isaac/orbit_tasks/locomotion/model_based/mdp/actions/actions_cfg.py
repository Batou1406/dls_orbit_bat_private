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

    # @configclass
    # class OffsetCfg:
    #     """The offset pose from parent frame to child frame.

    #     On many robots, end-effector frames are fictitious frames that do not have a corresponding
    #     rigid body. In such cases, it is easier to define this transform w.r.t. their parent rigid body.
    #     For instance, for the Franka Emika arm, the end-effector is defined at an offset to the the
    #     "panda_hand" frame.
    #     """

    #     pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    #     """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""
    #     rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    #     """Quaternion rotation ``(w, x, y, z)`` w.r.t. the parent frame. Defaults to (1.0, 0.0, 0.0, 0.0)."""

    # body_name: str = MISSING
    # """Name of the body or frame for which IK is performed."""

    # body_offset: OffsetCfg | None = None
    # """Offset of target frame w.r.t. to the body frame. Defaults to None, in which case no offset is applied."""




