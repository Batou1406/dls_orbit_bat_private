# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import carb

import omni.isaac.orbit.utils.string as string_utils
from omni.isaac.orbit.assets.articulation import Articulation
from omni.isaac.orbit.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import BaseEnv

    from . import actions_cfg


import jax
import jax.dlpack
import torch
import torch.utils.dlpack

def jax_to_torch(x):
    return torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(x))
def torch_to_jax(x):
    return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(x))



class ModelBaseAction(ActionTerm):
    """Base class for model base actions.

    The action term is responsible for processing the raw actions sent to the environment
    and applying them to the asset managed by the term. The action term is comprised of two
    operations:

    * Processing of actions: This operation is performed once per **environment step** and
      is responsible for pre-processing the raw actions sent to the environment.
    * Applying actions: This operation is performed once per **simulation step** and is
      responsible for applying the processed actions to the asset managed by the term.

    Properties :
        cfg
        _env
        _asset
        _scale, _offset
        num_env
        device
        action_dim
        raw_actions, _raw_actions
        processed_actions, _processed_actions
        f
        d
        p
        F
        z
        _joint_ids, _joint_names, _num_joints

    Method :
        reset(env_ids: Sequence[int] | None = None) -> None:
        __call__(*args) -> Any
        process_actions(actions: torch.Tensor)
        apply_actions()
    """

    cfg: actions_cfg.ModelBaseActionCfg
    """The configuration of the action term."""

    _asset: Articulation
    """The articulation asset on which the action term is applied. Asset is defined in ActionTerm base clase, here just the type is redefined"""

    _scale: torch.Tensor | float
    """The scaling factor applied to the input action."""

    _offset: torch.Tensor | float
    """The offset applied to the input action."""

    _num_legs = 4
    _prevision_horizon = 10


    def __init__(self, cfg: actions_cfg.ModelBaseActionCfg, env: BaseEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        """
        Model Base Variable
        """
        self.f = torch.zeros(self.num_envs, self._num_legs, device=self.device)
        self.d = torch.zeros(self.num_envs, self._num_legs, device=self.device)
        self.p = torch.zeros(self.num_envs, self._num_legs, self._prevision_horizon, device=self.device)
        self.F = torch.zeros(self.num_envs, self._num_legs, self._prevision_horizon, device=self.device)
        self.z = [self.f, self.d, self.p, self.F]

        # Instance of control class. Gets Z and output u
        self.controller = samplingController() 

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)  # joint_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self._num_joints = len(self._joint_ids)                                             # joint_names = ['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint', 'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint', 'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint']
        # log the resolved joint names for debugging
        carb.log_info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )
        # Avoid indexing across all joints for efficiency
        if self._num_joints == self._asset.num_joints:
            self._joint_ids = slice(None)

        # parse scale
        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        elif isinstance(cfg.scale, dict):
            self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.scale, self._joint_names)
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}. Supported types are float and dict.")
        
        # parse offset
        if isinstance(cfg.offset, (float, int)):
            self._offset = float(cfg.offset)
        elif isinstance(cfg.offset, dict):
            self._offset = torch.zeros_like(self._raw_actions)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.offset, self._joint_names)
            self._offset[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported offset type: {type(cfg.offset)}. Supported types are float and dict.")
        
    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return sum(variable.shape[1:].numel() for variable in self.z)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
    

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        """Processes the actions sent to the environment.
        Note: This function is called once per environment step by the manager.
        Args: actions: The actions to process.
        """
        # store the raw actions
        self._raw_actions[:] = actions

        # apply the affine transformations
        self._processed_actions = self._raw_actions * self._scale + self._offset

        # reconstruct the latent variable



    def apply_actions2(self):
        """Applies the actions to the asset managed by the term.
        Note: This is called at every simulation step by the manager.
        """

        # Use model controller to compute the torques from the latent variable
        output_torques = self.controller.compute_output(self.z)

        # Apply the computed torques
        # self._asset.set_joint_effort_target(output_torques, joint_ids=self._joint_ids)

    
    def apply_actions(self):
        """Applies the actions to the asset managed by the term.
        Note: This is called at every simulation step by the manager.
        """
        output_torques = (torch.rand(self.num_envs, self._num_joints, device=self.device))# * 80) - 40

        print('--- Torch ---')
        print('shape : ',output_torques.shape)
        print('device : ',output_torques.device)
        print('Type : ', type(output_torques))
        
        output_torques_jax = torch_to_jax(output_torques)
        output_torques_jax = (output_torques_jax * 80) - 40

        print('')
        print('--- Jax ---')
        print('Shape : ', output_torques_jax.shape)
        print('device : ',output_torques_jax.devices())
        print('Type : ', type(output_torques_jax))

        output_torques2 = jax_to_torch(output_torques_jax)

        # set joint effort targets (should be equivalent to torque) : Torque controlled robot
        self._asset.set_joint_effort_target(output_torques2, joint_ids=self._joint_ids)


class samplingController():
    """
    Some Description
    """
    def __init__(self):
        pass

    def compute_output(self, z):

        # 1. Sample from law given by actions
        # f_collection = jax.random
        
        # 2. Generate the trajectories from the samples

        # 3. evaluate the trajectories

        # 4. Pick the best trajectory 

        # 5. Return the first action from the best trajectory
        pass
