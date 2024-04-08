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

from model_base_controller import modelBaseController, samplingController

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
        - raw_actions, _raw_actions
              (torch.Tensor): Actions received from the RL policy   of shape (batch_size, action_dim)
        - processed_actions, _processed_actions
              (torch.Tensor): scaled and offseted actions from RL   of shape (batch_site, action_dim)
        _joint_ids, _joint_names, _num_joints
        - f   (torch.Tensor): Prior leg frequency                   of shape (batch_size, num_legs)
        - d   (torch.Tensor): Prior stepping duty cycle             of shape (batch_size, num_legs)
        - p   (torch.Tensor): Prior foot pos. sequence              of shape (batch_size, num_legs, 3, time_horizon)
        - F   (torch.Tensor): Prior Ground Reac. Forces (GRF) seq.  of shape (batch_size, num_legs, 3, time_horizon)
        - p_star (th.Tensor): Optimizied foot pos sequence          of shape (batch_size, num_legs, 3, time_horizon)
        - F_star (th.Tensor): Opt. Ground Reac. Forces (GRF) seq.   of shape (batch_size, num_legs, 3, time_horizon)
        - c_star (th.Tensor): Optimizied foot contact sequence      of shape (batch_size, num_legs, time_horizon)
        - pt_star(th.Tensor): Optimizied foot swing trajectory      of shape (batch_size, num_legs, 3, decimation)
        - z   (tuple)       : Latent variable : z=(f,d,p,F)         of shape (...)
        - u   (torch.Tensor): output joint torques                  of shaoe (batch_size, num_joints)
        - controller (modelBaseController): controller instance that compute u from z 

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

    controller: modelBaseController
    """Model base controller that compute u: output torques from z: latent variable""" 

    _num_legs = 4           # Should get that from articulation or robot config
    _prevision_horizon = 10 # Should get that from cfg
    _decimation = 10        # Should get that from the env


    def __init__(self, cfg: actions_cfg.ModelBaseActionCfg, env: BaseEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        # Latent variable
        self.f = torch.zeros(self.num_envs, self._num_legs, device=self.device)
        self.d = torch.zeros(self.num_envs, self._num_legs, device=self.device)
        self.p = torch.zeros(self.num_envs, self._num_legs, 3, self._prevision_horizon, device=self.device)
        self.F = torch.zeros(self.num_envs, self._num_legs, 3, self._prevision_horizon, device=self.device)
        self.z = [self.f, self.d, self.p, self.F]

        # Model-based optimized latent variable
        self.p_star = torch.zeros(self.num_envs, self._num_legs, 3, self._prevision_horizon, device=self.device)
        self.F_star = torch.zeros(self.num_envs, self._num_legs, 3, self._prevision_horizon, device=self.device)
        self.c_star = torch.zeros(self.num_envs, self._num_legs, self._prevision_horizon, device=self.device)
        self.pt_star= torch.zeros(self.num_envs, self._num_legs, self._decimation, device=self.device)

        # Control input u : joint torques
        self.u = torch.zeros(self.num_envs, self._num_joints, device=self.device)

        # Instance of control class. Gets Z and output u
        self.controller = cfg.controller

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

    # TODO : Faut-il modifier le 1 par un 0 ?
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

        Note:
            This function is called once per environment step. : Outer loop frequency
            1. Apply affine transformation (scale and offset)
            2. Reconstrut latent variable z = (f,d,p,F)
            3. Optimize the latent variable (call controller.optimize_control_output)
                and update optimizied solution p*, F*, c*, pt*

        Args:
            action (torch.Tensor): The actions received from RL policy of Shape (num_envs, total_action_dim)
        """

        # store the raw actions
        self._raw_actions[:] = actions

        # apply the affine transformations
        self._processed_actions = self._raw_actions * self._scale + self._offset

        # reconstruct the latent variable from the RL poliy actions
        self.f = self._processed_actions[:, :self._num_legs] # 0:3 
        self.d = self._processed_actions[:, self._num_legs:2*self._num_legs] # 4:7
        self.p = self._processed_actions[:, 2*self._num_legs:(2*self._num_legs + 3*self._num_legs*self._prevision_horizon)].reshape([self.num_envs, self._num_legs, 3, self._prevision_horizon])
        self.F = self._processed_actions[:, (2*self._num_legs + 3*self._num_legs*self._prevision_horizon):].reshape([self.num_envs, self._num_legs, 3, self._prevision_horizon])
        self.z = [self.f, self.d, self.p, self.F]

        # Optimize the latent variable with the model base controller
        self.p_star, self.F_star, self.c_star, self.pt_star = self.controller.optimize_latent_variable(f=self.f, d=self.d, p=self.p, F=self.F)


    def apply_actions(self):
        """Applies the actions to the asset managed by the term.

        Note: 
            This is called at every simulation step by the manager. : Inner loop frequency
            1. Extract optimized latent variable p0*, F0*, c0*, pt01*
            2. Compute joint torque to be applied (call controller.compute_control_output) and update u
            3. Apply joint torque to simulation 
        """
        # Use model controller to compute the torques from the latent variable
        self.u = self.controller.compute_control_output(F0_star=self.F_star[:,:,:,0], c0_star=self.c_star[:,:,0], pt01_star=self.pt_star[:,:,:,0])

        # Apply the computed torques
        self._asset.set_joint_effort_target(self.u, joint_ids=self._joint_ids)

    
    def apply_actions2(self):
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