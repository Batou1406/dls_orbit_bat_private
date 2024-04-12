# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# ./orbit.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-Model-Based-Base-Aliengo-v0  --num_envs 32

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import carb

import omni.isaac.orbit.utils.math as math_utils
import omni.isaac.orbit.utils.string as string_utils
from omni.isaac.orbit.assets.articulation import Articulation
from omni.isaac.orbit.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import BaseEnv

    from . import actions_cfg

from . import model_base_controller #import modelBaseController, samplingController

import jax
import jax.dlpack
import torch
import torch.utils.dlpack

import numpy as np


def jax_to_torch(x: jax.Array):
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
        - pt_star(th.Tensor): Optimizied foot swing trajectory      of shape (batch_size, num_legs, 9, decimation)  (9 = pos, vel, acc)
        - z   (tuple)       : Latent variable : z=(f,d,p,F)         of shape (...)
        - u   (torch.Tensor): output joint torques                  of shape (batch_size, num_joints)
        - controller (modelBaseController): controller instance that compute u from z 
        - _foot_idx         : List of index of the feet
        - _num_legs         : Number of legs of the robot  : useful for dimension definition
        - _num_joints_per_leg: Number of joints per leg    : useful for dimension definition
        - _joint_idx        : List of list of joints [FL_joints, FR_joints, ...]
        - jacobian_prev (ts): jacobian from prev. dt for jac_dot    of shape (batch_size, num_leg, 3, num_joints_per_leg) 
        - inner_loop        : Counter of inner loop wrt. outer loop

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

    controller: model_base_controller.modelBaseController
    """Model base controller that compute u: output torques from z: latent variable""" 

    _num_legs = 4           # Should get that from articulation or robot config
    _prevision_horizon = 10 # Should get that from cfg
    _decimation = 10        # Should get that from the env


    def __init__(self, cfg: actions_cfg.ModelBaseActionCfg, env: BaseEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

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

        # Retrieve series of information usefull for computation and generalisation
        # Feet Index in body, list [13, 14, 15, 16]
        self._foot_idx = self._asset.find_bodies(".*foot")[0]
        self._num_legs = len(self._foot_idx)
        self._num_joints_per_leg = self._num_joints // self._num_legs

        # variable to count the number of inner loop with respect to outer loop
        self.inner_loop = 0

        # Joint Index
        fl_joints = self._asset.find_joints("FL.*")[0]		# list [0, 4,  8]
        fr_joints = self._asset.find_joints("FR.*")[0]		# list [1, 5,  9]
        rl_joints = self._asset.find_joints("RL.*")[0]		# list [2, 6, 10]
        rr_joints = self._asset.find_joints("RR.*")[0]		# list [3, 7, 11]
        self._joints_idx = [fl_joints, fr_joints, rl_joints, rr_joints]

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
        self.pt_star= torch.zeros(self.num_envs, self._num_legs, 9, self._decimation, device=self.device)

        # Control input u : joint torques
        self.u = torch.zeros(self.num_envs, self._num_joints, device=self.device)

        # Variable for intermediary computaion
        self.jacobian_prev = torch.zeros(self.num_envs, self._num_legs, 3, self._num_joints_per_leg, device=self.device)

        # Instance of control class. Gets Z and output u
        self.controller = cfg.controller

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

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
        
        # # parse the body index
        # body_ids, body_names = self._asset.find_bodies(self.cfg.body_name)
        # if len(body_ids) != 1:
        #     raise ValueError(
        #         f"Expected one match for the body name: {self.cfg.body_name}. Found {len(body_ids)}: {body_names}."
        #     )
        # # save only the first body index
        # self._body_idx = body_ids[0]
        # self._body_name = body_names[0]
        # # check if articulation is fixed-base
        # # if fixed-base then the jacobian for the base is not computed
        # # this means that number of bodies is one less than the articulation's number of bodies
        # if self._asset.is_fixed_base:
        #     self._jacobi_body_idx = self._body_idx - 1
        # else:
        #     self._jacobi_body_idx = self._body_idx
        # carb.log_info(  # log info for debugging
        #     f"Resolved body name for the action term {self.__class__.__name__}: {self._body_name} [{self._body_idx}]"
        # )

        # # convert the fixed offsets to torch tensors of batched shape
        # if self.cfg.body_offset is not None:
        #     self._offset_pos = torch.tensor(self.cfg.body_offset.pos, device=self.device).repeat(self.num_envs, 1)
        #     self._offset_rot = torch.tensor(self.cfg.body_offset.rot, device=self.device).repeat(self.num_envs, 1)
        # else:
        #     self._offset_pos, self._offset_rot = None, None

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
        # self.p_star, self.F_star, self.c_star, self.pt_star = self.controller.optimize_latent_variable(f=self.f, d=self.d, p=self.p, F=self.F)

        # Reset the inner loop counter
        self.inner_loop = 0


    def apply_actions(self):
        """Applies the actions to the asset managed by the term.

        Note: 
            This is called at every simulation step by the manager. : Inner loop frequency
            1. Extract optimized latent variable p0*, F0*, c0*, pt01*
            2. Compute joint torque to be applied (call controller.compute_control_output) and update u
            3. Apply joint torque to simulation 
        """
        
        # Retrieve the robot states 
        p, p_dot, q_dot, jacobian, jacobian_dot, mass_matrix, h = self.get_robot_state() 

        # Extract optimal variables
        F0_star = self.F_star[:,:,:,0]
        c0_star = self.c_star[:,:,0]
        pt_i_star = self.pt_star[:,:,:,self.inner_loop]
        self.inner_loop += 1

        # Use model controller to compute the torques from the latent variable
        self.u = self.controller.compute_control_output(F0_star=F0_star, c0_star=c0_star, pt_i_star=pt_i_star, p=p, p_dot=p_dot, q_dot=q_dot, jacobian=jacobian, jacobian_dot=jacobian_dot, mass_matrix=mass_matrix, h=h)

        # Apply the computed torques
        self._asset.set_joint_effort_target(self.u, joint_ids=self._joint_ids)


    def apply_actions2(self):
        """Applies the actions to the asset managed by the term.
        Note: This is called at every simulation step by the manager.
        """
        output_torques = (torch.rand(self.num_envs, self._num_joints, device=self.device))# * 80) - 40

        # print('--- Torch ---')
        # print('shape : ',output_torques.shape)
        # print('device : ',output_torques.device)
        # print('Type : ', type(output_torques))
        
        output_torques_jax = torch_to_jax(output_torques)
        output_torques_jax = (output_torques_jax * 80) - 40

        # print('')
        # print('--- Jax ---')
        # print('Shape : ', output_torques_jax.shape)
        # print('device : ',output_torques_jax.devices())
        # print('Type : ', type(output_torques_jax))

        output_torques2 = jax_to_torch(output_torques_jax)

        self.get_robot_state2()

        # set joint effort targets (should be equivalent to torque) : Torque controlled robot
        self._asset.set_joint_effort_target(output_torques2, joint_ids=self._joint_ids)


    def get_robot_state(self):
        """ Retrieve the Robot states from the simulator

        Return :
            - p   (torch.Tensor): Feet Position  (latest from sim)      of shape(batch_size, num_legs, 3)
            - p_dot (tch.Tensor): Feet velocity  (latest from sim)      of shape(batch_size, num_legs, 3)
            - q_dot (tch.Tensor): Joint velocity (latest from sim)      of shape(batch_size, num_legs, num_joints_per_leg)
            - jacobian  (Tensor): Jacobian -> joint frame to foot frame of shape(batch_size, num_legs, 3, num_joints_per_leg)
            - jacobian_dot (Tsr): Jacobian derivative (forward euler)   of shape(batch_size, num_legs, 3, num_joints_per_leg)
            - mass_matrix (Tsor): Mass Matrix in joint space            of shape(batch_size, num_legs, num_joints_per_leg, num_joints_per_leg)
            - h   (torch.Tensor): C(q,q_dot) + G(q) (corr. and grav F.) of shape(batch_size, num_legs, num_joints_per_leg)
        """

        # Retrieve robot base position and orientation in order to compute world->base frame transformation
        robot_pos_w = self._asset.data.root_pos_w       # shape (batch_size, 3) (xyz)
        robot_orientation_w = self._asset.data.root_quat_w # shape (batch_size, 4) (quaternions)
        robot_vel_w = self._asset.data.root_lin_vel_w
        # robot_ang_vel_w = self._asset.data.root_ang_vel_w

        # Retrieve Feet position in world frame : [num_instances, num_bodies, 3] select right indexes to get 
        # shape(batch_size, num_legs, 3)
        # Finally apply frame transformation to get feet position in body frame
        p_w = self._asset.data.body_pos_w[:, self._foot_idx,:]
        p_orientation_w = self._asset.data.body_quat_w[:, self._foot_idx,:]
        p_b_0, _ = math_utils.subtract_frame_transforms(robot_pos_w, robot_orientation_w, p_w[:,0,:], p_orientation_w[:,0,:])
        p_b_1, _ = math_utils.subtract_frame_transforms(robot_pos_w, robot_orientation_w, p_w[:,1,:], p_orientation_w[:,1,:])
        p_b_2, _ = math_utils.subtract_frame_transforms(robot_pos_w, robot_orientation_w, p_w[:,2,:], p_orientation_w[:,2,:])
        p_b_3, _ = math_utils.subtract_frame_transforms(robot_pos_w, robot_orientation_w, p_w[:,3,:], p_orientation_w[:,3,:])
        p_b = torch.cat((p_b_0.unsqueeze(1), p_b_1.unsqueeze(1), p_b_2.unsqueeze(1), p_b_3.unsqueeze(1)), dim=1)

        # Retrieve Feet velocity in world frame : [num_instances, num_bodies, 3] select right indexes to get 
        # shape(batch_size, num_legs, 3)
        # Finally apply frame transformation to get feet position in body frame
        p_dot_w = self._asset.data.body_lin_vel_w[:, self._foot_idx,:]
        # p_dot_orientation_w = self._asset.data.body_ang_vel_w[:, self._foot_idx, :]
        p_dot_b_0, _ = math_utils.subtract_frame_transforms(robot_vel_w, robot_orientation_w, p_dot_w[:,0,:], p_orientation_w[:,0,:])
        p_dot_b_1, _ = math_utils.subtract_frame_transforms(robot_vel_w, robot_orientation_w, p_dot_w[:,1,:], p_orientation_w[:,1,:])
        p_dot_b_2, _ = math_utils.subtract_frame_transforms(robot_vel_w, robot_orientation_w, p_dot_w[:,2,:], p_orientation_w[:,2,:])
        p_dot_b_3, _ = math_utils.subtract_frame_transforms(robot_vel_w, robot_orientation_w, p_dot_w[:,3,:], p_orientation_w[:,3,:])
        p_dot_b = torch.cat((p_dot_b_0.unsqueeze(1), p_dot_b_1.unsqueeze(1), p_dot_b_2.unsqueeze(1), p_dot_b_3.unsqueeze(1)), dim=1)

        # Retrieve Joint velocities [num_instances, num_joints] -> reorganise the view and permute to get the
        # shape(batch_size, num_legs, num_joints_per_leg) : This is in joint space, no transformation required
        q_dot = self._asset.data.joint_vel.view(-1,self._num_joints_per_leg,self._num_legs).permute(0,2,1)
        
        # Retrieve Jacobian from sim
        # shape(batch_size, num_legs, 3, num_joints_per_leg)
        # Intermediary step : extract feet jacobian [batch_size, num_bodies=17, 6, num_joints+6=18] -> [..., 4, 3, 18]
        # Shift from 6 due to an offset. This is due to how the model is define I think
        jacobian_feet_full = self._asset.root_physx_view.get_jacobians()[:, self._foot_idx, :3, :]
        jacobian = torch.cat((jacobian_feet_full[:, 0, :, 6+np.asarray(self._joints_idx[0])].unsqueeze(1),
                              jacobian_feet_full[:, 1, :, 6+np.asarray(self._joints_idx[1])].unsqueeze(1),
                              jacobian_feet_full[:, 2, :, 6+np.asarray(self._joints_idx[2])].unsqueeze(1),
                              jacobian_feet_full[:, 3, :, 6+np.asarray(self._joints_idx[3])].unsqueeze(1)), dim=1)
        

        # Compute jacobian derivative, using forward euler. 
        jacobian_dot = ((jacobian - self.jacobian_prev) / self._env.physics_dt)
        self.jacobian_prev = jacobian

        # Retrieve the mass Matrix
        # Shape is (batch_size, num_joints, num_joints) (ie. 144 element), we have to extract num leg sub matrices from that to have 
        # shape (batch_size, num_leg, num_joints_per_leg, num_joints_per_leg) (ie. 36 elements)
        # This is done with complex indexing operations
        # mass_matrix_full = self._asset.root_physx_view.get_mass_matrices()
        # mass_matrix_FL = mass_matrix_full[:,[0,4,8],:][:, [0,4,8]]
        joints_idx_tensor = torch.Tensor(self._joints_idx).unsqueeze(2).unsqueeze(3).long() # long to use it to access indexes -> float trow an error
        mass_matrix = self._asset.root_physx_view.get_mass_matrices()[:, joints_idx_tensor, joints_idx_tensor.transpose(1,2)].squeeze(-1)
        
        # Retrieve Corriolis, centrifugial and gravitationnal term
        # get_coriolis_and_centrifugal_forces -> (batch_size, num_joints)
        # get_generalized_gravity_forces -> (batch_size, num_joints)
        h = self._asset.root_physx_view.get_coriolis_and_centrifugal_forces() + self._asset.root_physx_view.get_generalized_gravity_forces()

        return p_b, p_dot_b, q_dot, jacobian, jacobian_dot, mass_matrix, h


    def get_robot_state2(self):
        """ TODO Write description
        """

        # Joint Index
        fl_joints = self._asset.find_joints("FL.*")[0]		# list [0, 4,  8]
        fr_joints = self._asset.find_joints("FR.*")[0]		# list [1, 5,  9]
        rl_joints = self._asset.find_joints("RL.*")[0]		# list [2, 6, 10]
        rr_joints = self._asset.find_joints("RR.*")[0]		# list [3, 7, 11]

        # Body Index
        foot_idx = self._asset.find_bodies(".*foot")[0]

        # 'FL_foot', 'FR_foot', 'RL_foot', 'RR_foot'
        fl_jacobian = self._asset.root_physx_view.get_jacobians()[:, foot_idx[0], 0:3, fl_joints]# + 6]
        fr_jacobian = self._asset.root_physx_view.get_jacobians()[:, foot_idx[1], 0:3, fr_joints]# + 6]
        rl_jacobian = self._asset.root_physx_view.get_jacobians()[:, foot_idx[2], 0:3, rl_joints]# + 6]
        rr_jacobian = self._asset.root_physx_view.get_jacobians()[:, foot_idx[3], 0:3, rr_joints]# + 6]

        # foot position in wf
        fl_foot_pos_w = self._asset.data.body_state_w[:, foot_idx[0], 0:3]
        fr_foot_pos_w = self._asset.data.body_state_w[:, foot_idx[1], 0:3]
        rl_foot_pos_w = self._asset.data.body_state_w[:, foot_idx[2], 0:3]
        rr_foot_pos_w = self._asset.data.body_state_w[:, foot_idx[3], 0:3]

        # foot orientation in wf
        fl_foot_orient_w = self._asset.data.body_state_w[:, foot_idx[0], 3:7]
        fr_foot_orient_w = self._asset.data.body_state_w[:, foot_idx[1], 3:7]
        rl_foot_orient_w = self._asset.data.body_state_w[:, foot_idx[2], 3:7]
        rr_foot_orient_w = self._asset.data.body_state_w[:, foot_idx[3], 3:7]

        # Root state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 13)
        base_pose_w = self._asset.data.root_state_w[:, 0:3]
        base_orient_w = self._asset.data.root_state_w[:, 3:7]
        base_lin_vel_w = self._asset.data.root_state_w[:, 7:10]
        base_ang_vel_w = self._asset.data.root_state_w[:, 10:13]

        # foot position, orientation in bf
        fl_foot_pos_b, fl_foot_orient_b = math_utils.subtract_frame_transforms(base_pose_w, base_orient_w, fl_foot_pos_w, fl_foot_orient_w)
        fr_foot_pos_b, fr_foot_orient_b = math_utils.subtract_frame_transforms(base_pose_w, base_orient_w, fr_foot_pos_w, fr_foot_orient_w)
        rl_foot_pos_b, rl_foot_orient_b = math_utils.subtract_frame_transforms(base_pose_w, base_orient_w, rl_foot_pos_w, rl_foot_orient_w)
        rr_foot_pos_b, rr_foot_orient_b = math_utils.subtract_frame_transforms(base_pose_w, base_orient_w, rr_foot_pos_w, rr_foot_orient_w)

        # foot joint position
        fl_joint_pos = self._asset.data.joint_pos[:, fl_joints]
        fr_joint_pos = self._asset.data.joint_pos[:, fr_joints]
        rl_joint_pos = self._asset.data.joint_pos[:, rl_joints]
        rr_joint_pos = self._asset.data.joint_pos[:, rr_joints]

        # foot joint velocity
        fl_joint_vel = self._asset.data.joint_vel[:, fl_joints]
        fr_joint_vel = self._asset.data.joint_vel[:, fr_joints]
        rl_joint_vel = self._asset.data.joint_vel[:, rl_joints]
        rr_joint_vel = self._asset.data.joint_vel[:, rr_joints]

        print('alo')