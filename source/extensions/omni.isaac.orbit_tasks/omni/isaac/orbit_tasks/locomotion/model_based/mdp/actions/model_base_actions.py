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

## >>>Visualization
import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.orbit.sim import SimulationContext
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR, ISAAC_ORBIT_NUCLEUS_DIR
from omni.isaac.orbit.utils.math import quat_from_angle_axis
## <<<Visualization

def jax_to_torch(x: jax.Array):
    return torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(x))
def torch_to_jax(x):
    return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(x))

verbose_mb = True
verbose_loop = 40

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
    --Inherited--
        cfg                 (actions_cfg.ModelBaseActionCfg)                                                            Inherited from ManagerTermBase
        _env                (BaseEnv)                                                                                   Inherited from ManagerTermBase
        num_env             @Property                                                                                   Inherited from ManagerTermBase
        device              @Property                                                                                   Inherited from ManagerTermBase
        _asset              (Articulation)                                                                              Inherited from ActionTerm

    --Defined--
        _scale       (float): Re-scale the raw actions received from RL policy                                          Received from ModelBaseActionCfg
        _offset      (float): Offset   the raw actions received from RL policy                                          Received from ModelBaseActionCfg
        action_dim          @Property                                                                                   Inherited from ActionTerm (not implemented)
        raw_actions         @Property                                                                                   Inherited from ActionTerm (not implemented)
        _raw_actions (Tnsor): Actions received from the RL policy   of shape (batch_size, action_dim)                   
        processed_actions   @Property                                                                                   Inherited from ActionTerm (not implemented)
        _processed_actions  : scaled and offseted actions from RL   of shape (batch_site, action_dim)
        _joint_ids    (list): list of int corresponding to joint index
        _joint_names  (list): list of str corresponding to joint name                                                   Received from ModelBaseActionCfg
        _num_joints    (int): Number of joints
        _foot_idx     (list): List of index of the feet
        _num_legs      (int): Number of legs of the robot  : useful for dimension definition
        _num_joints_per_leg : Number of joints per leg     : useful for dimension definition
        _joint_idx    (list): List of list of joints [FL_joints=[FL_Hip,...], FR_joints, ...]
        _decimation    (int): Inner Loop steps per outer loop steps
        _prevision_horizon  : Prediction time horizon for the Model Base controller (runs at outer loop frequecy)       Received from ModelBaseActionCfg
        controller (modelBaseController): controller instance that compute u from z                                     Received from ModelBaseActionCfg
        
        f     (torch.Tensor): Prior leg frequency                   of shape (batch_size, num_legs)
        d     (torch.Tensor): Prior stepping duty cycle             of shape (batch_size, num_legs)
        p     (torch.Tensor): Prior foot pos. sequence              of shape (batch_size, num_legs, 3, time_horizon)
        F     (torch.Tensor): Prior Ground Reac. Forces (GRF) seq.  of shape (batch_size, num_legs, 3, time_horizon)
        p_star (trch.Tensor): Optimizied foot pos sequence          of shape (batch_size, num_legs, 3, time_horizon)
        F_star (trch.Tensor): Opt. Ground Reac. Forces (GRF) seq.   of shape (batch_size, num_legs, 3, time_horizon)
        c_star (trch.Tensor): Optimizied foot contact sequence      of shape (batch_size, num_legs, time_horizon)
        pt_star (tch.Tensor): Optimizied foot swing trajectory      of shape (batch_size, num_legs, 9, decimation)  (9 = pos, vel, acc)
        z            (tuple): Latent variable : z=(f,d,p,F)         of shape (...)
        u     (torch.Tensor): output joint torques                  of shape (batch_size, num_joints)
        jacobian_prev (Tsor): jacobian from prev. dt for jac_dot    of shape (batch_size, num_leg, 3, num_joints_per_leg) 
        inner_loop     (int): Counter of inner loop wrt. outer loop

    Method :
        reset(env_ids: Sequence[int] | None = None) -> None:                                                            Inherited from ManagerTermBaseCfg (not implemented)
        __call__(*args) -> Any                                                                                          Inherited from ManagerTermBaseCfg (not implemented)
        process_actions(actions: torch.Tensor)                                                                          Inherited from ActionTerm (not implemented)
        apply_actions()                                                                                                 Inherited from ActionTerm (not implemented)
        get_robot_state() -> tuple(torch.Tensor)
        get_jacobian() -> torch.Tensor
        get_reset_foot_position() -> torch.Tensors
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


    def __init__(self, cfg: actions_cfg.ModelBaseActionCfg, env: BaseEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        # Prevision Horizon
        self._prevision_horizon = self.cfg.prevision_horizon

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
  
        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)  # joint_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self._num_joints = len(self._joint_ids)                                             # joint_names = ['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint', 'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint', 'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint']
        # log the resolved joint names for debugging
        carb.log_info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )
        # Avoid indexing across all joints for efficiency #TODO Is it still usefull
        # if self._num_joints == self._asset.num_joints:
        #     self._joint_ids = slice(None)

        # Retrieve series of information usefull for computation and generalisation
        # Feet Index in body, list [13, 14, 15, 16]
        self._foot_idx = self._asset.find_bodies(".*foot")[0]
        self._num_legs = len(self._foot_idx)
        self._num_joints_per_leg = self._num_joints // self._num_legs
        self._decimation = self._env.cfg.decimation  

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

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)   

        # Model-based optimized latent variable
        self.p_star = torch.zeros(self.num_envs, self._num_legs, 3, self._prevision_horizon, device=self.device)
        self.F_star = torch.zeros(self.num_envs, self._num_legs, 3, self._prevision_horizon, device=self.device)
        self.c_star = torch.ones(self.num_envs, self._num_legs, self._prevision_horizon, device=self.device)
        self.pt_star= torch.zeros(self.num_envs, self._num_legs, 9, self._decimation, device=self.device)

        # Control input u : joint torques
        self.u = torch.zeros(self.num_envs, self._num_joints, device=self.device)

        # Variable for intermediary computaion
        self.jacobian_prev = torch.zeros(self.num_envs, self._num_legs, 3, self._num_joints_per_leg, device=self.device) # TODO Is it correct ? This would likely gice a huge jacobian_dot ?

        # Instance of control class. Gets Z and output u
        self.controller = cfg.controller
        self.controller.late_init(device=self.device, num_envs=self.num_envs, num_legs=self._num_legs, time_horizon=self._prevision_horizon, dt_out=self._decimation*self._env.physics_dt, decimation=self._decimation, dt_in=self._env.physics_dt, p_default=self.get_reset_foot_position()) 

        if verbose_mb:
            self.my_visualizer = {}
            self.my_visualizer['foot'] = define_markers('sphere', {'radius': 0.05, 'color': (1.0,1.0,0)})
            self.my_visualizer['jacobian'] = define_markers('arrow_x', {'scale':(0.04,0.04,0.3), 'color': (1.0,0,0)})
            self.my_visualizer['foot_traj'] = define_markers('sphere', {'radius': 0.02, 'color': (1.0,0.0,1.0)})


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
        # self.z = [self.f, self.d, self.p, self.F]

        # Optimize the latent variable with the model base controller
        self.p_star, self.F_star, self.c_star, self.pt_star = self.controller.optimize_latent_variable(f=self.f, d=self.d, p=self.p, F=self.F)

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
        # Transform the shape from (batch_size, num_legs, num_joints_per_leg) to (batch_size, num_joints)
        self.u = (self.controller.compute_control_output(F0_star=F0_star, c0_star=c0_star, pt_i_star=pt_i_star, p=p, p_dot=p_dot, q_dot=q_dot, jacobian=jacobian, jacobian_dot=jacobian_dot, mass_matrix=mass_matrix, h=h)).permute(0,2,1).reshape(self.num_envs,self._num_joints)

        # Apply the computed torques
        self._asset.set_joint_effort_target(self.u, joint_ids=self._joint_ids)

        # Debug
        if verbose_mb:
            self.debug_apply_action(p, p_dot, q_dot, jacobian, jacobian_dot, mass_matrix, h, F0_star, c0_star, pt_i_star)


    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Resets the manager term.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.
        """
        # check if specific environment ids are provided otherwise all environments must be reseted
        if env_ids is None:
            env_ids = slice(None)   # TODO Check the behaviour

        # variable to count the number of inner loop with respect to outer loop
        self.inner_loop = 0

        # Reset jacobian_prev : copied from get_robot_state() Retrieve Jacobian from sim : shape(batch_size, num_legs, 3, num_joints_per_leg)
        self.jacobian_prev[env_ids,:,:,:] = self.get_jacobian()[env_ids,:,:,:]      # TODO : This may be wrong, maybe the environment need to be stepped once for the jacobian to be updated.

        # Reset the model base controller
        self.controller.reset(env_ids, self.get_reset_foot_position())


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

        # Retrieve Jacobian from sim  shape(batch_size, num_legs, 3, num_joints_per_leg) -> see method for implementation
        jacobian = self.get_jacobian()

        # Compute jacobian derivative, using forward euler. shape(batch_size, num_legs, 3, num_joints_per_leg)
        jacobian_dot = ((jacobian - self.jacobian_prev) / self._env.physics_dt)

        # Save jacobian for next iteration : required to compute jacobian derivative shape(batch_size, num_legs, 3, num_joints_per_leg)
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
        # Reshape and tranpose to get the correct shape in correct joint order-> (batch_size, num_legs, num_joints_per_leg)
        h = (self._asset.root_physx_view.get_coriolis_and_centrifugal_forces() + self._asset.root_physx_view.get_generalized_gravity_forces()).view(self.num_envs, self._num_joints_per_leg, self._num_legs).permute(0,2,1)

        return p_b, p_dot_b, q_dot, jacobian, jacobian_dot, mass_matrix, h
    

    def get_jacobian(self) -> torch.Tensor:
        """ To avoid code duplicate (used is get_robot_state() and reset())
        """
        # Retrieve Jacobian from sim
        # shape(batch_size, num_legs, 3, num_joints_per_leg)
        # Intermediary step : extract feet jacobian [batch_size, num_bodies=17, 6, num_joints+6=18] -> [..., 4, 3, 18]
        # Shift from 6 due to an offset. This is due to how the model is define I think
        jacobian_feet_full = self._asset.root_physx_view.get_jacobians()[:, self._foot_idx, :3, :]
        jacobian = torch.cat((jacobian_feet_full[:, 0, :, 6+np.asarray(self._joints_idx[0])].unsqueeze(1),
                              jacobian_feet_full[:, 1, :, 6+np.asarray(self._joints_idx[1])].unsqueeze(1),
                              jacobian_feet_full[:, 2, :, 6+np.asarray(self._joints_idx[2])].unsqueeze(1),
                              jacobian_feet_full[:, 3, :, 6+np.asarray(self._joints_idx[3])].unsqueeze(1)), dim=1)
        return jacobian
    

    def get_reset_foot_position(self) -> torch.Tensor:
        """ Return The default position of the robot's feet. this is the position when the states are reseted
        TODO Now, this is hardcoded for Aliengo (given a joint default position) -> Should get this from simulation or forward kinematics

        Return :
            - p   (torch.Tensor): Default Feet position after reset     of shape(batch_size, num_legs, 3)
        """
        return torch.tensor([[0.2238, 0.1735, -0.3678],[0.2238, -0.1735, -0.3678],[-0.329, 0.1719, -0.3579],[-0.329, -0.1719, -0.3679]], device=self.device).unsqueeze(0).expand(self.num_envs, -1, -1)
        

#-------------------------------------------------- Helpers ------------------------------------------------------------
    def debug_apply_action(self, p, p_dot, q_dot, jacobian, jacobian_dot, mass_matrix, h, F0_star, c0_star, pt_i_star):
        global verbose_loop

        # Print duty cycée and leg frequency
        verbose_loop+=1
        if verbose_loop>=40:
            verbose_loop=0
            print('Contact sequence : ', c0_star.flatten())
            print('\nLeg frequency : ', self.f.flatten())
            print('\nduty cycle : ', self.d.flatten())

        # Visualize foot position
        p_b = p.clone().detach()
        robot_pos_w = self._asset.data.root_pos_w
        robot_orientation_w = self._asset.data.root_quat_w
        p_orientation_w = self._asset.data.body_quat_w[:, self._foot_idx,:]

        p_w_0, _ = math_utils.combine_frame_transforms(robot_pos_w, robot_orientation_w, p_b[:,0,:])
        p_w_1, _ = math_utils.combine_frame_transforms(robot_pos_w, robot_orientation_w, p_b[:,1,:])
        p_w_2, _ = math_utils.combine_frame_transforms(robot_pos_w, robot_orientation_w, p_b[:,2,:])
        p_w_3, _ = math_utils.combine_frame_transforms(robot_pos_w, robot_orientation_w, p_b[:,3,:])
        p_w = torch.cat((p_w_0.unsqueeze(1), p_w_1.unsqueeze(1), p_w_2.unsqueeze(1), p_w_3.unsqueeze(1)), dim=1)

        marker_locations = p_w[0,:,:]
        self.my_visualizer['foot'].visualize(marker_locations)

        # Visualise jacobian
        joint_pos_w = self._asset.data.body_pos_w[0,self._joint_ids,:] # shape (num_joints, 3)
        marker_locations = joint_pos_w
        jacobian_temp = jacobian[0,:,:,:].clone().detach()
        jacobian_temp_T = jacobian_temp.permute(0,2,1) # shape (num_legs, 3, num_joints_per_leg) -> (num_legs, num_joints_per_leg, 3)
        jacobian_temp_T = jacobian_temp.flatten(0,1) # shape (num_joints, 3)
        normalize_jacobian_temp_T = torch.nn.functional.normalize(jacobian_temp_T, p=2, dim=1) # Transform jacobian to unit vectors

        # angle : u dot v = cos(angle) -> angle = acos(u*v) : for unit vector
        angle = torch.acos(torch.tensordot(normalize_jacobian_temp_T, torch.tensor([1.0,0.0,0.0], device=self.device), dims=1)) # shape(num_joints, 3) -> (num_joints)
        # Axis : Cross product between u^v (for unit vectors)
        axis = torch.cross(normalize_jacobian_temp_T, torch.tensor([1.0,0.0,0.0], device=self.device).unsqueeze(0).expand(normalize_jacobian_temp_T.shape))
        marker_orientations = quat_from_angle_axis(angle=angle, axis=axis)
        self.my_visualizer['jacobian'].visualize(marker_locations, marker_orientations)

        # Visualize foot trajectory
        pt_i_b = self.pt_star.clone().detach()  # shape (batch_size, num_legs, 9, decimation) (9=px,py,pz,vx,vy,vz,ax,ay,az)
        pt_i_b = pt_i_b[0,:,0:3,:] # -> shape (num_legs, 3, decimation)
        pt_i_b = pt_i_b.permute(0,2,1).flatten(0,1) # -> shape (num_legs*decimation, 3)
        pt_i_w, _ = math_utils.combine_frame_transforms(robot_pos_w[0,:].unsqueeze(0).expand(pt_i_b.shape), robot_orientation_w[0,:].unsqueeze(0).expand(pt_i_b.shape[0], 4), pt_i_b)
        
        marker_locations = pt_i_w
        self.my_visualizer['foot_traj'].visualize(marker_locations)


def define_markers(marker_type, param_dict) -> VisualizationMarkers:
    """Define markers with various different shapes.
    Args :
        - marker_type : 'sphere', 'arrow_x', ...
        - param_dict : dict of parameter for the given marker :
            'sphere' : {'radius':float, 'color':(float, float, float)}
            'arrow_x' : {'scale':(float, float, float), 'color':(float, float, float)}
    """ 
    markers={}

    if marker_type == 'sphere':
        markers['sphere'] = sim_utils.SphereCfg(
                radius=param_dict['radius'],
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=param_dict['color']),
            )
        
    if marker_type == 'arrow_x':
        markers['arrow_x'] = sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
            scale=param_dict['scale'],
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=param_dict['color']),
            )

    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers=markers
    )
    return VisualizationMarkers(marker_cfg)