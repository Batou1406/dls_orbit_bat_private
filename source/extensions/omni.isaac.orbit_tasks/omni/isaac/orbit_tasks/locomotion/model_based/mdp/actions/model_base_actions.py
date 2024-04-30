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

verbose_mb = False
verbose_loop = 40
vizualise_debug = {'foot': False, 'jacobian': True, 'foot_traj': True, 'lift-off': True, 'touch-down': True, 'GRF': True}
torch.set_printoptions(precision=2, linewidth=200, sci_mode=False)

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
        _number_predict_step: number of predicted touch down position (used by sampling controller, prior by RL)        Received from ModelBaseActionCfg
        controller (modelBaseController): controller instance that compute u from z                                     Received from ModelBaseActionCfg
        
        f     (torch.Tensor): Prior leg frequency                   of shape (batch_size, num_legs)
        d     (torch.Tensor): Prior stepping duty cycle             of shape (batch_size, num_legs)
        p_lw  (torch.Tensor): Prior foot pos. sequence              of shape (batch_size, num_legs, 3, time_horizon)
        F_lw  (torch.Tensor): Prior Ground Reac. Forces (GRF) seq.  of shape (batch_size, num_legs, 3, time_horizon)
        p_star_lw (t.Tensor): Optimizied foot pos sequence          of shape (batch_size, num_legs, 3, time_horizon)
        F_star_lw (t.Tensor): Opt. Ground Reac. Forces (GRF) seq.   of shape (batch_size, num_legs, 3, time_horizon)
        c_star (trch.Tensor): Optimizied foot contact sequence      of shape (batch_size, num_legs, time_horizon)
        pt_star_lw  (Tensor): Optimizied foot swing trajectory      of shape (batch_size, num_legs, 9, decimation)  (9 = pos, vel, acc)
        z            (tuple): Latent variable : z=(f,d,p,F)         of shape (...)
        u     (torch.Tensor): output joint torques                  of shape (batch_size, num_joints)
        jacobian_prev_lw (T): jacobian from prev. dt for jac_dot    of shape (batch_size, num_leg, 3, num_joints_per_leg) 
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

    # controller: model_base_controller.modelBaseController
    controller: model_base_controller.samplingController
    """Model base controller that compute u: output torques from z: latent variable""" 


    def __init__(self, cfg: actions_cfg.ModelBaseActionCfg, env: BaseEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        # Prevision Horizon and number of predicted step
        self._prevision_horizon = self.cfg.prevision_horizon
        self._number_predict_step = self.cfg.number_predict_step

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
        # self._foot_idx = self._asset.find_bodies(".*foot")[0]
        self._foot_idx = self._asset.find_bodies(".*FOOT")[0]
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
        self.f = 1*torch.ones(self.num_envs, self._num_legs, device=self.device)
        self.d = 0.55*torch.ones(self.num_envs, self._num_legs, device=self.device)
        self.p_lw = torch.zeros(self.num_envs, self._num_legs, 3, self._number_predict_step, device=self.device)
        self.F_lw = torch.zeros(self.num_envs, self._num_legs, 3, self._prevision_horizon, device=self.device)
        self.z = [self.f, self.d, self.p_lw, self.F_lw]

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)   

        # Model-based optimized latent variable
        self.p_star_lw = torch.zeros(self.num_envs, self._num_legs, 3, self._number_predict_step, device=self.device)
        self.F_star_lw = torch.zeros(self.num_envs, self._num_legs, 3, self._prevision_horizon, device=self.device)
        self.c_star = torch.ones(self.num_envs, self._num_legs, self._prevision_horizon, device=self.device)
        self.pt_star_lw = torch.zeros(self.num_envs, self._num_legs, 9, self._decimation, device=self.device)
        self.full_pt_lw = torch.zeros(self.num_envs, self._num_legs, 9, 22, device=self.device)

        # Control input u : joint torques
        self.u = torch.zeros(self.num_envs, self._num_joints, device=self.device)

        # Variable for intermediary computaion
        self.jacobian_prev_lw = self.get_reset_jacobian() # Jacobian is translation independant thus jacobian_w = jacobian_lw

        # Instance of control class. Gets Z and output u
        self.controller = cfg.controller
        self.controller.late_init(device=self.device, num_envs=self.num_envs, num_legs=self._num_legs, time_horizon=self._prevision_horizon, dt_out=self._decimation*self._env.physics_dt, decimation=self._decimation, dt_in=self._env.physics_dt, p_default_lw=self.get_reset_foot_position()) 

        if verbose_mb:
            self.my_visualizer = {}
            self.my_visualizer['foot'] = {'foot_swing' : define_markers('sphere', {'radius': 0.03, 'color': (1.0,1.0,0)}),
                                          'foot_stance': define_markers('sphere', {'radius': 0.03, 'color': (0.0,1.0,0)})}
            self.my_visualizer['jacobian'] = define_markers('arrow_x', {'scale':(0.1,0.1,1.0), 'color': (1.0,0,0)})
            self.my_visualizer['foot_traj'] = define_markers('sphere', {'radius': 0.01, 'color': (0.0,1.0,0.0)})
            self.my_visualizer['lift-off'] = define_markers('sphere', {'radius': 0.02, 'color': (0.678,1.0,0.184)})
            self.my_visualizer['touch-down'] = define_markers('sphere', {'radius': 0.02, 'color': (0.196,0.804,0.196)})
            self.my_visualizer['GRF']= define_markers('arrow_x', {'scale':(0.1,0.1,1.0), 'color': (0.196,0.804,0.196)})#'scale':(0.03,0.03,0.15), 


    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        # return sum(variable.shape[1:].numel() for variable in self.z) # shape[1:], return all the dimension exept dim0=batch_size
        ##>>>DEBUG
        return self.F_lw.shape[1:].numel()
        ##<<<DEBUG

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
        # self.f = 2*(self._processed_actions[:, :self._num_legs]).clamp(0,10) # 0:3 and clip frequency to valid range [0,20]
        # self.d = (self._processed_actions[:, self._num_legs:2*self._num_legs]).clamp(0.1,0.9) # 4:7 and clip leg duty cycle to valid range [0,1]
        # self.p_lw = 0.5*self._processed_actions[:, 2*self._num_legs:(2*self._num_legs + 3*self._num_legs*self._number_predict_step)].reshape([self.num_envs, self._num_legs, 3, self._number_predict_step])
        # self.F_lw = 50*self._processed_actions[:, (2*self._num_legs + 3*self._num_legs*self._number_predict_step):].reshape([self.num_envs, self._num_legs, 3, self._prevision_horizon]) #TODO change as scale parameter
        # self.z = [self.f, self.d, self.p_lw, self.F_lw]

        ##>>>DEBUG
        #self.f = 2    Doesn't change from default
        #self.d = 0.55 Doesn't change from default 
        self.p_lw = torch.tensor([[0.243, 0.138, 0],[0.243, -0.138, 0],[-0.236, 0.137, 0],[-0.236, -0.137, 0]], device=self.device).unsqueeze(0).expand(self.num_envs, -1, -1).unsqueeze(-1) 
        self.F_lw = self._processed_actions.reshape([self.num_envs, self._num_legs, 3, self._prevision_horizon])*10
        ##<<<DEBUG

        # Optimize the latent variable with the model base controller
        self.p_star_lw, self.F_star_lw, self.c_star, self.pt_star_lw, self.full_pt_lw = self.controller.optimize_latent_variable(f=self.f, d=self.d, p_lw=self.p_lw, F_lw=self.F_lw)

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
        p_lw, p_dot_lw, q_dot, jacobian_lw, jacobian_dot_lw, mass_matrix, h = self.get_robot_state() 

        # Extract optimal variables
        F0_star_lw = self.F_star_lw[:,:,:,0]
        c0_star = self.c_star[:,:,0]
        pt_i_star_lw = self.pt_star_lw[:,:,:,self.inner_loop]
        self.inner_loop += 1            

        # Use model controller to compute the torques from the latent variable
        # Transform the shape from (batch_size, num_legs, num_joints_per_leg) to (batch_size, num_joints) # Permute and reshape to have the joint in right order [0,4,8][1,5,...] to [0,1,2,...]
        self.u = (self.controller.compute_control_output(F0_star_lw=F0_star_lw, c0_star=c0_star, pt_i_star_lw=pt_i_star_lw, p_lw=p_lw, p_dot_lw=p_dot_lw, q_dot=q_dot,
                                                          jacobian_lw=jacobian_lw, jacobian_dot_lw=jacobian_dot_lw, mass_matrix=mass_matrix, h=h)).permute(0,2,1).reshape(self.num_envs,self._num_joints)

        # Apply the computed torques
        self._asset.set_joint_effort_target(self.u)#, joint_ids=self._joint_ids) # Do use joint_ids to speed up the process

        # Debug
        if verbose_mb:
            self.debug_apply_action(p_lw, p_dot_lw, q_dot, jacobian_lw, jacobian_dot_lw, mass_matrix, h, F0_star_lw, c0_star, pt_i_star_lw)


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

        # Reset jacobian_prev_lw : Jacobian are not initialise before env. step, thus can get them from robot state.
        self.jacobian_prev_lw[env_ids,:,:,:] = self.get_reset_jacobian()[env_ids,:,:,:]

        # Reset the model base controller : expect default foot position in local world frame 
        self.controller.reset(env_ids, self.get_reset_foot_position())


    def get_robot_state(self):
        """ Retrieve the Robot states from the simulator

        Return :
            - p_lw (trch.Tensor): Feet Position  (latest from sim) in _lw   of shape(batch_size, num_legs, 3)
            - p_dot_lw  (Tensor): Feet velocity  (latest from sim) in _lw   of shape(batch_size, num_legs, 3)
            - q_dot (tch.Tensor): Joint velocity (latest from sim)          of shape(batch_size, num_legs, num_joints_per_leg)
            - jacobian_lw (Tsor): Jacobian -> joint frame to local world fr of shape(batch_size, num_legs, 3, num_joints_per_leg)
            - jacobian_dot_lw   : Jacobian derivative (forward euler)in _lw of shape(batch_size, num_legs, 3, num_joints_per_leg)
            - mass_matrix (Tsor): Mass Matrix in joint space                of shape(batch_size, num_legs, num_joints_per_leg, num_joints_per_leg)
            - h   (torch.Tensor): C(q,q_dot) + G(q) (corr. and grav F.)     of shape(batch_size, num_legs, num_joints_per_leg)
        """

        # Retrieve robot base position and orientation in order to compute world->base frame transformation
        # robot_pos_w = self._asset.data.root_pos_w       # shape (batch_size, 3) (xyz)
        # robot_orientation_w = self._asset.data.root_quat_w # shape (batch_size, 4) (quaternions)
        # robot_vel_w = self._asset.data.root_lin_vel_w
        # robot_ang_vel_w = self._asset.data.root_ang_vel_w

        # Retrieve Feet position in world frame : [num_instances, num_bodies, 3] select right indexes to get 
        # shape(batch_size, num_legs, 3) : position is translation depend, thus : pos_lw = pos_w - env.scene.env_origin
        p_w = self._asset.data.body_pos_w[:, self._foot_idx,:]
        p_lw = p_w - self._env.scene.env_origins.unsqueeze(1).expand(p_w.shape)

        # Transformation to get feet position in body frame
        # p_orientation_w = self._asset.data.body_quat_w[:, self._foot_idx,:]
        # p_b_0, _ = math_utils.subtract_frame_transforms(robot_pos_w, robot_orientation_w, p_w[:,0,:], p_orientation_w[:,0,:])
        # p_b_1, _ = math_utils.subtract_frame_transforms(robot_pos_w, robot_orientation_w, p_w[:,1,:], p_orientation_w[:,1,:])
        # p_b_2, _ = math_utils.subtract_frame_transforms(robot_pos_w, robot_orientation_w, p_w[:,2,:], p_orientation_w[:,2,:])
        # p_b_3, _ = math_utils.subtract_frame_transforms(robot_pos_w, robot_orientation_w, p_w[:,3,:], p_orientation_w[:,3,:])
        # p_b = torch.cat((p_b_0.unsqueeze(1), p_b_1.unsqueeze(1), p_b_2.unsqueeze(1), p_b_3.unsqueeze(1)), dim=1)

        # Retrieve Feet velocity in world frame : [num_instances, num_bodies, 3] select right indexes to get 
        # shape(batch_size, num_legs, 3) : Velocity is translation independant, thus p_dot_w = p_dot_lw
        p_dot_lw = self._asset.data.body_lin_vel_w[:, self._foot_idx,:]

        # Transformation to get feet velocity in body frame
        # p_dot_b_0, _ = math_utils.subtract_frame_transforms(robot_vel_w, robot_orientation_w, p_dot_w[:,0,:], p_orientation_w[:,0,:])
        # p_dot_b_1, _ = math_utils.subtract_frame_transforms(robot_vel_w, robot_orientation_w, p_dot_w[:,1,:], p_orientation_w[:,1,:])
        # p_dot_b_2, _ = math_utils.subtract_frame_transforms(robot_vel_w, robot_orientation_w, p_dot_w[:,2,:], p_orientation_w[:,2,:])
        # p_dot_b_3, _ = math_utils.subtract_frame_transforms(robot_vel_w, robot_orientation_w, p_dot_w[:,3,:], p_orientation_w[:,3,:])
        # p_dot_b = torch.cat((p_dot_b_0.unsqueeze(1), p_dot_b_1.unsqueeze(1), p_dot_b_2.unsqueeze(1), p_dot_b_3.unsqueeze(1)), dim=1)

        # Retrieve Joint velocities [num_instances, num_joints] -> reorganise the view and permute to get the
        # shape(batch_size, num_legs, num_joints_per_leg) : This is in joint space, no transformation required
        q_dot = self._asset.data.joint_vel.view(-1,self._num_joints_per_leg,self._num_legs).permute(0,2,1)

        # Retrieve Jacobian from sim  shape(batch_size, num_legs, 3, num_joints_per_leg) -> see method for implementation
        # Jacobian is translation independant thus jacobian_lw = jacobian_w
        jacobian_lw, jacobian_b = self.get_jacobian()

        # Compute jacobian derivative, using forward euler. shape(batch_size, num_legs, 3, num_joints_per_leg)
        jacobian_dot_lw = ((jacobian_lw - self.jacobian_prev_lw) / self._env.physics_dt)

        # Save jacobian for next iteration : required to compute jacobian derivative shape(batch_size, num_legs, 3, num_joints_per_leg)
        self.jacobian_prev_lw = jacobian_lw
        
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

        return p_lw, p_dot_lw, q_dot, jacobian_lw, jacobian_dot_lw, mass_matrix, h
    

    def get_jacobian(self) -> tuple[torch.Tensor, torch.Tensor]:
        """ Return the Jacobian that link the end-effector (ie. the foot) velocity to the joint velocity
        The jacobian are computed in world frame and base frame

        Defined as a function to avoid code duplicate (used is get_robot_state() and reset())

        Returns :
            - jacobian_w (Tensor): Jacobian in the world frame          of shape (batch_size, num_legs, 3, num_joints_per_leg)
            - jacobian_b (Tensor): Jacobian in the base frame           of shape (batch_size, num_legs, 3, num_joints_per_leg)            
        """
        # Retrieve Jacobian from sim
        # shape(batch_size, num_legs, 3, num_joints_per_leg)
        # Intermediary step : extract feet jacobian [batch_size, num_bodies=17, 6, num_joints+6=18] -> [..., 4, 3, 18]
        # Shift from 6 due to an offset. This is due to how the model is define I think
        jacobian_feet_full_w = self._asset.root_physx_view.get_jacobians()[:, self._foot_idx, :3, :]
        jacobian_w = torch.cat((jacobian_feet_full_w[:, 0, :, 6+np.asarray(self._joints_idx[0])].unsqueeze(1),
                                jacobian_feet_full_w[:, 1, :, 6+np.asarray(self._joints_idx[1])].unsqueeze(1),
                                jacobian_feet_full_w[:, 2, :, 6+np.asarray(self._joints_idx[2])].unsqueeze(1),
                                jacobian_feet_full_w[:, 3, :, 6+np.asarray(self._joints_idx[3])].unsqueeze(1)), dim=1)

        # Retrieve the robot base orientation in the world frame as quaternions : shape(batch_size, 4)
        quat_robot_base_w = self._asset.data.root_quat_w 

        # From the quaternions compute the rotation matrix that rotates from world frame w to base frame b : shape(batch_size, 3,3)
        R_b_to_w = math_utils.matrix_from_quat(quat_robot_base_w)
        R_w_to_b =  R_b_to_w.transpose(1,2) # Given the convention used, we get R_b_to_w from the quaternions, thus one need to transpose it to have R_w_to_b

        # Finally, rotate the jacobian from world frame (fixed) to base frame (attached at the robot's base)
        jacobian_b = torch.matmul(R_w_to_b.unsqueeze(1), jacobian_w) # (batch, 1, 3, 3) * (batch, legs, 3, 3) -> (batch, legs, 3, 3)

        return jacobian_w, jacobian_b
    

    def get_reset_foot_position(self) -> torch.Tensor:
        """ Return The default position of the robot's feet. this is the position when the states are reseted
        TODO Now, this is hardcoded for Aliengo (given a joint default position) -> Should get this from simulation or forward kinematics

        Return :
            - p_lw   (torch.Tensor): Default Feet position after reset in local world frame    of shape(batch_size, num_legs, 3)
        """
        p_b = torch.tensor([[0.243, 0.138, -0.325],[0.243, -0.138, -0.325],[-0.236, 0.137, -0.326],[-0.236, -0.137, -0.326]], device=self.device).unsqueeze(0).expand(self.num_envs, -1, -1) 
        
        # Retrieve the robot base orientation in the world frame as quaternions : shape(batch_size, 4)
        quat_robot_base_w = self._asset.data.root_quat_w 

        # From the quaternions compute the rotation matrix that rotates from base frame b to world frame w : shape(batch_size, 3,3)
        R_b_to_w = math_utils.matrix_from_quat(quat_robot_base_w)

        # Rotate p from base to world frame (no translation yet)
        p_rotated = torch.matmul(R_b_to_w, p_b.permute(0,2,1)).permute(0,2,1)
        
        # Compute translation from base frame to local world frame
        # Transform p_b into p_lw b appling final translation
        p_lw = (self._asset.data.root_pos_w.unsqueeze(1) - self._env.scene.env_origins.unsqueeze(1)) + p_rotated 

        return p_lw


    def get_reset_jacobian(self) -> torch.Tensor:
        """ Return The default Jacobian that link joint velocities to end-effector (ie. feet) velocities. 
        this is the Jacobian when the states are reseted
        TODO Now, this is hardcoded for Aliengo (given a joint default position)

        Return :
            - jacobian_w (Tensor): Default Jacobian after reset in world frame of shape(batch_size, num_legs, 3, num_joints_per_leg)
        """
        jacobian_default_b = torch.tensor([[0, -0.311, -0.155],[0.311, 0, 0],[0.083, 0, -0.196]], device=self.device).unsqueeze(0).unsqueeze(0).expand(self.num_envs, self._num_legs, -1, -1)

        # Retrieve the robot base orientation in the world frame as quaternions : shape(batch_size, 4)
        quat_robot_base_w = self._asset.data.root_quat_w 

        # From the quaternions compute the rotation matrix that rotates from base frame b to world frame w : shape(batch_size, 3,3)
        R_b_to_w = math_utils.matrix_from_quat(quat_robot_base_w)

        # Finally, rotate the jacobian from base to world frame : shape(batch_size, num_legs, 3, num_joints_per_leg)
        jacobian_w = torch.matmul(R_b_to_w.unsqueeze(1), jacobian_default_b) # (batch, 1, 3, 3) * (batch, legs, 3, joints_per_leg) -> (batch, legs, 3, joints_per_leg)
        
        return jacobian_w

#-------------------------------------------------- Helpers ------------------------------------------------------------
    def debug_apply_action(self, p_lw, p_dot_lw, q_dot, jacobian_lw, jacobian_dot_lw, mass_matrix, h, F0_star_lw, c0_star, pt_i_star_lw):
        global verbose_loop

        # --- Print --- 
        verbose_loop+=1
        if verbose_loop>=40:
            verbose_loop=0
            print('\nContact sequence : ', c0_star[0,...].flatten())
            print('  Leg  frequency : ', self.f[0,...].flatten())
            print('   duty   cycle  : ', self.d[0,...].flatten())
            print('Touch-down pos   : ', self.p_lw[0,0,:,0])
            print('Foot traj shape  : ', self.pt_star_lw.shape)
            print('Foot traj : ', self.pt_star_lw[0,0,:3,:])
            print('Foot Force :', self.F_star_lw[0,:,:])
            if (self.F_lw != self.F_star_lw).any():
                assert ValueError('F value don\'t match...')

        # --- Visualize foot position ---
        if vizualise_debug['foot']:
            # p_lw_ = p_lw.clone().detach()
            p_w = p_lw + self._env.scene.env_origins.unsqueeze(1)

            # Transformation if p in base frame
            # robot_pos_w = self._asset.data.root_pos_w
            # robot_orientation_w = self._asset.data.root_quat_w
            # p_orientation_w = self._asset.data.body_quat_w[:, self._foot_idx,:]
            # p_w_0, _ = math_utils.combine_frame_transforms(robot_pos_w, robot_orientation_w, p_b[:,0,:])
            # p_w_1, _ = math_utils.combine_frame_transforms(robot_pos_w, robot_orientation_w, p_b[:,1,:])
            # p_w_2, _ = math_utils.combine_frame_transforms(robot_pos_w, robot_orientation_w, p_b[:,2,:])
            # p_w_3, _ = math_utils.combine_frame_transforms(robot_pos_w, robot_orientation_w, p_b[:,3,:])
            # p_w = torch.cat((p_w_0.unsqueeze(1), p_w_1.unsqueeze(1), p_w_2.unsqueeze(1), p_w_3.unsqueeze(1)), dim=1)
            stance = self.c_star[:,:,0]

            marker_locations_stance = (p_w[:,:,:] * stance.unsqueeze(-1)).flatten(0,1)
            marker_locations_swing = (p_w[:,:,:] * (~stance.unsqueeze(-1))).flatten(0,1)
            self.my_visualizer['foot']['foot_stance'].visualize(marker_locations_stance)
            self.my_visualizer['foot']['foot_swing'].visualize(marker_locations_swing)

        # --- Visualise jacobian ---
        if vizualise_debug['jacobian']:
            # Get point where to draw the jacobian
            joint_pos_w = self._asset.data.body_pos_w[0,1+np.asarray(self._joint_ids),:] # shape (num_joints, 3) # +1 Not clean a way to change from joint to body view but it works
            marker_locations = joint_pos_w

            # From jacobian, retrieve orientation (angle and axis representation)
            jacobian_temp_lw = jacobian_lw[0,:,:,:].clone().detach()
            jacobian_temp_T_lw = jacobian_temp_lw.permute(0,2,1) # shape (num_legs, 3, num_joints_per_leg) -> (num_legs, num_joints_per_leg, 3)
            jacobian_temp_T_lw = jacobian_temp_T_lw.permute(1,0,2).flatten(0,1) # shape (num_joints, 3) # permute to have index in right order (ie. 0,1,2,... and not 0,4,8,1,...)
            normalize_jacobian_temp_T_lw = torch.nn.functional.normalize(jacobian_temp_T_lw, p=2, dim=1) # Transform jacobian to unit vectors
            # angle : u dot v = cos(angle) -> angle = acos(u*v) : for unit vector # Need to take the opposite angle in order to make appropriate rotation
            angle = -torch.acos(torch.tensordot(normalize_jacobian_temp_T_lw, torch.tensor([1.0,0.0,0.0], device=self.device), dims=1)) # shape(num_joints, 3) -> (num_joints)
            # Axis : Cross product between u^v (for unit vectors)
            axis = torch.cross(normalize_jacobian_temp_T_lw, torch.tensor([1.0,0.0,0.0], device=self.device).unsqueeze(0).expand(normalize_jacobian_temp_T_lw.shape))
            marker_orientations = quat_from_angle_axis(angle=angle, axis=axis)

            # Scale Jacobian arrow
            scale = torch.linalg.vector_norm(jacobian_temp_T_lw, dim=1).unsqueeze(-1).expand(jacobian_temp_T_lw.shape) / 1.8

            # The arrow point is define at its center. So to avoid having the arrow in the middle of the joint, we translate it by a factor along its pointing direction
            translation = scale*torch.tensor([0.25, 0.0, 0.0], device=self.device).unsqueeze(0).expand(joint_pos_w.shape)
            translation = math_utils.transform_points(points=translation.unsqueeze(1), pos=joint_pos_w, quat=marker_orientations).squeeze(1)
            marker_locations = translation

            self.my_visualizer['jacobian'].visualize(translations=marker_locations, orientations=marker_orientations, scales=scale)

        # --- Visualize foot trajectory ---
        if vizualise_debug['foot_traj']:
            pt_i_lw_= self.pt_star_lw.clone().detach()  # shape (batch_size, num_legs, 9, decimation) (9=px,py,pz,vx,vy,vz,ax,ay,az)
            pt_i_lw_ = pt_i_lw_[0,:,0:3,:] # -> shape (num_legs, 3, decimation)
            pt_i_lw_ = pt_i_lw_.permute(0,2,1).flatten(0,1) # -> shape (num_legs*decimation, 3)

            full_pt_lw_ = self.full_pt_lw.clone().detach()  # shape (batch_size, num_legs, 9, 22) (9=px,py,pz,vx,vy,vz,ax,ay,az)
            full_pt_lw_ = full_pt_lw_[:,:,0:3,:] # -> shape (batch_size, num_legs, 3, 22)
            full_pt_lw_ = full_pt_lw_.permute(0,1,3,2).flatten(1,2) # -> shape (batch_size, num_legs, 22, 3) -> (batch_size, num_legs*22, 3)
            full_pt_w_ = full_pt_lw_ + self._env.scene.env_origins.unsqueeze(1) 
            full_pt_w_ = full_pt_w_.flatten(0,1) # -> shape (batch_size*num_legs*22, 3)
            marker_locations = full_pt_w_

            # If in base frame
            # robot_pos_w = self._asset.data.root_pos_w
            # robot_orientation_w = self._asset.data.root_quat_w
            # pt_i_w, _ = math_utils.combine_frame_transforms(robot_pos_w[0,:].unsqueeze(0).expand(pt_i_b.shape), robot_orientation_w[0,:].unsqueeze(0).expand(pt_i_b.shape[0], 4), pt_i_b)
            
            pt_i_w = pt_i_lw_ + self._env.scene.env_origins.unsqueeze(1)
            
            # Visualize the traj only if it is used (ie. the foot is in swing -> c==0)
            marker_indices = ~((c0_star.unsqueeze(-1).expand(self.num_envs,self._num_legs,22)).flatten(1,2).flatten(0,1))
            # marker_locations = pt_i_w[0,...]        

            self.my_visualizer['foot_traj'].visualize(translations=marker_locations, marker_indices=marker_indices)

        # --- Visualize Lift-off position ---
        if vizualise_debug['lift-off']:
            p0_lw = self.controller.p0_lw.clone().detach()
            p0_w = p0_lw + self._env.scene.env_origins.unsqueeze(1)
            # If in base frame
            # p_b = self.controller.p0_lw.clone().detach()
            # robot_pos_w = self._asset.data.root_pos_w
            # robot_orientation_w = self._asset.data.root_quat_w
            # p_orientation_w = self._asset.data.body_quat_w[:, self._foot_idx,:]
            # p_w_0, _ = math_utils.combine_frame_transforms(robot_pos_w, robot_orientation_w, p_b[:,0,:])
            # p_w_1, _ = math_utils.combine_frame_transforms(robot_pos_w, robot_orientation_w, p_b[:,1,:])
            # p_w_2, _ = math_utils.combine_frame_transforms(robot_pos_w, robot_orientation_w, p_b[:,2,:])
            # p_w_3, _ = math_utils.combine_frame_transforms(robot_pos_w, robot_orientation_w, p_b[:,3,:])
            # p_w = torch.cat((p_w_0.unsqueeze(1), p_w_1.unsqueeze(1), p_w_2.unsqueeze(1), p_w_3.unsqueeze(1)), dim=1)

            # marker_locations = p0_w[0,:,:]
            marker_locations = p0_w.flatten(0,1)
            self.my_visualizer['lift-off'].visualize(marker_locations)

        #  --- Visualize touch-down position ---
        if vizualise_debug['touch-down']:
            p2_lw = self.p_lw[:,:,:,0].clone().detach()
            p2_w = p2_lw + self._env.scene.env_origins.unsqueeze(1)
            # p2_w[:,:,2] = 0.05 #small height to make them more visible
            marker_locations = p2_w[0,:,:]
            marker_locations = p2_w.flatten(0,1)
            self.my_visualizer['touch-down'].visualize(marker_locations)

        # --- Visualize Ground Reactions Forces (GRF) ---
        if vizualise_debug['GRF']:

            # GRF location are the feet position
            p3_w = p_lw + self._env.scene.env_origins.unsqueeze(1) # (batch, num_legs, 3)
            marker_locations = p3_w[0,...]

            # From GRF, retrieve orientation (angle and axis representation)
            F = F0_star_lw.clone().detach()[0,:] # (batch, num_legs, 3) -> (num_legs, 3)
            normalize_F = torch.nn.functional.normalize(F, p=2, dim=1) # Transform jacobian to unit vectors
            # angle : u dot v = cos(angle) -> angle = acos(u*v) : for unit vector # Need to take the opposite angle in order to make appropriate rotation
            angle = -torch.acos(torch.tensordot(normalize_F, torch.tensor([1.0,0.0,0.0], device=self.device), dims=1)) # shape(num_joints, 3) -> (num_joints)
            # Axis : Cross product between u^v (for unit vectors)
            axis = torch.cross(normalize_F, torch.tensor([1.0,0.0,0.0], device=self.device).unsqueeze(0).expand(normalize_F.shape))
            marker_orientations = quat_from_angle_axis(angle=angle, axis=axis)

            # Scale GRF
            scale = torch.linalg.vector_norm(F, dim=1).unsqueeze(-1).expand(F.shape) / 150

            # The arrow point is define at its center. So to avoid having the arrow in the middle of the feet, we translate it by a factor along its pointing direction
            translation = scale*torch.tensor([0.25, 0.0, 0.0], device=self.device).unsqueeze(0).expand(marker_locations.shape)
            translation = math_utils.transform_points(points=translation.unsqueeze(1), pos=marker_locations, quat=marker_orientations).squeeze(1)
            marker_locations = translation

            # Visualize the force only if it is used (ie. the foot is in contact -> c==1)
            marker_indices = c0_star[0,...]

            self.my_visualizer['GRF'].visualize(translations=marker_locations, orientations=marker_orientations, scales=scale, marker_indices=marker_indices)



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