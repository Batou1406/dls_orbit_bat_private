# Copyright (c) 2022-2024, The lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# ./lab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-Model-Based-Base-Aliengo-v0  --num_envs 32

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import carb

import omni.isaac.lab.utils.math as math_utils
import omni.isaac.lab.utils.string as string_utils
from omni.isaac.lab.assets.articulation import Articulation
from omni.isaac.lab.managers.action_manager import ActionTerm

import torch.distributions.constraints

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from typing import Literal

from . import actions_cfg

from . import model_base_controller #import modelBaseController, samplingController

import numpy as np

## >>>Visualization
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.utils.math import quat_from_angle_axis

import omni.kit.app
import weakref
## <<<Visualization


verbose_mb = False
verbose_loop = 40
vizualise_debug = {'foot': False, 'jacobian': False, 'foot_traj': False, 'lift-off': False, 'touch-down': False, 'GRF': False, 'touch-down polygon': False}
torch.set_printoptions(precision=4, linewidth=200, sci_mode=False)
if verbose_mb: import omni.isaac.debug_draw._debug_draw as omni_debug_draw
# import omni.isaac.debug_draw._debug_draw as omni_debug_draw

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
        foot_idx      (list): List of index of the feet
        _num_legs      (int): Number of legs of the robot  : useful for dimension definition
        _num_joints_per_leg : Number of joints per leg     : useful for dimension definition
        _joint_idx    (list): List of list of joints [FL_joints=[FL_Hip,...], FR_joints, ...]
        _hip_idx      (list): List of index (in body view) with the index of the robot's hip
        _decimation    (int): Inner Loop steps per outer loop steps
        _F_param            : Last dimension of F, can be discrete (ie. =1) or some kind of parametrisation             Received from ModelBaseActionCfg
        _p_param            : Last dimension of p, can be discrete (ie. =1) or some kind of parametrisation             Received from ModelBaseActionCfg
        controller (modelBaseController): controller instance that compute u from z                                     Received from ModelBaseActionCfg
        
        f_raw (torch.Tensor): Prior leg frequency                 (raw)     of shape (batch_size, num_legs)
        d_raw (torch.Tensor): Prior stepping duty cycle           (raw)     of shape (batch_size, num_legs)
        p_raw (torch.Tensor): Prior foot pos. sequence            (raw)     of shape (batch_size, num_legs, 2, p_param)
        F_raw (torch.Tensor): Prior Gnd Reac. Forces seq.         (raw)     of shape (batch_size, num_legs, 3, F_param)       

        f     (torch.Tensor): Prior leg frequency              (normalized) of shape (batch_size, num_legs)
        d     (torch.Tensor): Prior stepping duty cycle        (normalized) of shape (batch_size, num_legs)
        p_norm (trch.Tensor): Prior foot pos. seq. hip center  (normalized) of shape (batch_size, num_legs, 2, p_param)
        F_norm (trch.Tensor): Prior Gnd Reac. Forces seq.      (normalized) of shape (batch_size, num_legs, 3, F_param) 

        f_prev  (tch.Tensor): Previous Prior leg frequency     (normalized) of shape (batch_size, num_legs)
        d_prev  (tch.Tensor): Previous Prior step duty cycle   (normalized) of shape (batch_size, num_legs)
        p_norm_prev (Tensor): Previous Prior f p s hip center  (normalized) of shape (batch_size, num_legs, 2, p_param)
        F_norm_prev (Tensor): Previous Prior Gnd Reac. F. seq. (normalized) of shape (batch_size, num_legs, 3, F_param) 

        p_lw  (torch.Tensor): Prior foot pos. sequence                      of shape (batch_size, num_legs, 3, p_param)
        F_lw  (torch.Tensor): Prior Ground Reac. Forces (GRF) seq.          of shape (batch_size, num_legs, 3, F_param)

        f_len               : To ease the actions extraction
        d_len               : To ease the actions extraction
        p_len               : To ease the actions extraction
        F_len               : To ease the actions extraction
        
        p0_star_lw  (Tensor): Optimizied foot pos sequence                  of shape (batch_size, num_legs, 3, p_param)
        F0_star_lw  (Tensor): Opt. Ground Reac. Forces (GRF) seq.           of shape (batch_size, num_legs, 3, F_param)
        c0_star (tch.Tensor): Optimizied foot contact sequence              of shape (batch_size, num_legs, 1)
        pt_star_lw  (Tensor): Optimizied foot swing trajectory              of shape (batch_size, num_legs, 9, decimation)  (9 = pos, vel, acc)
        
        z            (tuple): Latent variable : z=(f,d,p,F)                 of shape (...)
        u     (torch.Tensor): output joint torques                          of shape (batch_size, num_joints)
        
        jacobian_prev_lw (T): jacobian from prev. dt for jac_dot            of shape (batch_size, num_leg, 3, num_joints_per_leg) 
        inner_loop     (int): Counter of inner loop wrt. outer loop
        hip0_pos_lw (Tensor): Hip position when leg lift-off                of shape (batch_size, num_legs, 3)
        hip0_yaw_quat_lw (T): Robot yaw when leg lift-off                   of shape (batch_size, num_legs, 4)
        heightScan  (HeightScanCfg): Config class with : resolution, size, hip_offset, scale_y, max_x, max_y

    Method :
        reset(env_ids: Sequence[int] | None = None) -> None:                                                            Inherited from ManagerTermBaseCfg (not implemented)
        __call__(*args) -> Any                                                                                          Inherited from ManagerTermBaseCfg (not implemented)
        process_actions(actions: torch.Tensor)                                                                          Inherited from ActionTerm (not implemented)
        apply_actions()                                                                                                 Inherited from ActionTerm (not implemented)
        get_robot_state() -> tuple(torch.Tensor)
        get_jacobian() -> torch.Tensor
        get_reset_foot_position() -> torch.Tensors
        normalize_actions()
        height_scan_index_from_pos_b()
        debug_apply_action()
    """

    cfg: actions_cfg.ModelBaseActionCfg
    """The configuration of the action term."""

    _asset: Articulation
    """The articulation asset on which the action term is applied. Asset is defined in ActionTerm base clase, here just the type is redefined"""

    _env : ManagerBasedRLEnv
    """To enable type hinting"""

    controller: model_base_controller.modelBaseController
    # controller: model_base_controller.samplingController
    """Model base controller that compute u: output torques from z: latent variable""" 


    def __init__(self, cfg: actions_cfg.ModelBaseActionCfg, env: ManagerBasedEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        # Define F and p size given how it is parametrised
        if self.cfg.optimizerCfg is not None:
            # Wether to bypass RL actions with some static gait
            self.debug_apply_action_status = self.cfg.optimizerCfg.debug_apply_action

            # Parametrized Action 
            if self.cfg.optimizerCfg.parametrization_F == 'cubic_spline':
                self._F_param = 4 
            # Discrete Action : Multiple actions - Multiple time step (one action per time step)
            elif self.cfg.optimizerCfg.parametrization_F == 'discrete':
                self._F_param = self.cfg.optimizerCfg.prevision_horizon
            else : raise NotImplementedError('Provided F parametrisation not implemented')

            # Parametrized Action 
            if self.cfg.optimizerCfg.parametrization_p == 'cubic_spline':
                self._p_param = 4
            # Discrete Action : Multiple actions - Multiple time step (one action per time step)
            elif self.cfg.optimizerCfg.parametrization_p == 'discrete':
                self._p_param = self.cfg.optimizerCfg.prevision_horizon
            else : raise NotImplementedError('Provided P parametrisation not implemented')
        else :
            # RL Actions applied 
            self.debug_apply_action_status = None 

            # Discrete Action : One action - One time step
            self._F_param = 1
            self._p_param = 1
  
        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)  # joint_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self._num_joints = len(self._joint_ids)                                             # joint_names = ['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint', 'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint', 'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint']
        # log the resolved joint names for debugging
        carb.log_info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )

        # Retrieve series of information usefull for computation and generalisation
        # Feet Index in body, list [13, 14, 15, 16]
        self.foot_idx = self._asset.find_bodies(".*foot")[0]
        self._num_legs = len(self.foot_idx)
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

        
        # raw RL output
        self.f_raw  = 1.0*torch.ones( self.num_envs, self._num_legs,                   device=self.device) 
        self.d_raw  = 0.6*torch.ones( self.num_envs, self._num_legs,                   device=self.device)
        self.p_raw  =     torch.zeros(self.num_envs, self._num_legs, 2, self._p_param, device=self.device)
        self.F_raw  =     torch.zeros(self.num_envs, self._num_legs, 3, self._F_param, device=self.device)

        # Normalized RL output
        self.f      = self.f_raw.clone().detach() 
        self.d      = self.d_raw.clone().detach()
        self.p_norm = self.p_raw.clone().detach() 
        self.F_norm = self.F_raw.clone().detach() 

        # Previous output - used for derivative computation
        self.f_prev      = self.f_raw.clone().detach() 
        self.d_prev      = self.d_raw.clone().detach() 
        self.p_norm_prev = self.p_raw.clone().detach() 
        self.F_norm_prev = self.F_raw.clone().detach() 

        # Normalized and transformed to frame RL output
        self.p_lw   =     torch.zeros(self.num_envs, self._num_legs, 3, self._p_param, device=self.device) 
        self.F_lw   =     torch.zeros(self.num_envs, self._num_legs, 3, self._F_param, device=self.device)

        # For ease of reshaping variables
        self.f_len = self.f_raw.shape[1:].numel()
        self.d_len = self.d_raw.shape[1:].numel()
        self.p_len = self.p_raw.shape[1:].numel()
        self.F_len = self.F_raw.shape[1:].numel()

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)   

        # Model-based optimized latent variable
        self.f_star     = torch.zeros(self.num_envs, self._num_legs,                     device=self.device)
        self.d_star     = torch.zeros(self.num_envs, self._num_legs,                     device=self.device)
        self.c0_star    = torch.ones( self.num_envs, self._num_legs,                     device=self.device)
        self.p0_star_lw = torch.zeros(self.num_envs, self._num_legs, 3,                  device=self.device)
        self.F0_star_lw = torch.zeros(self.num_envs, self._num_legs, 3,                  device=self.device)
        self.pt_star_lw = torch.zeros(self.num_envs, self._num_legs, 9, self._decimation,device=self.device)
        self.full_pt_lw = torch.zeros(self.num_envs, self._num_legs, 9, 22,              device=self.device)  # Used for plotting only

        # Control input u : joint torques
        self.u = torch.zeros(self.num_envs, self._num_joints, device=self.device)

        # Intermediary variable - Hip variable for p centering
        self._hip_idx = self._asset.find_bodies(".*thigh")[0]
        self.hip0_pos_lw = torch.zeros(self.num_envs, self._num_legs, 3, device=self.device)
        self.hip0_yaw_quat_lw = torch.zeros(self.num_envs, self._num_legs, 4, device=self.device)
        self.hip0_yaw_quat_lw[:,:,0] = 1 # initialize correctly the quaternion to avoid NaN in first operation

        # Variable for intermediary computaion
        self.jacobian_prev_lw = self.get_reset_jacobian() # Jacobian is translation independant thus jacobian_w = jacobian_lw

        # variable for the height_scanner
        if self.cfg.height_scan_available :
            self.heightScan = actions_cfg.ModelBaseActionCfg.HeightScanCfg(
                resolution = self._env.scene["height_scanner"].cfg.pattern_cfg.resolution,
                size = self._env.scene["height_scanner"].cfg.pattern_cfg.size,
                # hip_offset = self._asset.data.body_pos_w[0,  self._hip_idx, :2].unsqueeze(0) # shape(1, num_legs, 2=xy) # Sadly this isn't initialized at init time
                hip_offset = torch.tensor([[0.24,0.05],[0.24,-0.05],[-0.24,0.05],[-0.24,-0.05]], device=self.device).reshape(1,4,2), # So it's hardcoded for aliengo for now... TODO don't hardcode
            )
            self.heightScan.scale_y = int(self.heightScan.size[0]/self.heightScan.resolution) + 1
            self.heightScan.max_x = int(self.heightScan.size[0]//self.heightScan.resolution)
            self.heightScan.max_y = int(self.heightScan.size[1]//self.heightScan.resolution)

        # Instance of control class. Gets Z and output u
        if cfg.controller == model_base_controller.modelBaseController:
            self.controller = cfg.controller(
                verbose_md=verbose_mb, device=self.device, num_envs=self.num_envs, num_legs=self._num_legs,
                dt_out=self._decimation*self._env.physics_dt, decimation=self._decimation, dt_in=self._env.physics_dt,
                p_default_lw=self.get_reset_foot_position(), step_height=cfg.footTrajectoryCfg.step_height,
                foot_offset=cfg.footTrajectoryCfg.foot_offset, swing_ctrl_pos_gain_fb=self.cfg.swingControllerCfg.swing_ctrl_pos_gain_fb , 
                swing_ctrl_vel_gain_fb=self.cfg.swingControllerCfg.swing_ctrl_vel_gain_fb
            )
        elif cfg.controller == model_base_controller.samplingController:
            self.controller = cfg.controller(
                verbose_md=verbose_mb, device=self.device, num_envs=self.num_envs, num_legs=self._num_legs,
                dt_out=self._decimation*self._env.physics_dt, decimation=self._decimation, dt_in=self._env.physics_dt,
                p_default_lw=self.get_reset_foot_position(), step_height=cfg.footTrajectoryCfg.step_height,
                foot_offset=cfg.footTrajectoryCfg.foot_offset, swing_ctrl_pos_gain_fb=self.cfg.swingControllerCfg.swing_ctrl_pos_gain_fb , 
                swing_ctrl_vel_gain_fb=self.cfg.swingControllerCfg.swing_ctrl_vel_gain_fb,
                optimizerCfg=cfg.optimizerCfg
            ) 

        if verbose_mb:
            self.my_visualizer = {}
            self.my_visualizer['foot'] = {'foot_swing' : define_markers('sphere', {'radius': 0.03, 'color': (1.0,1.0,0)}),
                                          'foot_stance': define_markers('sphere', {'radius': 0.03, 'color': (0.0,1.0,0)})}
            self.my_visualizer['jacobian'] = define_markers('arrow_x', {'scale':(0.1,0.1,1.0), 'color': (1.0,0,0)})
            self.my_visualizer['foot_traj'] = define_markers('sphere', {'radius': 0.01, 'color': (0.0,1.0,0.0)})
            self.my_visualizer['lift-off'] = define_markers('sphere', {'radius': 0.02, 'color': (0.678,1.0,0.184)})
            self.my_visualizer['touch-down'] = define_markers('sphere', {'radius': 0.02, 'color': (0.196,0.804,0.196)})
            self.my_visualizer['GRF']= define_markers('arrow_x', {'scale':(0.1,0.1,1.0), 'color': (0.196,0.804,0.196)})#'scale':(0.03,0.03,0.15), 
            self.my_visualizer['touch-down polygon'] = omni_debug_draw.acquire_debug_draw_interface()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self._debug_vis_handle = None
        # set initial state of debug visualization
        self.set_debug_vis(self.cfg.debug_vis)


    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self.f_len + self.d_len  + self.p_len + self.F_len 

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
    
    @property
    def RL_applied_actions(self) -> torch.Tensor:
        """ Return the optimal action back in the RL frame
        after inverse_normalisation(inverse_transformation(optimization(transformation(normalisation(raw_actions))))) 
        Dimension is always 4+4+8+12 = (batch_size, 28) """

        f, d, p_h, F_h = self.inverse_transformation(f=self.f_star, d=self.d_star, p_lw=self.p0_star_lw.unsqueeze(-1), F_lw=self.F0_star_lw.unsqueeze(-1))
        f_raw, d_raw, p_raw, F_raw = self. inverse_normalization(f=f, d=d, p=p_h, F=F_h)
        return  torch.cat((f_raw.flatten(1,-1), d_raw.flatten(1,-1), p_raw.flatten(1,-1), F_raw.flatten(1,-1)),dim=1)

    @property
    def has_debug_vis_implementation(self) -> bool:
        """Whether the command generator has a debug visualization implemented."""
        return True

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        """Processes the actions sent to the environment.

        Note:
            This function is called once per environment step. : Outer loop frequency
            1. Reconstrut latent variable z = (f,d,p,F)
            2. Normalized and transformed to relevant frame the latent variable z = (f,d,p,F)
            3. Process the latent variable (call controller.optimize_control_output)
                and update optimizied solution p*, F*, c*, pt*

        Args:
            action (torch.Tensor): The actions received from RL policy of Shape (num_envs, total_action_dim)
        """
        # save the previous variable - used for derivative computation in penalty
        self.f_prev      = self.f
        self.d_prev      = self.d
        self.p_norm_prev = self.p_norm
        self.F_norm_prev = self.F_norm

        # store the raw actions
        self._raw_actions[:] = actions

        # Filter the invalid values
        if torch.isinf(self._raw_actions).any():
            print('Problem with Infinite value in raw actions')
            self._raw_actions[torch.nonzero(torch.isinf(self._raw_actions))] = 0

        if not torch.distributions.constraints.real.check(self._raw_actions).any():
            print('Problem with NaN value in raw actions')
            self._raw_actions = torch.nan_to_num(self._raw_actions)

        # reconstruct the latent variable from the RL poliy actions
        self.f_raw = (self._raw_actions[:, 0                                    : self.f_len                                       ]).reshape_as(self.f_raw)
        self.d_raw = (self._raw_actions[:, self.f_len                           : self.f_len + self.d_len                          ]).reshape_as(self.d_raw)
        self.p_raw = (self._raw_actions[:, self.f_len + self.d_len              : self.f_len + self.d_len + self.p_len             ]).reshape_as(self.p_raw)
        self.F_raw = (self._raw_actions[:, self.f_len + self.d_len + self.p_len : self.f_len + self.d_len + self.p_len + self.F_len]).reshape_as(self.F_raw)

        # Normalize the actions
        self.f, self.d, self.p_norm, self.F_norm = self.normalization(f=self.f_raw, d=self.d_raw, p=self.p_raw, F=self.F_raw)

        # Disable Action : useful for debug
        if self.debug_apply_action_status:
            self.f, self.d, self.p_norm = self.debug_disable_action(f=self.f, d=self.d, p_norm=self.p_norm, gait=self.debug_apply_action_status)

        # Apply Transformation to have the Actions in the correct Frame
        self.f, self.d, self.p_lw, self.F_lw = self.transformation(f=self.f, d=self.d, p_h=self.p_norm, F_h=self.F_norm)
                    
        # Store the processed Actions
        self._processed_actions = torch.cat((self.f.flatten(1,-1), self.d.flatten(1,-1), self.p_lw.flatten(1,-1), self.F_lw.flatten(1,-1)),dim=1)

        # Optimize the latent variable with the model base controller
        self.f_star, self.d_star, self.c0_star, self.p0_star_lw, self.F0_star_lw, self.pt_star_lw, self.full_pt_lw = self.controller.process_latent_variable(f=self.f, d=self.d, p_lw=self.p_lw, F_lw=self.F_lw, env=self._env, height_map=torch.zeros(17,11)) # TODO implement height map

        # Enforce friction cone constraints for GRF : enforce only with sampling controller, because training should learn not to slip
        # if type(self.controller) == model_base_controller.samplingController: self.F0_star_lw = self.enforce_friction_cone_constraints(F=self.F0_star_lw.unsqueeze(-1), mu=self.cfg.optimizerCfg.mu).squeeze(-1)

        # print('robot mass :',self._asset.data.default_mass)

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
        pt_i_star_lw = self.pt_star_lw[:,:,:,self.inner_loop]
        self.inner_loop += 1    

        # Use model controller to compute the torques from the latent variable
        # Transform the shape from (batch_size, num_legs, num_joints_per_leg) to (batch_size, num_joints) # Permute and reshape to have the joint in right order [0,4,8][1,5,...] to [0,1,2,...]
        self.u = (self.controller.compute_control_output(F0_star_lw=self.F0_star_lw, c0_star=self.c0_star, pt_i_star_lw=pt_i_star_lw, p_lw=p_lw, p_dot_lw=p_dot_lw, q_dot=q_dot,
                                                         jacobian_lw=jacobian_lw, jacobian_dot_lw=jacobian_dot_lw, mass_matrix=mass_matrix, h=h)).permute(0,2,1).reshape(self.num_envs,self._num_joints)

        # Apply the computed torques
        self._asset.set_joint_effort_target(self.u)#, joint_ids=self._joint_ids) # Don't use joint_ids to speed up the process     

        # Debug
        if verbose_mb:
            self.debug_apply_action(p_lw, p_dot_lw, q_dot, jacobian_lw, jacobian_dot_lw, mass_matrix, h, self.F0_star_lw, self.c0_star, pt_i_star_lw)


    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Resets the manager term.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.
        """
        # check if specific environment ids are provided otherwise all environments must be reseted
        if env_ids is None:
            env_ids = slice(None) 

        # variable to count the number of inner loop with respect to outer loop
        self.inner_loop = 0

        # Reset jacobian_prev_lw : Jacobian are not initialise before env. step, thus can get them from robot state.
        self.jacobian_prev_lw[env_ids,:,:,:] = self.get_reset_jacobian()[env_ids,:,:,:]

        # Reset the model base controller : expect default foot position in local world frame 
        self.controller.reset(env_ids, self.get_reset_foot_position())

        # raw RL output
        self.f_raw[env_ids,...] = 1.0
        self.d_raw[env_ids,...] = 0.6
        self.p_raw[env_ids,...] = 0.0    
        self.F_raw[env_ids,...] = 0.0     

        # Normalized RL output
        self.f[env_ids,...]      = 1.0
        self.d[env_ids,...]      = 0.6
        self.p_norm[env_ids,...] = 0.0
        self.F_norm[env_ids,...] = 0.0

        # Previous output - used for derivative computation
        self.f_prev[env_ids,...]      = 1.0 
        self.d_prev[env_ids,...]      = 0.6
        self.p_norm_prev[env_ids,...] = 0.0
        self.F_norm_prev[env_ids,...] = 0.0

        # Normalized and transformed to frame RL output
        self.p_lw[env_ids,...] = 0.0
        self.F_lw[env_ids,...] = 0.0

        # create tensors for raw and processed actions
        self._raw_actions[env_ids,...]       = 0.0
        self._processed_actions[env_ids,...] = 0.0

        # Model-based optimized latent variable
        self.f_star[env_ids,...]     = 0.0
        self.d_star[env_ids,...]     = 0.0
        self.c0_star[env_ids,...]    = 1.0
        self.p0_star_lw[env_ids,...] = 0.0
        self.F0_star_lw[env_ids,...] = 0.0
        self.pt_star_lw[env_ids,...] = 0.0
        self.full_pt_lw[env_ids,...] = 0.0 # Used for plotting only

        # Control input u : joint torques
        self.u[env_ids,...] = 0.0

        # Intermediary variable - Hip variable for p centering
        self.hip0_pos_lw[env_ids,...]      = 0.0
        self.hip0_yaw_quat_lw[env_ids,...] = 0.0
        self.hip0_yaw_quat_lw[env_ids,:,0] = 1 # initialize correctly the quaternion to avoid NaN in first operation


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
        p_w = self._asset.data.body_pos_w[:, self.foot_idx,:]
        p_lw = p_w - self._env.scene.env_origins.unsqueeze(1).expand(p_w.shape)
        p_lw = torch.nan_to_num(p_lw) # TODO solve this when sim reset get only NaN

        # Transformation to get feet position in body frame
        # p_orientation_w = self._asset.data.body_quat_w[:, self.foot_idx,:]
        # p_b_0, _ = math_utils.subtract_frame_transforms(robot_pos_w, robot_orientation_w, p_w[:,0,:], p_orientation_w[:,0,:])
        # p_b_1, _ = math_utils.subtract_frame_transforms(robot_pos_w, robot_orientation_w, p_w[:,1,:], p_orientation_w[:,1,:])
        # p_b_2, _ = math_utils.subtract_frame_transforms(robot_pos_w, robot_orientation_w, p_w[:,2,:], p_orientation_w[:,2,:])
        # p_b_3, _ = math_utils.subtract_frame_transforms(robot_pos_w, robot_orientation_w, p_w[:,3,:], p_orientation_w[:,3,:])
        # p_b = torch.cat((p_b_0.unsqueeze(1), p_b_1.unsqueeze(1), p_b_2.unsqueeze(1), p_b_3.unsqueeze(1)), dim=1)

        # Retrieve Feet velocity in world frame : [num_instances, num_bodies, 3] select right indexes to get 
        # shape(batch_size, num_legs, 3) : Velocity is translation independant, thus p_dot_w = p_dot_lw
        p_dot_lw = self._asset.data.body_lin_vel_w[:, self.foot_idx,:]
        p_dot_lw = torch.nan_to_num(p_dot_lw) # TODO solve this when sim reset get only NaN

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
        # mass_matrix = torch.tensor([1])

        # Retrieve Corriolis, centrifugial and gravitationnal term
        # get_coriolis_and_centrifugal_forces -> (batch_size, num_joints)
        # get_generalized_gravity_forces -> (batch_size, num_joints)
        # Reshape and tranpose to get the correct shape in correct joint order-> (batch_size, num_legs, num_joints_per_leg)
        h = (self._asset.root_physx_view.get_coriolis_and_centrifugal_forces() + self._asset.root_physx_view.get_generalized_gravity_forces()).view(self.num_envs, self._num_joints_per_leg, self._num_legs).permute(0,2,1)
        # h = torch.Tensor([1])

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
        jacobian_feet_full_w = self._asset.root_physx_view.get_jacobians()[:, self.foot_idx, :3, :]
        jacobian_w = torch.cat((jacobian_feet_full_w[:, 0, :, 6+np.asarray(self._joints_idx[0])].unsqueeze(1),          # FL (idx 13)
                                jacobian_feet_full_w[:, 1, :, 6+np.asarray(self._joints_idx[1])].unsqueeze(1),          # FR (idx 14)
                                jacobian_feet_full_w[:, 2, :, 6+np.asarray(self._joints_idx[2])].unsqueeze(1),          # RL (idx 15)
                                jacobian_feet_full_w[:, 3, :, 6+np.asarray(self._joints_idx[3])].unsqueeze(1)), dim=1)  # RR (idx 16)

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


    def normalization(self, f:torch.Tensor|None, d:torch.Tensor|None, p:torch.Tensor|None, F:torch.Tensor|None) -> tuple[torch.Tensor|None, torch.Tensor|None, torch.Tensor|None, torch.Tensor|None] :
        """ Given the action with mean=0 and std=1 (~ [-1,+1])
        Scale, offset and clip the actions in new range

        Args :
            - f (torch.Tensor): leg frequency  RL policy output in [-1,1] range of shape(batch_size, num_legs)
            - d (torch.Tensor): leg duty cycle RL policy output in [-1,1] range of shape(batch_size, num_legs)
            - p (torch.Tensor): touch down pos RL policy output in [-1,1] range of shape(batch_size, num_legs, 2, p_param)
            - F (torch.Tensor): GRF            RL policy output in [-1,1] range of shape(batch_size, num_legs, 3, F_param)

        Returns :
            - f (torch.Tensor): Scaled output in [Hz]    of shape(batch_size, num_legs)
            - d (torch.Tensor): Scaled output in [1/rad] of shape(batch_size, num_legs)
            - p (torch.Tensor): Scaled output in [m]     of shape(batch_size, num_legs, 2, p_param)
            - F (torch.Tensor): Scaled output in [N]     of shape(batch_size, num_legs, 3, F_param)
        """
        param: actions_cfg.ModelBaseActionCfg.actionNormalizationCfg = self.cfg.actionNormalizationCfg

        #--- Normalize f ---
        # f:[-1,1]->[std_n,std_p]       : mean=(std_n+std_p)/2, std=(std_p-std_n)/2     : clipped to (min, max)
        # shape(batch_size, num_legs)
        if f is not None:
            f = ((f * ((param.std_p_f-param.std_n_f)/2)) + ((param.std_p_f+param.std_n_f)/2)).clamp(param.min_f,param.max_f)


        #--- Normalize d ---
        # d:[-1,1]->[std_n,std_p]       : mean=(std_n+std_p)/2, std=(std_p-std_n)/2     : clipped to (min, max)
        # shape(batch_size, num_legs)
        if d is not None:
            d = ((d * ((param.std_p_d-param.std_n_d)/2)) + ((param.std_p_d+param.std_n_d)/2)).clamp(param.min_d,param.max_d)


        #--- Normalize F ---
        # F_xy:[-1,1]->[-std,+std]  : mean=0, std=std
        # F_z:[-1,1]->[mean-std,mean+std]      : mean=m*g/2, std=mean/10
        # shape(batch_size, num_legs, 3, F_param)
        if F is not None :            
            std_xy_F = (torch.tensor(param.std_xy_F, device=self.device)).unsqueeze(-1).unsqueeze(-1) # shape (batch_size,1,1)
            F_x = F[:,:,0,:] * std_xy_F
            F_y = F[:,:,1,:] * std_xy_F

            if param.min_xy_F is not None or param.max_xy_F is not None :
                F_x = F_x.clamp(min=param.min_xy_F, max=param.max_xy_F)
                F_y = F_y.clamp(min=param.min_xy_F, max=param.max_xy_F)

            mean_z_F = (torch.tensor(param.mean_z_F, device=self.device)).unsqueeze(-1).unsqueeze(-1) 
            std_z_F  = (torch.tensor(param.std_z_F,  device=self.device)).unsqueeze(-1).unsqueeze(-1)
            F_z = (F[:,:,2,:]  * (std_z_F)) + (mean_z_F)

            if param.min_z_F is not None or param.max_z_F is not None :
                F_z = F_z.clamp(min=param.min_z_F, max=param.max_z_F)

            F = torch.cat((F_x, F_y, F_z), dim=2).reshape_as(self.F_lw)


        #--- Normalize p ---
        # p:[-1,1]->[std_n, std_p]      : mean=(std_n+std_p)/2, std=(std_p-std_n)/2     : clipped to (min, max)
        # shape(batch_size, num_legs, 3, p_param)
        if p is not None:
            p_x = ((p[:,:,0,:] * ((param.std_p_x_p-param.std_n_x_p)/(2*1))) + ((param.std_p_x_p+param.std_n_x_p)/2)).clamp(min=param.min_x_p, max=param.max_x_p)
            p_y = ((p[:,:,1,:] * ((param.std_p_y_p-param.std_n_y_p)/(2*1))) + ((param.std_p_y_p+param.std_n_y_p)/2)).clamp(min=param.min_y_p, max=param.max_y_p)
            if p.shape[2] == 2 : p = torch.cat((p_x, p_y           ), dim=2).reshape_as(p)
            else               : p = torch.cat((p_x, p_y,p[:,:,2,:]), dim=2).reshape_as(p)

        return f, d, p, F
    

    def inverse_normalization(self, f:torch.Tensor|None, d:torch.Tensor|None, p:torch.Tensor|None, F:torch.Tensor|None) -> tuple[torch.Tensor|None, torch.Tensor|None, torch.Tensor|None, torch.Tensor|None] :
        """ Given the action already normalized, find the original action by applying 
         inverse scaling and offset

        Args :
            - f (torch.Tensor): leg frequency  RL policy output in [-1,1] range of shape(batch_size, num_legs)
            - d (torch.Tensor): leg duty cycle RL policy output in [-1,1] range of shape(batch_size, num_legs)
            - p (torch.Tensor): touch down pos RL policy output in [-1,1] range of shape(batch_size, num_legs, 2, p_param)
            - F (torch.Tensor): GRF            RL policy output in [-1,1] range of shape(batch_size, num_legs, 3, F_param)

        Returns :
            - f (torch.Tensor): Scaled output in [Hz]    of shape(batch_size, num_legs)
            - d (torch.Tensor): Scaled output in [1/rad] of shape(batch_size, num_legs)
            - p (torch.Tensor): Scaled output in [m]     of shape(batch_size, num_legs, 2, p_param)
            - F (torch.Tensor): Scaled output in [N]     of shape(batch_size, num_legs, 3, F_param)
        """
        param: actions_cfg.ModelBaseActionCfg.actionNormalizationCfg = self.cfg.actionNormalizationCfg

        #--- Normalize f ---
        # f:[-1,1]->[std_n,std_p]       : mean=(std_n+std_p)/2, std=(std_p-std_n)/2     : clipped to (min, max)
        # shape(batch_size, num_legs)
        if f is not None:
            f = (f - ((param.std_p_f+param.std_n_f)/2)) / ((param.std_p_f-param.std_n_f)/2)


        #--- Normalize d ---
        # d:[-1,1]->[std_n,std_p]       : mean=(std_n+std_p)/2, std=(std_p-std_n)/2     : clipped to (min, max)
        # shape(batch_size, num_legs)
        if d is not None:
            d = (d - ((param.std_p_d+param.std_n_d)/2)) / ((param.std_p_d-param.std_n_d)/2)


        #--- Normalize F ---
        # F_xy:[-1,1]->[-std,+std]  : mean=0, std=std
        # F_z:[-1,1]->[mean-std,mean+std]      : mean=m*g/2, std=mean/10
        # shape(batch_size, num_legs, 3, F_param)
        if F is not None :            
            std_xy_F = (torch.tensor(param.std_xy_F, device=self.device)).unsqueeze(-1).unsqueeze(-1) # shape (batch_size,1,1)
            F_x = F[:,:,0,:] / std_xy_F
            F_y = F[:,:,1,:] / std_xy_F

            mean_z_F = (torch.tensor(param.mean_z_F, device=self.device)).unsqueeze(-1).unsqueeze(-1) 
            std_z_F  = (torch.tensor(param.std_z_F,  device=self.device)).unsqueeze(-1).unsqueeze(-1)
            F_z = (F[:,:,2,:] - mean_z_F) / std_z_F 

            F = torch.cat((F_x, F_y, F_z), dim=2).reshape_as(F)


        #--- Normalize p ---
        # p:[-1,1]->[std_n, std_p]      : mean=(std_n+std_p)/2, std=(std_p-std_n)/2     : clipped to (min, max)
        # shape(batch_size, num_legs, 3, p_param)
        if p is not None:
            p_x = (p[:,:,0,:] - ((param.std_p_x_p+param.std_n_x_p)/2)) / ((param.std_p_x_p-param.std_n_x_p)/2)
            p_y = (p[:,:,1,:] - ((param.std_p_y_p+param.std_n_y_p)/2)) / ((param.std_p_y_p-param.std_n_y_p)/2)
            if p.shape[2] == 2 : p = torch.cat((p_x, p_y           ), dim=2).reshape_as(p)
            else               : p = torch.cat((p_x, p_y,p[:,:,2,:]), dim=2).reshape_as(p)

        return f, d, p, F


    def transformation(self, f:torch.Tensor, d:torch.Tensor, p_h:torch.Tensor, F_h:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Given the latent variable (f,d,p,F) in some frame usefull for the Network, return the latent variable (f,d,p,F)
        in a Frame usefull for the low level controller.
        
        Args : 
            f    (Tensor): No frame -> no transformation,                             shape(batch_size, num_legs)
            d    (Tensor): No frame -> no transformation,                             shape(batch_size, num_legs)
            p_h  (Tensor): in hip_centered(x,y) frame with (roll_w, pitch_w, yaw_b),  shape(batch_size, num_legs, 2, p_param)
            F_h  (Tensor): in Horizonzal orientation frame (roll_w, pitch_w, yaw_b),  shape(batch_size, num_legs, 3, F_param)

        Returns :
            f    (Tensor): No frame -> no transformation,                             shape(batch_size, num_legs)
            d    (Tensor): No frame -> no transformation,                             shape(batch_size, num_legs)
            p_lw (Tensor): foot touch down pos. in local world frame,                 shape(batch_size, num_legs, 3, p_param)
            F_lw (Tensor): GRF in local world frame,                                  shape(batch_size, num_legs, 3, p_param)
        """
        # --- f : no transformation


        # --- d : no transformation


        # --- p : Transform from hip centered (xy plane only) with robot's heading orientation (roll_w, pitch_w, yaw_b) into local world frame 
        # Hip position in local world frame : shape(batch, num_legs, 3)
        p_hip_lw = self._asset.data.body_pos_w[:, self._hip_idx, :] - self._env.scene.env_origins.unsqueeze(1)
        p_hip_lw = torch.nan_to_num(p_hip_lw) # TODO solve this when sim reset with NaN get only NaN afterwards

        # Project Hip position onto the xy plane : shape(batch, num_legs, 3)
        p_hip_lw[:,:,2] = 0

        # Retrieve the robot yaw as quaternion : shape (batch_size, 4) -> (batch_size, 1, 4)
        robot_yaw_in_w = math_utils.yaw_quat(self._asset.data.root_quat_w).unsqueeze(1)

        # Update hip position and orientation while leg in contact (so it's saved for the entire swing trajectory with lift-off position)
        in_contact = (self.c0_star==1).unsqueeze(-1)  # True if foot in contact, False in in swing, shape(batch, legs, 1)
        self.hip0_pos_lw = (p_hip_lw * in_contact) + (self.hip0_pos_lw * (~in_contact))                 # shape(batch, legs, 3)
        self.hip0_yaw_quat_lw = (robot_yaw_in_w * in_contact) + (self.hip0_yaw_quat_lw * (~in_contact)) # shape(batch, 1, 4)*(batch, legs, 1) -> (batch, legs, 4)

        # p_h is two dimensionnal, need to happend a dimension to use 'transform_points'
        p_h = torch.cat([p_h, torch.zeros_like(p_h[:, :, :1, :])], dim=2) 

        # Transpose dimension to be able to use 'transform_points'
        p_h_permuted = p_h.permute(0,3,1,2) # Shape (batch, predict, legs, 3)

        # Transform points from horizontal to local world frame (rotate + shift)
        p0_lw = math_utils.transform_points(p_h_permuted[:,:,0,:], pos=self.hip0_pos_lw[:,0,:], quat=self.hip0_yaw_quat_lw[:,0,:]).unsqueeze(1) # shape(batch_size, 1,p_param, 3)
        p1_lw = math_utils.transform_points(p_h_permuted[:,:,1,:], pos=self.hip0_pos_lw[:,1,:], quat=self.hip0_yaw_quat_lw[:,1,:]).unsqueeze(1) # shape(batch_size, 1,p_param, 3)
        p2_lw = math_utils.transform_points(p_h_permuted[:,:,2,:], pos=self.hip0_pos_lw[:,2,:], quat=self.hip0_yaw_quat_lw[:,2,:]).unsqueeze(1) # shape(batch_size, 1,p_param, 3)
        p3_lw = math_utils.transform_points(p_h_permuted[:,:,3,:], pos=self.hip0_pos_lw[:,3,:], quat=self.hip0_yaw_quat_lw[:,3,:]).unsqueeze(1) # shape(batch_size, 1,p_param, 3)
        
        # Reconstruct the tensor (concatenate + reshape)
        p_lw = torch.cat((p0_lw, p1_lw, p2_lw, p3_lw), dim=1).permute(0,1,3,2) # shape(batch, legs, 3, p_param)

        # p_lw height is nonsense (on purpose): fill the step height with the foot offset (between foot as a body and the ground)
        p_lw[:,:,2] = self.cfg.footTrajectoryCfg.foot_offset

        # If the height scan is available, add the terrain height to the feet touch down position
        if self.cfg.height_scan_available:
            # Retrieve the height_scan_index given the feet position in base centered frame of shape (batch, legs, 2) + (1, legs, 2)
            height_scan_index = self.height_scan_index_from_pos_b(pos_b=self.p_norm[:,:,:2,0] + self.heightScan.hip_offset) #return shape(batch, legs)

            # Retrieve the height at the feet touch-down position from the height scan and add it to of shape (batch, legs, 3, 1)
            terrain_height_grid = self._env.scene["height_scanner"].data.ray_hits_w # shape (batch, 183, 3)

            # Retrieve the touch down position height given their index in the height grid : The height_scan has the env_origins offset, that must be removed to be in lw
            terrain_height_feet = terrain_height_grid[torch.arange(self.num_envs).unsqueeze(1), height_scan_index, 2] - self._env.scene.env_origins[:,2].unsqueeze(-1) #shape (batch_size, num_legs)

            # If Sensor reach maximum value it will retrun 'inf' -> Filter invalid value in case there is one
            if torch.isnan(terrain_height_feet).any() or torch.isinf(terrain_height_feet).any():
                print("terrain_height_feet contains NaN or Inf values")
                terrain_height_feet = torch.nan_to_num(terrain_height_feet, posinf=0.0, neginf=0.0, nan=0.0)

            p_lw[:,:,2] += terrain_height_feet.unsqueeze(-1) #shape (batch_size, num_legs, num_predict_step)


        # --- F : Rotate GRF from horizonzal frame (roll_w, pitch_w, yaw_b) to local wolrd frame
        # Find the robot's yaw in wolrd frame
        robot_yaw_in_w = math_utils.yaw_quat(self._asset.data.root_quat_w)

        # Transpose and Flatten to be able use efficiently 'transform_points'
        F_h_flatten = F_h.transpose(2,3).reshape(F_h.shape[0], F_h.shape[1]*F_h.shape[3], F_h.shape[2])    # shape(batch_size, F_param*num_legs, 3)

        # Rotate from horizontal frame to local world frame
        F_lw_flatten = math_utils.transform_points(F_h_flatten, quat=robot_yaw_in_w)                       # shape(batch_size, F_param*num_legs, 3)

        # Reshape back into original shape
        F_lw = F_lw_flatten.reshape(F_h.shape[0], F_h.shape[1], F_h.shape[3], F_h.shape[2]).transpose(2,3) # shape(batch_size, num_legs, 3, F_param)
        
        return f, d, p_lw, F_lw


    def inverse_transformation(self, f:torch.Tensor, d:torch.Tensor, p_lw:torch.Tensor, F_lw:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Given the latent variable (f,d,p,F) in some frame usefull for the low level controller, return the latent 
        variable (f,d,p,F) in a Frame usefull for the Network.
        
        Args :
            f    (Tensor): No frame -> no transformation,                             shape(batch_size, num_legs)
            d    (Tensor): No frame -> no transformation,                             shape(batch_size, num_legs)
            p_lw (Tensor): foot touch down pos. in local world frame,                 shape(batch_size, num_legs, 3, p_param)
            F_lw (Tensor): GRF in local world frame,                                  shape(batch_size, num_legs, 3, p_param)

        Returns : 
            f    (Tensor): No frame -> no transformation,                             shape(batch_size, num_legs)
            d    (Tensor): No frame -> no transformation,                             shape(batch_size, num_legs)
            p_h  (Tensor): in hip_centered frame with (roll_w, pitch_w, yaw_b),       shape(batch_size, num_legs, 2, p_param)
            F_h  (Tensor): in Horizonzal orientation frame (roll_w, pitch_w, yaw_b),  shape(batch_size, num_legs, 3, F_param)
        """
        # --- f : no transformation


        # --- d : no transformation


        # --- p : From local world frame, transform into hip centered frame (xy), with world roll and pitch and base yaw (horizontal frame orientation)

        # Retrieve the world orientation wrt to the hip (ie. base frame orientation some timestep in the past)
        hip0_yaw_w_in_b = math_utils.quat_conjugate(self.hip0_yaw_quat_lw[:,0,:])
        hip1_yaw_w_in_b = math_utils.quat_conjugate(self.hip0_yaw_quat_lw[:,1,:])
        hip2_yaw_w_in_b = math_utils.quat_conjugate(self.hip0_yaw_quat_lw[:,2,:])
        hip3_yaw_w_in_b = math_utils.quat_conjugate(self.hip0_yaw_quat_lw[:,3,:])

        # p in hip frame origin, but with world's orientation
        p_hw = p_lw - self.hip0_pos_lw.unsqueeze(-1) #shape(batch, legs*p_param, 3, p_param)

        # Transpose p_param and flatten for ease of manipulation
        p_hw_permuted = p_hw.permute(0,1,3,2)    #shape(batch, legs, 3, p_param) -> (batch, legs, p_param, 3)

        # Rotate the position from world frame to hip frame (ie base frame at maybe previous several time steps)
        p0_b = math_utils.transform_points(p_hw_permuted[:,0,:,:], quat=hip0_yaw_w_in_b).unsqueeze(1) # shape(batch_size, 1,p_param, 3)
        p1_b = math_utils.transform_points(p_hw_permuted[:,1,:,:], quat=hip1_yaw_w_in_b).unsqueeze(1) # shape(batch_size, 1,p_param, 3)
        p2_b = math_utils.transform_points(p_hw_permuted[:,2,:,:], quat=hip2_yaw_w_in_b).unsqueeze(1) # shape(batch_size, 1,p_param, 3)
        p3_b = math_utils.transform_points(p_hw_permuted[:,3,:,:], quat=hip3_yaw_w_in_b).unsqueeze(1) # shape(batch_size, 1,p_param, 3)

        # reconstruct the vector
        p_h_3D = torch.cat((p0_b, p1_b, p2_b, p3_b), dim=1).permute(0,1,3,2) # shape(batch, legs, 3, p_param)

        # Keep only the xy dimension since p_h is 2D
        p_h = p_h_3D[:,:,:2,:]


        # --- F : From local world frame, rotate to world roll and pitch and base yaw (horizontal frame orientation)         

        # Find the world's yaw wrt to the robot's base
        robot_yaw_in_w = math_utils.yaw_quat(self._asset.data.root_quat_w)
        world_yaw_in_b = math_utils.quat_conjugate(robot_yaw_in_w)

        # Transpose and Flatten to be able use efficiently 'transform_points'
        F_lw_flatten = F_lw.transpose(2,3).reshape(F_lw.shape[0], F_lw.shape[1]*F_lw.shape[3], F_lw.shape[2]) # shape(batch_size, num_legs*F_param, 3)

        # Rotate from local world frame to horizontal frame
        F_h_flatten = math_utils.transform_points(F_lw_flatten, quat=world_yaw_in_b)                          # shape(batch_size, num_legs*F_param, 3)

        # Reshape back into original shape
        F_h = F_h_flatten.reshape(F_lw.shape[0], F_lw.shape[1], F_lw.shape[3], F_lw.shape[2]).transpose(2,3)  # shape(batch_size, num_legs, 3, F_param) 


        return f, d, p_h, F_h


    def enforce_friction_cone_constraints(self, F:torch.Tensor, mu:float) -> torch.Tensor:
        """ Enforce the friction cone constraints
        ||F_xy|| < F_z*mu
        Args :
            F (torch.Tensor): The GRF   of shape(batch_size, num_legs, 3, F_param)

        Returns :
            F (torch.Tensor): The GRF with enforced friction constraints of shape(batch_size, num_legs, 3, F_param)
        """

        F_x = F[:,:,0,:].unsqueeze(2)
        F_y = F[:,:,1,:].unsqueeze(2)
        F_z = F[:,:,2,:].unsqueeze(2).clamp(min=0)

        # Angle between vec_x and vec_F_xy
        alpha = torch.atan2(F[:,:,1,:], F[:,:,0,:]).unsqueeze(2) # atan2(y,x) = arctan(y/x)

        # Compute the maximal Force in the xy plane
        F_xy_max = mu*F_z

        # Clipped the violation for the x and y component (unsqueeze to avoid to loose that dimension) : To use clamp_max -> need to remove the sign...
        F_x_clipped =  F_x.sign()*(torch.abs(F_x).clamp_max(torch.abs(torch.cos(alpha)*F_xy_max)))
        F_y_clipped =  F_y.sign()*(torch.abs(F_y).clamp_max(torch.abs(torch.sin(alpha)*F_xy_max)))

        # Reconstruct the vector
        F = torch.cat((F_x_clipped, F_y_clipped, F_z), dim=2)

        return F

#-------------------------------------------------- Helpers ------------------------------------------------------------
    def height_scan_index_from_pos_b(self, pos_b: torch.Tensor) -> torch.Tensor:
        """ Given a position in the (almost) robot body frame (centered at the robot base, aligned with yaw, but pitch-roll are ignored),
        it returns the index to access the (closest) height from the height_scan sensor.
        The height scanner makes a 2D grid, that start at the bottom left corner (ie. -length/2, -width/2). This grid is then
        flattened as a 1D index : [(-l/2;-w/2), (-l/2+res;-w/2), (-l2/2+2*res;-w/2), ... ,(l/2,-w/2),(-l/2;-w/2+res),...,(l/2;w/2)]

        Args : 
            - pos_b       (torch.Tensor): of shape (batch_size, number_of_pos, 2) (2=xy)

        Return :
            - height_scan_index (Tensor): of shape(batch_size, number_of_pos)
        """
        # Retrieve the index given the x and y direction : shape(batch_size, number_of_pos) : p - (-l/2)
        index_x = (torch.round((pos_b[:,:,0] + (self.heightScan.size[0]/2) ) / self.heightScan.resolution)).clamp(0, self.heightScan.max_x)
        index_y = (torch.round((pos_b[:,:,1] + (self.heightScan.size[1]/2) ) / self.heightScan.resolution)).clamp(0, self.heightScan.max_y) 

        # Apply the scalling to the y direction induced by how the grid is flatten : shape(batch_size, number_of_pos)
        height_scan_index = index_x*1 + index_y*(self.heightScan.scale_y)

        return height_scan_index.to(torch.int) 


    def debug_disable_action(self, f: torch.Tensor,d: torch.Tensor, p_norm: torch.Tensor, gait: Literal['full_stance', 'trot']) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """For debugging purposes, fix duty cycle, leg frequency and foot touch down position
        
        Args:
            gait (str) : The imposed gait, can be 'full_stance' or 'trot'
        """

        if   gait == 'full_stance':
            f = 0.0 * torch.ones_like(f)
            d = 1.0 * torch.ones_like(d)
            p_norm = torch.zeros_like(p_norm) # shape (batch_size, num_legs, 2, p_param)

        elif gait == 'trot':
            f = 2.5 * torch.ones_like(f)
            d = 0.65 * torch.ones_like(d)
            p_norm = torch.zeros_like(p_norm) # shape (batch_size, num_legs, 2, p_param)
            
            speed_command_b = (self._env.command_manager.get_command("base_velocity")).squeeze(0) # shape(3)

            p_norm[:,:,0] = speed_command_b[0] * (1/(2*2.5))
            p_norm[:,:,1] = speed_command_b[1] * (1/(2*2.5))

        return f, d, p_norm


    def debug_apply_action(self, p_lw, p_dot_lw, q_dot, jacobian_lw, jacobian_dot_lw, mass_matrix, h, F0_star_lw, c0_star, pt_i_star_lw):
        global verbose_loop

        # --- Print --- 
        verbose_loop+=1
        if verbose_loop>=50:
            verbose_loop=0
            # print()
            # print('Contact sequence : ', c0_star[0,...].flatten())
            # print('  Leg  frequency : ', self.f[0,:])
            # print('   duty   cycle  : ', self.d[0,...].flatten())
            # print('terrain dificulty: ', torch.mean(self._env.scene.terrain.terrain_levels.float()))
            # print('  Max dificulty  : ', self._env.scene.terrain.difficulty.float()[:])
            # print('terrain dificulty: ', self._env.scene.terrain.terrain_levels.float()[:])
            # print('Terrain Progress : ', self._env.command_manager.get_term("base_velocity").metrics['cumulative_distance'][:4]/(self._env.scene.terrain.cfg.terrain_generator.size[0] / 2))
            # print('speed difficulty : ', self._env.command_manager.get_term("base_velocity").difficulty)
            # print('speed command    : ', self._env.command_manager.get_command("base_velocity")[:,0])
            # print('Actual Speed     : ', self._asset.data.root_lin_vel_b[:,0])
            # print('Touch-down pos   : ', self.p_lw[0,0,:,0])
            # print(' Foot  position  : ', p_lw[0,...])
            # print(' Robot position  : ', self._asset.data.root_pos_w[0,...])
            # print('Foot traj shape  : ', self.pt_star_lw.shape)
            # print('Foot traj : ', self.pt_star_lw[0,0,:3,:])
            # print('Foot Force :', self.F0_star_lw[0,:,:])
            # print('\nZ lin vel : ', self._asset.data.root_lin_vel_b[0, 2])
            # print(self._env.reward_manager.find_terms('track_lin_vel_xy_exp'))
            # try : 
                # print('Penalty Lin vel z  : ',self._env.reward_manager._episode_sums["penalty_lin_vel_z_l2"][0])
                # print('Track ang vel z    : ',self._env.reward_manager._episode_sums["track_ang_vel_z_exp"][0])
                # print('Penalty frequency  : ',self._env.reward_manager._episode_sums["penalty_frequency_variation"][0])
                # print('Track soft exp    : ',self._env.reward_manager._episode_sums["track_soft_vel_xy_exp"][0:4])
                # print('Track exp         : ',self._env.reward_manager._episode_sums["track_lin_vel_xy_exp"][0:4])
            # except : pass

            if (self.F_lw[:,:,:,0] != self.F0_star_lw).any():
                assert ValueError('F value don\'t match...')

        # --- Visualize foot position ---
        if vizualise_debug['foot']:
            # p_lw_ = p_lw.clone().detach()
            p_w = p_lw + self._env.scene.env_origins.unsqueeze(1)

            # Transformation if p in base frame
            # robot_pos_w = self._asset.data.root_pos_w
            # robot_orientation_w = self._asset.data.root_quat_w
            # p_orientation_w = self._asset.data.body_quat_w[:, self.foot_idx,:]
            # p_w_0, _ = math_utils.combine_frame_transforms(robot_pos_w, robot_orientation_w, p_b[:,0,:])
            # p_w_1, _ = math_utils.combine_frame_transforms(robot_pos_w, robot_orientation_w, p_b[:,1,:])
            # p_w_2, _ = math_utils.combine_frame_transforms(robot_pos_w, robot_orientation_w, p_b[:,2,:])
            # p_w_3, _ = math_utils.combine_frame_transforms(robot_pos_w, robot_orientation_w, p_b[:,3,:])
            # p_w = torch.cat((p_w_0.unsqueeze(1), p_w_1.unsqueeze(1), p_w_2.unsqueeze(1), p_w_3.unsqueeze(1)), dim=1)
            stance = self.c0_star

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
            marker_indices = ((c0_star.unsqueeze(-1).expand(self.num_envs,self._num_legs,22)).flatten(1,2).flatten(0,1))
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
            # p_orientation_w = self._asset.data.body_quat_w[:, self.foot_idx,:]
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
            marker_indices = ~c0_star[0,...]

            self.my_visualizer['GRF'].visualize(translations=marker_locations, orientations=marker_orientations, scales=scale, marker_indices=marker_indices)

        # --- Visualize Foot touch-down polygon ---
        if vizualise_debug['touch-down polygon']:
            """self.my_visualizer['touch-down polygon'] = omni_debug_draw.acquire_debug_draw_interface()"""

            pass # Need to redo transform_p_from_rl_frame_to_lw for it to work
            # FOOT_OFFSET = self.cfg.footTrajectoryCfg.foot_offset
            # # Find the corner points of the polygon - provide big values that will be clipped to corresponding bound
            # # p shape(num_corners, 3)
            # p_corner = torch.tensor([[10,10,FOOT_OFFSET],[10,-10,FOOT_OFFSET],[-10,-10,FOOT_OFFSET],[-10,10,FOOT_OFFSET]], device=self.device)

            # # Reshape p to be passed to transform_p_from_rl_to_lw -> (num_corner, num_legs, 3, 1)
            # p_corner = p_corner.unsqueeze(1).expand(4,4,3).unsqueeze(-1)

            # # Normalize to find the correct bound
            # _, _, _, p_corner_rl = self.normalize_actions(f=None, d=None, F=None, p=p_corner)
            # p_corner_rl[:,:,2,:] = FOOT_OFFSET # This is overwritten by the normalization

            # # shape (batch, num_corner, num_leg, 3, 1)
            # p_corner_batched_rl = p_corner_rl.unsqueeze(0).expand(self.num_envs,4,4,3,1)
            
            # # Needs p shape(batch, num_corner, num_legs, 3, 1) -> (batch, num_legs, 3, 1)
            # p_corner_1_lw = self.transform_p_from_rl_frame_to_lw(p_corner_batched_rl[:,0,:,:,:])
            # p_corner_2_lw = self.transform_p_from_rl_frame_to_lw(p_corner_batched_rl[:,1,:,:,:])
            # p_corner_3_lw = self.transform_p_from_rl_frame_to_lw(p_corner_batched_rl[:,2,:,:,:])
            # p_corner_4_lw = self.transform_p_from_rl_frame_to_lw(p_corner_batched_rl[:,3,:,:,:])
            # # p_lw = self.transform_p_from_rl_frame_to_lw(p_corner_rl)

            # # shape (batch_size, num_corner, num_legs, 3, 1)
            # p_lw = torch.cat((p_corner_1_lw.unsqueeze(1), p_corner_2_lw.unsqueeze(1), p_corner_3_lw.unsqueeze(1), p_corner_4_lw.unsqueeze(1)), dim=1)

            # # Reshape according to our needs -> shape(batch, num_legs, num_corner,3)
            # p_lw = p_lw.squeeze(-1).permute(0,2,1,3)

            # # Transform to world frame
            # p_w = p_lw + (self._env.scene.env_origins).unsqueeze(1).unsqueeze(2) #

            # # Create the list to display the line
            # source_pos = p_w.flatten(0,2)
            # target_pos = p_w.roll(-1,dims=2).flatten(0,2)

            # # Start by clearing the eventual previous line
            # self.my_visualizer['touch-down polygon'].clear_lines()

            # # plain color for lines
            # lines_colors = [[1.0, 1.0, 0.0, 1.0]] * source_pos.shape[0]
            # line_thicknesses = [2.0] * source_pos.shape[0]

            # self.my_visualizer['touch-down polygon'].draw_lines(source_pos.tolist(), target_pos.tolist(), lines_colors, line_thicknesses)


    def set_debug_vis(self, debug_vis: bool) -> bool:
        """Sets whether to visualize the command data.

        Args:
            debug_vis: Whether to visualize the command data.

        Returns:
            Whether the debug visualization was successfully set. False if the command
            generator does not support debug visualization.
        """

        # toggle debug visualization objects
        self._set_debug_vis_impl(debug_vis)

        # toggle debug visualization handles
        if debug_vis:
            # create a subscriber for the post update event if it doesn't exist
            if self._debug_vis_handle is None:
                app_interface = omni.kit.app.get_app_interface()
                self._debug_vis_handle = app_interface.get_post_update_event_stream().create_subscription_to_pop(
                    lambda event, obj=weakref.proxy(self): obj._debug_vis_callback(event)
                )
        else:
            # remove the subscriber if it exists
            if self._debug_vis_handle is not None:
                self._debug_vis_handle.unsubscribe()
                self._debug_vis_handle = None
        # return success
        return True
    

    def _set_debug_vis_impl(self, debug_vis: bool):
        global verbose_mb
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "foot_traj_visualizer"):
                # -- foot traj
                self.foot_traj_visualizer = define_markers('sphere', {'radius': 0.01, 'color': (0.0,1.0,0.0)})
                self.foot_GRF_visualizer = define_markers('arrow_x', {'scale':(0.1,0.1,1.0), 'color': (0.804,0.196,0.196)})#'scale':(0.03,0.03,0.15), 

            # set their visibility to true
            self.foot_traj_visualizer.set_visibility(True)
            self.foot_GRF_visualizer.set_visibility(True)

            verbose_mb = True
            try:self.controller.verbose_md=True
            except:pass
        else:
            if hasattr(self, "foot_traj_visualizer"):
                self.foot_traj_visualizer.set_visibility(False)
                self.foot_GRF_visualizer.set_visibility(False)

            verbose_mb = False
            try:self.controller.verbose_md=False
            except:pass


    def _debug_vis_callback(self, event):
        # --- Update foot trajectory
        full_pt_lw_ = self.full_pt_lw.clone().detach()  # shape (batch_size, num_legs, 9, 22) (9=px,py,pz,vx,vy,vz,ax,ay,az)
        full_pt_lw_ = full_pt_lw_[:,:,0:3,:] # -> shape (batch_size, num_legs, 3, 22)
        full_pt_lw_ = full_pt_lw_.permute(0,1,3,2).flatten(1,2) # -> shape (batch_size, num_legs, 22, 3) -> (batch_size, num_legs*22, 3)
        full_pt_w_ = full_pt_lw_ + self._env.scene.env_origins.unsqueeze(1) 
        full_pt_w_ = full_pt_w_.flatten(0,1) # -> shape (batch_size*num_legs*22, 3)
        marker_locations = full_pt_w_

        # Visualize the traj only if it is used (ie. the foot is in swing -> c==0)
        marker_indices = ((self.c0_star.unsqueeze(-1).expand(self.num_envs,self._num_legs,22)).flatten(1,2).flatten(0,1))       

        self.foot_traj_visualizer.visualize(translations=marker_locations, marker_indices=marker_indices)


        # --- Update foot force GRF
        p_w = self._asset.data.body_pos_w[:, self.foot_idx,:]
        p_lw = p_w - self._env.scene.env_origins.unsqueeze(1).expand(p_w.shape)

        # GRF location are the feet position
        p3_w = p_lw + self._env.scene.env_origins.unsqueeze(1) # (batch, num_legs, 3)
        # marker_locations = p3_w[0,...]
        marker_locations = p3_w.flatten(0,1) # (batch*num_legs, 3)

        # From GRF, retrieve orientation (angle and axis representation)
        F = self.F0_star_lw.clone().detach()[0,:] # (batch, num_legs, 3) -> (num_legs, 3)
        F = self.F0_star_lw.clone().detach().flatten(0,1) # (batch*num_legs, 3)
        normalize_F = torch.nn.functional.normalize(F, p=2, dim=1) # Transform GRF to unit vectors # (batch*num_legs, 3)
        # angle : u dot v = cos(angle) -> angle = acos(u*v) : for unit vector # Need to take the opposite angle in order to make appropriate rotation
        angle = -torch.acos(torch.tensordot(normalize_F, torch.tensor([1.0,0.0,0.0], device=self.device), dims=1)) # shape(batch*num_legs, 3) -> (batch*num_legs)
        # Axis : Cross product between u^v (for unit vectors)
        axis = torch.cross(normalize_F, torch.tensor([1.0,0.0,0.0], device=self.device).unsqueeze(0).expand(normalize_F.shape),  dim=1)
        marker_orientations = quat_from_angle_axis(angle=angle, axis=axis)

        # Scale GRF
        scale = torch.linalg.vector_norm(F, dim=1).unsqueeze(-1).expand(F.shape) / 250 # 150

        # The arrow point is define at its center. So to avoid having the arrow in the middle of the feet, we translate it by a factor along its pointing direction
        translation = scale*torch.tensor([0.25, 0.0, 0.0], device=self.device).unsqueeze(0).expand(marker_locations.shape)
        translation = math_utils.transform_points(points=translation.unsqueeze(1), pos=marker_locations, quat=marker_orientations).squeeze(1)
        marker_locations = translation

        # Visualize the force only if it is used (ie. the foot is in contact -> c==1)
        marker_indices = ~self.c0_star.flatten(0,1)

        self.foot_GRF_visualizer.visualize(translations=marker_locations, orientations=marker_orientations, scales=scale, marker_indices=marker_indices)
        


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