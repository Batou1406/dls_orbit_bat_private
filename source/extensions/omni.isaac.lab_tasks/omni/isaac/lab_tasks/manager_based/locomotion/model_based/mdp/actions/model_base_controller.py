from abc import ABC
from collections.abc import Sequence
import torch
import time
import numpy as np


# TODO put relevant thing into typechecking to avoid circular import ie. articulation and env
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    pass

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets.articulation import Articulation
from omni.isaac.lab.envs import ManagerBasedRLEnv

from .helper import inverse_conjugate_euler_xyz_rate_matrix, rotation_matrix_from_w_to_b, gait_generator, compute_cubic_spline, compute_discrete, normal_sampling, uniform_sampling
from .helper import fit_cubic, enforce_friction_cone_constraints_torch, from_zero_twopi_to_minuspi_pluspi, compute_unique


class baseController(ABC):
    """
    Abstract controller class for model base control implementation
    
    Properties : 
        - verbose_md    : Verbose variable. Save some computation that aren't necessary if not in debug mode.
        - _device
        - _num_envs
        - _num_legs
        - _dt_out       : Outer Loop time step 
        - _decimation   : Inner Loop time horizon
        - _dt_in        : Inner Loop time step

    Method :
        - late_init(device, num_envs, num_legs) : save environment variable and allow for lazy initialisation of variables
        - reset(env_ids) : Reset controller variables upon environment reset if needed
        - process_latent_variable(f, d, p, F) -> p*, F*, c*, pt*
        - compute_control_output(F0*, c0*, pt01*) -> T
    """

    def __init__(self, verbose_md, device, num_envs, num_legs, robot_mass, mu, dt_out, decimation, dt_in):
        """ Initialise Model Base variable after the model base action class has been initialised

        Args : 
            - verbose_md  (bool): Debug mode
            - device            : Cpu or GPu
            - num_envs     (int): Number of parallel environments
            - time_horiton (int): Prediction time horizon for the Model Base controller (runs at outer loop frequecy)
            - dt_out       (int): Outer loop delta t (decimation * dt_in)
            - decimation   (int): Inner Loop steps per outer loop steps
            - dt_in        (int): Inner loop delta t
        """
        super().__init__()
        self.verbose_md = verbose_md
        self._num_envs = num_envs
        self._device = device
        self._num_legs = num_legs
        self._robot_mass = robot_mass
        self._mu = mu
        self._dt_out = dt_out
        self._decimation = decimation
        self._dt_in = dt_in


    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """ The environment is reseted -> this requires to reset some controller variables
        """
        pass 


    def process_latent_variable(self, f: torch.Tensor, d: torch.Tensor, p_b: torch.Tensor, F_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Given the latent variable z=[f,d,p,F], return the optimized latent variable p*, F*, c*, pt*

        Args:
            - f   (torch.Tensor): Prior leg frequency                   of shape (batch_size, num_legs)
            - d   (torch.Tensor): Prior stepping duty cycle             of shape (batch_size, num_legs)
            - p_b (torch.Tensor): Prior foot pos. seq. in base frame    of shape (batch_size, num_legs, 3, p_param)
            - F_w (torch.Tensor): Prior Ground Reac. Forces (GRF) seq.  of shape (batch_size, num_legs, 3, F_param)
                                  In world frame

        Returns:
            - p*  (torch.Tensor): Optimized foot position sequence      of shape (batch_size, num_legs, 3, p_param)
            - F*  (torch.Tensor): Opt. Ground Reac. Forces (GRF) seq.   of shape (batch_size, num_legs, 3, F_param)
            - c*  (torch.Tensor): Optimized foot contact sequence       of shape (batch_size, num_legs, 1)
        """
        raise NotImplementedError


    def compute_control_output(self, F0_star: torch.Tensor, c0_star: torch.Tensor, pt_i_star: torch.Tensor) -> torch.Tensor:
        """ Compute the output torque to be applied to the system
        typically, it would compute :
            - T_stance_phase = stance_leg_controller(GRF, q, c) # Update the jacobian with the new joint position.
            - T_swing_phase = swing_leg_controller(trajectory, q, q_dot, c) # Feedback lineraization control - trajectory computed with a spline to be followed - new updated joint controller.
            - T = (T_stance_phase * c0_star) + (T_swing_phase * (~c0_star))
        and return T
        
        Args:
            - F0* (torch.Tensor): Opt. Ground Reac. Forces (GRF)        of shape(batch_size, num_legs, 3)
            - c0* (torch.bool)  : Optimized foot contact sequence       of shape(batch_size, num_legs)
            - pt_i* (tch.Tensor): Opt. Foot point in swing phase        of shape(batch_size, num_legs, 9) (9 = pos, vel, acc)
            - ...

        Returns:
            - T   (torch.Tensor): control output (ie. Joint Torques)    of shape(batch_size, num_joints)
        """
        raise NotImplementedError

    
class modelBaseController(baseController):
    """
    Implement a model based controller based on the latent variable z = [f,d,p,F]

    - Gait Generator :
        From the latent variables f and d (leg frequency & duty cycle) and the phase property, compute the next leg
        phases and contact sequence.

    - Sampling Controller :
        Optimize the latent variable z. Generates samples, simulate the samples, evaluate them and return the best one.

    Properties : 
        - verbose_md    : Verbose variable. Save some computation that aren't necessary if not in debug mode.           Inherited from baseController
        - _device                                                                                                       Inherited from baseController
        - _num_envs                                                                                                     Inherited from baseController
        - _num_legs                                                                                                     Inherited from baseController
        - _dt_out       : Outer Loop time step                                                                          Inherited from baseController
        - _decimation   : Inner Loop time horizon                                                                       Inherited from baseController
        - _dt_in        : Inner Loop time step                                                                          Inherited from baseController
        - phase (Tensor): Leg phase                                             of shape (batch_size, num_legs)
        - p0_lw (Tensor): Lift-off position                                     of shape (batch_size, num_legs, 3)
        - swing_time (T): time progression of the leg in swing phase            of shape (batch_size, num_legs)  
        - p_lw_sim_prev : Last foot position from sim. upd in comp_ctrl in _lw  of shape (batch_size, num_legs, 3)
        - step_height   : Apex height of the swing trajectory
        - FOOT_OFFSET   : Offset between the foot (as return by the sim.) and the ground when in contact

    Method :
        - late_init(device, num_envs, num_legs) : save environment variable and allow for lazy init of variables        Inherited from baseController
        - reset(env_ids) : Reset controller variables upon environment reset if needed                                  Inherited from baseController (not implemented)
        - process_latent_variable(f, d, p, F) -> p*, F*, c*, pt*                                                       Inherited from baseController (not implemented)
        - compute_control_output(F0*, c0*, pt01*) -> T                                                                  Inherited from baseController (not implemented)
        - full_swing_trajectory_generator(p_b, c, decimation) -> pt_b
        - swing_leg_controller(c0*, pt01*) -> T_swing
        - stance_leg_controller(F0*, c0*) -> T_stance
    """

    phase : torch.Tensor
    p0_lw : torch.Tensor
    swing_time : torch.Tensor
    p_lw_sim_prev : torch.Tensor

    def __init__(self, verbose_md, device, num_envs, num_legs, robot_mass, mu, dt_out, decimation, dt_in, p_default_lw: torch.Tensor, step_height, foot_offset, swing_ctrl_pos_gain_fb, swing_ctrl_vel_gain_fb):
        """ Initialise Model Base variable after the model base action class has been initialised
        Note :
            The variable are in the 'local' world frame _wl. This notation is introduced to avoid confusion with the 'global' world frame, where all the batches coexists.

        Args : 
            - verbose_md  (bool): Debug mode
            - device            : Cpu or GPu
            - num_envs     (int): Number of parallel environments
            - time_horiton (int): Prediction time horizon for the Model Base controller (runs at outer loop frequecy)
            - dt_out       (int): Outer loop delta t (decimation * dt_in)
            - decimation   (int): Inner Loop steps per outer loop steps
            - dt_in        (int): Inner loop delta t
            - p_default (Tensor): Default feet pos of robot when reset  of Shape (batch_size, num_legs, 3)
        """
        super().__init__(verbose_md, device, num_envs, num_legs, robot_mass, mu, dt_out, decimation, dt_in)
        self.phase = torch.zeros(num_envs, num_legs, device=device)
        self.phase[:,(0,3)] = 0.5 # Init phase [0.5, 0, 0, 0.5]
        self.p0_lw = p_default_lw.clone().detach()
        self.swing_time = torch.zeros(num_envs, num_legs, device=device)
        self.p_lw_sim_prev = p_default_lw.clone().detach()
        self.step_height = step_height
        self.FOOT_OFFSET = foot_offset
        self.swing_ctrl_pos_gain_fb = swing_ctrl_pos_gain_fb
        self.swing_ctrl_vel_gain_fb = swing_ctrl_vel_gain_fb


    def reset(self, env_ids: Sequence[int] | None,  p_default_lw: torch.Tensor) -> None:
        """ The environment is reseted -> this requires to reset some controller variables
        Note :
            The variable are in the 'local' world frame _wl. This notation is introduced to avoid confusion with the 'global' world frame, where all the batches coexists.

        Args :
            - env_ids           : Index of the env beeing reseted
            - p_default_lw (Tsr): Default feet position in local world frame of shape (batch_size, num_legs, 3)
        """
        # Reset gait phase          : Shape (batch_size, num_legs)
        self.phase[env_ids,:] = torch.zeros_like(self.phase, device=self._device)[env_ids,:]
        self.phase[env_ids,0] = 0.5
        self.phase[env_ids,3] = 0.5 # Init phase [0.5, 0, 0.5, 0]

        # Reset lift-off pos       : Shape (batch_size, num_legs, 3)
        self.p0_lw[env_ids,:,:] = p_default_lw[env_ids,:,:].clone().detach()

        # Reset swing time         : Shape (batch_size, num_legs)
        self.swing_time[env_ids,:] = torch.zeros_like(self.swing_time, device=self._device)[env_ids,:]

        # Reset previous foot position  : Shape (batch_size, num_legs, 3)
        self.p_lw_sim_prev[env_ids,:,:] = p_default_lw[env_ids,:,:].clone().detach()


# ----------------------------------- Outer Loop ------------------------------
    def process_latent_variable(self, f: torch.Tensor, d: torch.Tensor, p_lw: torch.Tensor, delta_F_lw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Given the latent variable z=[f,d,p,F], return the optimize latent variable f*, d*, c*, p*, F*, pt* full_pt
        Note :
            The variable are in the 'horizontal' frame _h : (x_b, y_b, z_w, 0, 0, yaw_b)

        Args:
            f            (Tensor): Prior leg frequency                             of shape (batch_size, num_legs)
            d            (Tensor): Prior stepping duty cycle                       of shape (batch_size, num_legs)
            p_lw         (Tensor): Prior foot touch down delta seq. in _h          of shape (batch_size, num_legs, 3, p_param)
            delta_F_lw   (Tensor): Prior Ground Reac. Forces delta(GRF) seq.       of shape (batch_size, num_legs, 3, F_param)

        Returns:
            f*           (Tensor): Opt. leg frequency                              of shape (batch_size, num_legs)
            d*           (Tensor): Opt. leg duty cycle                             of shape (batch_size, num_legs)
            c0*          (Tensor): Optimized foot contact sequence                 of shape (batch_size, num_legs)
            p0*_lw       (Tensor): Optimized foot touch down delta seq. in _h      of shape (batch_size, num_legs, 3)
            F0*_lw       (Tensor): Opt. Gnd Reac. F. delta(GRF) seq.   in _h       of shape (batch_size, num_legs, 3)
            pt*_lw (torch.Tensor): Optimized foot swing traj.     in _lw of shape (batch_size, num_legs, 9, decimation)  (9 = pos, vel, acc)
            full_pt_lw (t.Tensor): Full swing traj for ploting           of shape (batch_size, num_legs, 9, 22)
        """

        # No optimizer
        f_star, d_star, p0_star_lw, delta_F0_star_lw = f, d, p_lw[:,:,:,0], delta_F_lw[:,:,:,0]

        # Compute the contact sequence and update the phase (c at current time, phase at next time (after one f*dt increment)) (need at least an horizon 2 to have c_0 and phase_1)
        c_star, next_phase = gait_generator(f=f_star, d=d_star, phase=self.phase, horizon=2, dt=self._dt_out)
        c0_star = c_star[:,:,0] # shape (batch_size, num_legs)

        # Generate the swing trajectory
        pt_star_lw, full_pt_lw = self.full_swing_trajectory_generator(p_lw=p0_star_lw, c0=c0_star, d=d_star, f=f_star)

        # Add the gravity compensation
        num_legs_in_contact = torch.sum(c0_star, dim=1).clamp_min(1) #shape(batch_size)
        F0_star_lw = delta_F0_star_lw # shape(batch_size, num_legs, 3)
        F0_star_lw[:,:,2] += (c0_star * (self._robot_mass * 9.81) / num_legs_in_contact.unsqueeze(-1)) # shape(batch_size, num_legs)
        F0_star_lw[:,:,2] = F0_star_lw[:,:,2].clamp(min=0)

        # Enforce the friction cone constraints
        F0_star_lw = enforce_friction_cone_constraints_torch(F=F0_star_lw, mu=self._mu, F_z_min=0.0, F_z_max=9.81*self._robot_mass) # shape(num_samples, num_legs, 3)

        # update the phase variable
        self.phase = next_phase

        return f_star, d_star, c0_star, p0_star_lw, F0_star_lw, pt_star_lw, full_pt_lw
    

    def full_swing_trajectory_generator(self, p_lw: torch.Tensor, c0: torch.Tensor, f: torch.Tensor, d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ Given feet position sequence and contact sequence -> compute swing trajectories by fitting a cubic spline between
        the lift-off and the touch down define in the contact sequence. 
        - Swing frequency and duty cycle are used to compute the swing period
        - A middle point is used for the interpolation : which is heuristically defined. It defines the step height
        - p1 (middle point) and p2 (touch-down) are updated each time, while p0 is conserved (always the same lift off position)
        Note :
            The variable are in the 'local' world frame _wl. This notation is introduced to avoid confusion with the 'global' world frame, where all the batches coexists.
        
        Args:
            - p_lw (trch.Tensor): Foot touch down postion in _lw        of shape(batch_size, num_legs, 3)
            - c0  (torch.Tensor): Foot contact sequence                 of shape(batch_size, num_legs)
            - f   (torch.Tensor): Leg frequency           in R+         of shape(batch_size, num_legs)
            - d   (torch.Tensor): Stepping duty cycle     in[0,1]       of shape(batch_size, num_legs)

        Returns:
            - pt_lw (tch.Tensor): Desired Swing Leg traj. in _lw frame  of shape(batch_size, num_legs, 9, decimation)   (9 = xyz_pos, xzy_vel, xyz_acc)
        """
        # --- Step 1. Compute the phase trajectory : shape (batch_size, num_legs, decimation)
        # ie. retrieve the actual leg phase -> and compute the trajectory (phase evolution) for the next outer loop period (ie. for decimation inner loop iteration) 

        # swing phase in [0,1] (leg is in swing when phase = [d, 1] -> scale to have swing_phase in [0,1]), shape(batch_size, num_legs)
        swing_phase = (self.phase - d) / (1 - d + 1e-10)  

        swing_frequency = f / (1 - d + 1e-10)           # [Hz] : swing frequency,   shape(batch_size, num_legs)
        delta_phase = swing_frequency * self._dt_in      # delta_phase = swing_freq [Hz] * dt [s],  shape(batch_size, num_legs)

        # swing phase trajectpry [phase, phase + delta_phase, ...],   shape (batch_size, num_legs, decimation)
        # (batch_size, num_legs, 1) + [(1, 1, decimation) * (batch_size, num_legs, 1)] -> (batch_size, num_legs, decimation)
        swing_phase_traj = (swing_phase.unsqueeze(-1)) + ((0+torch.arange(self._decimation, device=self._device).unsqueeze(0).unsqueeze(0)) * delta_phase.unsqueeze(-1))


        # --- Step 2. Retrieve the three interpolation points : p0, p1, p2 (lift-off, middle point, touch down)

        # Retrieve p0 : update p0 with latest foot position when in contact, don't update when in swing
        # p0 shape (batch_size, num_legs, 3)
        in_contact = (c0==1).unsqueeze(-1)    # True if foot in contact, False in in swing, shape (batch_size, num_legs, 1)
        self.p0_lw = (self.p_lw_sim_prev * in_contact) + (self.p0_lw * (~in_contact)) # p_lw_sim_prev : last data from sim available

        # Retrieve p2 : this is simply the foot touch down prior given as input
        # p2 shape (batch_size, num_legs, 3) 
        p2_lw = p_lw 

        # Retrieve p1 : (x,y) position are define as the middle point between p0 and p1 (lift-off and touch-down). z is heuristcally define
        # p1 shape (batch_size, num_legs, 3)
        # TODO Not only choose height as step heigh but use +the terrain height or +the feet height at touch down
        p1_lw = (self.p0_lw[:,:,:2] + p2_lw[:,:,:2]) / 2     # p1(x,y) is in the middle of p0 and p2
        # p1_lw = torch.cat((p1_lw, self.step_height*torch.ones_like(p1_lw[:,:,:1]) + self.FOOT_OFFSET), dim=2) # Append a third dimension z : defined as step_height
        p1_lw = torch.cat((p1_lw, self.step_height*torch.ones_like(p1_lw[:,:,:1]) + (torch.max(self.p0_lw[:,:,2], p2_lw[:,:,2])).unsqueeze(-1)), dim=2) # Append a third dimension z : defined as step_height + max(lift_off_height,touch_down_height)


        # --- Step 3. Compute the parameters for the interpolation (control points)
        # Compute the a,b,c,d polynimial coefficient for the cubic interpolation S(x) = a*x^3 + b*x^2 + c*x + d, x in [0,1]

        # If swing_time < swing period/2 -> S_0(t) (ie. first interpolation), otherwise -> S_1(t - delta_t/2) (ie. second interpolation)
        # is_S0 may vary during the trajectory if we are close to the middle point, (ie if phase goes through 0.5), this is why is_S0 has decimation dimension
        is_S0 = (swing_phase_traj <=  0.5).unsqueeze(2)  # shape (batch_size, num_legs, 1, decimation)

        # cp_x shape (batch_size, num_legs, 3, decimation)
        # cp_x already has decimation dimension thanks to is_S0, px.unsqueeze(-1) : shape(batch, legs, 3) -> shape(batch, legs, 3, 1)
        #     --------------- S0 ---------------                                              ------------- S1 -------------
        cp1 = (self.p0_lw.unsqueeze(-1) * is_S0)                                            + (p1_lw.unsqueeze(-1) * ~is_S0)
        cp2 = (self.p0_lw.unsqueeze(-1) * is_S0)                                            + (torch.cat((p2_lw[:,:,:2], p1_lw[:,:,2:]), dim=2).unsqueeze(-1) * ~is_S0)
        cp3 = (torch.cat((self.p0_lw[:,:,:2], p1_lw[:,:,2:]), dim=2).unsqueeze(-1) * is_S0) + (p2_lw.unsqueeze(-1) * ~is_S0)
        cp4 = (p1_lw.unsqueeze(-1) * is_S0)                                                 + (p2_lw.unsqueeze(-1) * ~is_S0)


        # --- Step 4. Prepare parameters to compute interpolation trajectory in one operation -> matrix multiplication
        # Prepare swing phase traj to be multplied by cp_x : shape(batch_size, num_leg, decimation) -> (batch_size, num_leg, 1, decimation) (unsqueezed(2) but is_S0 is already unsqueezed (ie in the right shape))
        # swing phase may be > 1 if we reach the end of the traj, thus we clamp it to 1. 
        # Moreover, S0 and S1 takes values in [0,1], thus swing phase need to be double (and modulo 1) to be corrected
        phase_traj = (2 * swing_phase_traj.unsqueeze(2) - 1*(~is_S0)).clamp(0,1) # ie. double_swing_phase_traj


        # --- Step 5. Compute the interpolation trajectory
        # shape (batch_size, num_legs, 3, decimation)
        desired_foot_pos_traj_lw = cp1*(1 - phase_traj)**3 + 3*cp2*(phase_traj)*(1 - phase_traj)**2 + 3*cp3*((phase_traj)**2)*(1 - phase_traj) + cp4*(phase_traj)**3
        desired_foot_vel_traj_lw = 3*(cp2 - cp1)*(1 - phase_traj)**2 + 6*(cp3 - cp2)*(1 - phase_traj)*(phase_traj) + 3*(cp4 - cp3)*(phase_traj)**2
        desired_foot_acc_traj_lw = 6*(1 - phase_traj) * (cp3 - 2*cp2 + cp1) + 6 * (phase_traj) * (cp4 - 2*cp3 + cp2)

        # shape (batch_size, num_legs, 9, decimation) (9 = xyz_pos, xzy_vel, xyz_acc)
        pt_lw = torch.cat((desired_foot_pos_traj_lw, desired_foot_vel_traj_lw, desired_foot_acc_traj_lw), dim=2)

        
        # --- Compute the full trajectory for plotting and debugging purposes ---
        if self.verbose_md:
            # Shape (1,1,1,22)
            full_phase_traj = torch.cat((torch.arange(start=0, end=1.01, step=0.1, device=self._device), torch.arange(start=0, end=1.01, step=0.1, device=self._device))).unsqueeze(0).unsqueeze(1).unsqueeze(2) # [0, 0.1, 0.2, ..., 0.9, 1.0, 0.0, 0.1, ..., 1.0]
            is_S0 = (torch.arange(start=0, end=22, step=1, device=self._device) < 11).unsqueeze(0).unsqueeze(1).unsqueeze(2)

            # Recompute cp_x with the new 'decimation' that comes from the new is_S0
            # cp_x shape (batch_size, num_legs, 3, 22)
            # cp_x already has decimation dimension thanks to is_S0, px.unsqueeze(-1) : shape(batch, legs, 3) -> shape(batch, legs, 3, 1)
            #     --------------- S0 ---------------                                              ------------- S1 -------------
            cp1 = (self.p0_lw.unsqueeze(-1) * is_S0)                                            + (p1_lw.unsqueeze(-1) * ~is_S0)
            cp2 = (self.p0_lw.unsqueeze(-1) * is_S0)                                            + (torch.cat((p2_lw[:,:,:2], p1_lw[:,:,2:]), dim=2).unsqueeze(-1) * ~is_S0)
            cp3 = (torch.cat((self.p0_lw[:,:,:2], p1_lw[:,:,2:]), dim=2).unsqueeze(-1) * is_S0) + (p2_lw.unsqueeze(-1) * ~is_S0)
            cp4 = (p1_lw.unsqueeze(-1) * is_S0)                                                 + (p2_lw.unsqueeze(-1) * ~is_S0)

            # Compute the full trajectory
            # shape (batch_size, num_legs, 3, 22)
            desired_foot_pos_traj_lw = cp1*(1 - full_phase_traj)**3 + 3*cp2*(full_phase_traj)*(1 - full_phase_traj)**2 + 3*cp3*((full_phase_traj)**2)*(1 - full_phase_traj) + cp4*(full_phase_traj)**3
            desired_foot_vel_traj_lw = 3*(cp2 - cp1)*(1 - full_phase_traj)**2 + 6*(cp3 - cp2)*(1 - full_phase_traj)*(full_phase_traj) + 3*(cp4 - cp3)*(full_phase_traj)**2
            desired_foot_acc_traj_lw = 6*(1 - full_phase_traj) * (cp3 - 2*cp2 + cp1) + 6 * (full_phase_traj) * (cp4 - 2*cp3 + cp2)

            full_pt_lw = torch.cat((desired_foot_pos_traj_lw, desired_foot_vel_traj_lw, desired_foot_acc_traj_lw), dim=2)
        else : full_pt_lw = torch.empty((self._num_envs, self._num_legs, 9, 22),device=self._device)

        return pt_lw, full_pt_lw


# ----------------------------------- Inner Loop ------------------------------
    def compute_control_output(self, F0_star_lw: torch.Tensor, c0_star: torch.Tensor, pt_i_star_lw: torch.Tensor, p_lw:torch.Tensor, p_dot_lw:torch.Tensor, q_dot: torch.Tensor, jacobian_lw: torch.Tensor, jacobian_dot_lw: torch.Tensor, mass_matrix: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """ Compute the output torque to be applied to the system
        typically, it would compute :
            - T_stance_phase = stance_leg_controller(GRF, q, c) # Update the jacobian with the new joint position.
            - T_swing_phase = swing_leg_controller(trajectory, q, q_dot, c) # Feedback lineraization control - trajectory computed with a spline to be followed - new updated joint controller.
            - T = (T_stance_phase * c0_star) + (T_swing_phase * (~c0_star))
        and return T
        Note :
            The variable are in the 'local' world frame _wl. This notation is introduced to avoid confusion with the 'global' world frame, where all the batches coexists.
        
        Args:
            - F0*_lw (th.Tensor): Opt. Ground Reac. Forces (GRF)   in _lw   of shape(batch_size, num_legs, 3)
            - c0* (torch.bool)  : Optimized foot contact sequence           of shape(batch_size, num_legs)
            - pt_i*_lw  (Tensor): Opt. Foot point in swing phase   in _lw   of shape(batch_size, num_legs, 9) (9 = pos, vel, acc)
            - p_lw (trch.Tensor): Feet Position  (latest from sim) in _lw   of shape(batch_size, num_legs, 3)
            - p_dot_lw  (Tensor): Feet velocity  (latest from sim) in _lw   of shape(batch_size, num_legs, 3)
            - q_dot (tch.Tensor): Joint velocity (latest from sim)          of shape(batch_size, num_legs, num_joints_per_leg)
            - jacobian_lw (Tsor): Jacobian -> joint frame to in _lw frame   of shape(batch_size, num_legs, 3, num_joints_per_leg)
            - jacobian_dot_lw   : Jacobian derivative (forward euler)in _lw of shape(batch_size, num_legs, 3, num_joints_per_leg)
            - mass_matrix (Tsor): Mass Matrix in joint space                of shape(batch_size, num_legs, num_joints_per_leg, num_joints_per_leg)
            - h   (torch.Tensor): C(q,q_dot) + G(q) (corr. and grav F.)     of shape(batch_size, num_legs, num_joints_per_leg)

        Returns:
            - T   (torch.Tensor): control output (ie. Joint Torques)        of shape(batch_size, num_legs, num_joints_per_leg)
        """

        # Get the swing torque from the swing controller 
        # T_swing Shape (batch_size, num_legs, num_joints_per_leg)
        T_swing = self.swing_leg_controller(pt_i_star_lw=pt_i_star_lw, p_lw=p_lw, p_dot_lw=p_dot_lw, q_dot=q_dot, jacobian_lw=jacobian_lw, jacobian_dot_lw=jacobian_dot_lw, mass_matrix=mass_matrix, h=h)

        # Get the stance torque from the stance controller 
        # T_stance Shape (batch_size, num_legs, num_joints_per_leg)
        T_stance = self.stance_leg_controller(F0_star_lw=F0_star_lw, jacobian_lw=jacobian_lw)

        # Compute the final torque : keep T_stance for leg in stance and T_swing for leg in swing
        # T shape (batch_size, num_legs, num_joints_per_leg) , c0* shape(batch_size, num_legs) -> unsqueezed(-1) -> (batch_size, num_legs, 1)
        T = (T_stance * c0_star.unsqueeze(-1))  +  (T_swing * (~c0_star.unsqueeze(-1))) 

        # Save variables
        self.p_lw_sim_prev = p_lw # Used in genereate trajectory

        return T


    def swing_leg_controller(self, pt_i_star_lw: torch.Tensor, p_lw:torch.Tensor, p_dot_lw:torch.Tensor, q_dot: torch.Tensor,
                             jacobian_lw: torch.Tensor, jacobian_dot_lw: torch.Tensor, mass_matrix: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """ Given feet contact, and desired feet trajectory : compute joint torque with feedback linearization control
        T = M(q)*J⁻¹[p_dot_dot - J_dot(q)*q_dot] + C(q,q_dot) + G(q)
        Note :
            The variable are in the 'local' world frame _wl. This notation is introduced to avoid confusion with the 'global' world frame, where all the batches coexists.

        Args:
            - pt_i_lw*  (Tensor): Opt. Foot point in swing phase in _lw     of shape(batch_size, num_legs, 9) (9 = pos, vel, acc)
            - p_lw (trch.Tensor): Feet Position  in _lw                     of shape(batch_size, num_legs, 3)
            - p_dot_lw  (Tensor): Feet velocity  in _lw                     of shape(batch_size, num_legs, 3)
            - q_dot (tch.Tensor): Joint velocity                            of shape(batch_size, num_legs, num_joints_per_leg)
            - jacobian_lw (Tsor): Jacobian -> joint frame to _lw frame      of shape(batch_size, num_legs, 3, num_joints_per_leg)
            - jacobian_dot_lw   : Jacobian derivative (forward euler)in _lw of shape(batch_size, num_legs, 3, num_joints_per_leg)
            - mass_matrix (Tsor): Mass Matrix in joint space                of shape(batch_size, num_legs, num_joints_per_leg, num_joints_per_leg)
            - h   (torch.Tensor): C(q,q_dot) + G(q) (corr. and grav F.)     of shape(batch_size, num_legs, num_joints_per_leg)

        Returns:
            - T_swing (t.Tensor): Swing Leg joint torques               of shape(batch_size, num_legs, num_joints_per_leg)
        """

        # Intermediate variables : position, velocity errors : frame independant, desired acceleration in 'local' world frame
        pos_err = pt_i_star_lw[:,:,0:3] - p_lw
        vel_err = pt_i_star_lw[:,:,3:6] - p_dot_lw
        des_foot_acc_lw = pt_i_star_lw[:,:,6:9]

        # Intermediary step : p_dot_dot
        # Compute the desired acceleration : with a PD controller thanks to the feedback linearization
        # Shape (batch_size, num_legs, 3)
        p_dot_dot_lw = des_foot_acc_lw + self.swing_ctrl_pos_gain_fb * (pos_err) + self.swing_ctrl_vel_gain_fb * (vel_err)
        # p_dot_dot_lw = self.swing_ctrl_pos_gain_fb * (pos_err) + self.swing_ctrl_vel_gain_fb * (vel_err)
        # p_dot_dot_lw = self.swing_ctrl_pos_gain_fb * (pos_err) 

        # Compute  the inverse jacobian. This synchronise CPU and GPU
        # Compute pseudo-inverse -> to be resilient to any number of joint per legs (not restricted to square matrix)
        # Changed shape from (batch_size, num_legs, 3, num_joints_per_leg) to -> (batch_size, num_legs, num_joints_per_leg, 3)
        jacobian_inv = torch.linalg.pinv(jacobian_lw)
        
        # Intermediary step : J(q)*q_dot            (batch_size, num_legs, 3, num_joints_per_leg) * (batch_size, num_legs, num_joints_per_leg)
        # Must unsqueeze q_dot to perform matmul (ie add a singleton dimension on last position)
        # change q_dot shape from (batch_size, num_legs, num_joints_per_leg) to (batch_size, num_legs, num_joints_per_leg, 1)
        # J_dot_x_q_dot is of shape (batch_size, num_legs, 3) (The singleton dim is dropped by the squeeze operation)
        J_dot_x_q_dot = torch.matmul(jacobian_dot_lw, q_dot.unsqueeze(-1)).squeeze(-1)

        # Intermediary step : [p_dot_dot - J(q)*q_dot]          : Shape (batch_size, num_legs, 3)
        p_dot_dot_min_J_dot_x_q_dot = p_dot_dot_lw - J_dot_x_q_dot

        # Intermediary step : J⁻¹[p_dot_dot - J(q)*q_dot]       : (batch_size, num_legs, num_joints_per_leg, 3) * (batch_size, num_legs, 3)
        # Shape is (batch_size, num_legs, num_joints_per_leg)
        J_inv_p_dot_dot_min_J_dot_x_q_dot = torch.matmul(jacobian_inv, p_dot_dot_min_J_dot_x_q_dot.unsqueeze(-1)).squeeze(-1)

        # Intermediary step : M(q)*J⁻¹[p_dot_dot - J(q)*q_dot]  : (batch_size, num_legs, num_joints_per_leg, num_joints_per_leg) * (batch_size, num_legs, num_joints_per_leg)
        # Shape is (batch_size, num_legs, num_joints_per_leg)
        M_J_inv_p_dot_dot_min_J_dot_x_q_dot = torch.matmul(mass_matrix, J_inv_p_dot_dot_min_J_dot_x_q_dot.unsqueeze(-1)).squeeze(-1)

        # Final step        : # Shape is (batch_size, num_legs, num_joints_per_leg, num_joints_per_leg)
        T_swing = torch.add(M_J_inv_p_dot_dot_min_J_dot_x_q_dot, h)

        return T_swing
    

    def stance_leg_controller(self, F0_star_lw: torch.Tensor, jacobian_lw: torch.Tensor) -> torch.Tensor:
        """ Given GRF and contact sequence -> compute joint torques using the jacobian : T = -J*F
        1. compute the jacobian using the simulation tool : end effector jacobian wrt to robot base
        2. compute the stance torque : T = -J*F
        Note :
            The variable are in the 'local' world frame _wl. This notation is introduced to avoid confusion with the 'global' world frame, where all the batches coexists.

        Args:
            - F0*_lw (th.Tensor): Opt. Ground Reac. Forces (GRF) in w fr.   of shape(batch_size, num_legs, 3)
            - jacobian_lw (Tsor): Jacobian -> joint space to world frame    of shape(batch_size, num_legs, 3, num_joints_per_leg)

        Returns:
            - T_stance(t.Tensor): Stance Leg joint Torques              of shape(batch_size, num_legs, num_joints_per_leg)
        """

        # Transpose the jacobian -> In batch operation : permut the last two dimensions 
        # shape(batch_size, num_legs, 3, num_joints_per_leg) -> shape(batch_size, num_legs, num_joints_per_leg, 3)
        jacobian_lw_T = jacobian_lw.transpose(-1,-2)

        # Add a singleton dimension on last position to enable matmul operation 
        # shape(batch_size, num_legs, 3) -> shape(batch_size, num_legs, 3, 1)
        F0_star_lw_unsqueezed = F0_star_lw.unsqueeze(-1)

        # Perform the matrix multiplication T = - J^T * F
        # shape(batch_size, num_legs, num_joints_per_leg, 1)
        T_unsqueezed= -torch.matmul(jacobian_lw_T, F0_star_lw_unsqueezed)

        # Supress the singleton dimension added for the matmul operation
        # shape(batch_size, num_legs, num_joints_per_leg, 1) -> shape(batch_size, num_legs, num_joints_per_leg)
        T_stance = T_unsqueezed.squeeze(-1)

        return T_stance


class samplingController(modelBaseController):
    """
    TODO

    Properties
        samplingOptimizer
    """

    def __init__(self, verbose_md, device, num_envs, num_legs, robot_mass, mu, dt_out, decimation, dt_in, p_default_lw: torch.Tensor, step_height, foot_offset, swing_ctrl_pos_gain_fb, swing_ctrl_vel_gain_fb, env: ManagerBasedRLEnv, optimizerCfg):
        """ Initial the Model Base Controller and define an optimizer """

        # Initialise the Model Base Controller
        super().__init__(verbose_md, device, num_envs, num_legs, robot_mass, mu, dt_out, decimation, dt_in, p_default_lw, step_height, foot_offset, swing_ctrl_pos_gain_fb, swing_ctrl_vel_gain_fb)

        self.samplingOptimizer = SamplingOptimizer(device=device,num_legs=num_legs, robot_mass=robot_mass, mu=mu,env=env, optimizerCfg=optimizerCfg)  

        # To enable or not the optimizer at run time
        self.optimizer_active = True


    def reset(self, env_ids: Sequence[int] | None,  p_default_lw: torch.Tensor) -> None:
        """ Reset the sampling optimizer internal values"""
        super().reset(env_ids,  p_default_lw)
        self.samplingOptimizer.reset()


    def process_latent_variable(self, f: torch.Tensor, d: torch.Tensor, p_lw: torch.Tensor, delta_F_lw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Given the latent variable z=[f,d,p,F], return the optimize latent variable f*, d*, c*, p*, F*, pt* full_pt
        Note :
            The variable are in the 'horizontal' frame _h : (x_b, y_b, z_w, 0, 0, yaw_b)

        Args:
            f            (Tensor): Prior leg frequency                             of shape (batch_size, num_legs)
            d            (Tensor): Prior stepping duty cycle                       of shape (batch_size, num_legs)
            p_lw         (Tensor): Prior foot touch down delta seq. in _h          of shape (batch_size, num_legs, 3, p_param)
            delta_F_lw   (Tensor): Prior Ground Reac. Forces delta(GRF) seq.       of shape (batch_size, num_legs, 3, F_param)

        Returns:
            f*           (Tensor): Opt. leg frequency                              of shape (batch_size, num_legs)
            d*           (Tensor): Opt. leg duty cycle                             of shape (batch_size, num_legs)
            c0*          (Tensor): Optimized foot contact sequence                 of shape (batch_size, num_legs)
            p0*_lw       (Tensor): Optimized foot touch down delta seq. in _h      of shape (batch_size, num_legs, 3)
            F0*_lw       (Tensor): Opt. Gnd Reac. F. delta(GRF) seq.   in _h       of shape (batch_size, num_legs, 3)
            pt*_lw (torch.Tensor): Optimized foot swing traj.     in _lw of shape (batch_size, num_legs, 9, decimation)  (9 = pos, vel, acc)
            full_pt_lw (t.Tensor): Full swing traj for ploting           of shape (batch_size, num_legs, 9, 22)
        """

        # Call the optimizer
        if self.optimizer_active: # To enable or not optimization in real time
            f_star, d_star, p0_star_lw, F0_star_lw = self.samplingOptimizer.optimize_latent_variable(f=f, d=d, p_lw=p_lw, delta_F_lw=delta_F_lw, phase=self.phase)
        else : 
            f_star, d_star = f, d
            p0_star_lw = self.samplingOptimizer.interpolation_p(parameters=p_lw,       step=0, horizon=1) # shape(batch, num_legs, 3)
            delta_F0_star_lw = self.samplingOptimizer.interpolation_F(parameters=delta_F_lw, step=0, horizon=1) # shape(batch, num_legs, 3)

        # Compute the contact sequence and update the phase (c at current time, phase at next time (after one f*dt increment)) (need at least an horizon 2 to have c_0 and phase_1)
        c_star, next_phase = gait_generator(f=f_star, d=d_star, phase=self.phase, horizon=2, dt=self._dt_out)
        c0_star = c_star[:,:,0] # shape (batch_size, num_legs)

        # Generate the swing trajectory
        pt_star_lw, full_pt_lw = self.full_swing_trajectory_generator(p_lw=p0_star_lw, c0=c0_star, d=d_star, f=f_star)

        # If no optimizer -> Add the gravity compensation + friction cone constraint
        if not self.optimizer_active:
            num_legs_in_contact = torch.sum(c0_star, dim=1).clamp_min(1) #shape(batch_size)
            F0_star_lw = delta_F0_star_lw # shape(batch_size, num_legs, 3)
            F0_star_lw[:,:,2] += (c0_star * (self._robot_mass * 9.81) / num_legs_in_contact.unsqueeze(-1)) # shape(batch_size, num_legs, F_param)
            F0_star_lw[:,:,2] = F0_star_lw[:,:,2].clamp(min=0)

            # Enforce the friction cone constraints
            F0_star_lw = enforce_friction_cone_constraints_torch(F=F0_star_lw, mu=self._mu, F_z_min=0.0, F_z_max=9.81*self._robot_mass) # shape(num_samples, num_legs, 3)

        # update the phase variable
        self.phase = next_phase

        return f_star, d_star, c0_star, p0_star_lw, F0_star_lw, pt_star_lw, full_pt_lw 

class samplingTrainer(modelBaseController):
    """
    """
    def __init__(self, verbose_md, device, num_envs, num_legs, robot_mass, mu, dt_out, decimation, dt_in, p_default_lw: torch.Tensor, step_height, foot_offset, swing_ctrl_pos_gain_fb, swing_ctrl_vel_gain_fb, env: ManagerBasedRLEnv, optimizerCfg):
        """ Initial the Model Base Controller and define an optimizer """

        # Initialise the Model Base Controller
        super().__init__(verbose_md, device, num_envs, num_legs, robot_mass, mu, dt_out, decimation, dt_in, p_default_lw, step_height, foot_offset, swing_ctrl_pos_gain_fb, swing_ctrl_vel_gain_fb)

        self.samplingOptimizer = SamplingBatchedTrainer(device=device,num_legs=num_legs, robot_mass=robot_mass, mu=mu, env=env, optimizerCfg=optimizerCfg)  

        self.batched_cost = torch.zeros((env.num_envs), device=device)

    def process_latent_variable(self, f: torch.Tensor, d: torch.Tensor, p_lw: torch.Tensor, delta_F_lw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Given the latent variable z=[f,d,p,F], return the optimize latent variable f*, d*, c*, p*, F*, pt* full_pt
        Note :
            The variable are in the 'horizontal' frame _h : (x_b, y_b, z_w, 0, 0, yaw_b)

        Args:
            f            (Tensor): Prior leg frequency                             of shape (batch_size, num_legs)
            d            (Tensor): Prior stepping duty cycle                       of shape (batch_size, num_legs)
            p_lw         (Tensor): Prior foot touch down delta seq. in _h          of shape (batch_size, num_legs, 3, p_param)
            delta_F_lw   (Tensor): Prior Ground Reac. Forces delta(GRF) seq.       of shape (batch_size, num_legs, 3, F_param)

        Returns:
            f*           (Tensor): Opt. leg frequency                              of shape (batch_size, num_legs)
            d*           (Tensor): Opt. leg duty cycle                             of shape (batch_size, num_legs)
            c0*          (Tensor): Optimized foot contact sequence                 of shape (batch_size, num_legs)
            p0*_lw       (Tensor): Optimized foot touch down delta seq. in _h      of shape (batch_size, num_legs, 3)
            F0*_lw       (Tensor): Opt. Gnd Reac. F. delta(GRF) seq.   in _h       of shape (batch_size, num_legs, 3)
            pt*_lw (torch.Tensor): Optimized foot swing traj.     in _lw of shape (batch_size, num_legs, 9, decimation)  (9 = pos, vel, acc)
            full_pt_lw (t.Tensor): Full swing traj for ploting           of shape (batch_size, num_legs, 9, 22)
        """

        # Call the optimizer
        batched_cost, p0_star_lw, F0_star_lw = self.samplingOptimizer.compute_batched_rollout_cost(f=f, d=d, p_lw=p_lw, delta_F_lw=delta_F_lw, phase=self.phase)

        f_star = f
        d_star = d
        self.batched_cost = batched_cost

        # Compute the contact sequence and update the phase (c at current time, phase at next time (after one f*dt increment)) (need at least an horizon 2 to have c_0 and phase_1)
        c_star, next_phase = gait_generator(f=f, d=d, phase=self.phase, horizon=2, dt=self._dt_out)
        c0_star = c_star[:,:,0] # shape (batch_size, num_legs)

        # Generate the swing trajectory
        pt_star_lw, full_pt_lw = self.full_swing_trajectory_generator(p_lw=p0_star_lw, c0=c0_star, d=d_star, f=f_star)

        # update the phase variable
        self.phase = next_phase

        return f_star, d_star, c0_star, p0_star_lw, F0_star_lw, pt_star_lw, full_pt_lw 
    
    
# ---------------------------------- Optimizer --------------------------------
class SamplingOptimizer():
    """ Model Based optimizer based on the centroidal model 
    
    Properties
        c_actual (Tensor): Contact sequence determined at previous iteration of shape (batch_size, num_leg)
        
    """

    def __init__(self, device, num_legs, robot_mass, mu, env:ManagerBasedRLEnv, optimizerCfg):
        """ 
        Args :
            device                 : 'cuda' or 'cpu' 
            num_legs               : Number of legs of the robot. Used for variable dimension definition
            num_samples            : Number of samples used for the sampling optimizer
            sampling_horizon       : Number of time steps the sampling optimizer is going to predict
            discretization_time    : Duration of a time step
            interpolation_F_method : Method to reconstruct GRF action from provided GRF warm start :
                                    can be 'discrete' (one action per time step is provided) or 'cubic spine' (a set of parameters for every time step)
            interpolation_p_method : Method to reconstruct foot touch down position action from provided warm start :
                                    can be 'discrete' (one action per time step is provided) or 'cubic spine' (a set of parameters for every time step)
        """

        # Save the environment
        self._env: ManagerBasedRLEnv = env

        # General variables
        self.device = device
        self.num_legs = num_legs
        
        # Save Config
        self.cfg = optimizerCfg

        # Optimizer configuration
        self.num_samples = optimizerCfg.num_samples
        self.sampling_horizon = optimizerCfg.prevision_horizon
        self.mpc_dt = optimizerCfg.discretization_time 
        self.num_optimizer_iterations = optimizerCfg.num_optimizer_iterations
        self.mu = mu #optimizerCfg.mu

        self.number_of_policy = optimizerCfg.multipolicy
        self.wining_policy = 0

        # Define Interpolation method for GRF and interfer GRF input size 
        if   optimizerCfg.parametrization_F == 'cubic_spline': 
            self.interpolation_F=compute_cubic_spline
            self.F_param = 4
        elif optimizerCfg.parametrization_F == 'discrete':
            self.interpolation_F=compute_discrete
            self.F_param = self.sampling_horizon
        elif optimizerCfg.parametrization_F == 'from_discrete_fit_spline':
            self.interpolation_F=compute_cubic_spline
            self.F_param = 4
        elif optimizerCfg.parametrization_F == 'from_single_expand_discrete':
            self.interpolation_F=compute_discrete
            self.F_param = self.sampling_horizon

        else : raise NotImplementedError('Request interpolation method is not implemented yet')

        # Define Interpolation method for foot touch down position and interfer foot touch down position input size 
        if   optimizerCfg.parametrization_p == 'cubic_spline': 
            self.interpolation_p=compute_cubic_spline
            self.p_param = 4
        elif optimizerCfg.parametrization_p == 'discrete': 
            self.interpolation_p=compute_discrete
            self.p_param = self.sampling_horizon
        elif optimizerCfg.parametrization_p == 'from_discrete_fit_spline': 
            self.interpolation_p=compute_cubic_spline
            self.p_param = 4
        elif optimizerCfg.parametrization_p == 'from_single_expand_discrete':
            self.interpolation_p=compute_discrete
            self.p_param = self.sampling_horizon
        elif optimizerCfg.parametrization_p == 'first':
            self.interpolation_p=compute_unique
            self.p_param = 1
        else : raise NotImplementedError('Request interpolation method is not implemented yet')

        # Input and State dimension for centroidal model : hardcoded because the centroidal model is hardcoded # TODO get rid of these
        self.state_dim = 24 # CoM_pos(3) + lin_vel(3) + CoM_pose(3) + ang_vel(3) + foot_pos(12) 
        self.input_dim = 12 # GRF(12) (foot touch down pos is state and an input)


        # TODO Get these value properly
        self.gravity_lw = torch.tensor((0.0, 0.0, -9.81), device=self.device) # shape(3) #self._env.sim.cfg.gravity
        # self.robot_mass = 24.64
        # self.robot_mass = 20.6380
        self.robot_mass = robot_mass
        self.robot_inertia = torch.tensor([[ 0.2310941359705289,   -0.0014987128245817424, -0.021400468992761768 ], # shape (3,3)
                                           [-0.0014987128245817424, 1.4485084687476608,     0.0004641447134275615],
                                           [-0.021400468992761768,  0.0004641447134275615,  1.503217877350808    ]],device=self.device)
        self.inv_robot_inertia = torch.linalg.inv(self.robot_inertia) # shape(3,3)
        self.F_z_min = 0
        self.F_z_max = self.robot_mass*9.81

        # Boolean to enable variable optimization or not
        self.optimize_f = optimizerCfg.optimize_f
        self.optimize_d = optimizerCfg.optimize_d
        self.optimize_p = optimizerCfg.optimize_p
        self.optimize_F = optimizerCfg.optimize_F

        # Sampling law
        if   optimizerCfg.sampling_law == 'normal' : self.sampling_law = self.normal_sampling
        elif optimizerCfg.sampling_law == 'uniform': self.sampling_law = self.uniform_sampling

        # Wether to force clip the sample to std
        self.clip_sample = optimizerCfg.clip_sample

        # How much of the previous solution is used to generate samples compare to the provided guess (in [0,1])
        self.propotion_previous_solution = optimizerCfg.propotion_previous_solution

        # Define the height reference for the tracking
        self.height_ref = optimizerCfg.height_ref

        # Define Variance for the sampling law TODO Get the value properly
        self.std_f = torch.tensor((0.05), device=device)
        self.std_d = torch.tensor((0.05), device=device)
        self.std_p = torch.tensor((0.02), device=device)
        self.std_F = torch.tensor((5.00), device=device)
        self.spline_std_F = 10.0

        # Current contact sequence : used to reset solution when switch to swing shape (batch_size, num_leg)
        self.c_actual = torch.ones((1, self.num_legs), device=device)

        # State weight - shape(state_dim)
        self.Q_vec = torch.zeros(self.state_dim, device=self.device)
        self.Q_vec[0]  = 0.0        #com_x
        self.Q_vec[1]  = 0.0        #com_y
        self.Q_vec[2]  = 111500     #com_z
        self.Q_vec[3]  = 5000       #com_vel_x
        self.Q_vec[4]  = 5000       #com_vel_y
        self.Q_vec[5]  = 200        #com_vel_z
        self.Q_vec[6]  = 11200      #base_angle_roll
        self.Q_vec[7]  = 11200      #base_angle_pitch
        self.Q_vec[8]  = 0.0        #base_angle_yaw
        self.Q_vec[9]  = 20         #base_angle_rates_x
        self.Q_vec[10] = 20         #base_angle_rates_y
        self.Q_vec[11] = 600        #base_angle_rates_z

        # Input weight - shape(input_dim)
        self.R_vec = torch.zeros(self.input_dim, device=self.device)

        # Initialize the best solution
        self.f_best = 1.5*torch.ones( (1,self.num_legs),                 device=device)
        self.d_best = 0.6*torch.ones( (1,self.num_legs),                 device=device)
        self.p_best =     torch.zeros((1,self.num_legs,3,self.p_param ), device=device)
        self.F_best =     torch.zeros((1,self.num_legs,3,self.F_param ), device=device)
        # self.F_best[:,:,2,:] = 50.0

        # For plotting
        self.live_plot = True
        self.robot_height_list = [0.0]
        self.robot_height_ref_list = [0.0]
        self.cost_list = [0.0]
        self.wining_policy_list = [0]
        self.FL_foot_list = [np.array([0.0, 0.0, 0.0])]
        self.FR_foot_list = [np.array([0.0, 0.0, 0.0])]
        self.RL_foot_list = [np.array([0.0, 0.0, 0.0])]
        self.RR_foot_list = [np.array([0.0, 0.0, 0.0])]

        self.previous_best_cost = torch.tensor(0.0, device=device)


    def reset(self):
        # Reset the best solution
        self.f_best = 1.5*torch.ones( (1,self.num_legs),                 device=self.device)
        self.d_best = 0.6*torch.ones( (1,self.num_legs),                 device=self.device)
        self.p_best =     torch.zeros((1,self.num_legs,3,self.p_param ), device=self.device)
        self.F_best =     torch.zeros((1,self.num_legs,3,self.F_param ), device=self.device) 
        # self.F_best[:,:,2,:] = 50.0


    def optimize_latent_variable(self, f:torch.Tensor, d:torch.Tensor, p_lw:torch.Tensor, delta_F_lw:torch.Tensor, phase:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Given latent variable f,d,F,p, returns f*,d*,F*,p*, optimized with a sampling optimization 
        
        Args :
            f      (Tensor): Leg frequency                of shape(batch_size, num_leg)
            d      (Tensor): Leg duty cycle               of shape(batch_size, num_leg)
            p_lw   (Tensor): Foot touch down position     of shape(batch_size, num_leg, 3, p_param)
            F_lw   (Tensor): ground Reaction Forces       of shape(batch_size, num_leg, 3, F_param)
            phase  (Tensor): Current feet phase           of shape(batch_size, num_leg)

        Returns :
            f_star    (Tensor): Leg frequency                of shape(batch_size, num_leg)
            d_star    (Tensor): Leg duty cycle               of shape(batch_size, num_leg)
            p_star_lw (Tensor): Foot touch down position     of shape(batch_size, num_leg, 3, p_param)
            F_star_lw (Tensor): ground Reaction Forces       of shape(batch_size, num_leg, 3, F_param)
        """
        print()

        for i in range(self.num_optimizer_iterations):
            print(f'\niteration {i}')
            # --- Step 1 : Generate the samples and bound them to valid input
            f_samples, d_samples, p_lw_samples, delta_F_lw_samples = self.generate_samples(iter=i, f=f, d=d, p_lw=p_lw, delta_F_lw=delta_F_lw)

            # --- Step 2 : Given f and d samples -> generate the contact sequence for the samples, contact sequence start at time t_0, next phase is phase at time t_1
            c_samples, next_phase = gait_generator(f=f_samples, d=d_samples, phase=phase.squeeze(0), horizon=max(2, self.sampling_horizon), dt=self.mpc_dt)

            # --- Step 2 : prepare the variables 
            initial_state, reference_seq_state, reference_seq_input_samples, action_param_samples = self.prepare_variable_for_compute_rollout(c_samples=c_samples, p_lw_samples=p_lw_samples, delta_F_lw_samples=delta_F_lw_samples, feet_in_contact=self.c_actual[0,])

            # --- Step 3 : Compute the rollouts to find the rollout cost : can't used named argument with VMAP...
            cost_samples = self.compute_rollout( initial_state, reference_seq_state, reference_seq_input_samples, action_param_samples, c_samples)

            # --- Step 4 : Given the samples cost, find the best control action
            f_star, d_star, p0_star_lw, F0_star_lw = self.find_best_actions(cost_samples, f_samples, d_samples, c_samples, p_lw_samples, delta_F_lw_samples)


        return f_star, d_star, p0_star_lw, F0_star_lw # p_star_lw, F_star_lw


    def prepare_variable_for_compute_rollout(self, c_samples:torch.Tensor, p_lw_samples:torch.Tensor, delta_F_lw_samples:torch.Tensor, feet_in_contact:torch.Tensor) -> tuple[dict, dict, dict, dict]:
        """ Helper function to modify the embedded state, reference and action to be used with the 'compute_rollout' function

        Note :
            Initial state and reference can be retrieved only with the environment
            _w   : World frame (inertial frame)
            _lw  : World frame centered at the environment center -> local world frame
            _b   : Base frame (attached to robot's base)
            _h   : Horizontal frame -> Base frame position for xy, world frame for z, roll, pitch, base frame for yaw
            _bw  : Base/world frame -> Base frame position, world frame rotation

        Args :
            env  (ManagerBasedRLEnv): Environment manager to retrieve all necessary simulation variable
            c_samples       (t.bool): Foot contact sequence sample                                                      of shape(num_samples, num_legs, sampling_horizon)
            p_lw_samples    (Tensor): Foot touch down position                                                          of shape(num_samples, num_legs, 3, p_param)
            F_lw_samples    (Tensor): ground Reaction Forces                                                            of shape(num_samples, num_legs, 3, F_param)
            feet_in_contact (Tensor): Feet in contact, This is equivalent to c0                                         of shape(num_legs)

        
        Return :
            initial_state         (dict): Dictionnary containing the current robot's state
                pos_com_lw      (Tensor): CoM position in local world frame                             of shape(3)
                lin_com_vel_lw  (Tensor): CoM linear velocity in local world frame                      of shape(3)
                euler_xyz_angle (Tensor): CoM orientation (wrt. to l. world frame) as XYZ euler angle   of shape(3)
                ang_vel_com_b   (Tensor): CoM angular velocity as roll pitch yaw                        of shape(3)
                p_lw            (Tensor): Feet position in local world frame                            of shape(num_legs, 3)

            reference_seq_state   (dict): Dictionnary containing the robot's reference state along the prediction horizon
                pos_com_lw      (Tensor): CoM position in local world frame                             of shape(3, sampling_horizon)
                lin_com_vel_lw  (Tensor): CoM linear velocity in local world frame                      of shape(3, sampling_horizon)
                euler_xyz_angle (Tensor): CoM orientation (wrt. to l. world frame) as XYZ euler angle   of shape(3, sampling_horizon)
                ang_vel_com_b   (Tensor): CoM angular velocity as roll pitch yaw                        of shape(3, sampling_horizon)
                p_lw            (Tensor): Feet position in local world frame                            of shape(num_legs, 3, sampling_horizon)                

            reference_seq_input_samples (dict) 
                F_lw            (Tensor): Reference GRF sequence samples along the prediction horizon   of shape(num_sample, num_legs, 3, sampling_horizon)  

            action_param_samples  (dict): Dictionnary containing the robot's actions along the prediction horizon 
                p_lw            (Tensor): Foot touch down position in local world frame                 of shape(num_samples, num_legs, 3, p_param)
                delta_F_lw      (Tensor): delta GRF parameters in local world frame                     of shape(num_samples, num_legs, 3, F_param)
        """
        initial_state = {}
        reference_seq_state = {}
        reference_seq_input_samples = {}
        action_param_samples = {}

        # Check that a single robot was provided
        if self._env.num_envs > 1:
            assert ValueError('More than a single environment was provided to the sampling controller')

        # Retrieve robot from the scene : specify type to enable type hinting
        robot: Articulation = self._env.scene["robot"]

        # Retrieve indexes
        foot_idx = robot.find_bodies(".*foot")[0]


        # ----- Step 1 : Retrieve the initial state
        # Retrieve the robot position in local world frame of shape(3)
        com_pos_lw = (robot.data.root_pos_w - self._env.scene.env_origins).squeeze(0) # shape(3)
        # Compute proprioceptive height
        if (feet_in_contact == 0).all() : feet_in_contact = torch.ones_like(feet_in_contact) # if no feet in contact : robot is in the air, we use all feet to compute the height, not correct but avoid div by zero
        com_pos_lw[2] = robot.data.root_pos_w[:,2] - (torch.sum(((robot.data.body_pos_w[:, foot_idx,2]).squeeze(0)) * feet_in_contact)) / (torch.sum(feet_in_contact)) # height is proprioceptive

        # Retrieve the robot orientation in lw as euler angle ZXY of shape(3)
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(robot.data.root_quat_w) 
        roll, pitch, yaw = from_zero_twopi_to_minuspi_pluspi(roll, pitch, yaw) 
        euler_xyz_angle = torch.tensor((roll, pitch, yaw), device=self.device)

        # Retrieve the robot linear and angular velocity in base frame of shape(3)
        com_lin_vel_lw = (robot.data.root_lin_vel_w).squeeze(0) # shape(3) 
        com_ang_vel_b  = (robot.data.root_ang_vel_b).squeeze(0) # shape(3)

        # Retrieve the feet position in local world frame of shape(num_legs*3)
        p_w = (robot.data.body_pos_w[:, foot_idx,:]).squeeze(0) # shape(num_legs, 3) - foot position in w
        p_lw = p_w - self._env.scene.env_origins                      # shape(num_legs, 3) - foot position in lw

        # Prepare the state (at time t)
        initial_state['pos_com_lw']      = com_pos_lw       # shape(3)              # CoM position in local world frame, height is propriocetive
        initial_state['lin_com_vel_lw']  = com_lin_vel_lw   # shape(3)              # Linear Velocity in local world frame
        initial_state['euler_xyz_angle'] = euler_xyz_angle  # shape(3)              # Euler angle in XYZ convention
        initial_state['ang_vel_com_b']   = com_ang_vel_b    # shape(3)              # Angular velocity in base frame
        initial_state['p_lw']            = p_lw             # shape(num_legs, 3)    # Foot position in local world frame


        # ----- Step 2 : Retrieve the robot's reference along the integration horizon
        # Retrieve the speed command : (lin_vel_x_b, lin_vel_y_b, ang_vel_yaw_b)
        speed_command_b = (self._env.command_manager.get_command("base_velocity")).squeeze(0) # shape(3)

        # CoM reference position : tracked only for the height -> xy can be anything eg 0
        com_pos_ref_seq_lw = torch.tensor((0,0,self.height_ref), device=self.device).unsqueeze(1).expand(3, self.sampling_horizon) # shape(3, sampling_horizon)

        # The pose reference is (0,0) for roll and pitch, but the yaw must be integrated along the horizon (in world frame)
        euler_xyz_angle_ref_seq      = torch.zeros_like(com_pos_ref_seq_lw) # shape(3, sampling_horizon)
        euler_xyz_angle_ref_seq[2,:] = euler_xyz_angle[2] + (torch.arange(self.sampling_horizon, device=self.device) * (self.mpc_dt * speed_command_b[2])) # shape(sampling_horizon)

        # The linear speed reference is (x_dot_ref_b, y_dot_ref_b, 0) in base frame and must be rotated in world frame
        com_lin_vel_ref_seq_b = torch.zeros(3, device=self.device) # shape(3)
        com_lin_vel_ref_seq_b[0] = speed_command_b[0] # foward  velocity reference
        com_lin_vel_ref_seq_b[1] = speed_command_b[1] # lateral velocity reference

        # Retrieve the robot base orientation in the world frame as quaternions : shape(batch_size, 4)
        quat_robot_base_w = robot.data.root_quat_w 

        # From the quaternions compute the rotation matrix that rotates from base frame b to world frame w: shape(batch_size, 3,3) -> shape(3,3)
        R_b_to_w = math_utils.matrix_from_quat(quat_robot_base_w).squeeze(0)

        # Rotate the speed reference in world frame and expand to get the sampling horizon
        com_lin_vel_ref_seq_w = torch.matmul(R_b_to_w, com_lin_vel_ref_seq_b.unsqueeze(-1)).expand(3, self.sampling_horizon).clone().detach() # shape (3,3)@(3,1) -> (3,1) -> shape(3, sampling_horizon)

        # Add the effect of the yaw rate in world frame
        delta_yaw_seq = self.mpc_dt * speed_command_b[2] * torch.arange(0, self.sampling_horizon, device=self.device) # shape(sampling_horizon)

        com_lin_vel_ref_seq_w[0,:] = ((com_lin_vel_ref_seq_w[0,:] * torch.cos(delta_yaw_seq)) - (com_lin_vel_ref_seq_w[1,:] * torch.sin(delta_yaw_seq))) # shape(sampling_horizon)
        com_lin_vel_ref_seq_w[1,:] = ((com_lin_vel_ref_seq_w[0,:] * torch.sin(delta_yaw_seq)) + (com_lin_vel_ref_seq_w[1,:] * torch.cos(delta_yaw_seq))) # shape(sampling_horizon)

        # Get the angular velocity reference in base frame as euler rates : (0, 0, yaw rate)
        com_ang_vel_ref_seq_b = torch.zeros_like(com_pos_ref_seq_lw) # shape(3, sampling_horizon)

        # Angular velocity reference is base frame (roll_rate=0, pitch_rate=0, yaw_rate=yaw_rate_ref)
        com_ang_vel_ref_seq_b[2,:] = speed_command_b[2]  # shape(3, sampling_horizon)

        # Defining the foot position sequence is tricky.. Since we only have number of predicted step < sampling_horizon
        p_ref_seq_lw = torch.zeros((self.num_legs, 3, self.sampling_horizon), device=self.device) # shape(num_legs, 3, sampling_horizon) TODO Define this !

        # Compute the gravity compensation GRF along the horizon : of shape (num_samples, num_legs, 3, sampling_horizon)
        num_leg_contact_seq_samples = (torch.sum(c_samples, dim=1)).clamp(min=1) # Compute the number of leg in contact, clamp by minimum 1 to avoid division by zero. shape(num_samples, sampling_horizon)
        gravity_compensation_F_samples = torch.zeros((self.num_samples, self.num_legs, 3, self.sampling_horizon), device=self.device) # shape (num_samples, num_legs, 3, sampling_horizon)
        gravity_compensation_F_samples[:,:,2,:] =  c_samples * ((self.robot_mass * 9.81)/num_leg_contact_seq_samples).unsqueeze(1)    # shape (num_samples, num_legs, sampling_horizon)
        
        # Prepare the reference sequence (at time t, t+dt, etc.)
        reference_seq_state['pos_com_lw']      = com_pos_ref_seq_lw      # shape(3, sampling_horizon)           # CoM position reference in local world frame
        reference_seq_state['lin_com_vel_lw']  = com_lin_vel_ref_seq_w   # shape(3, sampling_horizon)           # Linear Velocity reference in local world frame
        reference_seq_state['euler_xyz_angle'] = euler_xyz_angle_ref_seq # shape(3, sampling_horizon)           # Euler angle reference in XYZ convention
        reference_seq_state['ang_vel_com_b']   = com_ang_vel_ref_seq_b   # shape(3, sampling_horizon)           # Angular velocity reference in base frame
        reference_seq_state['p_lw']            = p_ref_seq_lw            # shape(num_legs, 3, sampling_horizon) # Foot position in local world frame

        # Prepare the input sequence reference for the samples
        reference_seq_input_samples['F_lw'] = gravity_compensation_F_samples # shape(num_samples, num_legs, 3, sampling_horizon) # gravity compensation per samples along the horizon


        # ----- Step 3 : Retrieve the actions and prepare them with the correct method
        action_param_samples['p_lw']       = p_lw_samples.clone().detach()       # Foot touch down position samples          # shape(num_samples, num_legs, 3 p_param)
        action_param_samples['delta_F_lw'] = delta_F_lw_samples.clone().detach() # Delta Ground Reaction Forces samples      # shape(num_samples, num_legs, 3 F_param)


        # ----- Step 4 : Save variable for live plotting
        if self.live_plot : 
            # Save Robot's Height
            self.robot_height_list.append(com_pos_lw[2].cpu().numpy())
            if len(self.robot_height_list) > 100 : self.robot_height_list.pop(0)
            np.savetxt('live_variable/height.csv', [self.robot_height_list], delimiter=',', fmt='%.3f')

            # Save Robot's Height reference
            self.robot_height_ref_list.append(self.height_ref)
            if len(self.robot_height_ref_list) > 100 : self.robot_height_ref_list.pop(0)
            np.savetxt('live_variable/height_ref.csv', [self.robot_height_ref_list], delimiter=',', fmt='%.3f')

            # Save foot position's
            self.FL_foot_list.append(initial_state['p_lw'][0,:].cpu().numpy())
            if len(self.FL_foot_list) > 100 : self.FL_foot_list.pop(0)
            np.savetxt('live_variable/FL_foot.csv', self.FL_foot_list, delimiter=',', fmt='%.3f')
            self.FR_foot_list.append(initial_state['p_lw'][1,:].cpu().numpy())
            if len(self.FR_foot_list) > 100 : self.FR_foot_list.pop(0)
            np.savetxt('live_variable/FR_foot.csv', self.FR_foot_list, delimiter=',', fmt='%.3f')
            self.RL_foot_list.append(initial_state['p_lw'][2,:].cpu().numpy())
            if len(self.RL_foot_list) > 100 : self.RL_foot_list.pop(0)
            np.savetxt('live_variable/RL_foot.csv', self.RL_foot_list, delimiter=',', fmt='%.3f')
            self.RR_foot_list.append(initial_state['p_lw'][3,:].cpu().numpy())
            if len(self.RR_foot_list) > 100 : self.RR_foot_list.pop(0)
            np.savetxt('live_variable/RR_foot.csv', self.RR_foot_list, delimiter=',', fmt='%.3f')


        return initial_state, reference_seq_state, reference_seq_input_samples, action_param_samples


    def find_best_actions(self,cost_samples, f_samples: torch.Tensor, d_samples: torch.Tensor, c_samples: torch.Tensor, p_lw_samples: torch.Tensor, delta_F_lw_samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
        """ Given action samples and associated cost, filter invalid values and retrieves the best cost and associated actions
        
        Args 
            cost_samples       (Tensor): Associated trajectory cost          of shape(num_samples)
            f_samples          (Tensor): Leg frequency samples               of shape(num_samples, num_leg)
            d_samples          (Tensor): Leg duty cycle samples              of shape(num_samples, num_leg)
            c_samples          (Tensor): Leg contact sequence samples        of shape(num_samples, num_legs, time_horizon)
            p_lw_samples       (Tensor): Foot touch down position samples    of shape(num_samples, num_leg, 3, p_param)
            delta_F_lw_samples (Tensor): Ground Reaction forces samples      of shape(num_samples, num_leg, 3, F_param)
             
        Returns
            f_star             (Tensor): Best leg frequency                  of shape(1, num_leg)
            d_star             (Tensor): Best leg duty cycle                 of shape(1, num_leg)
            p0_star_lw         (Tensor): Best foot touch down position       of shape(1, num_leg, 3)
            F0_star_lw         (Tensor): Best ground Reaction Forces         of shape(1, num_leg, 3)  
        """

        # Saturate the cost in case of NaN or inf
        cost_samples[torch.isnan(cost_samples) | torch.isinf(cost_samples)] = 1e10

        # Take the best found control parameters
        best_index = torch.argmin(cost_samples)
        best_cost = cost_samples[best_index] # cost_samples.take(best_index)

        print('cost sample ',cost_samples)
        print('Best cost :', best_cost, ', best index :', best_index)

        # Retrieve from wich law it has been sampled
        num_samples_previous_best = int(self.num_samples * self.propotion_previous_solution)
        num_samples_RL = self.num_samples - num_samples_previous_best

        self.wining_policy = min((best_index // (num_samples_RL // self.number_of_policy)).cpu().numpy(), self.number_of_policy) # min to avoid having different number if previous best win



        # Retrieve best sample, given the best index
        f_star          = f_samples[best_index].unsqueeze(0).clone().detach()           # shape(1, num_leg)
        d_star          = d_samples[best_index].unsqueeze(0).clone().detach()           # shape(1, num_leg)
        c_star          = c_samples[best_index].unsqueeze(0).clone().detach()           # shape(1, num_leg, sampling_horizon)
        p_star_lw       = p_lw_samples[best_index].unsqueeze(0).clone().detach()        # shape(1, num_leg, 3, p_param)
        delta_F_star_lw = delta_F_lw_samples[best_index].unsqueeze(0).clone().detach()  # shape(1, num_leg, 3, F_param)

        # Save live variable to enable live plotting
        if self.live_plot:
            # Save best cost
            self.cost_list.append(best_cost.cpu().numpy())
            if len(self.cost_list) > 100 : self.cost_list.pop(0)
            np.savetxt('live_variable/cost.csv', [self.cost_list], delimiter=',', fmt='%.3f')

            # Save winning policy
            self.wining_policy_list.append(self.wining_policy)
            if len(self.wining_policy_list) > 100 : self.wining_policy_list.pop(0)
            np.savetxt('live_variable/winning_policy.csv', [self.wining_policy_list], delimiter=',', fmt='%d')
            
            # Save best GRF
            np.savetxt("live_variable/F_best_FL.csv", delta_F_star_lw[0,0,:,:].cpu().numpy(), delimiter=",")
            np.savetxt("live_variable/F_best_FR.csv", delta_F_star_lw[0,1,:,:].cpu().numpy(), delimiter=",")
            np.savetxt("live_variable/F_best_RL.csv", delta_F_star_lw[0,2,:,:].cpu().numpy(), delimiter=",")
            np.savetxt("live_variable/F_best_RR.csv", delta_F_star_lw[0,3,:,:].cpu().numpy(), delimiter=",")

        # Update previous best solution
        self.f_best, self.d_best, self.p_best, self.F_best = f_star, d_star, p_star_lw, delta_F_star_lw
        # self.f_best, self.d_best, self.p_best, self.F_best = self.shift_actions(f=self.f_best, d=self.d_best, p=self.p_best, F=self.F_best)

        # reset the delta of the actions if contact ended (ie. started swing phase)
        lift_off_mask = ((self.c_actual[:,:] == 1) * (c_star[:,:,1] == 0)) # shape (1,num_legs)
        self.F_best[lift_off_mask] = 0.0 

        # Retrive action to be applied at next time step
        p0_star_lw = self.interpolation_p(parameters=p_star_lw,       step=0, horizon=self.sampling_horizon).clone().detach() # shape(1, num_legs, 3)
        F0_star_lw = self.interpolation_F(parameters=delta_F_star_lw, step=0, horizon=self.sampling_horizon).clone().detach() # shape(1, num_legs, 3)

        # Print difference with RL warm start
        if self.optimize_f : print('f - cum. diff. : %3.2f' % torch.sum(torch.abs(f_star - f_samples[0,...])))
        if self.optimize_d : print('d - cum. diff. : %3.2f' % torch.sum(torch.abs(d_star - d_samples[0,...])))
        if self.optimize_p : print('p - cum. diff. : %3.2f' % torch.sum(torch.abs(p_star_lw - p_lw_samples[0,...])))
        if self.optimize_F : print('F - cum. diff. : %5.1f' % torch.sum(torch.abs(delta_F_star_lw - delta_F_lw_samples[0,...])))

        print(c_star)

        # Add gravity compensation
        F0_star_lw -= c_star[:,:,0].unsqueeze(-1) * self.gravity_lw.unsqueeze(0).unsqueeze(0) * self.robot_mass / (torch.sum(c_star[:,:,0]).clamp_min(min=1)).unsqueeze(0).unsqueeze(-1) # shape(1, num_legs, 3)

        # Enforce force constraints (Friction cone constraints)
        # F0_star_lw = self.enforce_friction_cone_constraints_torch(F=F0_star_lw, mu=self.mu)                           # shape(num_samples, num_legs, 3)
        F0_star_lw = enforce_friction_cone_constraints_torch(F=F0_star_lw, mu=self.mu, F_z_min=self.F_z_min, F_z_max=self.F_z_max) # shape(num_samples, num_legs, 3)

        # Update c_actual for next iteration
        self.c_actual = c_star[:,:,1] # shape (batch_size, num_legs)

        return f_star, d_star, p0_star_lw, F0_star_lw


    def shift_actions(self, f: torch.Tensor, d: torch.Tensor, p: torch.Tensor,F: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Shift the actions from one time step and copy the last action
        """
        f[...,0:-1] = f[...,1:].clone()
        d[...,0:-1] = d[...,1:].clone()
        p[...,0:-1] = p[...,1:].clone()
        F[...,0:-1] = F[...,1:].clone()

        return f, d, p, F


    def generate_samples(self, iter:int, f:torch.Tensor, d:torch.Tensor, p_lw:torch.Tensor, delta_F_lw:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Given action (f,d,p,F), generate action sequence samples (f_samples, d_samples, p_samples, F_samples)
        If multiple action sequence are provided (because several policies are blended together), generate samples
        from these polices with equal proportions. TODO
        
        Args :
            f             (Tensor): Leg frequency                                of shape(batch_size, num_leg)
            d             (Tensor): Leg duty cycle                               of shape(batch_size, num_leg)
            p_lw          (Tensor): Foot touch down position                     of shape(batch_size, num_leg, 3, p_param)
            delta_F_lw    (Tensor): ground Reaction Forces                       of shape(batch_size, num_leg, 3, F_param)
            
        Returns :
            f_samples    (Tensor) : Leg frequency samples               of shape(num_samples, num_leg)
            d_samples    (Tensor) : Leg duty cycle samples              of shape(num_samples, num_leg)
            p_lw_samples (Tensor) : Foot touch down position samples    of shape(num_samples, num_leg, 3, p_param)
            F_lw_samples (Tensor) : Ground Reaction forces samples      of shape(num_samples, num_leg, 3, F_param)
        """
        # Prepare RL input if needed
        if self.cfg.parametrization_F == 'from_discrete_fit_spline' :
            delta_F_lw = fit_cubic(delta_F_lw)  # shape(batch, leg , 3, sampling_horizon) -> shape(batch, leg , 3, F_param)
        
        if self.cfg.parametrization_p == 'from_discrete_fit_spline' :
            p_lw = fit_cubic(p_lw)  # shape(batch, leg , 3, sampling_horizon) -> shape(batch, leg , 3, p_param)

        if self.cfg.parametrization_F == 'from_single_expand_discrete' :
            # delta_F_lw = delta_F_lw.expand_as(self.F_best) # shape(batch, leg , 3, 1) -> shape(batch, leg , 3, F_param)
            delta_F_lw = delta_F_lw.expand(self.number_of_policy, self.num_legs, 3, self.F_param) # shape(batch, leg , 3, 1) -> shape(batch, leg , 3, F_param)
        
        if self.cfg.parametrization_p == 'from_single_expand_discrete':
            # p_lw = p_lw.expand_as(self.p_best) # shape(batch, leg , 3, 1) -> shape(batch, leg , 3, p_param)
            p_lw = p_lw.expand(self.number_of_policy, self.num_legs, 3, self.p_param) # shape(batch, leg , 3, 1) -> shape(batch, leg , 3, p_param)

        if self.cfg.parametrization_F == 'cubic_spline':
            # Since the cubic spline fitting has been normalised with first and last coefficient 10 times smaller than what they should be
            delta_F_lw[:,:,:,0] = 10*delta_F_lw[:,:,:,0]
            delta_F_lw[:,:,:,3] = 10*delta_F_lw[:,:,:,3] 

        # Define how much samples from the RL or from the previous solution we're going to sample
        if iter == 0:
            num_samples_RL = self.num_samples - int(self.num_samples * self.propotion_previous_solution)
            num_samples_previous_best = self.num_samples  - ((num_samples_RL//self.number_of_policy)*self.number_of_policy) # To avoid to loose a few sample due to rounding
        else :
            num_samples_previous_best = self.num_samples
            num_samples_RL = 0

        # Fix to have to different range for F splines
        std_F = self.std_F
        if self.cfg.parametrization_F == 'cubic_spline' or self.cfg.parametrization_F == 'from_discrete_fit_spline' : 
            std_F = self.std_F.repeat((1, self.num_legs,3, self.F_param)).clone().detach() #shape (1, num_leg, 3, F_param)
            std_F[:,:,:,(0,3)] = self.spline_std_F * std_F[:,:,:,(0,3)]

        # Samples from the previous best solution
        f_samples_best          = self.sampling_law(num_samples=num_samples_previous_best, mean=self.f_best[0], std=self.std_f, clip=self.clip_sample)
        d_samples_best          = self.sampling_law(num_samples=num_samples_previous_best, mean=self.d_best[0], std=self.std_d, clip=self.clip_sample)
        p_lw_samples_best       = self.sampling_law(num_samples=num_samples_previous_best, mean=self.p_best[0], std=self.std_p, clip=self.clip_sample)
        delta_F_lw_samples_best = self.sampling_law(num_samples=num_samples_previous_best, mean=self.F_best[0], std=std_F, clip=self.clip_sample)

        f_samples_rl_list, d_samples_rl_list, p_lw_samples_rl_list, delta_F_lw_samples_rl_list = [], [], [], []
        for i in range(self.number_of_policy):
            f_samples_rl_list.append(          self.sampling_law(num_samples=num_samples_RL//self.number_of_policy, mean=f[i],          std=self.std_f, clip=self.clip_sample) )
            d_samples_rl_list.append(          self.sampling_law(num_samples=num_samples_RL//self.number_of_policy, mean=d[i],          std=self.std_d, clip=self.clip_sample) )
            p_lw_samples_rl_list.append(       self.sampling_law(num_samples=num_samples_RL//self.number_of_policy, mean=p_lw[i],       std=self.std_p, clip=self.clip_sample) )
            delta_F_lw_samples_rl_list.append( self.sampling_law(num_samples=num_samples_RL//self.number_of_policy, mean=delta_F_lw[i], std=std_F,      clip=self.clip_sample) )

        f_samples_rl          = torch.cat(f_samples_rl_list, dim=0)
        d_samples_rl          = torch.cat(d_samples_rl_list, dim=0)
        p_lw_samples_rl       = torch.cat(p_lw_samples_rl_list, dim=0)
        delta_F_lw_samples_rl = torch.cat(delta_F_lw_samples_rl_list, dim=0)

        # # Samples from the provided guess
        # f_samples_rl          = self.sampling_law(num_samples=num_samples_RL, mean=f[0],    std=self.std_f, clip=self.clip_sample)
        # d_samples_rl          = self.sampling_law(num_samples=num_samples_RL, mean=d[0],    std=self.std_d, clip=self.clip_sample)
        # p_lw_samples_rl       = self.sampling_law(num_samples=num_samples_RL, mean=p_lw[0], std=self.std_p, clip=self.clip_sample)
        # delta_F_lw_samples_rl = self.sampling_law(num_samples=num_samples_RL, mean=delta_F_lw[0], std=std_F, clip=self.clip_sample)

        # Concatenate the samples
        f_samples          = torch.cat((f_samples_rl,          f_samples_best),          dim=0)
        d_samples          = torch.cat((d_samples_rl,          d_samples_best),          dim=0)
        p_lw_samples       = torch.cat((p_lw_samples_rl,       p_lw_samples_best),       dim=0)
        delta_F_lw_samples = torch.cat((delta_F_lw_samples_rl, delta_F_lw_samples_best), dim=0)

        # Clamp the input to valid range
        f_samples, d_samples, p_lw_samples, delta_F_lw_samples = self.enforce_valid_input(f_samples=f_samples, d_samples=d_samples, p_lw_samples=p_lw_samples, delta_F_lw_samples=delta_F_lw_samples)

        # Set the foot height to the nominal foot height # TODO change
        p_lw_samples[:,:,2,:] = p_lw[0,:,2,:]

        # Put the Previous best actions as the last samples
        f_samples[-1,:]              = self.f_best[0,:]
        d_samples[-1,:]              = self.d_best[0,:]
        p_lw_samples[-1,:,:,:]       = self.p_best[0,:,:,:]
        delta_F_lw_samples[-1,:,:,:] = self.F_best[0,:,:,:]

        # Put the RL actions as the first samples
        if not self.propotion_previous_solution > 0.999 :
            f_samples[0,:]              = f[0,:]
            d_samples[0,:]              = d[0,:]
            p_lw_samples[0,:,:,:]       = p_lw[0,:,:,:]
            delta_F_lw_samples[0,:,:,:] = delta_F_lw[0,:,:,:]

        # If optimization is set to false, samples are feed with initial guess
        if not self.optimize_f :
            for i in range(self.number_of_policy):
                f_samples[i*(num_samples_RL//self.number_of_policy):(i+1)*(num_samples_RL//self.number_of_policy),:] = f[i].clone().detach()
            f_samples[self.num_samples-num_samples_previous_best:,:] = f[0].clone().detach()

        if not self.optimize_d :
            for i in range(self.number_of_policy):
                d_samples[i*(num_samples_RL//self.number_of_policy):(i+1)*(num_samples_RL//self.number_of_policy),:] = d[i].clone().detach()
            d_samples[self.num_samples-num_samples_previous_best:,:] = d[0].clone().detach()
        if not self.optimize_p :
            for i in range(self.number_of_policy): 
                p_lw_samples[i*(num_samples_RL//self.number_of_policy):(i+1)*(num_samples_RL//self.number_of_policy),:,:,:] = p_lw[i].clone().detach()
            p_lw_samples[self.num_samples-num_samples_previous_best:,:] = p_lw[0].clone().detach()
        if not self.optimize_F : 
            for i in range(self.number_of_policy): 
                delta_F_lw_samples[i*(num_samples_RL//self.number_of_policy):(i+1)*(num_samples_RL//self.number_of_policy),:,:,:] = delta_F_lw[i].clone().detach()
            delta_F_lw_samples[self.num_samples-num_samples_previous_best:,:] = delta_F_lw[0].clone().detach()

        return f_samples, d_samples, p_lw_samples, delta_F_lw_samples


    def normal_sampling(self, num_samples:int, mean:torch.Tensor, std:torch.Tensor|None=None, seed:int|None=None, clip=False) -> torch.Tensor:
        """ Normal sampling law given mean and std -> return a samples
        
        Args :
            mean     (Tensor): Mean of normal sampling law          of shape(num_dim1, num_dim2, etc.)
            std      (Tensor): Standard dev of normal sampling law  of shape(num_dim1, num_dim2, etc.)
            num_samples (int): Number of samples to generate
            seed        (int): seed to generate random numbers

        Return :
            samples  (Tensor): Samples generated with mean and std  of shape(num_sammple, num_dim1, num_dim2, etc.)
        """

        # Seed if provided
        if seed : 
            torch.manual_seed(seed)

        if std is None :
            std = torch.ones_like(mean)

        # Sample from a normal law with the provided parameters
        if clip == True :
            samples = mean + (std * torch.randn((num_samples,) + mean.shape, device=self.device)).clamp(min=-2*std, max=2*std)
        else :
            samples = mean + (std * torch.randn((num_samples,) + mean.shape, device=self.device))

        return samples
    

    def uniform_sampling(self, num_samples:int, mean:torch.Tensor, std:torch.Tensor|None=None, seed:int|None=None, clip=False) -> torch.Tensor:
        """ Normal sampling law given mean and std -> return a samples
        
        Args :
            mean     (Tensor): Mean of normal sampling law          of shape(num_dim1, num_dim2, etc.)
            std      (Tensor): Standard dev of normal sampling law  of shape(num_dim1, num_dim2, etc.)
            num_samples (int): Number of samples to generate
            seed        (int): seed to generate random numbers

        Return :
            samples  (Tensor): Samples generated with mean and std  of shape(num_sammple, num_dim1, num_dim2, etc.)
        """

        # Seed if provided
        if seed : 
            torch.manual_seed(seed)

        if std is None :
            std = torch.ones_like(mean)

        # Sample from a uniform law with the provided parameters
        samples = mean + (std * torch.empty((num_samples,) + mean.shape, device=self.device).uniform_(-1.0, 1.0))

        return samples


    def compute_rollout(self, initial_state: dict, reference_seq_state: dict, reference_seq_input_samples: dict, action_param_samples: dict, c_samples: torch.Tensor) -> torch.Tensor:
        """
        Calculate cost of rollouts of given action sequence samples 

        Args :
            initial_state         (dict): Dictionnary containing the current robot's state
                pos_com_lw      (Tensor): CoM position in local world frame                             of shape(3)
                lin_com_vel_lw  (Tensor): CoM linear velocity in local world frame                      of shape(3)
                euler_xyz_angle (Tensor): CoM orientation (wrt. to l. world frame) as XYZ euler angle   of shape(3)
                ang_vel_com_b   (Tensor): CoM angular velocity as roll pitch yaw                        of shape(3)
                p_lw            (Tensor): Feet position in local world frame                            of shape(num_legs, 3)

            reference_seq_state   (dict): Dictionnary containing the robot's reference state along the prediction horizon
                pos_com_lw      (Tensor): CoM position in local world frame                             of shape(3, sampling_horizon)
                lin_com_vel_lw  (Tensor): CoM linear velocity in local world frame                      of shape(3, sampling_horizon)
                euler_xyz_angle (Tensor): CoM orientation (wrt. to l. world frame) as XYZ euler angle   of shape(3, sampling_horizon)
                ang_vel_com_b   (Tensor): CoM angular velocity as roll pitch yaw                        of shape(3, sampling_horizon)
                p_lw            (Tensor): Feet position in local world frame                            of shape(num_legs, 3, sampling_horizon)                

            reference_seq_input_samples (dict) 
                F_lw            (Tensor): Reference GRF sequence samples along the prediction horizon   of shape(num_sample, num_legs, 3, sampling_horizon)  

            action_param_samples  (dict): Dictionnary containing the robot's actions along the prediction horizon 
                p_lw            (Tensor): Foot touch down position in local world frame                 of shape(num_samples, num_legs, 3, p_param)
                delta_F_lw      (Tensor): Delta GRF parameters in local world frame                     of shape(num_samples, num_legs, 3, F_param)

            c_samples       (t.bool): Foot contact sequence sample                                                      of shape(num_samples, num_legs, sampling_horizon)

        Return 
            cost_samples        (Tensor): Cost associated at each sample (along the prediction horizon) of shape(num_samples)
        """
        cost_samples = torch.zeros(self.num_samples, device=self.device)
        state = {}
        state['pos_com_lw']      = initial_state['pos_com_lw'].unsqueeze(0).expand(self.num_samples, 3)
        state['lin_com_vel_lw']  = initial_state['lin_com_vel_lw'].unsqueeze(0).expand(self.num_samples, 3)
        state['euler_xyz_angle'] = initial_state['euler_xyz_angle'].unsqueeze(0).expand(self.num_samples, 3)
        state['ang_vel_com_b']   = initial_state['ang_vel_com_b'].unsqueeze(0).expand(self.num_samples, 3)
        state['p_lw']            = initial_state['p_lw'].unsqueeze(0).expand(self.num_samples, self.num_legs, 3)
        input = {}

        for i in range(self.sampling_horizon):
            # --- Step 1 : prepare the inputs
            # Find the current action given the actions parameters
            input['p_lw'] = self.interpolation_p(parameters=action_param_samples['p_lw'],       step=i, horizon=self.sampling_horizon)# shape(num_samples, num_legs, 3)
            input['F_lw'] = self.interpolation_F(parameters=action_param_samples['delta_F_lw'], step=i, horizon=self.sampling_horizon)# shape(num_samples, num_legs, 3)
            contact = c_samples[:,:,i]                                                                                          # shape(num_samples, num_legs)

            # Add gravity compensation
            input['F_lw'] -= contact.unsqueeze(-1) * (self.gravity_lw.unsqueeze(0).unsqueeze(0) * self.robot_mass) / (torch.sum(contact, dim=1).clamp_min(min=1)).unsqueeze(-1).unsqueeze(-1)      

            # Enforce force constraints (Friction cone constraints)
            # input['F_lw'] = self.enforce_friction_cone_constraints_torch(F=input['F_lw'], mu=self.mu)                           # shape(num_samples, num_legs, 3)
            input['F_lw'] = enforce_friction_cone_constraints_torch(F=input['F_lw'], mu=self.mu, F_z_min=self.F_z_min, F_z_max=self.F_z_max) # shape(num_samples, num_legs, 3)


            # --- Step 2 : Step the model
            new_state = self.centroidal_model_step(state=state, input=input, contact=contact)
            state = new_state


            # --- Step 3 : compute the step cost
            state_vector     = torch.cat([vector.view(self.num_samples, -1) for vector in state.values()], dim=1)               # Shape: (num_samples, state_dim)
            ref_state_vector = torch.cat([vector.select(-1, i).view(-1) for vector in reference_seq_state.values()], dim=0)     # Shape: (state_dim)
            
            # Compute the state cost
            state_error = state_vector - ref_state_vector.unsqueeze(0)                                                          # shape (num_samples, state_dim)
            state_cost  = torch.sum(self.Q_vec.unsqueeze(0) * (state_error ** 2), dim=1)                                        # Shape (num_samples)

            # Compute the input cost
            input_error = (input['F_lw'] - reference_seq_input_samples['F_lw'][:,:,:,i]).flatten(1,2)                           # shape (num_samples, action_dim)
            input_cost  = torch.sum(self.R_vec.unsqueeze(0) * (input_error ** 2), dim=1)                                        # Shape (num_samples)

            step_cost = state_cost #+ input_cost                                                                                # shape(num_samples)

            # Update the trajectory cost
            cost_samples += step_cost                                                                                           # shape(num_samples)

            # Saturate the cost in case of NaN or inf
            cost_samples[torch.isnan(cost_samples) | torch.isinf(cost_samples)] = 1e10

            # Take the best found control parameters
            best_index = torch.argmin(cost_samples)
            best_cost = cost_samples[best_index] # cost_samples.take(best_index)

            if best_cost > 2000 :
                print('oulalala')


            if best_cost > 10*self.previous_best_cost:
                print('Oh no')

            self.previous_best_cost = best_cost

        return cost_samples
     

    def enforce_valid_input(self, f_samples: torch.Tensor, d_samples: torch.Tensor, p_lw_samples: torch.Tensor, delta_F_lw_samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Enforce the input f, d, p_lw, F_lw to valid ranges. Ie. clip
            - f to [0,3] [Hz]
            - d to [0,1]
            - p_lw can't be clipped because already in lw frame
            - F_lw -> F_lw_z to [0, +inf]
        Moreover, ensure additionnal constraints
            - p_z on the ground (not implemented yet) TODO Implement
            - F_lw : Friction cone constraints

        Args
            f_samples          (torch.Tensor): Leg frequency samples               of shape(num_samples, num_leg)
            d_samples          (torch.Tensor): Leg duty cycle samples              of shape(num_samples, num_leg)
            p_lw_samples       (torch.Tensor): Foot touch down position samples    of shape(num_samples, num_leg, 3, p_param)
            delta_F_lw_samples (torch.Tensor): Ground Reaction forces samples      of shape(num_samples, num_leg, 3, F_param)

        Return
            f_samples          (torch.Tensor): Clipped Leg frequency samples               of shape(num_samples, num_leg)
            d_samples          (torch.Tensor): Clipped Leg duty cycle samples              of shape(num_samples, num_leg)
            p_lw_samples       (torch.Tensor): Clipped Foot touch down position samples    of shape(num_samples, num_leg, 3, p_param)
            delta_F_lw_samples (torch.Tensor): Clipped Ground Reaction forces samples      of shape(num_samples, num_leg, 3, F_param)
        """
        # --- Step 1 : Clip Action to valid range
        # Clip f
        f_samples = f_samples.clamp(min=0, max=3)

        # Clip d
        d_samples = d_samples.clamp(min=0, max=1)

        # Clip p - is already in lw frame... can't clip it in this frame
        # ...

        # Clip F - /!\ Can't work if we are sampling over a delta ! Must be after the gravity compensation /!\
        # F_lw_samples[:,:,2,:] = F_lw_samples[:,:,2,:].clamp(min=self.F_z_min, max=self.F_z_max) # TODO This clip also spline param 0 and 3, which may be restrictive


        # --- Step 2 : Add Constraints
        # p :  Ensure p on the ground TODO Implement
        # p_lw_samples[:,:,2,:] = 0.0*torch.ones_like(p_lw_samples[:,:,2,:])
      
        # F : Ensure Friction Cone constraints (Bounding spline means quasi bounding trajectory) /!\ Can't work if we are sampling over a delta ! Must be after the gravity compensation /!\
        # F_lw_samples = self.enforce_friction_cone_constraints_torch(F=F_lw_samples, mu=self.mu)


        return f_samples, d_samples, p_lw_samples, delta_F_lw_samples


    # def enforce_friction_cone_constraints_torch(self, F:torch.Tensor, mu:float) -> torch.Tensor:
        """ Enforce the friction cone constraints
        ||F_xy|| < F_z*mu
        Args :
            F (torch.Tensor): The GRF                                    of shape(num_samples, num_legs, 3,(optinally F_param))

        Returns :
            F (torch.Tensor): The GRF with enforced friction constraints of shape(num_samples, num_legs, 3,(optinally F_param))
        """

        F_x = F[:,:,0].unsqueeze(2)
        F_y = F[:,:,1].unsqueeze(2)
        F_z = F[:,:,2].unsqueeze(2).clamp(min=self.F_z_min, max=self.F_z_max)

        # Angle between vec_x and vec_F_xy
        alpha = torch.atan2(F[:,:,1], F[:,:,0]).unsqueeze(2) # atan2(y,x) = arctan(y/x)

        # Compute the maximal Force in the xy plane
        F_xy_max = mu*F_z

        # Clipped the violation for the x and y component (unsqueeze to avoid to loose that dimension) : To use clamp_max -> need to remove the sign...
        F_x_clipped =  F_x.sign()*(torch.abs(F_x).clamp_max(torch.abs(torch.cos(alpha)*F_xy_max)))
        F_y_clipped =  F_y.sign()*(torch.abs(F_y).clamp_max(torch.abs(torch.sin(alpha)*F_xy_max)))

        # Reconstruct the vector
        F = torch.cat((F_x_clipped, F_y_clipped, F_z), dim=2)

        return F


    # def from_zero_twopi_to_minuspi_pluspi(self, roll, pitch, yaw):
        """ Change the function space from [0, 2pi[ to ]-pi, pi] 
        
        Args :
            roll  (Tensor): roll in [0, 2pi[    shape(x)
            pitch (Tensor): roll in [0, 2pi[    shape(x)
            yaw   (Tensor): roll in [0, 2pi[    shape(x)
        
        Returns :   
            roll  (Tensor): roll in ]-pi, pi]   shape(x)
            pitch (Tensor): roll in ]-pi, pi]   shape(x)
            yaw   (Tensor): roll in ]-pi, pi]   shape(x)    
        """

        # Apply the transformation
        roll  = ((roll  - torch.pi) % (2*torch.pi)) - torch.pi
        pitch = ((pitch - torch.pi) % (2*torch.pi)) - torch.pi 
        yaw   = ((yaw   - torch.pi) % (2*torch.pi)) - torch.pi

        return roll, pitch, yaw
    

    def centroidal_model_step(self, state, input, contact):
        """
        Simulate one step of the forward dynamics, using the controidal model. 

        Note 
            Model used is described in 'Model Predictive Control With Environment Adaptation for Legged Locomotion'
            ttps://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9564053
            
        Args : 
            state     (dict): Dictionnary containing the robot's state
                pos_com_lw      (tensor): CoM position in local world frame                             of shape(num_samples, 3)
                lin_com_vel_lw  (tensor): CoM linear velocity in local world frame                      of shape(num_samples, 3)
                euler_xyz_angle (tensor): CoM orientation (wrt. to l. world frame) as XYZ euler angle   of shape(num_samples, 3)
                ang_vel_com_b   (tensor): CoM angular velocity as roll pitch yaw                        of shape(num_samples, 3)
                p_lw            (tensor): Feet position in local world frame                            of shape(num_samples, num_legs, 3)

            input     (dict): Dictionnary containing the robot's input
                p_lw            (tensor): Feet touch down position in local world frame                 of shape(num_samples, num_legs, 3)
                F_lw            (tensor): Ground Reaction Forces in local world frame                   of shape(num_samples, num_legs, 3)

            contact (tensor): Foot contact status (stance=1, swing=0)                                   of shape(num_samples, num_legs)

        Return :
            new_state (dict): Dictionnary containing the robot's updated state after one iteration
                pos_com_lw      (tensor): CoM position in local world frame                             of shape(num_samples, 3)
                lin_com_vel_lw  (tensor): CoM linear velocity in local world frame                      of shape(num_samples, 3)
                euler_xyz_angle (tensor): CoM orientation (wrt. to l. world frame) as XYZ euler angle   of shape(num_samples, 3)
                ang_vel_com_b   (tensor): CoM angular velocity as roll pitch yaw                        of shape(num_samples, 3)
                p_lw            (tensor): Feet position in local world frame                            of shape(num_samples, num_legs, 3)
        """
        new_state = {}

        # --- Step 1 : Compute linear velocity
        lin_com_vel_lw = state['lin_com_vel_lw']     # shape (num_samples, 3)


        # --- Step 2 : Compute linear acceleration  as sum of forces divide by mass
        linear_com_acc_lw = (torch.sum(input['F_lw'] * contact.unsqueeze(-1), dim=1) / self.robot_mass) + self.gravity_lw.unsqueeze(0) # shape (num_samples, 3)


        # --- Step 3 : Compute angular velocity (as euler rate to increment euler angles) : Given by eq. 27 in 'Note' paper
        # euler_xyz_rate = inverse(conjugate(euler_xyz_rate_matrix) * omega_b
        euler_xyz_rate = torch.bmm(inverse_conjugate_euler_xyz_rate_matrix(state['euler_xyz_angle']), state['ang_vel_com_b'].unsqueeze(-1)).squeeze(-1 ) # shape (s,3,3)*(s,3,1)->(s,3,1) -> (num_samples, 3)


        # --- Step 4 : Compute angular acceleration : Given by eq. 4 in 'Note' paper
        # Compute the sum of the moment induced by GRF (1. Moment=dist_from_F cross F, 2. Keep moment only for feet in contact, 3. sum over 4 legs)
        sum_of_GRF_moment_lw = torch.sum(contact.unsqueeze(-1) * torch.cross((state['p_lw'] - state['pos_com_lw'].unsqueeze(1)), input['F_lw'], dim=-1), dim=1) # shape sum{(s,l,1)*[(s,l,3)^(s,l,3)], dim=l} -> (num_sample, 3)

        # Transform the sum of moment from local world frame to base frame
        sum_of_GRF_moment_b  = torch.bmm(rotation_matrix_from_w_to_b(state['euler_xyz_angle']), sum_of_GRF_moment_lw.unsqueeze(-1)).squeeze(-1) # shape(num_samples, 3)

        # Compute intermediary variable : w_b^(I*w_b) : unsqueeze robot inertia for batched operation, squeeze and unsqueeze matmul to perform dot product (1,3,3)*(s,3,1)->(s,3,1)->(s,3)
        omega_b_cross_inertia_dot_omega_b = torch.cross(state['ang_vel_com_b'], torch.matmul(self.robot_inertia.unsqueeze(0), state['ang_vel_com_b'].unsqueeze(-1)).squeeze(-1), dim=-1) # shape(num_samples, 3)

        # Finally compute the angular acc : unsqueeze robot inertia for batched operation, squeeze and unsqueeze matmul to perform dot product (1,3,3)*(s,3,1)->(s,3,1)->(s,3)
        ang_acc_com_b = torch.matmul(self.inv_robot_inertia.unsqueeze(0), (sum_of_GRF_moment_b - (omega_b_cross_inertia_dot_omega_b)).unsqueeze(-1)).squeeze(-1) # shape(num_samples, 3)


        # --- Step 5 : Perform forward integration of the model (as simple forward euler)
        new_state['pos_com_lw']      = state['pos_com_lw']      + self.mpc_dt*lin_com_vel_lw
        new_state['lin_com_vel_lw']  = state['lin_com_vel_lw']  + self.mpc_dt*linear_com_acc_lw
        new_state['euler_xyz_angle'] = state['euler_xyz_angle'] + self.mpc_dt*euler_xyz_rate
        new_state['ang_vel_com_b']   = state['ang_vel_com_b']   + self.mpc_dt*ang_acc_com_b
        # new_state['p_lw']            = state['p_lw']
        new_state['p_lw']            = state['p_lw']*(contact.unsqueeze(-1)) + input['p_lw']*(~contact.unsqueeze(-1))

        return new_state
    

class SamplingBatchedTrainer():
    """
    """

    def __init__(self, device, num_legs, robot_mass, mu, env:ManagerBasedRLEnv, optimizerCfg):
        """ 
        Args :
            device                 : 'cuda' or 'cpu' 
            num_legs               : Number of legs of the robot. Used for variable dimension definition
            num_samples            : Number of samples used for the sampling optimizer
            sampling_horizon       : Number of time steps the sampling optimizer is going to predict
            discretization_time    : Duration of a time step
            interpolation_F_method : Method to reconstruct GRF action from provided GRF warm start :
                                    can be 'discrete' (one action per time step is provided) or 'cubic spine' (a set of parameters for every time step)
            interpolation_p_method : Method to reconstruct foot touch down position action from provided warm start :
                                    can be 'discrete' (one action per time step is provided) or 'cubic spine' (a set of parameters for every time step)
        """

        # Save the environment
        self._env: ManagerBasedRLEnv = env

        # General variables
        self.device = device
        self.num_legs = num_legs
        
        # Save Config
        self.cfg = optimizerCfg

        # Optimizer configuration
        self.num_samples = optimizerCfg.num_samples
        self.sampling_horizon = optimizerCfg.prevision_horizon
        self.mpc_dt = optimizerCfg.discretization_time 
        self.mu = mu #optimizerCfg.mu

        # Define Interpolation method for GRF and interfer GRF input size 
        if   optimizerCfg.parametrization_F == 'cubic_spline': 
            self.interpolation_F=compute_cubic_spline
            self.F_param = 4
        elif optimizerCfg.parametrization_F == 'discrete':
            self.interpolation_F=compute_discrete
            self.F_param = self.sampling_horizon
        elif optimizerCfg.parametrization_F == 'from_discrete_fit_spline':
            self.interpolation_F=compute_cubic_spline
            self.F_param = 4
        elif optimizerCfg.parametrization_F == 'from_single_expand_discrete':
            self.interpolation_F=compute_discrete
            self.F_param = self.sampling_horizon
        else : raise NotImplementedError('Request interpolation method is not implemented yet')

        # Define Interpolation method for foot touch down position and interfer foot touch down position input size 
        if   optimizerCfg.parametrization_p == 'cubic_spline': 
            self.interpolation_p=compute_cubic_spline
            self.p_param = 4
        elif optimizerCfg.parametrization_p == 'discrete': 
            self.interpolation_p=compute_discrete
            self.p_param = self.sampling_horizon
        elif optimizerCfg.parametrization_p == 'from_discrete_fit_spline': 
            self.interpolation_p=compute_cubic_spline
            self.p_param = 4
        elif optimizerCfg.parametrization_p == 'from_single_expand_discrete':
            self.interpolation_p=compute_discrete
            self.p_param = self.sampling_horizon
        elif optimizerCfg.parametrization_p == 'first':
            self.interpolation_p=compute_unique
            self.p_param = 1
        else : raise NotImplementedError('Request interpolation method is not implemented yet')

        # Input and State dimension for centroidal model : hardcoded because the centroidal model is hardcoded # TODO get rid of these
        self.state_dim = 24 # CoM_pos(3) + lin_vel(3) + CoM_pose(3) + ang_vel(3) + foot_pos(12) 
        self.input_dim = 12 # GRF(12) (foot touch down pos is state and an input)


        # TODO Get these value properly
        self.gravity_lw = torch.tensor((0.0, 0.0, -9.81), device=self.device) # shape(3) #self._env.sim.cfg.gravity
        # self.robot_mass = 24.64
        # self.robot_mass = 20.6380
        self.robot_mass = robot_mass
        self.robot_inertia = torch.tensor([[ 0.2310941359705289,   -0.0014987128245817424, -0.021400468992761768 ], # shape (3,3)
                                           [-0.0014987128245817424, 1.4485084687476608,     0.0004641447134275615],
                                           [-0.021400468992761768,  0.0004641447134275615,  1.503217877350808    ]],device=self.device)
        self.inv_robot_inertia = torch.linalg.inv(self.robot_inertia) # shape(3,3)
        self.F_z_min = 0
        self.F_z_max = self.robot_mass*9.81

        # How much of the previous solution is used to generate samples compare to the provided guess (in [0,1])
        self.propotion_previous_solution = optimizerCfg.propotion_previous_solution

        # Define the height reference for the tracking
        self.height_ref = optimizerCfg.height_ref

        # Current contact sequence : used to reset solution when switch to swing shape (batch_size, num_leg)
        self.c_actual = torch.ones((self._env.num_envs, self.num_legs), device=device)

        # State weight - shape(state_dim)
        self.Q_vec = torch.zeros(self.state_dim, device=self.device)
        self.Q_vec[0]  = 0.0        #com_x
        self.Q_vec[1]  = 0.0        #com_y
        self.Q_vec[2]  = 111500     #com_z
        self.Q_vec[3]  = 5000       #com_vel_x
        self.Q_vec[4]  = 5000       #com_vel_y
        self.Q_vec[5]  = 200        #com_vel_z
        self.Q_vec[6]  = 11200      #base_angle_roll
        self.Q_vec[7]  = 11200      #base_angle_pitch
        self.Q_vec[8]  = 0.0        #base_angle_yaw
        self.Q_vec[9]  = 20         #base_angle_rates_x
        self.Q_vec[10] = 20         #base_angle_rates_y
        self.Q_vec[11] = 600        #base_angle_rates_z

        # Input weight - shape(input_dim)
        self.R_vec = torch.zeros(self.input_dim, device=self.device)

        # Initialize the best solution
        self.f_best = 1.5*torch.ones( (1,self.num_legs),                 device=device)
        self.d_best = 0.6*torch.ones( (1,self.num_legs),                 device=device)
        self.p_best =     torch.zeros((1,self.num_legs,3,self.p_param ), device=device)
        self.F_best =     torch.zeros((1,self.num_legs,3,self.F_param ), device=device)
        self.F_best[:,:,2,:] = 0.0

        # For plotting
        self.live_plot = False
        self.robot_height_list = [0.0]
        self.robot_height_ref_list = [0.0]
        self.cost_list = [0.0]
        self.FL_foot_list = [np.array([0.0, 0.0, 0.0])]
        self.FR_foot_list = [np.array([0.0, 0.0, 0.0])]
        self.RL_foot_list = [np.array([0.0, 0.0, 0.0])]
        self.RR_foot_list = [np.array([0.0, 0.0, 0.0])]


    def compute_batched_rollout_cost(self, f:torch.Tensor, d:torch.Tensor, p_lw:torch.Tensor, delta_F_lw:torch.Tensor, phase:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Given latent variable f,d,F,p, returns f*,d*,F*,p*, optimized with a sampling optimization 
        
        Args :
            f      (Tensor): Leg frequency                of shape(batch_size, num_leg)
            d      (Tensor): Leg duty cycle               of shape(batch_size, num_leg)
            p_lw   (Tensor): Foot touch down position     of shape(batch_size, num_leg, 3, p_param)
            F_lw   (Tensor): ground Reaction Forces       of shape(batch_size, num_leg, 3, F_param)
            phase  (Tensor): Current feet phase           of shape(batch_size, num_leg)

        Returns :
            f_star    (Tensor): Leg frequency                of shape(batch_size, num_leg)
            d_star    (Tensor): Leg duty cycle               of shape(batch_size, num_leg)
            p_star_lw (Tensor): Foot touch down position     of shape(batch_size, num_leg, 3, p_param)
            F_star_lw (Tensor): ground Reaction Forces       of shape(batch_size, num_leg, 3, F_param)
        """
        # --- Step 0 : modify F and p if necessary.
        if self.cfg.parametrization_p == 'from_single_expand_discrete' :
            p_lw = p_lw.expand(self._env.num_envs, self.num_legs, 3, self.sampling_horizon) # shape(batch, leg , 3, 1) -> shape(batch, leg , 3, p_param)
 
        if self.cfg.parametrization_F == 'from_single_expand_discrete' :
            delta_F_lw = delta_F_lw.expand(self._env.num_envs, self.num_legs, 3, self.sampling_horizon) # shape(batch, leg , 3, 1) -> shape(batch, leg , 3, F_param)
        if self.cfg.parametrization_F == 'cubic_spline':
            # Since the cubic spline fitting has been normalised with first and last coefficient 10 times smaller than what they should be
            delta_F_lw[:,:,:,0] = 10*delta_F_lw[:,:,:,0]
            delta_F_lw[:,:,:,3] = 10*delta_F_lw[:,:,:,3] 


        # --- Step 1 : Given f and d samples -> generate the contact sequence for the samples
        batched_c, next_phase = gait_generator(f=f, d=d, phase=phase, horizon=max(2,self.sampling_horizon), dt=self.mpc_dt)
        batched_c = batched_c[:self.sampling_horizon] # in case horizon == 1, batched_c will have horizon 2

        # --- Step 2 : prepare the variables 
        batched_initial_state, batched_reference_seq_state, batched_reference_seq_input, batched_action_param = self.prepare_variable_for_compute_rollout(batched_c=batched_c, batched_p_lw=p_lw, batched_delta_F_lw=delta_F_lw, feet_in_contact=self.c_actual)

        # --- Step 3 : Compute the rollouts to find the rollout cost : can't used named argument with VMAP...
        bacthed_cost = self.batched_compute_rollout(batched_initial_state, batched_reference_seq_state, batched_reference_seq_input, batched_action_param, batched_c)
        bacthed_cost[torch.isnan(bacthed_cost) | torch.isinf(bacthed_cost)] = 1e10

        p0_star_lw = self.interpolation_p(parameters=p_lw,       step=0, horizon=1) # shape(batch, num_legs, 3)
        F0_star_lw = self.interpolation_F(parameters=delta_F_lw, step=0, horizon=1) # shape(batch, num_legs, 3)

        # Add gravity compensation
        F0_star_lw -= batched_c[...,0].unsqueeze(-1) * (self.gravity_lw.unsqueeze(0).unsqueeze(0) * self.robot_mass) / (torch.sum(batched_c[...,0], dim=1).clamp_min(min=1)).unsqueeze(-1).unsqueeze(-1)      

        # Enforce force constraints (Friction cone constraints)
        F0_star_lw = enforce_friction_cone_constraints_torch(F=F0_star_lw, mu=self.mu, F_z_min=self.F_z_min, F_z_max=self.F_z_max) # shape(batch, num_legs, 3)

        # print('batched cost :', bacthed_cost)
        # Save live variable to enable live plotting
        if self.live_plot:
            # Save best cost
            self.cost_list.append(bacthed_cost[0].cpu().numpy())
            if len(self.cost_list) > 100 : self.cost_list.pop(0)
            np.savetxt('live_variable/cost.csv', [self.cost_list], delimiter=',', fmt='%.3f')

        # Update for next iteration
        self.c_actual = batched_c[:,:,1]

        return bacthed_cost, p0_star_lw, F0_star_lw
    

    def prepare_variable_for_compute_rollout(self, batched_c:torch.Tensor, batched_p_lw:torch.Tensor, batched_delta_F_lw:torch.Tensor, feet_in_contact:torch.Tensor) -> tuple[dict, dict, dict, dict]:
        """ Helper function to modify the embedded state, reference and action to be used with the 'compute_rollout' function

        Note :
            Initial state and reference can be retrieved only with the environment
            _w   : World frame (inertial frame)
            _lw  : World frame centered at the environment center -> local world frame
            _b   : Base frame (attached to robot's base)
            _h   : Horizontal frame -> Base frame position for xy, world frame for z, roll, pitch, base frame for yaw
            _bw  : Base/world frame -> Base frame position, world frame rotation

        Args :
            env  (ManagerBasedRLEnv): Environment manager to retrieve all necessary simulation variable
            batched_c       (t.bool): Foot contact sequence sample                                                      of shape(batch, num_legs, sampling_horizon)
            batched_p_lw    (Tensor): Foot touch down position                                                          of shape(batch, num_legs, 3, p_param)
            batched_delta_F_lw (Tsr): ground Reaction Forces                                                            of shape(batch, num_legs, 3, F_param)
            feet_in_contact (Tensor): Feet in contact, determined by prevous solution                                   of shape(batch, num_legs)

        
        Return :
            initial_state         (dict): Dictionnary containing the current robot's state
                pos_com_lw      (Tensor): CoM position in local world frame                             of shape(batch, 3)
                lin_com_vel_lw  (Tensor): CoM linear velocity in local world frame                      of shape(batch, 3)
                euler_xyz_angle (Tensor): CoM orientation (wrt. to l. world frame) as XYZ euler angle   of shape(batch, 3)
                ang_vel_com_b   (Tensor): CoM angular velocity as roll pitch yaw                        of shape(batch, 3)
                p_lw            (Tensor): Feet position in local world frame                            of shape(batch, num_legs, 3)

            reference_seq_state   (dict): Dictionnary containing the robot's reference state along the prediction horizon
                pos_com_lw      (Tensor): CoM position in local world frame                             of shape(3, sampling_horizon)
                lin_com_vel_lw  (Tensor): CoM linear velocity in local world frame                      of shape(3, sampling_horizon)
                euler_xyz_angle (Tensor): CoM orientation (wrt. to l. world frame) as XYZ euler angle   of shape(3, sampling_horizon)
                ang_vel_com_b   (Tensor): CoM angular velocity as roll pitch yaw                        of shape(3, sampling_horizon)
                p_lw            (Tensor): Feet position in local world frame                            of shape(num_legs, 3, sampling_horizon)                

            reference_seq_input_samples (dict) 
                F_lw            (Tensor): Reference GRF sequence samples along the prediction horizon   of shape(num_sample, num_legs, 3, sampling_horizon)  

            action_param_samples  (dict): Dictionnary containing the robot's actions along the prediction horizon 
                p_lw            (Tensor): Foot touch down position in local world frame                 of shape(num_samples, num_legs, 3, p_param)
                delta_F_lw      (Tensor): delta GRF parameters in local world frame                     of shape(num_samples, num_legs, 3, F_param)
        """
        batched_initial_state = {}
        batched_reference_seq_state = {}
        bacthed_reference_seq_input = {}
        batched_action_param = {}

        # Retrieve robot from the scene : specify type to enable type hinting
        robot: Articulation = self._env.scene["robot"]

        # Retrieve indexes
        foot_idx = robot.find_bodies(".*foot")[0]


        # ----- Step 1 : Retrieve the initial state
        # Retrieve the robot position in local world frame of shape(3)
        com_pos_lw = (robot.data.root_pos_w - self._env.scene.env_origins) # shape(batch, 3)
        # Compute proprioceptive height
        com_pos_lw[:,2] = robot.data.root_pos_w[:,2] - (torch.sum((robot.data.body_pos_w[:, foot_idx,2]) * feet_in_contact, dim=1)) / (torch.sum(feet_in_contact, dim=1)) # height is proprioceptive shape(batch)

        # Retrieve the robot orientation in lw as euler angle ZXY of shape(3)
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(robot.data.root_quat_w) 
        roll, pitch, yaw = from_zero_twopi_to_minuspi_pluspi(roll, pitch, yaw) 
        # euler_xyz_angle = torch.tensor((roll, pitch, yaw), device=self.device)
        euler_xyz_angle = torch.stack((roll, pitch, yaw), dim=1) # shape (batch, 3)

        # Retrieve the robot linear and angular velocity in base frame of shape(3)
        com_lin_vel_lw = (robot.data.root_lin_vel_w) # shape(batch, 3) 
        com_ang_vel_b  = (robot.data.root_ang_vel_b) # shape(batch, 3)

        # Retrieve the feet position in local world frame of shape(num_legs*3)
        p_w = (robot.data.body_pos_w[:, foot_idx,:])          # shape(batch, num_legs, 3) - foot position in w
        p_lw = p_w - self._env.scene.env_origins.unsqueeze(1) # shape(batch, num_legs, 3) - foot position in lw

        # Prepare the state (at time t)
        batched_initial_state['pos_com_lw']      = com_pos_lw       # shape(batch, 3)              # CoM position in local world frame, height is propriocetive
        batched_initial_state['lin_com_vel_lw']  = com_lin_vel_lw   # shape(batch, 3)              # Linear Velocity in local world frame
        batched_initial_state['euler_xyz_angle'] = euler_xyz_angle  # shape(batch, 3)              # Euler angle in XYZ convention
        batched_initial_state['ang_vel_com_b']   = com_ang_vel_b    # shape(batch, 3)              # Angular velocity in base frame
        batched_initial_state['p_lw']            = p_lw             # shape(batch, num_legs, 3)    # Foot position in local world frame


        # ----- Step 2 : Retrieve the robot's reference along the integration horizon
        # Retrieve the speed command : (lin_vel_x_b, lin_vel_y_b, ang_vel_yaw_b)
        speed_command_b = (self._env.command_manager.get_command("base_velocity")) # shape(batch, 3)

        # CoM reference position : tracked only for the height -> xy can be anything eg 0
        com_pos_ref_seq_lw = torch.tensor((0,0,self.height_ref), device=self.device).unsqueeze(-1).unsqueeze(0).expand(self._env.num_envs, 3, self.sampling_horizon) # shape(batch, 3, sampling_horizon)

        # The pose reference is (0,0) for roll and pitch, but the yaw must be integrated along the horizon (in world frame)
        euler_xyz_angle_ref_seq      = torch.zeros_like(com_pos_ref_seq_lw) # shape(batch, 3, sampling_horizon)
        #                                 (batch, 1)                         + ( (batch, sampling_horizon)                                                                                               * (batch, 1) )
        euler_xyz_angle_ref_seq[:, 2,:] = euler_xyz_angle[:,2].unsqueeze(-1) + ((torch.arange(self.sampling_horizon, device=self.device).unsqueeze(0).expand(self._env.num_envs, self.sampling_horizon)) * (self.mpc_dt * speed_command_b[:,2].unsqueeze(-1))) # shape(batch, sampling_horizon)

        # The linear speed reference is (x_dot_ref_b, y_dot_ref_b, 0) in base frame and must be rotated in world frame
        com_lin_vel_ref_seq_b = torch.zeros((self._env.num_envs, 3), device=self.device) # shape(batch, 3)
        com_lin_vel_ref_seq_b[:, 0] = speed_command_b[:, 0] # foward  velocity reference # shape (batch)
        com_lin_vel_ref_seq_b[:, 1] = speed_command_b[:, 1] # lateral velocity reference # shape (batch)

        # Retrieve the robot base orientation in the world frame as quaternions : shape(batch_size, 4)
        quat_robot_base_w = robot.data.root_quat_w # shape(batch_size, 4)

        # From the quaternions compute the rotation matrix that rotates from base frame b to world frame w: shape(batch_size, 3,3)
        R_b_to_w = math_utils.matrix_from_quat(quat_robot_base_w) # shape(batch_size, 3,3)

        # Rotate the speed reference in world frame and expand to get the sampling horizon
        com_lin_vel_ref_seq_w = torch.matmul(R_b_to_w, com_lin_vel_ref_seq_b.unsqueeze(-1)).expand(self._env.num_envs, 3, self.sampling_horizon).clone().detach() # shape (batch,3,3)@(batch,3,1) -> (batch,3,1) -> shape(batch, 3, sampling_horizon)

        # Add the effect of the yaw rate in world frame
        delta_yaw_seq = self.mpc_dt * speed_command_b[:,2].unsqueeze(-1) * torch.arange(0, self.sampling_horizon, device=self.device).unsqueeze(0) # (b,1)*(1,h)  -> shape(batch, sampling_horizon)

        com_lin_vel_ref_seq_w[:,0,:] = ((com_lin_vel_ref_seq_w[:,0,:] * torch.cos(delta_yaw_seq)) - (com_lin_vel_ref_seq_w[:,1,:] * torch.sin(delta_yaw_seq))) # shape(batch, sampling_horizon)
        com_lin_vel_ref_seq_w[:,1,:] = ((com_lin_vel_ref_seq_w[:,0,:] * torch.sin(delta_yaw_seq)) + (com_lin_vel_ref_seq_w[:,1,:] * torch.cos(delta_yaw_seq))) # shape(batch, sampling_horizon)

        # Get the angular velocity reference in base frame as euler rates : (0, 0, yaw rate)
        com_ang_vel_ref_seq_b = torch.zeros_like(com_pos_ref_seq_lw) # shape(batch, 3, sampling_horizon)

        # Angular velocity reference is base frame (roll_rate=0, pitch_rate=0, yaw_rate=yaw_rate_ref)
        com_ang_vel_ref_seq_b[:,2,:] = speed_command_b[:,2].unsqueeze(-1)  # shape(batch, sampling_horizon)

        # Defining the foot position sequence is tricky.. Since we only have number of predicted step < sampling_horizon
        p_ref_seq_lw = torch.zeros((self._env.num_envs, self.num_legs, 3, self.sampling_horizon), device=self.device) # shape(batch, num_legs, 3, sampling_horizon) TODO Define this !

        # Compute the gravity compensation GRF along the horizon : of shape (batch, num_legs, 3, sampling_horizon)
        num_leg_contact_seq_samples = (torch.sum(batched_c, dim=1)).clamp(min=1) # Compute the number of leg in contact, clamp by minimum 1 to avoid division by zero. shape(batch, sampling_horizon)
        gravity_compensation_F_samples = torch.zeros((self._env.num_envs, self.num_legs, 3, self.sampling_horizon), device=self.device) # shape (batch, num_legs, 3, sampling_horizon)
        gravity_compensation_F_samples[:,:,2,:] =  batched_c * ((self.robot_mass * 9.81)/num_leg_contact_seq_samples).unsqueeze(1)    # shape (batch, num_legs, sampling_horizon)
        
        # Prepare the reference sequence (at time t, t+dt, etc.)
        batched_reference_seq_state['pos_com_lw']      = com_pos_ref_seq_lw      # shape(batch, 3, sampling_horizon)           # CoM position reference in local world frame
        batched_reference_seq_state['lin_com_vel_lw']  = com_lin_vel_ref_seq_w   # shape(batch, 3, sampling_horizon)           # Linear Velocity reference in local world frame
        batched_reference_seq_state['euler_xyz_angle'] = euler_xyz_angle_ref_seq # shape(batch, 3, sampling_horizon)           # Euler angle reference in XYZ convention
        batched_reference_seq_state['ang_vel_com_b']   = com_ang_vel_ref_seq_b   # shape(batch, 3, sampling_horizon)           # Angular velocity reference in base frame
        batched_reference_seq_state['p_lw']            = p_ref_seq_lw            # shape(batch, num_legs, 3, sampling_horizon) # Foot position in local world frame

        # Prepare the input sequence reference for the samples
        bacthed_reference_seq_input['F_lw'] = gravity_compensation_F_samples # shape(batch, num_legs, 3, sampling_horizon) # gravity compensation per samples along the horizon


        # ----- Step 3 : Retrieve the actions and prepare them with the correct method
        batched_action_param['p_lw']       = batched_p_lw.clone().detach()       # Foot touch down position samples          # shape(batch, num_legs, 3 p_param)
        batched_action_param['delta_F_lw'] = batched_delta_F_lw.clone().detach() # Delta Ground Reaction Forces samples      # shape(batch, num_legs, 3 F_param)

        # ----- Step 4 : Save variable for live plotting
        if self.live_plot : 
            # Save Robot's Height
            self.robot_height_list.append(com_pos_lw[0,2].cpu().numpy())
            if len(self.robot_height_list) > 100 : self.robot_height_list.pop(0)
            np.savetxt('live_variable/height.csv', [self.robot_height_list], delimiter=',', fmt='%.3f')

            # Save Robot's Height reference
            self.robot_height_ref_list.append(self.height_ref)
            if len(self.robot_height_ref_list) > 100 : self.robot_height_ref_list.pop(0)
            np.savetxt('live_variable/height_ref.csv', [self.robot_height_ref_list], delimiter=',', fmt='%.3f')

            # Save foot position's
            self.FL_foot_list.append(batched_initial_state['p_lw'][0,0,:].cpu().numpy())
            if len(self.FL_foot_list) > 100 : self.FL_foot_list.pop(0)
            np.savetxt('live_variable/FL_foot.csv', self.FL_foot_list, delimiter=',', fmt='%.3f')
            self.FR_foot_list.append(batched_initial_state['p_lw'][0,1,:].cpu().numpy())
            if len(self.FR_foot_list) > 100 : self.FR_foot_list.pop(0)
            np.savetxt('live_variable/FR_foot.csv', self.FR_foot_list, delimiter=',', fmt='%.3f')
            self.RL_foot_list.append(batched_initial_state['p_lw'][0,2,:].cpu().numpy())
            if len(self.RL_foot_list) > 100 : self.RL_foot_list.pop(0)
            np.savetxt('live_variable/RL_foot.csv', self.RL_foot_list, delimiter=',', fmt='%.3f')
            self.RR_foot_list.append(batched_initial_state['p_lw'][0,3,:].cpu().numpy())
            if len(self.RR_foot_list) > 100 : self.RR_foot_list.pop(0)
            np.savetxt('live_variable/RR_foot.csv', self.RR_foot_list, delimiter=',', fmt='%.3f')

        return batched_initial_state, batched_reference_seq_state, bacthed_reference_seq_input, batched_action_param
    
    
    def batched_compute_rollout(self, batched_initial_state: dict, batched_reference_seq_state: dict, batched_reference_seq_input: dict, batched_action_param: dict, batched_c: torch.Tensor) -> torch.Tensor:
        """
        Calculate cost of rollouts of given action sequence samples 

        Args :
            batched_initial_state       (dict): Dictionnary containing the current robot's state
                pos_com_lw      (Tensor): CoM position in local world frame                             of shape(batch, 3)
                lin_com_vel_lw  (Tensor): CoM linear velocity in local world frame                      of shape(batch, 3)
                euler_xyz_angle (Tensor): CoM orientation (wrt. to l. world frame) as XYZ euler angle   of shape(batch, 3)
                ang_vel_com_b   (Tensor): CoM angular velocity as roll pitch yaw                        of shape(batch, 3)
                p_lw            (Tensor): Feet position in local world frame                            of shape(batch, num_legs, 3)

            batched_reference_seq_state (dict): Dictionnary containing the robot's reference state along the prediction horizon
                pos_com_lw      (Tensor): CoM position in local world frame                             of shape(batch, 3, sampling_horizon)
                lin_com_vel_lw  (Tensor): CoM linear velocity in local world frame                      of shape(batch, 3, sampling_horizon)
                euler_xyz_angle (Tensor): CoM orientation (wrt. to l. world frame) as XYZ euler angle   of shape(batch, 3, sampling_horizon)
                ang_vel_com_b   (Tensor): CoM angular velocity as roll pitch yaw                        of shape(batch, 3, sampling_horizon)
                p_lw            (Tensor): Feet position in local world frame                            of shape(batch, num_legs, 3, sampling_horizon)                

            batched_reference_seq_input (dict): 
                F_lw            (Tensor): Reference GRF sequence samples along the prediction horizon   of shape(batch, num_legs, 3, sampling_horizon)  

            batched_action_param  (dict): Dictionnary containing the robot's actions along the prediction horizon 
                p_lw            (Tensor): Foot touch down position in local world frame                 of shape(batch, num_legs, 3, p_param)
                delta_F_lw      (Tensor): Delta GRF parameters in local world frame                     of shape(batch, num_legs, 3, F_param)

            batched_c       (t.bool): Foot contact sequence sample                                      of shape(batch, num_legs, sampling_horizon)

        Return 
            bacthed_cost        (Tensor): Cost associated at each sample (along the prediction horizon) of shape(batch)
        """
        bacthed_cost = torch.zeros(self._env.num_envs, device=self.device)
        state = {}
        state['pos_com_lw']      = batched_initial_state['pos_com_lw']
        state['lin_com_vel_lw']  = batched_initial_state['lin_com_vel_lw']
        state['euler_xyz_angle'] = batched_initial_state['euler_xyz_angle']
        state['ang_vel_com_b']   = batched_initial_state['ang_vel_com_b']
        state['p_lw']            = batched_initial_state['p_lw']
        input = {}

        for i in range(self.sampling_horizon):
            # --- Step 1 : prepare the inputs
            # Find the current action given the actions parameters
            input['p_lw'] = self.interpolation_p(parameters=batched_action_param['p_lw'],       step=i, horizon=self.sampling_horizon)# shape(batch, num_legs, 3)
            input['F_lw'] = self.interpolation_F(parameters=batched_action_param['delta_F_lw'], step=i, horizon=self.sampling_horizon)# shape(batch, num_legs, 3)
            contact = batched_c[:,:,i]                                                                                                # shape(batch, num_legs)

            # Add gravity compensation
            input['F_lw'] -= contact.unsqueeze(-1) * (self.gravity_lw.unsqueeze(0).unsqueeze(0) * self.robot_mass) / (torch.sum(contact, dim=1).clamp_min(min=1)).unsqueeze(-1).unsqueeze(-1)      

            # Enforce force constraints (Friction cone constraints)
            input['F_lw'] = enforce_friction_cone_constraints_torch(F=input['F_lw'], mu=self.mu, F_z_min=self.F_z_min, F_z_max=self.F_z_max) # shape(batch, num_legs, 3)


            # --- Step 2 : Step the model
            new_state = self.centroidal_model_step(state=state, input=input, contact=contact)
            state = new_state


            # --- Step 3 : compute the step cost
            state_vector     = torch.cat([vector.view(self._env.num_envs, -1) for vector in state.values()], dim=1)             # Shape: (batch, state_dim)
            ref_state_vector = torch.cat([vector[...,i].view(self._env.num_envs, -1) for vector in batched_reference_seq_state.values()], dim=1)            # Shape: (batch, state_dim)
            
            # Compute the state cost
            state_error = state_vector - ref_state_vector                                                                       # shape (num_samples, state_dim)
            state_cost  = torch.sum(self.Q_vec.unsqueeze(0) * (state_error ** 2), dim=1)                                        # Shape (num_samples)

            # Compute the input cost
            input_error = (input['F_lw'] - batched_reference_seq_input['F_lw'][:,:,:,i]).flatten(1,2)                           # shape (num_samples, action_dim)
            input_cost  = torch.sum(self.R_vec.unsqueeze(0) * (input_error ** 2), dim=1)                                        # Shape (num_samples)

            step_cost = state_cost #+ input_cost                                                                                # shape(num_samples)

            # Update the trajectory cost
            bacthed_cost += step_cost                                                                                           # shape(num_samples)


            # print('Lin vel ', state['lin_com_vel_lw'])
            # print('Lin vel ref ', batched_reference_seq_state['lin_com_vel_lw'][...,i])
            # print('state error ', state_error)
            # print()

        return bacthed_cost
    

    def centroidal_model_step(self, state, input, contact):
        """
        Simulate one step of the forward dynamics, using the controidal model. 

        Note 
            Model used is described in 'Model Predictive Control With Environment Adaptation for Legged Locomotion'
            ttps://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9564053
            
        Args : 
            state     (dict): Dictionnary containing the robot's state
                pos_com_lw      (tensor): CoM position in local world frame                             of shape(num_samples, 3)
                lin_com_vel_lw  (tensor): CoM linear velocity in local world frame                      of shape(num_samples, 3)
                euler_xyz_angle (tensor): CoM orientation (wrt. to l. world frame) as XYZ euler angle   of shape(num_samples, 3)
                ang_vel_com_b   (tensor): CoM angular velocity as roll pitch yaw                        of shape(num_samples, 3)
                p_lw            (tensor): Feet position in local world frame                            of shape(num_samples, num_legs, 3)

            input     (dict): Dictionnary containing the robot's input
                p_lw            (tensor): Feet touch down position in local world frame                 of shape(num_samples, num_legs, 3)
                F_lw            (tensor): Ground Reaction Forces in local world frame                   of shape(num_samples, num_legs, 3)

            contact (tensor): Foot contact status (stance=1, swing=0)                                   of shape(num_samples, num_legs)

        Return :
            new_state (dict): Dictionnary containing the robot's updated state after one iteration
                pos_com_lw      (tensor): CoM position in local world frame                             of shape(num_samples, 3)
                lin_com_vel_lw  (tensor): CoM linear velocity in local world frame                      of shape(num_samples, 3)
                euler_xyz_angle (tensor): CoM orientation (wrt. to l. world frame) as XYZ euler angle   of shape(num_samples, 3)
                ang_vel_com_b   (tensor): CoM angular velocity as roll pitch yaw                        of shape(num_samples, 3)
                p_lw            (tensor): Feet position in local world frame                            of shape(num_samples, num_legs, 3)
        """
        new_state = {}

        # --- Step 1 : Compute linear velocity
        lin_com_vel_lw = state['lin_com_vel_lw']     # shape (num_samples, 3)


        # --- Step 2 : Compute linear acceleration  as sum of forces divide by mass
        linear_com_acc_lw = (torch.sum(input['F_lw'] * contact.unsqueeze(-1), dim=1) / self.robot_mass) + self.gravity_lw.unsqueeze(0) # shape (num_samples, 3)


        # --- Step 3 : Compute angular velocity (as euler rate to increment euler angles) : Given by eq. 27 in 'Note' paper
        # euler_xyz_rate = inverse(conjugate(euler_xyz_rate_matrix) * omega_b
        euler_xyz_rate = torch.bmm(inverse_conjugate_euler_xyz_rate_matrix(state['euler_xyz_angle']), state['ang_vel_com_b'].unsqueeze(-1)).squeeze(-1 ) # shape (s,3,3)*(s,3,1)->(s,3,1) -> (num_samples, 3)


        # --- Step 4 : Compute angular acceleration : Given by eq. 4 in 'Note' paper
        # Compute the sum of the moment induced by GRF (1. Moment=dist_from_F cross F, 2. Keep moment only for feet in contact, 3. sum over 4 legs)
        sum_of_GRF_moment_lw = torch.sum(contact.unsqueeze(-1) * torch.cross((state['p_lw'] - state['pos_com_lw'].unsqueeze(1)), input['F_lw'], dim=-1), dim=1) # shape sum{(s,l,1)*[(s,l,3)^(s,l,3)], dim=l} -> (num_sample, 3)

        # Transform the sum of moment from local world frame to base frame
        sum_of_GRF_moment_b  = torch.bmm(rotation_matrix_from_w_to_b(state['euler_xyz_angle']), sum_of_GRF_moment_lw.unsqueeze(-1)).squeeze(-1) # shape(num_samples, 3)

        # Compute intermediary variable : w_b^(I*w_b) : unsqueeze robot inertia for batched operation, squeeze and unsqueeze matmul to perform dot product (1,3,3)*(s,3,1)->(s,3,1)->(s,3)
        omega_b_cross_inertia_dot_omega_b = torch.cross(state['ang_vel_com_b'], torch.matmul(self.robot_inertia.unsqueeze(0), state['ang_vel_com_b'].unsqueeze(-1)).squeeze(-1), dim=-1) # shape(num_samples, 3)

        # Finally compute the angular acc : unsqueeze robot inertia for batched operation, squeeze and unsqueeze matmul to perform dot product (1,3,3)*(s,3,1)->(s,3,1)->(s,3)
        ang_acc_com_b = torch.matmul(self.inv_robot_inertia.unsqueeze(0), (sum_of_GRF_moment_b - (omega_b_cross_inertia_dot_omega_b)).unsqueeze(-1)).squeeze(-1) # shape(num_samples, 3)


        # --- Step 5 : Perform forward integration of the model (as simple forward euler)
        new_state['pos_com_lw']      = state['pos_com_lw']      + self.mpc_dt*lin_com_vel_lw
        new_state['lin_com_vel_lw']  = state['lin_com_vel_lw']  + self.mpc_dt*linear_com_acc_lw
        new_state['euler_xyz_angle'] = state['euler_xyz_angle'] + self.mpc_dt*euler_xyz_rate
        new_state['ang_vel_com_b']   = state['ang_vel_com_b']   + self.mpc_dt*ang_acc_com_b
        # new_state['p_lw']            = state['p_lw']
        new_state['p_lw']            = state['p_lw']*(contact.unsqueeze(-1)) + input['p_lw']*(~contact.unsqueeze(-1))

        return new_state