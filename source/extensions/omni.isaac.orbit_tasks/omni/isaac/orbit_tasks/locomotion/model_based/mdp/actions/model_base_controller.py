from abc import ABC
from collections.abc import Sequence
import torch
from torch.distributions.constraints import real

# import jax.numpy as jnp
# import numpy as np

import omni.isaac.orbit.utils.math as math_utils

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    pass
from omni.isaac.orbit.envs import RLTaskEnv
from omni.isaac.orbit.assets.articulation import Articulation

 


# import jax
# import jax.dlpack
# import torch
# import torch.utils.dlpack

# def jax_to_torch(x: jax.Array):
#     return torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(x))
# def torch_to_jax(x):
#     return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(x))

# import numpy as np
# import matplotlib.pyplot as plt
# np.set_printoptions(precision=2, linewidth=200)
# force=[[],[],[],[],[],[],[],[],[],[],[],[]]
# torque=[[],[],[],[],[],[],[],[],[],[],[],[]]
# pos_tracking_error = [[],[],[],[]]
# vel_tracking_error = [[],[],[],[]]
# acc_tracking_error = [[],[],[],[]]

class modelBaseController(ABC):
    """
    Abstract controller class for model base control implementation
    
    Properties : 
        - verbose_md    : Verbose variable. Save some computation that aren't necessary if not in debug mode.
        - _device
        - _num_envs
        - _num_legs
        - _time_horizon : Outer Loop prediction time horizon
        - _dt_out       : Outer Loop time step 
        - _decimation   : Inner Loop time horizon
        - _dt_in        : Inner Loop time step

    Method :
        - late_init(device, num_envs, num_legs) : save environment variable and allow for lazy initialisation of variables
        - reset(env_ids) : Reset controller variables upon environment reset if needed
        - optimize_latent_variable(f, d, p, F) -> p*, F*, c*, pt*
        - compute_control_output(F0*, c0*, pt01*) -> T
        - gait_generator(f, d, phase) -> c, new_phase

    """

    def __init__(self, verbose_md, device, num_envs, num_legs, time_horizon, dt_out, decimation, dt_in):
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
        self._time_horizon = time_horizon
        self._dt_out = dt_out
        self._decimation = decimation
        self._dt_in = dt_in


    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """ The environment is reseted -> this requires to reset some controller variables
        """
        pass 


    def optimize_latent_variable(self, f: torch.Tensor, d: torch.Tensor, p_b: torch.Tensor, F_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Given the latent variable z=[f,d,p,F], return the optimized latent variable p*, F*, c*, pt*

        Args:
            - f   (torch.Tensor): Prior leg frequency                   of shape (batch_size, num_legs)
            - d   (torch.Tensor): Prior stepping duty cycle             of shape (batch_size, num_legs)
            - p_b (torch.Tensor): Prior foot pos. seq. in base frame    of shape (batch_size, num_legs, 3, time_horizon)
            - F_w (torch.Tensor): Prior Ground Reac. Forces (GRF) seq.  of shape (batch_size, num_legs, 3, time_horizon)
                                  In world frame

        Returns:
            - p*  (torch.Tensor): Optimized foot position sequence      of shape (batch_size, num_legs, 3, time_horizon)
            - F*  (torch.Tensor): Opt. Ground Reac. Forces (GRF) seq.   of shape (batch_size, num_legs, 3, time_horizon)
            - c*  (torch.Tensor): Optimized foot contact sequence       of shape (batch_size, num_legs, time_horizon)
        """
        raise NotImplementedError


    def compute_control_output(self, F0_star: torch.Tensor, c0_star: torch.Tensor, pt_i_star: torch.Tensor) -> torch.Tensor:
        """ Compute the output torque to be applied to the system
        typically, it would compute :
            - T_stance_phase = stance_leg_controller(GRF, q, c) # Update the jacobian with the new joint position.
            - T_swing_phase = swing_leg_controller(trajectory, q, q_dot, c) # Feedback lineraization control - trajectory computed with a spline to be followed - new updated joint controller.
            - T = (T_stance_phase * c_star) + (T_swing_phase * (~c_star))
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

    
class samplingController(modelBaseController):
    """
    Implement a model based controller based on the latent variable z = [f,d,p,F]

    - Gait Generator :
        From the latent variables f and d (leg frequency & duty cycle) and the phase property, compute the next leg
        phases and contact sequence.

    - Sampling Controller :
        Optimize the latent variable z. Generates samples, simulate the samples, evaluate them and return the best one.

    Properties : 
        - verbose_md    : Verbose variable. Save some computation that aren't necessary if not in debug mode.           Inherited from modelBaseController
        - _device                                                                                                       Inherited from modelBaseController
        - _num_envs                                                                                                     Inherited from modelBaseController
        - _num_legs                                                                                                     Inherited from modelBaseController
        - _time_horizon : Outer Loop prediction time horizon                                                            Inherited from modelBaseController
        - _dt_out       : Outer Loop time step                                                                          Inherited from modelBaseController
        - _decimation   : Inner Loop time horizon                                                                       Inherited from modelBaseController
        - _dt_in        : Inner Loop time step                                                                          Inherited from modelBaseController
        - phase (Tensor): Leg phase                                             of shape (batch_size, num_legs)
        - p0_lw (Tensor): Lift-off position                                     of shape (batch_size, num_legs, 3)
        - swing_time (T): time progression of the leg in swing phase            of shape (batch_size, num_legs)  
        - p_lw_sim_prev : Last foot position from sim. upd in comp_ctrl in _lw  of shape (batch_size, num_legs, 3)
        - step_height   : Apex height of the swing trajectory
        - FOOT_OFFSET   : Offset between the foot (as return by the sim.) and the ground when in contact

    Method :
        - late_init(device, num_envs, num_legs) : save environment variable and allow for lazy init of variables        Inherited from modelBaseController
        - reset(env_ids) : Reset controller variables upon environment reset if needed                                  Inherited from modelBaseController (not implemented)
        - optimize_latent_variable(f, d, p, F) -> p*, F*, c*, pt*                                                       Inherited from modelBaseController (not implemented)
        - compute_control_output(F0*, c0*, pt01*) -> T                                                                  Inherited from modelBaseController (not implemented)
        - gait_generator(f, d, phase) -> c, new_phase                                                                   Inherited from modelBaseController (not implemented)
        - swing_trajectory_generator(p_b, c, decimation) -> pt_b
        - swing_leg_controller(c0*, pt01*) -> T_swing
        - stance_leg_controller(F0*, c0*) -> T_stance
    """

    phase : torch.Tensor
    p0_lw : torch.Tensor
    swing_time : torch.Tensor
    p_lw_sim_prev : torch.Tensor

    def __init__(self, verbose_md, device, num_envs, num_legs, time_horizon, dt_out, decimation, dt_in, p_default_lw: torch.Tensor, step_height, foot_offset, swing_ctrl_pos_gain_fb, swing_ctrl_vel_gain_fb):
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
        super().__init__(verbose_md, device, num_envs, num_legs, time_horizon, dt_out, decimation, dt_in)
        self.phase = torch.zeros(num_envs, num_legs, device=device)
        self.phase[:,(0,3)] = 0.5 # Init phase [0.5, 0, 0.5, 0]
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
    def optimize_latent_variable(self, f: torch.Tensor, d: torch.Tensor, p_lw: torch.Tensor, F_lw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Given the latent variable z=[f,d,p,F], return the optimized latent variable p*, F*, c*, pt*
        Note :
            The variable are in the 'local' world frame _wl. This notation is introduced to avoid confusion with the 'global' world frame, where all the batches coexists.

        Args:
            - f   (torch.Tensor): Prior leg frequency                   of shape (batch_size, num_legs)
            - d   (torch.Tensor): Prior stepping duty cycle             of shape (batch_size, num_legs)
            - p_lw (trch.Tensor): Prior foot touch down seq. in _lw     of shape (batch_size, num_legs, 3, number_predict_step)
            - F_lw (trch.Tensor): Prior Ground Reac. Forces (GRF) seq.  of shape (batch_size, num_legs, 3, time_horizon)
                                  In local world frame

        Returns:
            - p*_lw (tch.Tensor): Optimized foot touch down seq. in _lw of shape (batch_size, num_legs, 3, number_predict_step)
            - F*_lw (tch.Tensor): Opt. Gnd Reac. F. (GRF) seq.   in _lw of shape (batch_size, num_legs, 3, time_horizon)
            - c*  (torch.Tensor): Optimized foot contact sequence       of shape (batch_size, num_legs, time_horizon)
            - pt*_lw (th.Tensor): Optimized foot swing traj.     in _lw of shape (batch_size, num_legs, 9, decimation)  (9 = pos, vel, acc)
        """

        # Call the optimizer
        # F_star_lw, p_star_lw, c_star = self.samplingOptimizer()

        # Compute the contact sequence and update the phase
        c, self.phase = self.gait_generator(f=f, d=d, phase=self.phase, time_horizon=self._time_horizon, dt=self._dt_out)

        # Generate the swing trajectory
        # pt_lw = self.swing_trajectory_generator(p_lw=p_lw[:,:,:,0], c=c, d=d, f=f)
        pt_lw, full_pt_lw = self.full_swing_trajectory_generator(p_lw=p_lw[:,:,:,0], c=c, d=d, f=f)

        p_star_lw = p_lw
        F_star_lw = F_lw
        c_star = c
        pt_star_lw = pt_lw

        return p_star_lw, F_star_lw, c_star, pt_star_lw, full_pt_lw
    

    def gait_generator(self, f: torch.Tensor, d: torch.Tensor, phase: torch.Tensor, time_horizon: int, dt) -> tuple[torch.Tensor, torch.Tensor]:
        """ Implement a gait generator that return a contact sequence given a leg frequency and a leg duty cycle
        Increment phase by dt*f 
        restart if needed
        return contact : 1 if phase < duty cyle, 0 otherwise  
        c == 1 : Leg is in contact (stance)
        c == 0 : Leg is in swing

        Note:
            No properties used, no for loop : purely functional -> made to be jitted
            parallel_rollout : this is optional, it will work without the parallel rollout dimension

        Args:
            - f   (torch.Tensor): Leg frequency                         of shape(batch_size, (parallel_rollout), num_legs)
            - d   (torch.Tensor): Stepping duty cycle in [0,1]          of shape(batch_size, (parallel_rollout), num_legs)
            - phase (tch.Tensor): phase of leg in [0,1]                 of shape(batch_size, (parallel_rollout), num_legs)
            - time_horizon (int): Time horizon for the contact sequence

        Returns:
            - c     (torch.bool): Foot contact sequence                 of shape(batch_size, (parallel_rollout), num_legs, time_horizon)
            - phase (tch.Tensor): The phase updated by one time steps   of shape(batch_size, (parallel_rollout), num_legs)
        """
        
        # Increment phase of f*dt: new_phases[0] : incremented of 1 step, new_phases[1] incremented of 2 steps, etc. without a for loop.
        # new_phases = phase + f*dt*[1,2,...,time_horizon]
        # phase and f must be exanded from (batch_size, num_legs, parallel_rollout) to (batch_size, num_legs, parallel_rollout, time_horizon) in order to perform the operations
        new_phases = phase.unsqueeze(-1).expand(*[-1] * len(phase.shape),time_horizon) + f.unsqueeze(-1).expand(*[-1] * len(f.shape),time_horizon)*torch.linspace(start=1, end=time_horizon, steps=time_horizon, device=self._device)*dt

        # Make the phases circular (like sine) (% is modulo operation)
        new_phases = new_phases%1

        # Save first phase
        new_phase = new_phases[..., 0]

        # Make comparaison to return discret contat sequence : c = 1 if phase < d, 0 otherwise
        c = new_phases <= d.unsqueeze(-1).expand(*[-1] * len(d.shape), time_horizon)

        return c, new_phase


    def full_swing_trajectory_generator(self, p_lw: torch.Tensor, c: torch.Tensor, f: torch.Tensor, d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ Given feet position sequence and contact sequence -> compute swing trajectories by fitting a cubic spline between
        the lift-off and the touch down define in the contact sequence. 
        - Swing frequency and duty cycle are used to compute the swing period
        - A middle point is used for the interpolation : which is heuristically defined. It defines the step height
        - p1 (middle point) and p2 (touch-down) are updated each time, while p0 is conserved (always the same lift off position)
        Note :
            The variable are in the 'local' world frame _wl. This notation is introduced to avoid confusion with the 'global' world frame, where all the batches coexists.
        
        Args:
            - p_lw (trch.Tensor): Foot touch down postion in _lw        of shape(batch_size, num_legs, 3)
            - c   (torch.Tensor): Foot contact sequence                 of shape(batch_size, num_legs, time_horizon)
            - f   (torch.Tensor): Leg frequency           in R+         of shape(batch_size, num_legs)
            - d   (torch.Tensor): Stepping duty cycle     in[0,1]       of shape(batch_size, num_legs)

        Returns:
            - pt_lw (tch.Tensor): Desired Swing Leg traj. in _lw frame  of shape(batch_size, num_legs, 9, decimation)   (9 = xyz_pos, xzy_vel, xyz_acc)
        """
        # Step 1. Compute the phase trajectory : shape (batch_size, num_legs, decimation)
        # ie. retrieve the actual leg phase -> and compute the trajectory (phase evolution) for the next outer loop period (ie. for decimation inner loop iteration) 

        # swing phase in [0,1] (leg is in swing when phase = [d, 1] -> scale to have swing_phase in [0,1]), shape(batch_size, num_legs)
        swing_phase = (self.phase - d) / (1 - d + 1e-10)  

        swing_frequency = f / (1 - d + 1e-10)           # [Hz] : swing frequency,   shape(batch_size, num_legs)
        delta_phase = swing_frequency * self._dt_in      # delta_phase = swing_freq [Hz] * dt [s],  shape(batch_size, num_legs)

        # swing phase trajectpry [phase, phase + delta_phase, ...],   shape (batch_size, num_legs, decimation)
        # (batch_size, num_legs, 1) + [(1, 1, decimation) * (batch_size, num_legs, 1)] -> (batch_size, num_legs, decimation)
        swing_phase_traj = (swing_phase.unsqueeze(-1)) + ((0+torch.arange(self._decimation, device=self._device).unsqueeze(0).unsqueeze(0)) * delta_phase.unsqueeze(-1))


        # Step 2. Retrieve the three interpolation points : p0, p1, p2 (lift-off, middle point, touch down)

        # Retrieve p0 : update p0 with latest foot position when in contact, don't update when in swing
        # p0 shape (batch_size, num_legs, 3)
        in_contact = (c[:,:,0]==1).unsqueeze(-1)    # True if foot in contact, False in in swing, shape (batch_size, num_legs, 1)
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

        # Step 3. Compute the parameters for the interpolation (control points)
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


        # Step 4. Prepare parameters to compute interpolation trajectory in one operation -> matrix multiplication
        # Prepare swing phase traj to be multplied by cp_x : shape(batch_size, num_leg, decimation) -> (batch_size, num_leg, 1, decimation) (unsqueezed(2) but is_S0 is already unsqueezed (ie in the right shape))
        # swing phase may be > 1 if we reach the end of the traj, thus we clamp it to 1. 
        # Moreover, S0 and S1 takes values in [0,1], thus swing phase need to be double (and modulo 1) to be corrected
        phase_traj = (2 * swing_phase_traj.unsqueeze(2) - 1*(~is_S0)).clamp(0,1) # ie. double_swing_phase_traj


        # Step 5. Compute the interpolation trajectory
        # shape (batch_size, num_legs, 3, decimation)
        desired_foot_pos_traj_lw = cp1*(1 - phase_traj)**3 + 3*cp2*(phase_traj)*(1 - phase_traj)**2 + 3*cp3*((phase_traj)**2)*(1 - phase_traj) + cp4*(phase_traj)**3
        desired_foot_vel_traj_lw = 3*(cp2 - cp1)*(1 - phase_traj)**2 + 6*(cp3 - cp2)*(1 - phase_traj)*(phase_traj) + 3*(cp4 - cp3)*(phase_traj)**2
        desired_foot_acc_traj_lw = 6*(1 - phase_traj) * (cp3 - 2*cp2 + cp1) + 6 * (phase_traj) * (cp4 - 2*cp3 + cp2)

        # shape (batch_size, num_legs, 9, decimation) (9 = xyz_pos, xzy_vel, xyz_acc)
        pt_lw = torch.cat((desired_foot_pos_traj_lw, desired_foot_vel_traj_lw, desired_foot_acc_traj_lw), dim=2)

        # There are some NaN in the trajectory -> Replace them with zeros
        if not real.check(pt_lw).all() :
            print('Problem with NaN')
            NaN_index = torch.nonzero(~real.check(pt_lw)) # shape (number of nan, 4) (4 = batch + legs + 9 + decimation)

            # Check if there is faulty value for a leg in swing, we're only interested in the first two index of NaN index. 
            if not (in_contact[NaN_index[:,0], NaN_index[:,1],:].all()) :
                breakpoint()
                raise ValueError('A NaN was found on a trajectory for a swing leg, thus there is a problem with the math')
            
            # Else, this is not a problem since the swing torque for a leg in stance should not be applied, thus one can simple filter the faulty values out.
            pt_lw[~real.check(pt_lw)] = 0

        
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
        else : full_pt_lw = torch.empty(1)

        return pt_lw, full_pt_lw


# ----------------------------------- Inner Loop ------------------------------
    def compute_control_output(self, F0_star_lw: torch.Tensor, c0_star: torch.Tensor, pt_i_star_lw: torch.Tensor, p_lw:torch.Tensor, p_dot_lw:torch.Tensor, q_dot: torch.Tensor, jacobian_lw: torch.Tensor, jacobian_dot_lw: torch.Tensor, mass_matrix: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """ Compute the output torque to be applied to the system
        typically, it would compute :
            - T_stance_phase = stance_leg_controller(GRF, q, c) # Update the jacobian with the new joint position.
            - T_swing_phase = swing_leg_controller(trajectory, q, q_dot, c) # Feedback lineraization control - trajectory computed with a spline to be followed - new updated joint controller.
            - T = (T_stance_phase * c_star) + (T_swing_phase * (~c_star))
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

        # ---- Plot Torques ----
        # if c0_star[0,0]:
        #     torque[0].append(T_stance.cpu()[0,0,0])
        #     torque[1].append(T_stance.cpu()[0,0,1])
        #     torque[2].append(T_stance.cpu()[0,0,2])
        # if c0_star[0,1]:
        #     torque[3].append(T_stance.cpu()[0,1,0])
        #     torque[4].append(T_stance.cpu()[0,1,1])
        #     torque[5].append(T_stance.cpu()[0,1,2])
        # if c0_star[0,2]:
        #     torque[6].append(T_stance.cpu()[0,2,0])
        #     torque[7].append(T_stance.cpu()[0,2,1])
        #     torque[8].append(T_stance.cpu()[0,2,2])
        # if c0_star[0,3]:
        #     torque[9].append(T_stance.cpu()[0,3,0])
        #     torque[10].append(T_stance.cpu()[0,3,1])
        #     torque[11].append(T_stance.cpu()[0,3,2])
        #
        # if len(torque[0]) == 1000:
        #     row_labels = ['FL [Nm]', 'FR [Nm]', 'RL [Nm]', 'RR [Nm]']
        #     col_labels = ['Hip', 'Thigh', 'Calf']
        #     fig, axs = plt.subplots(4, 3,sharey='col')
        #     for i, ax in enumerate(axs.flat):
        #         ax.plot(torque[i])
        #         if (i%3) == 0:
        #             ax.set_ylabel(row_labels[i//3])
        #         if i >=9 :
        #             ax.set_xlabel(col_labels[i-9])
        #     fig.suptitle('Robot\'s Joint Torque over time', fontsize=16)
        #     for i in range(len(torque)):
        #         print('%s %s - mean:%2.2f \t std:%.2f' % (row_labels[i//3],col_labels[i%3],np.mean(torque[i]),np.std(torque[i])))
        #     # plt.show()
        #     plt.savefig("mygraph.png")

        # ---- plot swing tracking error ----
        # pos_err_rmse = (pt_i_star_lw[:,:,0:3] - p_lw).pow(2).mean(dim=-1).sqrt()
        # vel_err_rmse = (pt_i_star_lw[:,:,3:6] - p_dot_lw).pow(2).mean(dim=-1).sqrt()
        # if c0_star[0,0]:
        #     if pos_err_rmse[0,0] < 0.2:
        #         pos_tracking_error[0].append(pos_err_rmse[0,0].cpu())
        #         vel_tracking_error[0].append(vel_err_rmse[0,0].cpu())
        # if c0_star[0,1]:
        #     if pos_err_rmse[0,1] < 0.2:
        #         pos_tracking_error[1].append(pos_err_rmse[0,1].cpu())
        #         vel_tracking_error[1].append(vel_err_rmse[0,1].cpu())
        # if c0_star[0,2]:
        #     if pos_err_rmse[0,2] < 0.2:
        #         pos_tracking_error[2].append(pos_err_rmse[0,2].cpu())
        #         vel_tracking_error[2].append(vel_err_rmse[0,2].cpu())
        # if c0_star[0,3]:
        #     if pos_err_rmse[0,3] < 0.2:
        #         pos_tracking_error[3].append(pos_err_rmse[0,3].cpu())
        #         vel_tracking_error[3].append(vel_err_rmse[0,3].cpu())
        # if len(pos_tracking_error[0]) == 1000:
        #     row_labels = ['FL [Nm]', 'FR [Nm]', 'RL [Nm]', 'RR [Nm]']
        #     col_labels = ['Position RMSE', 'Velocity RMSE']
        #     fig, axs = plt.subplots(4, 2, figsize=(19.20,10.80))#,sharey='col')
        #     for i, ax in enumerate(axs.flat):
        #         if (i%2) == 0:
        #             ax.set_ylabel(row_labels[i//3])
        #             ax.plot(pos_tracking_error[i//2][10:])
        #         else:
        #             ax.plot(vel_tracking_error[(i//2)][10:])
        #         if i >=6 :
        #             ax.set_xlabel(col_labels[i-6])
        #     fig.suptitle('Swing Tracking Error', fontsize=16)
        #     # for i in range(len(pos_tracking_error)):
        #     #     print('%s %s - mean:%2.2f \t std:%.2f' % (row_labels[i//2],col_labels[i%2],(pos_tracking_error[i]).mean(),(pos_tracking_error[i]).std))
        #     # plt.show()
        #     plt.savefig("Tracking Error - Kp=10'000, Kd=-1.0, kff=1.png",dpi=600)

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


# # ---------------------------------- Optimizer --------------------------------
# class samplingOptimizer():
#     """ TODO """
#     def __init__(self):
#         """ TODO """

#         self.time_horizon = 10
#         self.num_predict_step = 3

#         self.num_samples = 1000

#         self.F_param = self.time_horizon
#         self.P_param = self.num_predict_step

#         self.dt = 0.02

#         self.device = 'cuda'


#     def optimize_latent_variable(self, env: RLTaskEnv, f:torch.Tensor, d:torch.Tensor, p_lw:torch.Tensor, F_lw:torch.Tensor, phase:torch.Tensor, c_prev:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#         """ Given latent variable f,d,F,p, returns f*,d*,F*,p*, optimized with a sampling optimization 
        
#         Args :
#             f      (Tensor): Leg frequency                of shape(batch_size, num_leg)
#             d      (Tensor): Leg duty cycle               of shape(batch_size, num_leg)
#             p_lw   (Tensor): Foot touch down position     of shape(batch_size, num_leg, 3, TODO)
#             F_lw   (Tensor): ground Reaction Forces       of shape(batch_size, num_leg, 3, TODO)
#             phase  (Tensor): Current feet phase           of shape(batch_size, num_leg)
#             c_prev (Tensor): Contact sequence determined at previous iteration of shape (batch_size, num_leg)

#         Returns :
#             f_star    (Tensor): Leg frequency                of shape(batch_size, num_leg)
#             d_star    (Tensor): Leg duty cycle               of shape(batch_size, num_leg)
#             p_star_lw (Tensor): Foot touch down position     of shape(batch_size, num_leg, 3, TODO)
#             F_star_lw (Tensor): ground Reaction Forces       of shape(batch_size, num_leg, 3, TODO)
#         """

#         # --- Step 1 : Generate the samples
#         f_samples, d_samples, p_lw_samples, F_lw_samples = self.generate_samples(f=f, d=d, p_lw=p_lw, F_lw=F_lw)

#         # --- Step 2 : Given f and d samples -> generate the contact sequence for the samples
#         c_samples, new_phase = self.gait_generator(f_samples=f_samples, d_samples=d_samples, phase=phase, time_horizon=self.time_horizon, dt=self.dt)

#         # --- Step 2 : prepare the variables : convert from torch.Tensor to Jax
#         initial_state_jax, reference_seq_jax, action_seq_samples_jax = self.prepare_variable_for_compute_rollout(env=env, c_samples=c_samples, p_lw_samples=p_lw_samples, F_lw_samples=F_lw_samples, feet_in_contact=c_prev)

#         # --- Step 3 : Compute the rollouts to find the rollout cost
#         cost_samples_jax = self.compute_rollout(initial_state_jax=initial_state_jax, reference_seq_jax=reference_seq_jax, action_seq_samples_jax=action_seq_samples_jax)
        
#         # --- Step 4 : Given the samples cost, find the best control action
#         best_action_seq_jax, best_index = self.find_best_actions(action_seq_samples_jax, cost_samples_jax)

#         # --- Step 4 : Convert the optimal value back to torch.Tensor
#         f_star, d_star, p_star_lw, F_star_lw = self.retrieve_z_from_action_seq(best_action_seq_jax)

#         return f_star, d_star, p_star_lw, F_star_lw


#     def prepare_variable_for_compute_rollout(self, env: RLTaskEnv, c_samples:torch.Tensor, p_lw_samples:torch.Tensor, F_lw_samples:torch.Tensor, feet_in_contact:torch.Tensor) -> tuple[jnp.array, jnp.array, jnp.array]:
#         """ Helper function to modify the embedded state, reference and action to be used with the 'compute_rollout' function

#         Note :
#             Initial state and reference can be retrieved only with the environment

#         Args :
#             env (RLTaskEnv): Environment manager to retrieve all necessary simulation variable
#             c_samples       (Tensor): Leg frequency                of shape(batch_size, num_leg)
#             p_lw_samples    (Tensor): Foot touch down position     of shape(batch_size, num_leg, 3, p_param)
#             F_lw_samples    (Tensor): ground Reaction Forces       of shape(batch_size, num_leg, 3, F_param)
#             feet_in_contact (Tensor): Feet in contact, determined by prevous solution of shape(batch_size, num_legs)

        
#         Return :
#             initial_state_jax      (jnp.array): Current state of the robot (CoM pos-vel, foot pos) as required by 'compute_rollout'
#             reference_seq_jax      (jnp.array): Reference state along the prediction horizon as required by 'compute_rollout'
#             action_seq_samples_jax (jnp.array): Action sequence along the prediction horizon as required by 'compute_rollout'
#         """
#         # Check that a single robot was provided
#         if env.num_envs > 1:
#             assert ValueError('More than a single environment was provided to the sampling controller')

#         # Retrieve robot from the scene : specify type to enable type hinting
#         robot: Articulation = env.scene["robot"]


#         # ----- Step 1 : Retrieve the initial state
#         # Retrieve the robot position in local world frame of shape(3)
#         com_pos_lw = (robot.data.root_pos_w - env.scene.env_origins).squeeze(0)

#         # Robot height is proprioceptive : need to compute it
#         feet_pos_lw = robot.data.body_pos_w[:, robot.find_bodies(".*foot")[0]] - env.scene.env_origins.unsqueeze(1) # shape(num_env, num_leg, 3)
#         com_pos_lw[2] = com_pos_lw[2] - (torch.sum(feet_pos_lw[2] * feet_in_contact) / torch.sum(feet_in_contact))  # shape(1) -> don't work if num_envs > 1

#         # Retrieve the robot orientation in lw as euler angle ZXY of shape(3)
#         roll, pitch, yaw = math_utils.euler_xyz_from_quat(robot.data.root_quat_w) # TODO Check the angles !
#         com_pose_lw = torch.tensor((roll, pitch, yaw))

#         # Retrieve the robot linear and angular velocity in base frame of shape(6)
#         com_vel_b = (robot.data.root_vel_b).squeeze(0)

#         # Retrieve the feet position in local world frame of shape(num_legs, 3)
#         foot_idx = robot.find_bodies(".*foot")[0]
#         p_w = robot.data.body_pos_w[:, foot_idx,:] # shape(1,4,3) - retrieve position
#         p_lw = (p_w - env.scene.env_origins.unsqueeze(1).expand(p_w.shape)).squeeze(0) #shape(4,3) - convert to lw
#         p_lw = p_lw # shape(12) TODO, reshape correctly

#         # Prepare the state (at time t)
#         initial_state = torch.cat((
#             com_pos_lw,    # Position                       TODO Should be center at the COM -> Thus (0,0,0) -> height tracking is proprioceptive -> would be done with the feet
#             com_vel_b[:3], # Linear Velocity                TODO in world frame
#             com_pose_lw,   # Orientation as euler_angle ZXY TODO in world frame
#             com_vel_b[3:], # Angular velocity               TODO in base frame (Roll, Pitch, Yaw)
#             p_lw,          # Foot position                  TODO Should be center at the COM
#         )) # of shape(TODO)


#         # ----- Step 2 : Retrieve the robot's reference along the integration horizon
#         time_horizon = 10   # TODO Get that from the right place
#         dt = 0.02           # TODO Get that from the right place

#         # The reference position is tracked only for the height
#         com_pos_ref_lw = torch.tensor((0,0,0.4), device=env.device).expand(3,time_horizon) # TODO : get the height reference from the right place + in the right frame

#         # The speed reference is tracked for x_b, y_b and yaw
#         speed_command = (env.command_manager.get_command("base_velocity")).squeeze(0) # shape(3)
#         com_vel_ref_b = torch.tensor((speed_command[0], speed_command[1], 0, 0, 0, speed_command[2]), device=env.device).expand(6, time_horizon)

#         # The pose reference is (0,0) for roll and pitch, but the yaw must be integrated along the horizon
#         com_pose_ref_lw = torch.tensor((0,0,1), device=env.device).expand(3, time_horizon)
#         com_pose_ref_lw = com_pose_ref_lw * (torch.arange(time_horizon, device=env.device) * (dt * speed_command[2]))

#         # Defining the foot position sequence is tricky.. Since we only have number of predicted step < time_horizon
#         p_ref_lw = torch.empty((3, time_horizon), device=env.device) #TODO Define this !
        
#         # Prepare the reference sequence (at time t, t+dt, etc.)
#         reference_seq = torch.cat((
#             com_pos_ref_lw,    # Position reference
#             com_vel_ref_b[:3], # Linear Velocity reference
#             com_pose_ref_lw,   # Orientation reference as euler_angle
#             com_vel_ref_b[3:], # Angular velocity reference
#             p_ref_lw,          # Foot position reference
#         )) # of shape(TODO)


#         # ----- Step 3 : Retrieve the actions 
#         action_seq_samples = torch.cat((
#             c_samples,      # Contact sequence samples          of shape(TODO)
#             p_lw_samples,   # Foot touch down position samples  of shape(TODO)
#             F_lw_samples,   # Ground Reaction Forces            of shape(TODO)
#         )) # of shape(TODO)


#         # ----- Step 4 : Convert torch tensor to jax.array
#         initial_state_jax       = torch_to_jax(initial_state)
#         reference_seq_jax       = torch_to_jax(reference_seq)
#         action_seq_samples_jax  = torch_to_jax(action_seq_samples)             

#         return initial_state_jax, reference_seq_jax, action_seq_samples_jax


#     def retrieve_z_from_action_seq(self, action_sequence) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#         """ Given an action sequence return by the sampling optimizer as a Jax array, return the latent variabl z=(f,d,p,F)
#         As torch.Tensor, usable by the model based controller
         
#         Args : 
#             action_sequence (TODO): Action sequence containing z as returned by the sampling optimizer
          
#         Returns:
#             f_star    (Tensor): Leg frequency                of shape(num_policy, num_leg)
#             d_star    (Tensor): Leg duty cycle               of shape(num_policy, num_leg)
#             p_star_lw (Tensor): Foot touch down position     of shape(num_policy, num_leg, 3, TODO)
#             F_star_lw (Tensor): ground Reaction Forces       of shape(num_policy, num_leg, 3, TODO)            
#         """

#         f_star, d_star, p_star_lw, F_star_lw = ...

#         return f_star, d_star, p_star_lw, F_star_lw


#     def find_best_actions(self, action_seq_samples, cost_samples) : 
#         """ Given action samples and associated cost, filter invalid values and retrieves the best cost and associated actions
        
#         Args : 
#             action_seq_samples (TODO): Samples of actions   of shape(TODO)
#             cost_samples       (TODO): Associated cost      of shape(TODO)
             
#         Returns :
#             best_action_seq    (TODO):  Action with the smallest cost of shape(TODO)
#             best_index          (int):
#         """

#         # Saturate the cost in case of NaN or inf
#         cost_samples = jnp.where(jnp.isnan(cost_samples), 1000000, cost_samples)
#         cost_samples = jnp.where(jnp.isinf(cost_samples), 1000000, cost_samples)
        

#         # Take the best found control parameters
#         best_index = jnp.nanargmin(cost_samples)
#         best_cost = cost_samples.take(best_index)
#         best_action_seq = action_seq_samples[best_index]

#         return best_action_seq, best_index


#     def generate_samples(self, f:torch.Tensor, d:torch.Tensor, p_lw:torch.Tensor, F_lw:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#         """ Given action (f,d,p,F), generate action sequence samples (f_samples, d_samples, p_samples, F_samples)
#         If multiple action sequence are provided (because several policies are blended together), generate samples
#         from these polices with equal proportions. TODO
        
#         Args :
#             f    (Tensor): Leg frequency                of shape(batch_size, num_leg)
#             d    (Tensor): Leg duty cycle               of shape(batch_size, num_leg)
#             p_lw (Tensor): Foot touch down position     of shape(batch_size, num_leg, 3, p_param)
#             F_lw (Tensor): ground Reaction Forces       of shape(batch_size, num_leg, 3, F_param)
            
#         Returns :
#             f_samples    (Tensor) : Leg frequency samples               of shape(num_samples, num_leg)
#             d_samples    (Tensor) : Leg duty cycle samples              of shape(num_samples, num_leg)
#             p_lw_samples (Tensor) : Foot touch down position samples    of shape(num_samples, num_leg, 3, p_param)
#             F_lw_samples (Tensor) : Ground Reaction forces samples      of shape(num_samples, 3, F_param)
#         """

#         f_samples = ...
#         d_samples = ...
#         p_lw_samples = ...
#         F_lw_samples = ...

#         return f_samples, d_samples, p_lw_samples, F_lw_samples


#     # TODO Try this !
#     def gait_generator(self, f_samples: torch.Tensor, d_samples: torch.Tensor, phase: torch.Tensor, time_horizon: int, dt) -> tuple[torch.Tensor, torch.Tensor]:
#         """ Implement a gait generator that return a contact sequence given a leg frequency and a leg duty cycle
#         Increment phase by dt*f 
#         restart if needed
#         return contact : 1 if phase < duty cyle, 0 otherwise  
#         c == 1 : Leg is in contact (stance)
#         c == 0 : Leg is in swing

#         Note:
#             No properties used, no for loop : purely functional -> made to be jitted
#             parallel_rollout : this is optional, it will work without the parallel rollout dimension

#         Args:
#             - f_samples     (Tensor): Leg frequency samples                 of shape(num_samples, num_legs)
#             - d_samples     (Tensor): Stepping duty cycle samples in [0,1]  of shape(num_samples, num_legs)
#             - phase         (Tensor): phase of leg samples in [0,1]         of shape(num_legs)
#             - time_horizon     (int): Time horizon for the contact sequence

#         Returns:
#             - c_samples     (t.bool): Foot contact sequence samples         of shape(num_samples, num_legs, time_horizon)
#             - phase_samples (Tensor): The phase samples updated by 1 dt     of shape(num_samples, num_legs)
#         """
        
#         # Increment phase of f*dt: new_phases[0] : incremented of 1 step, new_phases[1] incremented of 2 steps, etc. without a for loop.
#         # new_phases = phase + f*dt*[1,2,...,time_horizon]
#         # phase and f must be exanded from (batch_size, num_legs, parallel_rollout) to (batch_size, num_legs, parallel_rollout, time_horizon) in order to perform the operations
#         new_phases = phase.unsqueeze(-1).expand(*[-1] * len(phase.shape),time_horizon) + f_samples.unsqueeze(-1).expand(*[-1] * len(f_samples.shape),time_horizon)*torch.linspace(start=1, end=time_horizon, steps=time_horizon, device=self.device)*dt

#         # Make the phases circular (like sine) (% is modulo operation)
#         new_phases = new_phases%1

#         # Save first phase
#         new_phase = new_phases[..., 0]

#         # Make comparaison to return discret contat sequence : c = 1 if phase < d, 0 otherwise
#         c_samples = new_phases <= d_samples.unsqueeze(-1).expand(*[-1] * len(d_samples.shape), time_horizon)

#         return c_samples, new_phase


#     def compute_rollout(self, initial_state_jax: jnp.array, reference_seq_jax: jnp.array, action_seq_samples_jax: jnp.array) -> jnp.array:
#         """Calculate cost of rollouts of given action sequence samples 

#         Args :
#             initial_state_jax      (jnp.array): Inital state of the robot                   of shape(TODO)
#             reference_seq_jax      (jnp.array): reference sequence for the robot state      of shape(TODO)
#             action_seq_samples_jax (jnp.array): Action sequence to apply to the robot       of shape(TODO)            
            
#         Returns:
#             cost_samples_jax       (jnp.array): costs of the rollouts                       of shape(num_samples)   
#         """  

#         cost_samples_jax = ...
        
#         return cost_samples_jax
    
