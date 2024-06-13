from abc import ABC
from collections.abc import Sequence
import torch
from torch.distributions.constraints import real

import jax.numpy as jnp
import numpy as np

import omni.isaac.orbit.utils.math as math_utils

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    pass
from omni.isaac.orbit.assets.articulation import Articulation
from omni.isaac.orbit.envs import RLTaskEnv


import jax
import jax.dlpack
import torch
import torch.utils.dlpack

def jax_to_torch(x):#: jax.Array):
    return torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(x))
def torch_to_jax_old(x):
    return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(x))

# Hopefuly this should solved the stride problem : https://github.com/google/jax/issues/14399
def torch_to_jax(x_torch):
    shape = x_torch.shape
    x_torch_flat = torch.flatten(x_torch)
    x_jax = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(x_torch_flat))
    return x_jax.reshape(shape)

from .quadrupedpympc.sampling.centroidal_model_jax import Centroidal_Model_JAX

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

        self.samplingOptimizer = SamplingOptimizer()  
        self.c_prev = torch.ones(num_envs, num_legs, device=device)


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
    def optimize_latent_variable(self, f: torch.Tensor, d: torch.Tensor, p_lw: torch.Tensor, F_lw: torch.Tensor, env: RLTaskEnv, height_map:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Given the latent variable z=[f,d,p,F], return the optimized latent variable p*, F*, c*, pt*
        Note :
            The variable are in the 'local' world frame _wl. This notation is introduced to avoid confusion with the 'global' world frame, where all the batches coexists.

        Args:
            - f   (torch.Tensor): Prior leg frequency                   of shape (batch_size, num_legs)
            - d   (torch.Tensor): Prior stepping duty cycle             of shape (batch_size, num_legs)
            - p_lw (trch.Tensor): Prior foot touch down seq. in _lw     of shape (batch_size, num_legs, 3, number_predict_step)
            - F_lw (trch.Tensor): Prior Ground Reac. Forces (GRF) seq.  of shape (batch_size, num_legs, 3, time_horizon)
                                  In local world frame
            - height_map (Tnsor): Height map arround the robot          of shape(x, y)

        Returns:
            - p*_lw (tch.Tensor): Optimized foot touch down seq. in _lw of shape (batch_size, num_legs, 3, number_predict_step)
            - F*_lw (tch.Tensor): Opt. Gnd Reac. F. (GRF) seq.   in _lw of shape (batch_size, num_legs, 3, time_horizon)
            - c*  (torch.Tensor): Optimized foot contact sequence       of shape (batch_size, num_legs, time_horizon)
            - pt*_lw (th.Tensor): Optimized foot swing traj.     in _lw of shape (batch_size, num_legs, 9, decimation)  (9 = pos, vel, acc)
        """

        # Call the optimizer
        # F_lw = F_lw.expand(1,4,3,5)
        # p_lw = p_lw.expand(1,4,3,5)

        d2=d # to be able to call d in the debugger
        # f_star, d_star, p_star_lw, F_star_lw = self.samplingOptimizer.optimize_latent_variable(env=env, f=f, d=d, p_lw=p_lw, F_lw=F_lw, phase=self.phase, c_prev=self.c_prev, height_map=height_map)
        f_star, d_star, F_star_lw, p_star_lw = f, d, F_lw, p_lw
        # p_star_lw = p_lw
        # f_star, d_star = f, d

        # Compute the contact sequence and update the phase
        c_star, self.phase = self.gait_generator(f=f_star, d=d_star, phase=self.phase, time_horizon=self._time_horizon, dt=self._dt_out)

        # Update c_prev
        self.c_prev = c_star

        # Generate the swing trajectory
        # pt_lw = self.swing_trajectory_generator(p_lw=p_lw[:,:,:,0], c=c, d=d, f=f)
        pt_star_lw, full_pt_lw = self.full_swing_trajectory_generator(p_lw=p_star_lw[:,:,:,0], c=c_star, d=d_star, f=f_star)

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


# ---------------------------------- Optimizer --------------------------------
class SamplingOptimizer():
    """ TODO """
    def __init__(self):
        """ TODO """

        self.time_horizon = 5
        self.num_predict_step = 3

        self.num_samples = 1

        self.F_param = self.time_horizon
        self.p_param = self.time_horizon

        self.dt = 0.02

        self.device = 'cuda'
        self.device_jax = jax.devices('gpu')[0]

        self.num_legs = 4

        self.sampling_horizon = self.time_horizon

        self.state_dim = 24
        self.input_dim = 12

        self.dtype_general = 'float32'

        interpolation_F_method = 'discrete' # 'discrete', 'cubic spline'
        if interpolation_F_method=='cubic spline' : self.interpolation_F=self.compute_cubic_spline
        if interpolation_F_method=='discrete'     : self.interpolation_F=self.compute_discrete

        # Initialize the robot model
        self.robot_model = Centroidal_Model_JAX(self.dt,self.device_jax)

        # Add the 'samples' dimension on the last for input variable, and on the output
        self.parallel_compute_rollout = jax.vmap(self.compute_rollout, in_axes=(None, None, 0, 0, 0, 0), out_axes=0)
        self.jit_parallel_compute_rollout = jax.jit(self.parallel_compute_rollout, device=self.device_jax)
        # self.jit_parallel_compute_rollout = self.parallel_compute_rollout

        self.F_z_min = 0
        self.F_z_max = self.robot_model.mass*9.81
        self.mu = 0.5

        self.height_ref = 0.4

        # State weight matrix (JAX)
        self.Q = jnp.identity(self.state_dim, dtype=self.dtype_general)*0
        self.Q = self.Q.at[0,0].set(0.0)
        self.Q = self.Q.at[1,1].set(0.0)
        self.Q = self.Q.at[2,2].set(111500) #com_z
        self.Q = self.Q.at[3,3].set(5000) #com_vel_x
        self.Q = self.Q.at[4,4].set(5000) #com_vel_y
        self.Q = self.Q.at[5,5].set(200) #com_vel_z
        self.Q = self.Q.at[6,6].set(11200) #base_angle_roll
        self.Q = self.Q.at[7,7].set(11200) #base_angle_pitch
        self.Q = self.Q.at[8,8].set(0.0) #base_angle_yaw
        self.Q = self.Q.at[9,9].set(20) #base_angle_rates_x
        self.Q = self.Q.at[10,10].set(20) #base_angle_rates_y
        self.Q = self.Q.at[11,11].set(600) #base_angle_rates_z

        # Input weight matrix (JAX)
        self.R = jnp.identity(self.input_dim, dtype=self.dtype_general)
        self.R = self.R.at[0,0].set(0.0) #foot_pos_x_FL
        self.R = self.R.at[1,1].set(0.0) #foot_pos_y_FL
        self.R = self.R.at[2,2].set(0.0) #foot_pos_z_FL
        self.R = self.R.at[3,3].set(0.0) #foot_pos_x_FR
        self.R = self.R.at[4,4].set(0.0) #foot_pos_y_FR
        self.R = self.R.at[5,5].set(0.0) #foot_pos_z_FR
        self.R = self.R.at[6,6].set(0.0) #foot_pos_x_RL
        self.R = self.R.at[7,7].set(0.0) #foot_pos_y_RL
        self.R = self.R.at[8,8].set(0.0) #foot_pos_z_RL
        self.R = self.R.at[9,9].set(0.0) #foot_pos_x_RR
        self.R = self.R.at[10,10].set(0.0) #foot_pos_y_RR
        self.R = self.R.at[11,11].set(0.0) #foot_pos_z_RR

        self.R = self.R.at[12,12].set(0.1) #foot_force_x_FL
        self.R = self.R.at[13,13].set(0.1) #foot_force_y_FL
        self.R = self.R.at[14,14].set(0.001) #foot_force_z_FL
        self.R = self.R.at[15,15].set(0.1) #foot_force_x_FR
        self.R = self.R.at[16,16].set(0.1) #foot_force_y_FR
        self.R = self.R.at[17,17].set(0.001) #foot_force_z_FR
        self.R = self.R.at[18,18].set(0.1) #foot_force_x_RL
        self.R = self.R.at[19,19].set(0.1) #foot_force_y_RL
        self.R = self.R.at[20,20].set(0.001) #foot_force_z_RL
        self.R = self.R.at[21,21].set(0.1) #foot_force_x_RR
        self.R = self.R.at[22,22].set(0.1) #foot_force_y_RR
        self.R = self.R.at[23,23].set(0.001) #foot_force_z_RR


    def optimize_latent_variable(self, env: RLTaskEnv, f:torch.Tensor, d:torch.Tensor, p_lw:torch.Tensor, F_lw:torch.Tensor, phase:torch.Tensor, c_prev:torch.Tensor, height_map) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Given latent variable f,d,F,p, returns f*,d*,F*,p*, optimized with a sampling optimization 
        
        Args :
            f      (Tensor): Leg frequency                of shape(batch_size, num_leg)
            d      (Tensor): Leg duty cycle               of shape(batch_size, num_leg)
            p_lw   (Tensor): Foot touch down position     of shape(batch_size, num_leg, 3, p_param)
            F_lw   (Tensor): ground Reaction Forces       of shape(batch_size, num_leg, 3, F_param)
            phase  (Tensor): Current feet phase           of shape(batch_size, num_leg)
            c_prev (Tensor): Contact sequence determined at previous iteration of shape (batch_size, num_leg)
            height_map (Tr): Height map arround the robot of shape(x, y)

        Returns :
            f_star    (Tensor): Leg frequency                of shape(batch_size, num_leg)
            d_star    (Tensor): Leg duty cycle               of shape(batch_size, num_leg)
            p_star_lw (Tensor): Foot touch down position     of shape(batch_size, num_leg, 3, p_param)
            F_star_lw (Tensor): ground Reaction Forces       of shape(batch_size, num_leg, 3, F_param)
        """

        # --- Step 1 : Generate the samples and bound them to valid input
        f_samples, d_samples, p_lw_samples, F_lw_samples = self.generate_samples(f=f, d=d, p_lw=p_lw, F_lw=F_lw, height_map=height_map)

        # --- Step 2 : Given f and d samples -> generate the contact sequence for the samples
        c_samples, new_phase = self.gait_generator(f_samples=f_samples, d_samples=d_samples, phase=phase.squeeze(0), time_horizon=self.sampling_horizon, dt=self.dt)

        # --- Step 2 : prepare the variables : convert from torch.Tensor to Jax
        initial_state_jax, reference_seq_state_jax, reference_seq_input_samples_jax, \
        action_seq_c_samples_jax, action_p_lw_samples_jax, action_F_lw_samples_jax = self.prepare_variable_for_compute_rollout(env=env, c_samples=c_samples, p_lw_samples=p_lw_samples, F_lw_samples=F_lw_samples, feet_in_contact=c_prev[0,])

        # --- Step 3 : Compute the rollouts to find the rollout cost : can't used named argument with VMAP...
        cost_samples_jax = self.jit_parallel_compute_rollout(initial_state_jax, reference_seq_state_jax, reference_seq_input_samples_jax,
                                                             action_seq_c_samples_jax, action_p_lw_samples_jax, action_F_lw_samples_jax)
        
        # --- Step 4 : Given the samples cost, find the best control action
        best_action_seq_jax, best_index = self.find_best_actions(action_seq_c_samples_jax, cost_samples_jax)

        # --- Step 4 : Convert the optimal value back to torch.Tensor
        f_star, d_star, p_star_lw, F_star_lw = self.retrieve_z_from_action_seq(best_index, f_samples, d_samples, p_lw_samples, F_lw_samples)

        return f_star, d_star, p_star_lw, F_star_lw


    def prepare_variable_for_compute_rollout_old(self, env: RLTaskEnv, c_samples:torch.Tensor, p_lw_samples:torch.Tensor, F_lw_samples:torch.Tensor, feet_in_contact:torch.Tensor) -> tuple[jnp.array, jnp.array, jnp.array, jnp.array, jnp.array, jnp.array]:
        """ Helper function to modify the embedded state, reference and action to be used with the 'compute_rollout' function

        Note :
            Initial state and reference can be retrieved only with the environment
            _w   : World frame
            _lw  : World frame centered at the environment center -> local world frame
            _b   : Base frame 
            _h   : Horizontal frame -> Base frame position for xy, world frame for z, roll, pitch, base frame for yaw
            _bw  : Base/world frame -> Base frame position, world frame rotation

        Args :
            env (RLTaskEnv): Environment manager to retrieve all necessary simulation variable
            c_samples       (t.bool): Foot contact sequence sample                                                      of shape(num_samples, num_legs, time_horizon)
            p_lw_samples    (Tensor): Foot touch down position                                                          of shape(num_samples, num_leg, 3, p_param)
            F_lw_samples    (Tensor): ground Reaction Forces                                                            of shape(num_samples, num_leg, 3, F_param)
            feet_in_contact (Tensor): Feet in contact, determined by prevous solution                                   of shape(num_legs)

        
        Return :
            initial_state_jax               (jnp.array): Current state of the robot (CoM pos-vel, foot pos)             of shape(state_dim)
            reference_seq_state_jax         (jnp.array): Reference state sequence along the prediction horizon          of shape(time_horizon, state_dim)
            reference_seq_input_samples_jax (jnp.array): Reference GRF sequence samples along the prediction horizon    of shape(num_samples, time_horizon, input_dim)
            action_seq_c_samples_jax        (jnp.array): contact sequence samples along the prediction horizon          of shape(num_samples, time_horizon, num_legs)
            action_p_lw_samples_jax         (jnp.array): Foot touch down position parameters samples                    of shape(num_samples, num_legs, 3*p_param)
            action_F_lw_samples_jax         (jnp.array): GRF parameters samples                                         of shape(num_samples, num_legs, 3*F_param)
        """
        # Check that a single robot was provided
        if env.num_envs > 1:
            assert ValueError('More than a single environment was provided to the sampling controller')

        # Retrieve robot from the scene : specify type to enable type hinting
        robot: Articulation = env.scene["robot"]

        # Retrieve indexes
        foot_idx = robot.find_bodies(".*foot")[0]
        hip_idx  = robot.find_bodies(".*thigh")[0]

        # ----- Step 1 : Retrieve the initial state
        # Retrieve the robot position in local world frame of shape(3)
        com_pos_lw = (robot.data.root_pos_w - env.scene.env_origins).squeeze(0)
        com_pos_lw[2] = robot.data.root_pos_w[:,2] - (torch.sum(((robot.data.body_pos_w[:, foot_idx,:]).squeeze(0))[:,2] * feet_in_contact)) / (torch.sum(feet_in_contact))
        com_pos_w = (robot.data.root_pos_w).squeeze(0) # shape(3)
        com_pos_b = torch.zeros_like(com_pos_w)        # shape(3)
        com_pos_bw = com_pos_b                         # shape(3)

        # Robot height is proprioceptive : need to compute it
        # feet_pos_lw = (robot.data.body_pos_w[:, foot_idx,:]).squeeze(0) - env.scene.env_origins # ((4,3) - (1,3)) -> shape(4,3) - retrieve position
        feet_pos_w = (robot.data.body_pos_w[:, foot_idx,:]).squeeze(0) # ((4,3) - (1,3)) -> shape(4,3) - retrieve position
        height_mean_foot_w = (torch.sum(feet_pos_w[:,2] * feet_in_contact)) / (torch.sum(feet_in_contact)) #shape(1)
        com_pos_height_h = com_pos_w[2] - height_mean_foot_w # shape(1)

        # Compute the robot CoM position into the horiontal frame -> (0,0, proprioceptive height)
        com_pos_h = torch.stack((com_pos_b[0], com_pos_b[1], com_pos_height_h), dim=0)

        # Retrieve the robot orientation in lw as euler angle ZXY of shape(3)
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(robot.data.root_quat_w) # TODO Check the angles !
        com_pose_w = torch.tensor((roll, pitch, yaw), device=self.device)
        com_pose_lw = com_pose_w
        com_pose_bw = com_pose_w

        # Retrieve the robot linear and angular velocity in base frame of shape(6)
        com_vel_b = (robot.data.root_vel_b).squeeze(0)

        # Retrieve the hip position in horizontal frame (used to compute feet in hip centered frame)
        hip_position_w = robot.data.body_pos_w[:, hip_idx, :]

        # Retrieve the feet position in local world frame of shape(num_legs, 3)
        p_w = feet_pos_w #shape(4,3) - foot position in lw
        p_lw = p_w - env.scene.env_origins
        # p_hip_w_rot = p_w - hip_position_w #foot position in horzontal frame (shape(4,3)-shape(1,3) -> shape(4,3))
        # p_hip = math_utils.quat_apply_yaw(quat=robot.data.root_quat_w, vec=p_hip_w_rot) # Foot position in hip centered (horizontal) frame shape(4,3)
        # p_hip = p_hip.flatten(0,1) # shape(12) TODO, check that it's reshaped correctly
        p_lw = p_lw.flatten(0,1) # shape(12) TODO, check that it's reshaped correctly
        p_bw = p_w - com_pos_w.unsqueeze(0) # shape(4,3)-shape(1,3) -> shape(4,3)
        p_bw = p_bw.flatten(0,1) # shape(12) TODO, check that it's reshaped correctly

        # Prepare the state (at time t)
        initial_state_old = torch.cat((
            com_pos_bw,     # Position in horizontal frame (Centered at CoM thus (0,0,h)), height tracking is proprioceptive -> would be done with the feet
            com_vel_b[:3], # Linear Velocity in base frame               
            com_pose_bw,   # Orientation as euler_angle (roll, pitch, yaw) in world frame
            com_vel_b[3:], # Angular velocity as (roll, pitch, yaw)      
            p_bw,         # Foot position centered at the hip ~(0,0,-h)
        )) # of shape(24) -> 3 + 3 + 3 + 3 + (4*3)
        initial_state = torch.cat((
            com_pos_lw,     # Position in horizontal frame (Centered at CoM thus (0,0,h)), height tracking is proprioceptive -> would be done with the feet
            com_vel_b[:3], # Linear Velocity in base frame               
            com_pose_lw,   # Orientation as euler_angle (roll, pitch, yaw) in world frame
            com_vel_b[3:], # Angular velocity as (roll, pitch, yaw)      
            p_lw,         # Foot position centered at the hip ~(0,0,-h)
        )) # of shape(24) -> 3 + 3 + 3 + 3 + (4*3)


        # ----- Step 2 : Retrieve the robot's reference along the integration horizon

        # The reference position is tracked only for the height
        com_pos_ref_bw = com_pos_bw # shape(3)
        com_pos_ref_bw[2] = self.height_ref - (com_pos_w[2] - height_mean_foot_w) # shape(1)TODO transform this into a sequence !!!!

        com_pos_ref_bw = com_pos_ref_bw.unsqueeze(1).expand(3, self.sampling_horizon) # TODO Verify this !

        com_pos_ref_lw = torch.tensor((0,0,self.height_ref), device=self.device).unsqueeze(1).expand(3,self.sampling_horizon) # TODO should COM_heigt=0 and feet_h=-height_ref or com_height=height_ref  and feet_h=0 ??
        # The speed reference is tracked for x_b, y_b and yaw   # shape(6, time_horizon)
        speed_command = (env.command_manager.get_command("base_velocity")).squeeze(0) # shape(3)
        com_vel_ref_b = torch.tensor((speed_command[0], speed_command[1], 0, 0, 0, speed_command[2]), device=self.device).unsqueeze(1).expand(6, self.sampling_horizon) 

        # The pose reference is (0,0) for roll and pitch, but the yaw must be integrated along the horizon
        com_pose_ref_w = torch.zeros_like(com_pos_ref_lw) # shape(3, time_horizon)
        com_pose_ref_w[2] =  com_pose_w[2] + (torch.arange(self.sampling_horizon, device=env.device) * (self.dt * speed_command[2])) # shape(time_horizon)
        com_pose_ref_bw = com_pose_ref_w # shape(3, time_horizon)
        com_pose_ref_lw = com_pose_ref_w # shape(3, time_horizon)

        # Defining the foot position sequence is tricky.. Since we only have number of predicted step < time_horizon
        p_ref_bw = torch.zeros((4,3, self.sampling_horizon), device=env.device) # # shape(4, 3, sampling_horizon) TODO Define this !
        p_ref_bw = p_ref_bw.flatten(0,1) # shape(12, sampling_horizon) TODO, check that it's reshaped correctly
        p_ref_lw = p_ref_bw

        # Compute the gravity compensation GRF along the horizon : of shape (num_samples, num_legs, 3, time_horizon)
        number_of_leg_in_contact_samples = (torch.sum(c_samples, dim=1)).clamp(min=1) # Compute the number of leg in contact, clamp by minimum 1 to avoid division by zero. shape(num_samples, time_horizon)
        gravity_compensation_F_samples = torch.zeros((self.num_samples, self.num_legs, 3, self.sampling_horizon), device=self.device) # shape (num_samples, num_legs, 3, time_horizon)
        gravity_compensation_F_samples[:,:,2,:] = ((self.robot_model.mass * 9.81) / number_of_leg_in_contact_samples).unsqueeze(1) # shape (num_samples, 1, time_horizon)
        
        # Prepare the reference sequence (at time t, t+dt, etc.)
        reference_seq_state_old = torch.cat((
            com_pos_ref_bw,    # Position reference                                                     of shape( 3, time_horizon)
            com_vel_ref_b[:3], # Linear Velocity reference                                              of shape( 3, time_horizon)
            com_pose_ref_bw,   # Orientation reference as euler_angle                                   of shape( 3, time_horizon)
            com_vel_ref_b[3:], # Angular velocity reference                                             of shape( 3, time_horizon)
            p_ref_bw,          # Foot position reference (xy plane in horizontal plane, hip centered)   of shape(12, time_horizon)
        )).permute(1,0) # of shape(time_horizon, 24) -> 3 + 3 + 3 + 3 + (4*3)

        reference_seq_state = torch.cat((
            com_pos_ref_lw,    # Position reference                                                     of shape( 3, time_horizon)
            com_vel_ref_b[:3], # Linear Velocity reference                                              of shape( 3, time_horizon)
            com_pose_ref_lw,   # Orientation reference as euler_angle                                   of shape( 3, time_horizon)
            com_vel_ref_b[3:], # Angular velocity reference                                             of shape( 3, time_horizon)
            p_ref_lw,          # Foot position reference (xy plane in horizontal plane, hip centered)   of shape(12, time_horizon)
        )).permute(1,0) # of shape(time_horizon, 24) -> 3 + 3 + 3 + 3 + (4*3)

        reference_seq_input_samples = torch.cat((
            gravity_compensation_F_samples.flatten(1,2), #                                              of shape(num_samples, num_legs*3, time_horizon)
        )).permute(0,2,1) # of shape(num_samples, time_horizon, num_legs*3)


        # ----- Step 3 : Retrieve the actions and prepare them with the correct method

        # TODO One could prepare the action here (discrete, spline, etc.)

        action_seq_c_samples = c_samples.permute(0,2,1).int() # Contact sequence samples         of shape(num_samples, time_horizon, num_legs) (converted to int for jax conversion)
        action_p_lw_samples  = p_lw_samples.flatten(2,3)      # Foot touch down position samples of shape(num_samples, num_legs, 3*p_param)
        action_F_lw_samples  = F_lw_samples.flatten(2,3)      # Ground Reaction Forces           of shape(num_samples, num_legs, 3*F_param)



        # ----- Step 4 : Convert torch tensor to jax.array
        initial_state_jax               = torch_to_jax(initial_state)                          # of shape(state_dim)
        reference_seq_state_jax         = torch_to_jax(reference_seq_state)                    # of shape(time_horizon, state_dim)
        reference_seq_input_samples_jax = torch_to_jax(reference_seq_input_samples)            # of shape(num_samples, time_horizon, input_dim)
        action_seq_c_samples_jax        = torch_to_jax(action_seq_c_samples)                   # of shape(num_samples, time_horizon, num_legs)
        action_p_lw_samples_jax         = torch_to_jax(action_p_lw_samples)                    # of shape(num_samples, num_legs, 3*p_param)
        action_F_lw_samples_jax         = torch_to_jax(action_F_lw_samples)                    # of shape(num_samples, num_legs, 3*F_param)


        return initial_state_jax, reference_seq_state_jax, reference_seq_input_samples_jax, action_seq_c_samples_jax, action_p_lw_samples_jax, action_F_lw_samples_jax


    def prepare_variable_for_compute_rollout(self, env: RLTaskEnv, c_samples:torch.Tensor, p_lw_samples:torch.Tensor, F_lw_samples:torch.Tensor, feet_in_contact:torch.Tensor) -> tuple[jnp.array, jnp.array, jnp.array, jnp.array, jnp.array, jnp.array]:
        """ Helper function to modify the embedded state, reference and action to be used with the 'compute_rollout' function

        Note :
            Initial state and reference can be retrieved only with the environment
            _w   : World frame
            _lw  : World frame centered at the environment center -> local world frame
            _b   : Base frame 
            _h   : Horizontal frame -> Base frame position for xy, world frame for z, roll, pitch, base frame for yaw
            _bw  : Base/world frame -> Base frame position, world frame rotation

        Args :
            env (RLTaskEnv): Environment manager to retrieve all necessary simulation variable
            c_samples       (t.bool): Foot contact sequence sample                                                      of shape(num_samples, num_legs, time_horizon)
            p_lw_samples    (Tensor): Foot touch down position                                                          of shape(num_samples, num_leg, 3, p_param)
            F_lw_samples    (Tensor): ground Reaction Forces                                                            of shape(num_samples, num_leg, 3, F_param)
            feet_in_contact (Tensor): Feet in contact, determined by prevous solution                                   of shape(num_legs)

        
        Return :
            initial_state_jax               (jnp.array): Current state of the robot (CoM pos-vel, foot pos)             of shape(state_dim)
            reference_seq_state_jax         (jnp.array): Reference state sequence along the prediction horizon          of shape(time_horizon, state_dim)
            reference_seq_input_samples_jax (jnp.array): Reference GRF sequence samples along the prediction horizon    of shape(num_samples, time_horizon, input_dim)
            action_seq_c_samples_jax        (jnp.array): contact sequence samples along the prediction horizon          of shape(num_samples, time_horizon, num_legs)
            action_p_lw_samples_jax         (jnp.array): Foot touch down position parameters samples                    of shape(num_samples, num_legs, 3*p_param)
            action_F_lw_samples_jax         (jnp.array): GRF parameters samples                                         of shape(num_samples, num_legs, 3*F_param)
        """
        # Check that a single robot was provided
        if env.num_envs > 1:
            assert ValueError('More than a single environment was provided to the sampling controller')

        # Retrieve robot from the scene : specify type to enable type hinting
        robot: Articulation = env.scene["robot"]

        # Retrieve indexes
        foot_idx = robot.find_bodies(".*foot")[0]

        # ----- Step 1 : Retrieve the initial state
        # Retrieve the robot position in local world frame of shape(3)
        com_pos_lw = (robot.data.root_pos_w - env.scene.env_origins).squeeze(0) # shape(3)
        com_pos_lw[2] = robot.data.root_pos_w[:,2] - (torch.sum(((robot.data.body_pos_w[:, foot_idx,:]).squeeze(0))[:,2] * feet_in_contact)) / (torch.sum(feet_in_contact)) # height is proprioceptive

        # Retrieve the robot orientation in lw as euler angle ZXY of shape(3)
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(robot.data.root_quat_w) # TODO Check the angles !
        com_pose_w = torch.tensor((roll, pitch, yaw), device=self.device)
        com_pose_lw = com_pose_w

        # Retrieve the robot linear and angular velocity in base frame of shape(6)
        com_ang_vel_b = (robot.data.root_ang_vel_b).squeeze(0)
        com_lin_vel_w = (robot.data.root_lin_vel_w).squeeze(0)

        # Retrieve the feet position in local world frame of shape(num_legs*3)
        p_w = (robot.data.body_pos_w[:, foot_idx,:]).squeeze(0) # shape(4,3) - foot position in w
        p_lw = p_w - env.scene.env_origins                      # shape(4,3) - foot position in lw
        p_lw = p_lw.flatten(0,1)                                # shape(12)  - foot position in lw

        # Prepare the state (at time t)
        initial_state = torch.cat((
            com_pos_lw,    # CoM position in horizontal in local world frame, height is proprioceptive
            com_lin_vel_w, # Linear Velocity in world frame               
            com_pose_lw,   # Orientation as euler_angle (roll, pitch, yaw) in world frame
            com_ang_vel_b, # Angular velocity as (roll, pitch, yaw) in base frame -> would be converted to euler rate in the centroidal model     
            p_lw,          # Foot position in local world frame
        )) # of shape(24) -> 3 + 3 + 3 + 3 + (4*3)


        # ----- Step 2 : Retrieve the robot's reference along the integration horizon
        # Retrieve the speed command : (lin_vel_x_b, lin_vel_y_b, ang_vel_yaw_b)
        speed_command_b = (env.command_manager.get_command("base_velocity")).squeeze(0) # shape(3)

        # CoM reference position : tracked only for the height -> xy can be anything eg 0
        com_pos_ref_seq_lw = torch.tensor((0,0,self.height_ref), device=self.device).unsqueeze(1).expand(3,self.sampling_horizon)

        # The pose reference is (0,0) for roll and pitch, but the yaw must be integrated along the horizon (in world frame)
        com_pose_ref_w = torch.zeros_like(com_pos_ref_seq_lw) # shape(3, time_horizon)
        com_pose_ref_w[2] =  com_pose_w[2] + (torch.arange(self.sampling_horizon, device=env.device) * (self.dt * speed_command_b[2])) # shape(time_horizon)
        com_pose_ref_lw = com_pose_ref_w # shape(3, time_horizon)

        # The speed reference is tracked for x_b, y_b and yaw -> must be converted accordingly  # shape(3, time_horizon)
        com_lin_vel_ref_seq_w = torch.zeros_like(com_pos_ref_seq_lw)
        com_ang_vel_ref_seq_b = torch.zeros_like(com_pos_ref_seq_lw)

        com_lin_vel_ref_seq_w[0] = speed_command_b[0]*torch.cos(com_pose_ref_w[2]) - speed_command_b[1]*torch.sin(com_pose_ref_w[2]) # shape(t_h*t_h - t_h*t_h) -> t_h #TODO Check that the rotation is correct
        com_lin_vel_ref_seq_w[1] = speed_command_b[0]*torch.sin(com_pose_ref_w[2]) + speed_command_b[1]*torch.cos(com_pose_ref_w[2]) # shape(t_h*t_h - t_h*t_h) -> t_h

        com_ang_vel_ref_seq_b[2] = speed_command_b[2]

        # Defining the foot position sequence is tricky.. Since we only have number of predicted step < time_horizon
        p_ref_seq_lw = torch.zeros((4,3, self.sampling_horizon), device=env.device) # shape(4, 3, sampling_horizon) TODO Define this !
        p_ref_seq_lw = p_ref_seq_lw.flatten(0,1)                                    # shape(12, sampling_horizon)

        # Compute the gravity compensation GRF along the horizon : of shape (num_samples, num_legs, 3, time_horizon)
        number_of_leg_in_contact_samples = (torch.sum(c_samples, dim=1)).clamp(min=1) # Compute the number of leg in contact, clamp by minimum 1 to avoid division by zero. shape(num_samples, time_horizon)
        gravity_compensation_F_samples = torch.zeros((self.num_samples, self.num_legs, 3, self.sampling_horizon), device=self.device) # shape (num_samples, num_legs, 3, time_horizon)
        gravity_compensation_F_samples[:,:,2,:] = ((self.robot_model.mass * 9.81) / number_of_leg_in_contact_samples).unsqueeze(1) # shape (num_samples, 1, time_horizon)
        
        # Prepare the reference sequence (at time t, t+dt, etc.)
        reference_seq_state = torch.cat((
            com_pos_ref_seq_lw,    # Position reference                                                     of shape( 3, time_horizon)
            com_lin_vel_ref_seq_w, # Linear Velocity reference                                              of shape( 3, time_horizon)
            com_pose_ref_lw,       # Orientation reference as euler_angle                                   of shape( 3, time_horizon)
            com_ang_vel_ref_seq_b, # Angular velocity reference                                             of shape( 3, time_horizon)
            p_ref_seq_lw,          # Foot position reference (xy plane in horizontal plane, hip centered)   of shape(12, time_horizon)
        )).permute(1,0) # of shape(time_horizon, 24) -> 3 + 3 + 3 + 3 + (4*3)

        reference_seq_input_samples = torch.cat((
            gravity_compensation_F_samples.flatten(1,2), #                                              of shape(num_samples, num_legs*3, time_horizon)
        )).permute(0,2,1) # of shape(num_samples, time_horizon, num_legs*3)


        # ----- Step 3 : Retrieve the actions and prepare them with the correct method

        # TODO One could prepare the action here (discrete, spline, etc.)

        action_seq_c_samples = c_samples.permute(0,2,1).int() # Contact sequence samples         of shape(num_samples, time_horizon, num_legs) (converted to int for jax conversion)
        action_p_lw_samples  = p_lw_samples.flatten(2,3)      # Foot touch down position samples of shape(num_samples, num_legs, 3*p_param)
        action_F_lw_samples  = F_lw_samples.flatten(2,3)      # Ground Reaction Forces           of shape(num_samples, num_legs, 3*F_param)



        # ----- Step 4 : Convert torch tensor to jax.array
        initial_state_jax               = torch_to_jax(initial_state)                          # of shape(state_dim)
        reference_seq_state_jax         = torch_to_jax(reference_seq_state)                    # of shape(time_horizon, state_dim)
        reference_seq_input_samples_jax = torch_to_jax(reference_seq_input_samples)            # of shape(num_samples, time_horizon, input_dim)
        action_seq_c_samples_jax        = torch_to_jax(action_seq_c_samples)                   # of shape(num_samples, time_horizon, num_legs)
        action_p_lw_samples_jax         = torch_to_jax(action_p_lw_samples)                    # of shape(num_samples, num_legs, 3*p_param)
        action_F_lw_samples_jax         = torch_to_jax(action_F_lw_samples)                    # of shape(num_samples, num_legs, 3*F_param)


        return initial_state_jax, reference_seq_state_jax, reference_seq_input_samples_jax, action_seq_c_samples_jax, action_p_lw_samples_jax, action_F_lw_samples_jax


    def retrieve_z_from_action_seq(self, best_index, f_samples, d_samples, p_lw_samples, F_lw_samples) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Given the index of the best sample and the samples, return the 'optimized' latent variabl z*=(f*,d*,p*,F*)
        As torch.Tensor, usable by the model based controller
         
        Args : 
            best_index            : Index of the best sample
            f_samples    (Tensor) : Leg frequency samples               of shape(num_samples, num_leg)
            d_samples    (Tensor) : Leg duty cycle samples              of shape(num_samples, num_leg)
            p_lw_samples (Tensor) : Foot touch down position samples    of shape(num_samples, num_leg, 3, p_param)
            F_lw_samples (Tensor) : Ground Reaction forces samples      of shape(num_samples, 3, F_param
          
        Returns:
            f_star    (Tensor): Best leg frequency                      of shape(1, num_leg)
            d_star    (Tensor): Best leg duty cycle                     of shape(1, num_leg)
            p_star_lw (Tensor): Best foot touch down position           of shape(1, num_leg, 3, p_param)
            F_star_lw (Tensor): Best ground Reaction Forces             of shape(1, num_leg, 3, F_param)            
        """

        f_star = f_samples[best_index.item()].unsqueeze(0)
        d_star = d_samples[best_index.item()].unsqueeze(0)
        p_star_lw = p_lw_samples[best_index.item()].unsqueeze(0)
        F_star_lw = F_lw_samples[best_index.item()].unsqueeze(0)

        return f_star, d_star, p_star_lw, F_star_lw


    def find_best_actions(self, action_seq_samples, cost_samples) : 
        """ Given action samples and associated cost, filter invalid values and retrieves the best cost and associated actions
        
        Args : 
            action_seq_samples (TODO): Samples of actions   of shape(TODO)
            cost_samples       (TODO): Associated cost      of shape(TODO)
             
        Returns :
            best_action_seq    (TODO):  Action with the smallest cost of shape(TODO)
            best_index          (int):
        """

        # Saturate the cost in case of NaN or inf
        cost_samples = jnp.where(jnp.isnan(cost_samples), 1000000, cost_samples)
        cost_samples = jnp.where(jnp.isinf(cost_samples), 1000000, cost_samples)
        

        # Take the best found control parameters
        best_index = jnp.nanargmin(cost_samples)
        best_cost = cost_samples.take(best_index)
        best_action_seq = action_seq_samples[best_index]

        return best_action_seq, best_index


    def generate_samples(self, f:torch.Tensor, d:torch.Tensor, p_lw:torch.Tensor, F_lw:torch.Tensor, height_map:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Given action (f,d,p,F), generate action sequence samples (f_samples, d_samples, p_samples, F_samples)
        If multiple action sequence are provided (because several policies are blended together), generate samples
        from these polices with equal proportions. TODO
        
        Args :
            f    (Tensor): Leg frequency                of shape(batch_size, num_leg)
            d    (Tensor): Leg duty cycle               of shape(batch_size, num_leg)
            p_lw (Tensor): Foot touch down position     of shape(batch_size, num_leg, 3, p_param)
            F_lw (Tensor): ground Reaction Forces       of shape(batch_size, num_leg, 3, F_param)
            height_map   (torch.Tensor): Height map arround the robot        of shape(x, y)
            
        Returns :
            f_samples    (Tensor) : Leg frequency samples               of shape(num_samples, num_leg)
            d_samples    (Tensor) : Leg duty cycle samples              of shape(num_samples, num_leg)
            p_lw_samples (Tensor) : Foot touch down position samples    of shape(num_samples, num_leg, 3, p_param)
            F_lw_samples (Tensor) : Ground Reaction forces samples      of shape(num_samples, 3, F_param)
        """

        if self.num_samples == 1:
            return f, d, p_lw, F_lw

        f_samples = self.normal_sampling(num_samples=self.num_samples, mean=f[0], var=torch.tensor((0.05), device=self.device))
        d_samples = self.normal_sampling(num_samples=self.num_samples, mean=d[0], var=torch.tensor((0.02), device=self.device))
        p_lw_samples = self.normal_sampling(num_samples=self.num_samples, mean=p_lw[0], var=torch.tensor((0.01), device=self.device))
        F_lw_samples = self.normal_sampling(num_samples=self.num_samples, mean=F_lw[0], var=torch.tensor((1.0), device=self.device))

        # Clamp the input to valid range and make sure p[2] is on the ground
        f_samples, d_samples, p_lw_samples, F_lw_samples = self.enforce_valid_input(f_samples=f_samples, d_samples=d_samples, p_lw_samples=p_lw_samples, F_lw_samples=F_lw_samples, height_map=height_map)

        return f_samples, d_samples, p_lw_samples, F_lw_samples


    def normal_sampling(self, num_samples:int, mean:torch.Tensor, var:torch.Tensor|None=None, seed:int|None=None) -> torch.Tensor:
        """ Normal sampling law given mean and var -> return a samples
        
        Args :
            mean     (Tensor): Mean of normal sampling law          of shape(num_dim1, num_dim2, etc.)
            var      (Tensor): Varriance of normal sampling law     of shape(num_dim1, num_dim2, etc.)
            num_samples (int): Number of samples to generate
            seed        (int): seed to generate random numbers

        Return :
            samples  (Tensor): Samples generated with mean and var  of shape(num_sammple, num_dim1, num_dim2, etc.)
        """

        # Seed if provided
        if seed : 
            torch.manual_seed(seed)

        if var is None :
            var = torch.ones_like(mean)

        # Sample from a normal law with the provided parameters
        samples = mean + torch.sqrt(var) * torch.randn((num_samples,) + mean.shape, device=self.device)

        return samples


    def gait_generator(self, f_samples: torch.Tensor, d_samples: torch.Tensor, phase: torch.Tensor, time_horizon: int, dt) -> tuple[torch.Tensor, torch.Tensor]:
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
            - f_samples     (Tensor): Leg frequency samples                 of shape(num_samples, num_legs)
            - d_samples     (Tensor): Stepping duty cycle samples in [0,1]  of shape(num_samples, num_legs)
            - phase         (Tensor): phase of leg samples in [0,1]         of shape(num_legs)
            - time_horizon     (int): Time horizon for the contact sequence

        Returns:
            - c_samples     (t.bool): Foot contact sequence samples         of shape(num_samples, num_legs, time_horizon)
            - phase_samples (Tensor): The phase samples updated by 1 dt     of shape(num_samples, num_legs)
        """
        
        # Increment phase of f*dt: new_phases[0] : incremented of 1 step, new_phases[1] incremented of 2 steps, etc. without a for loop.
        # new_phases = phase + f*dt*[1,2,...,time_horizon]
        #            (1, num_legs, 1)                  +  (samples, legs, 1)      * (1, 1, time_horizon) -> shape(samples, legs, time_horizon)
        new_phases_samples = phase.unsqueeze(0).unsqueeze(-1) + (f_samples.unsqueeze(-1) * torch.linspace(start=1, end=time_horizon, steps=time_horizon, device=self.device).unsqueeze(0).unsqueeze(1)*dt)

        # Make the phases circular (like sine) (% is modulo operation)
        new_phases_samples = new_phases_samples%1

        # Save first phase -> shape(num_samples, num_legs)
        new_phase_samples = new_phases_samples[..., 0]

        # Make comparaison to return discret contat sequence : c = 1 if phase < d, 0 otherwise
        #(samples, legs, time_horizon) <= (samples, legs, 1) -> shape(num_samples, num_legs, time_horizon)
        c_samples = new_phases_samples <= d_samples.unsqueeze(-1)

        return c_samples, new_phase_samples


    def compute_rollout(self, initial_state_jax: jnp.array, reference_seq_state_jax: jnp.array, reference_seq_input_jax: jnp.array, action_seq_c_jax: jnp.array, action_p_lw_jax: jnp.array, action_F_lw_jax: jnp.array) -> jnp.array:
        """Calculate cost of rollouts of given action sequence samples 

        Args :
            initial_state_jax       (jnp.array): Inital state of the robot                  of shape(state_dim)
            reference_seq_state_jax (jnp.array): reference sequence for the robot state     of shape(time_horizon, state_dim)                                           
            reference_seq_input_jax (jnp.array): GRF references                             of shape(time_horizon, input_dim)  Will be augmented with 'num_sample' dimension on dim0
            action_seq_c_jax        (jnp.array): Contact sequence samples                   of shape(time_horizon, num_legs)   Will be augmented with 'num_sample' dimension on dim0
            action_p_lw_jax         (jnp.array): Foot touch down position samples           of shape(num_legs, 3*p_param)      Will be augmented with 'num_sample' dimension on dim0
            action_F_lw_jax         (jnp.array): Ground Reaction Forces                     of shape(num_legs, 3*F_param)      Will be augmented with 'num_sample' dimension on dim0      
            
        Returns:
            cost_samples_jax                (jnp.array): costs of the rollouts                      of shape(num_samples)   
        """  

        def iterate_fun(n, carry):
            # --- Step 1 : Prepare variables
            # Extract the last state and cost from the carried over variable
            cost, state = carry 

            # Embed current contact into variable for the centroidal model
            current_contact = jnp.array([
                action_seq_c_jax[n]
            ], dtype=self.dtype_general)[0] # shape(4) (returns shape (1,4) -> thus the [0])

            # --- Step 2 : Retrieve the input given the interpolation parameters
            step = n # TODO : do this properly
            horizon = self.sampling_horizon # TODO : do this properly

            # Compute the GRFs given the interpolation parameter and the point in the curve
            F_lw_x_FL, F_lw_y_FL, F_lw_z_FL = self.interpolation_F(action_F_lw_jax[0], step, horizon) # TODO Single spline for now : Implement multiple spline function
            F_lw_x_FR, F_lw_y_FR, F_lw_z_FR = self.interpolation_F(action_F_lw_jax[1], step, horizon)
            F_lw_x_RL, F_lw_y_RL, F_lw_z_RL = self.interpolation_F(action_F_lw_jax[2], step, horizon)
            F_lw_x_RR, F_lw_y_RR, F_lw_z_RR = self.interpolation_F(action_F_lw_jax[3], step, horizon)

            # Apply F_lw only if in contact
            F_lw_x_FL = F_lw_x_FL * current_contact[0]
            F_lw_y_FL = F_lw_y_FL * current_contact[0]
            F_lw_z_FL = F_lw_z_FL * current_contact[0]
            
            F_lw_x_FR = F_lw_x_FR * current_contact[1]
            F_lw_y_FR = F_lw_y_FR * current_contact[1]
            F_lw_z_FR = F_lw_z_FR * current_contact[1]

            F_lw_x_RL = F_lw_x_RL * current_contact[2]
            F_lw_y_RL = F_lw_y_RL * current_contact[2]
            F_lw_z_RL = F_lw_z_RL * current_contact[2]

            F_lw_x_RR = F_lw_x_RR * current_contact[3]
            F_lw_y_RR = F_lw_y_RR * current_contact[3]
            F_lw_z_RR = F_lw_z_RR * current_contact[3]

            # Enforce force constraints
            F_lw_x_FL, F_lw_y_FL, F_lw_z_FL, \
            F_lw_x_FR, F_lw_y_FR, F_lw_z_FR, \
            F_lw_x_RL, F_lw_y_RL, F_lw_z_RL, \
            F_lw_x_RR, F_lw_y_RR, F_lw_z_RR = self.enforce_force_constraints(F_lw_x_FL, F_lw_y_FL, F_lw_z_FL,
                                                                    F_lw_x_FR, F_lw_y_FR, F_lw_z_FR,
                                                                    F_lw_x_RL, F_lw_y_RL, F_lw_z_RL,
                                                                    F_lw_x_RR, F_lw_y_RR, F_lw_z_RR)

            # Embed input into variable for the centroidal model
            input = jnp.array([
                jnp.float32(0), jnp.float32(0), jnp.float32(0), # action_p_lw_jax TODO : implement this 
                jnp.float32(0), jnp.float32(0), jnp.float32(0),
                jnp.float32(0), jnp.float32(0), jnp.float32(0),
                jnp.float32(0), jnp.float32(0), jnp.float32(0),
                F_lw_x_FL, F_lw_y_FL, F_lw_z_FL, # foot position fl
                F_lw_x_FR, F_lw_y_FR, F_lw_z_FR, # foot position fr
                F_lw_x_RL, F_lw_y_RL, F_lw_z_RL, # foot position rl
                F_lw_x_RR, F_lw_y_RR, F_lw_z_RR, # foot position rr
            ], dtype=self.dtype_general)


            # --- Step 2 : Integrate the dynamics with the centroidal model
            state_next = self.robot_model.integrate_jax(state, input, current_contact)


            # --- Step 3 : Compute the cost

            # Compute the state cost
            state_error = state_next - reference_seq_state_jax[n]
            state_cost  = state_error.T @ self.Q @ state_error

            # Compute the input cost
            input_error = input[12:] - reference_seq_input_jax[n] 
            input_cost  = input_error.T @ self.R @ input_error

            step_cost = state_cost + input_cost

            return (cost + step_cost, state_next)

        # Prepare the inital variable
        initial_cost  = jnp.float32(0.0)
        carry = (initial_cost, initial_state_jax)

        # Iterate the model over the time horizon and retrieve the cost
        cost, state = jax.lax.fori_loop(0, self.sampling_horizon, iterate_fun, carry)

        cost_samples_jax = cost

        return cost_samples_jax


    def compute_cubic_spline(self, parameters, step, horizon):
        """ Given a set of spline parameters, and the point in the trajectory return the function value 
        
        Args :
            parameters (jnp.array): of shape(TODO)
            step             (int): The point in the curve in [0, horizon]
            horizon          (int): The length of the curve
            
        Returns : 
        
        """

        # Find the point in the curve q in [0,1]
        tau = step/(horizon)        
        q = (tau - 0.0)/(1.0-0.0)
        
        # Compute the spline interpolation parameters
        a = 2*q*q*q - 3*q*q + 1
        b = (q*q*q - 2*q*q + q)*0.5
        c = -2*q*q*q + 3*q*q
        d = (q*q*q - q*q)*0.5

        # Compute the phi parameters
        phi_x = (1./2.)*(((parameters[2] - parameters[1])/0.5) + ((parameters[1] - parameters[0])/0.5))
        phi_next_x = (1./2.)*(((parameters[3] - parameters[2])/0.5) + ((parameters[2] - parameters[1])/0.5))

        phi_y = (1./2.)*(((parameters[6] - parameters[5])/0.5) + ((parameters[5] - parameters[4])/0.5))
        phi_next_y = (1./2.)*(((parameters[7] - parameters[6])/0.5) + ((parameters[6] - parameters[5])/0.5))

        phi_z = (1./2.)*(((parameters[10] - parameters[9])/0.5) + ((parameters[9] - parameters[8])/0.5))
        phi_next_z = (1./2.)*(((parameters[11] - parameters[10])/0.5) + ((parameters[10] - parameters[9])/0.5))

        # Compute the function value f(x)
        f_x = a*parameters[1] + b*phi_x + c*parameters[2]  + d*phi_next_x
        f_y = a*parameters[5] + b*phi_y + c*parameters[6]  + d*phi_next_y
        f_z = a*parameters[9] + b*phi_z + c*parameters[10] + d*phi_next_z
       
        return f_x, f_y, f_z  


    def compute_discrete(self, parameters, step, horizon):
        """ If actions are discrete actions, no interpolation are required.
        This function simply return the action

        Args :
            parameters (jnp.array): The action of shape(3*time_horizon)
            step             (int): The current step index along horizon
            horizon          (int): Not used : here for compatibility

        Returns :
            parameters (jnp.array): The action of shape(TODO)
        """

        # return parameters[step:step+3]
        return parameters[:3]


    def enforce_force_constraints(self, F_x_FL, F_y_FL, F_z_FL,
                                        F_x_FR, F_y_FR, F_z_FR,
                                        F_x_RL, F_y_RL, F_z_RL,
                                        F_x_RR, F_y_RR, F_z_RR):
       
        # Enforce push-only of the ground!

        F_z_FL = jnp.where(F_z_FL > self.F_z_min, F_z_FL, self.F_z_min)
        F_z_FR = jnp.where(F_z_FR > self.F_z_min, F_z_FR, self.F_z_min)
        F_z_RL = jnp.where(F_z_RL > self.F_z_min, F_z_RL, self.F_z_min)
        F_z_RR = jnp.where(F_z_RR > self.F_z_min, F_z_RR, self.F_z_min)

        # Enforce maximum force per leg!
        F_z_FL = jnp.where(F_z_FL<self.F_z_max, F_z_FL, self.F_z_max)
        F_z_FR = jnp.where(F_z_FR<self.F_z_max, F_z_FR, self.F_z_max)
        F_z_RL = jnp.where(F_z_RL<self.F_z_max, F_z_RL, self.F_z_max)
        F_z_RR = jnp.where(F_z_RR<self.F_z_max, F_z_RR, self.F_z_max)
        


        # Enforce friction cone
        #( F_{\text{min}} \leq F_z \leq F_{\text{max}} )
        # ( -\mu F_{\text{z}} \leq F_x \leq \mu F_{\text{z}} )
        # ( -\mu F_{\text{z}} \leq F_y \leq \mu F_{\text{z}} )
        
        # TODO REDO this ! This is wrong ! This is not the friction cone but the friction pyramide 
        F_x_FL = jnp.where(F_x_FL > -self.mu*F_z_FL, F_x_FL, -self.mu*F_z_FL)
        F_x_FL = jnp.where(F_x_FL <  self.mu*F_z_FL, F_x_FL,  self.mu*F_z_FL)
        F_y_FL = jnp.where(F_y_FL > -self.mu*F_z_FL, F_y_FL, -self.mu*F_z_FL)
        F_y_FL = jnp.where(F_y_FL <  self.mu*F_z_FL, F_y_FL,  self.mu*F_z_FL)

        F_x_FR = jnp.where(F_x_FR > -self.mu*F_z_FR, F_x_FR, -self.mu*F_z_FR)
        F_x_FR = jnp.where(F_x_FR <  self.mu*F_z_FR, F_x_FR,  self.mu*F_z_FR)
        F_y_FR = jnp.where(F_y_FR > -self.mu*F_z_FR, F_y_FR, -self.mu*F_z_FR)
        F_y_FR = jnp.where(F_y_FR <  self.mu*F_z_FR, F_y_FR,  self.mu*F_z_FR)

        F_x_RL = jnp.where(F_x_RL > -self.mu*F_z_RL, F_x_RL, -self.mu*F_z_RL)
        F_x_RL = jnp.where(F_x_RL <  self.mu*F_z_RL, F_x_RL,  self.mu*F_z_RL)
        F_y_RL = jnp.where(F_y_RL > -self.mu*F_z_RL, F_y_RL, -self.mu*F_z_RL)
        F_y_RL = jnp.where(F_y_RL <  self.mu*F_z_RL, F_y_RL,  self.mu*F_z_RL)

        F_x_RR = jnp.where(F_x_RR > -self.mu*F_z_RR, F_x_RR, -self.mu*F_z_RR)
        F_x_RR = jnp.where(F_x_RR <  self.mu*F_z_RR, F_x_RR,  self.mu*F_z_RR)
        F_y_RR = jnp.where(F_y_RR > -self.mu*F_z_RR, F_y_RR, -self.mu*F_z_RR)
        F_y_RR = jnp.where(F_y_RR <  self.mu*F_z_RR, F_y_RR,  self.mu*F_z_RR)

        
        return  F_x_FL, F_y_FL, F_z_FL, \
                F_x_FR, F_y_FR, F_z_FR, \
                F_x_RL, F_y_RL, F_z_RL, \
                F_x_RR, F_y_RR, F_z_RR


    def enforce_force_constraints_new(self, F_lw: jnp.array, c: jnp.array) -> jnp.array:
        """ Given raw GRFs in local world frame and the contact sequence, return the GRF clamped by the friction cone
        and set to zero if not in contact
        
        Args :
            F_lw (jnp.array): Ground Reaction forces samples in lw frame     of shape(num_legs*3)
            c    (jnp.array): contact sequence samples                       of shape(num_legs)
            
        Return
            F_lw (jnp.array): Clamped ground reaction forces                 of shape(num_legs*3)"""

        # --- Step 1 : Enforce the friction cone constraints
        # Compute the maximum Force in the xz plane
        F_xy_lw_max = self.mu * F_z

        # Compute the actual force in the xy plane
        F_xy_lw = norm(F_lw[:2])

        # Compute the angle in the xy plane of the Force
        alpha = acos(F_lw[0]/F_xy_lw)

        # Apply the constraint in the xy plane
        F_xy_lw_clamped = min(F_xy_lw, F_xy_lw_max)

        # Project these clamped forces in the xy plane back as x,y component
        F_x_lw_clamped = F_xy_lw_clamped*cos(alpha)
        F_y_lw_clamped = F_xy_lw_clamped*sin(alpha)

        # Finally reconstruct the vector
        F_lw_clamped = (F_x_lw_clamped, F_y_lw_clamped, F_lw[2])


        # --- Step 2 : Set force to zero for feet not in contact
        F_lw_constrained = F_y_lw_clamped * c 

        return F_lw_constrained
        

    def enforce_valid_input(self, f_samples: torch.Tensor, d_samples: torch.Tensor, p_lw_samples: torch.Tensor, F_lw_samples: torch.Tensor, height_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Enforce the input f, d, p_lw, F_lw to valid ranges. Ie. clip
            - f to [0,3] [Hz]
            - d to [0,1]
            /!\ NOT CLAMPED /!\ - p_lw to (p_x=[-0.24,+0.36], p_y=[-0.20,+0.20],[]) because not in the right frame
            - F_lw -> F_lw_z to [0, +inf]
        Moreover, ensure, p_z on the ground

        Args
            f_samples    (torch.Tensor): Leg frequency samples               of shape(num_samples, num_leg)
            d_samples    (torch.Tensor): Leg duty cycle samples              of shape(num_samples, num_leg)
            p_lw_samples (torch.Tensor): Foot touch down position samples    of shape(num_samples, num_leg, 3, p_param)
            F_lw_samples (torch.Tensor): Ground Reaction forces samples      of shape(num_samples, 3, F_param)
            height_map   (torch.Tensor): Height map arround the robot        of shape(x, y)

        Return
            f_samples    (torch.Tensor): Clipped Leg frequency samples               of shape(num_samples, num_leg)
            d_samples    (torch.Tensor): Clipped Leg duty cycle samples              of shape(num_samples, num_leg)
            p_lw_samples (torch.Tensor): Clipped Foot touch down position samples    of shape(num_samples, num_leg, 3, p_param)
            F_lw_samples (torch.Tensor): Clipped Ground Reaction forces samples      of shape(num_samples, 3, F_param)
        """
        
        # Clip f
        f_samples = f_samples.clamp(min=0, max=3)
        # f_samples = torch.zeros_like(f_samples)

        # Clip d
        d_samples = d_samples.clamp(min=0, max=1)
        # d_samples = torch.ones_like(d_samples)

        # Clip p
        # p_lw_samples[:,0] = p_lw_samples[:,0].clamp(min=-0.24,max=+0.36)
        # p_lw_samples[:,1] = p_lw_samples[:,1].clamp(min=-0.20,max=+0.20)

        # Clip F
        F_lw_samples = F_lw_samples.clamp(min=0)

        # Ensure p on the ground TODO Implement
        p_lw_samples[:,:,2,:] = -0.4*torch.ones_like(p_lw_samples[:,:,2,:])

        return f_samples, d_samples, p_lw_samples, F_lw_samples


if __name__ == "__main__":
    print('alo')