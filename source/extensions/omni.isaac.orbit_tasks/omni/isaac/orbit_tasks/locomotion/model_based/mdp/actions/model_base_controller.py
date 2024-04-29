from abc import ABC
from collections.abc import Sequence
import torch


class modelBaseController(ABC):
    """
    Abstract controller class for model base control implementation
    
    Properties : 
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

    def __init__(self):
        super().__init__()


    def late_init(self, device, num_envs, num_legs, time_horizon, dt_out, decimation, dt_in):
        """ Initialise Model Base variable after the model base action class has been initialised

        Args : 
            - device            : Cpu or GPu
            - num_envs     (int): Number of parallel environments
            - time_horiton (int): Prediction time horizon for the Model Base controller (runs at outer loop frequecy)
            - dt_out       (int): Outer loop delta t (decimation * dt_in)
            - decimation   (int): Inner Loop steps per outer loop steps
            - dt_in        (int): Inner loop delta t
        """
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
    

    def gait_generator(self, f: torch.Tensor, d: torch.Tensor, phase: torch.tensor, time_horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
        """ Implement a gait generator that return a contact sequence given a leg frequency and a leg duty cycle
        Increment phase by dt*f 
        restart if needed
        return contact : 1 if phase < duty cyle, 0 otherwise 

        Args:
            - f   (torch.Tensor): Leg frequency                         of shape(batch_size, num_legs, parallel_rollout)
            - d   (torch.Tensor): Stepping duty cycle                   of shape(batch_size, num_legs, parallel_rollout)
            - phase (tch.Tensor): phase of leg                          of shape(batch_size, num_legs, parallel_rollout)
            - time_horizon (int): Time horizon for the contact sequence

        Returns:
            - c     (torch.bool): Foot contact sequence                 of shape(batch_size, num_legs, parallel_rollout, time_horizon)
            - phase (tch.Tensor): The phase updated by one time steps   of shape(batch_size, num_legs, parallel_rollout)
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

    # Late init
    phase : torch.Tensor
    p0_lw : torch.Tensor
    swing_time : torch.Tensor
    p_lw_sim_prev : torch.Tensor

    def __init__(self, swing_ctrl_pos_gain_fb = 5000, swing_ctrl_vel_gain_fb=100):
        super().__init__()

        self.swing_ctrl_pos_gain_fb = swing_ctrl_pos_gain_fb
        self.swing_ctrl_vel_gain_fb = swing_ctrl_vel_gain_fb


    def late_init(self, device, num_envs, num_legs, time_horizon, dt_out, decimation, dt_in, p_default_lw: torch.Tensor):
        """ Initialise Model Base variable after the model base action class has been initialised
        Note :
            The variable are in the 'local' world frame _wl. This notation is introduced to avoid confusion with the 'global' world frame, where all the batches coexists.

        Args : 
            - device            : Cpu or GPu
            - num_envs     (int): Number of parallel environments
            - time_horiton (int): Prediction time horizon for the Model Base controller (runs at outer loop frequecy)
            - dt_out       (int): Outer loop delta t (decimation * dt_in)
            - decimation   (int): Inner Loop steps per outer loop steps
            - dt_in        (int): Inner loop delta t
            - p_default (Tensor): Default feet pos of robot when reset  of Shape (batch_size, num_legs, 3)
        """
        super().late_init(device, num_envs, num_legs, time_horizon, dt_out, decimation, dt_in)
        self.phase = torch.zeros(num_envs, num_legs, device=device)
        self.phase[:,(0,3)] = 0.5 # Init phase [0.5, 0, 0.5, 0]
        self.p0_lw = p_default_lw.clone().detach()
        self.swing_time = torch.zeros(num_envs, num_legs, device=device)
        self.p_lw_sim_prev = p_default_lw.clone().detach()


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
    def optimize_latent_variable(self, f: torch.Tensor, d: torch.Tensor, p_lw: torch.Tensor, F_lw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        # Compute the contact sequence and update the phase
        c, self.phase = self.gait_generator(f=f, d=d, phase=self.phase, time_horizon=self._time_horizon, dt=self._dt_out)

        # Generate the swing trajectory
        pt_lw = self.swing_trajectory_generator(p_lw=p_lw[:,:,:,0], c=c, d=d, f=f)

        p_star_lw = p_lw
        F_star_lw = F_lw
        c_star = c
        pt_star_lw = pt_lw

        return p_star_lw, F_star_lw, c_star, pt_star_lw
    

    def gait_generator(self, f: torch.Tensor, d: torch.Tensor, phase: torch.tensor, time_horizon: int, dt) -> tuple[torch.Tensor, torch.Tensor]:
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


    def swing_trajectory_generator(self, p_lw: torch.Tensor, c: torch.Tensor, f: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
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

        # Step 0. Define and Compute usefull variables

        # Heuristic TODO Save that on the right place, could also be a RL variable
        step_height = 0.05

        # True if foot in contact, False in in swing, shape (batch_size, num_legs, 1)
        in_contact = (c[:,:,0]==1).unsqueeze(-1)


        # Step 1. Compute the time trajectory (utimately the trajectory is in phase space [0,1])

        # Swing Period : Time during wich the leg is in swing.(add small numerical value to denominator to avoid division by 0)
        # Shape (batch_size, num_legs)
        swing_period = ((1-d) / (f+1e-10))                                                                  # [s]  Time during wich the leg is in swing
        half_swing_period = swing_period / 2                                                                # [s]  Time during wich the leg is in one of the two spline = swing period/2
        double_swing_frequency = 1 / ((swing_period+1e-10) / 2) #bezier_time_factor                         # [Hz] Frequency of the spline = 2*swing freq

        # Swing time : time since the leg is in swing [s]. 
        # Increment by dt_out if in swing, Reset if in contact (at -dt_out, so first time in swing time=0) 
        # (batch_size, num_legs)  # (squeeze in_contact : (batch_size, num_legs, 1)->(batch_size, num_legs))
        self.swing_time = (self.swing_time + self._dt_out) * (~in_contact.squeeze(-1)) - (self._dt_out*in_contact.squeeze(-1))  # [s] time since the leg is in swing
        half_swing_time = self.swing_time % (half_swing_period + 1e-10)  # add small numerical value to avoid nan when % 0      # [s] time since the leg is in one of the two spline

        # Generate the time trajectory t -> [t, t + dt, t+ 2*dt,...] : ie. increment the swing time by the inner loop time freq for the given number of inner steps
        # shape (batch_size, num_legs, decimation)
        time_traj = (half_swing_time.unsqueeze(-1)) + (torch.arange(self._decimation, device=self._device)*self._dt_in).unsqueeze(0).unsqueeze(0)   # [s] futur time at which the leg would be in swing

        # Convert the time trajectory (in [0, half_sing_period]) to a phase trajectory (in [0,1]),   shape (batch_size, num_legs, decimation)
        phase_traj = double_swing_frequency.unsqueeze(-1) * time_traj # (batch, legs)[Hz] * (batch, legs, decimation)[s] -> (batch, legs, decimation) in [0,1]
        

        # Step 2. Retrieve the three interpolation points : p0, p1, p2 (lift-off, middle point, touch down)

        # Retrieve p0 : update p0 with latest foot position when in contact, don't update when in swing
        # p0 shape (batch_size, num_legs, 3)
        self.p0_lw = (self.p_lw_sim_prev * in_contact) + (self.p0_lw * (~in_contact))

        # Retrieve p2 : this is simply the foot touch down prior given as input
        # p2 shape (batch_size, num_legs, 3) 
        p2_lw = p_lw 

        # Retrieve p1 : (x,y) position are define as the middle point between p0 and p1 (lift-off and touch-down). z is heuristcally define
        # p1 shape (batch_size, num_legs, 3)
        # TODO Not only choose height as step heigh but use +the terrain height or +the feet height at touch down
        p1_lw = (self.p0_lw[:,:,:2] + p2_lw[:,:,:2]) / 2     # p1(x,y) is in the middle of p0 and p2
        p1_lw = torch.cat((p1_lw, step_height*torch.ones_like(p1_lw[:,:,:1])), dim=2) # Append a third dimension z : defined as step_height


        # Step 3. Compute the parameters for the interpolation (control points)

        # Compute the a,b,c,d polynimial coefficient for the cubic interpolation S(t) = a*t^3 + b*t^2 + c*t + d
        # If swing_time < swing period/2 -> S_0(t) (ie. first interpolation), otherwise -> S_1(t - delta_t/2) (ie. second interpolation)
        # cp_x shape (batch_size, num_legs, 3)
        is_S0 = (self.swing_time <=  half_swing_period).unsqueeze(-1).expand(*[-1] * len(self.swing_time.shape), 3)  # shape (batch_size, num_legs, 3)
        cp1 = (self.p0_lw * is_S0)                                            + (p1_lw * ~is_S0)
        cp2 = (self.p0_lw * is_S0)                                            + (torch.cat((p2_lw[:,:,:2], p1_lw[:,:,2:]), dim=2)* ~is_S0)
        cp3 = (torch.cat((self.p0_lw[:,:,:2], p1_lw[:,:,2:]), dim=2) * is_S0) + (p2_lw * ~is_S0)
        cp4 = (p1_lw * is_S0)                                                 + (p2_lw * ~is_S0)

        # Step 4. Prepare parameters to compute interpolation trajectory in one operation -> matrix multiplication

        # Prepare cp_x to be mutltiplied by the time traj :  shape(batch_size, num_leg, 3) -> (batch_size, num_leg, 3, 1)
        cp1 = cp1.unsqueeze(-1)
        cp2 = cp2.unsqueeze(-1)
        cp3 = cp3.unsqueeze(-1)
        cp4 = cp4.unsqueeze(-1)

        # Prepare time traj to be multplied by cp_x : shape(batch_size, num_leg, decimation) -> (batch_size, num_leg, 1, decimation)
        phase_traj = phase_traj.unsqueeze(2)


        # Step 5. Compute the interpolation trajectory
        # shape (batch_size, num_legs, 3, decimation)
        desired_foot_pos_traj_lw = cp1*(1 - phase_traj)**3 + 3*cp2*(phase_traj)*(1 - phase_traj)**2 + 3*cp3*((phase_traj)**2)*(1 - phase_traj) + cp4*(phase_traj)**3
        desired_foot_vel_traj_lw = 3*(cp2 - cp1)*(1 - phase_traj)**2 + 6*(cp3 - cp2)*(1 - phase_traj)*(phase_traj) + 3*(cp4 - cp3)*(phase_traj)**2
        desired_foot_acc_traj_lw = 6*(1 - phase_traj) * (cp3 - 2*cp2 + cp1) + 6 * (phase_traj) * (cp4 - 2*cp3 + cp2)

        # shape (batch_size, num_legs, 9, decimation) (9 = xyz_pos, xzy_vel, xyz_acc)
        pt_lw = torch.cat((desired_foot_pos_traj_lw, desired_foot_vel_traj_lw, desired_foot_acc_traj_lw), dim=2)

        return pt_lw
     

    def swing_trajectory_generator_legacy(self, p_lw: torch.Tensor, c: torch.Tensor, f: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
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

        # Step 0. Define and Compute usefull variables

        # Heuristic TODO Save that on the right place, could also be a RL variable
        step_height = 0.05

        # Time during wich the leg is in swing.(add small numerical value to denominator to avoid division by 0)
        # Shape (batch_size, num_legs)
        swing_period = ((1-d) / (f+1e-10))
        half_swing_period = swing_period / 2
        time_fac = 1 / ((swing_period+1e-10) / 2) #bezier_time_factor


        # Step 1. Retrieve the three interpolation points : p0, p1, p2 (lift-off, middle point, touch down)

        # Retrieve p0 : update p0 with latest foot position when in contact, don't update when in swing
        # p0 shape (batch_size, num_legs, 3)
        in_contact = (c[:,:,0]==1).unsqueeze(-1) # shape (batch_size, num_legs, 1)
        self.p0_lw = (self.p_lw_sim_prev * in_contact) + (self.p0_lw * (~in_contact))

        # Retrieve p2 : this is simply the foot touch down prior given as input
        # p2 shape (batch_size, num_legs, 3) 
        p2_lw = p_lw 

        # Retrieve p1 : (x,y) position are define as the middle point between p0 and p1 (lift-off and touch-down). z is heuristcally define
        # p1 shape (batch_size, num_legs, 3)
        # TODO Not only choose height as step heigh but use +the terrain height or +the feet height at touch down
        p1_lw = (self.p0_lw[:,:,:2] + p2_lw[:,:,:2]) / 2     # p1(x,y) is in the middle of p0 and p2
        p1_lw = torch.cat((p1_lw, step_height*torch.ones_like(p1_lw[:,:,:1])), dim=2) # Append a third dimension z : defined as step_height


        # Step 2. Compute the parameters for the interpolation

        # Swing time : reset if in contact or increment by one time step (outer loop)  (squeeze in_contact : (batch_size, num_legs, 1)->(batch_size, num_legs)) # TODO should we (reset then increment), or (reset or increment)
        # then compute t in [0, Delta_t/2], which would be use for the spline interpolation
        # t & swing_time shape (batch_size, num_legs)
        self.swing_time = (self.swing_time + self._dt_out) * (~in_contact.squeeze(-1))
        t = self.swing_time % (half_swing_period + 1e-10)  # Swing time (half) : add small numerical value to avoid nan when % 0

        # Compute the a,b,c,d polynimial coefficient for the cubic interpolation S(t) = a*t^3 + b*t^2 + c*t + d
        # If swing_time < swing period/2 -> S_0(t) (ie. first interpolation), otherwise -> S_1(t - delta_t/2) (ie. second interpolation)
        # cp_x shape (batch_size, num_legs, 3)
        is_S0 = (self.swing_time <=  half_swing_period).unsqueeze(-1).expand(*[-1] * len(self.swing_time.shape), 3)  # shape (batch_size, num_legs, 3)
        cp1 = (self.p0_lw * is_S0)                                            + (p1_lw * ~is_S0)
        cp2 = (self.p0_lw * is_S0)                                            + (torch.cat((p2_lw[:,:,:2], p1_lw[:,:,2:]), dim=2)* ~is_S0)
        cp3 = (torch.cat((self.p0_lw[:,:,:2], p1_lw[:,:,2:]), dim=2) * is_S0) + (p2_lw * ~is_S0)
        cp4 = (p1_lw * is_S0)                                                 + (p2_lw * ~is_S0)

        # Step 3. Prepare parameters to compute interpolation trajectory in one operation -> matrix multiplication
        
        # Generate the time trajectory t -> [t, t + dt, t+ 2*dt,...]
        # time_fac, t : shape(batch_size, num_legs) -> unsqueezed(-1) -> Shape (batch_size, num_legs, 1)
        # (arrange = [0,1,2,...])*dt.unsqueeze(0).unsqueeze(-1)       -> Shape (1, 1, decimation)
        # time traj : Shape (batch_size, num_legs, decimation)
        t_traj = (time_fac*t).unsqueeze(-1) + (torch.arange(self._decimation, device=self._device)*self._dt_in).unsqueeze(0).unsqueeze(0)

        # Prepare cp_x to be mutltiplied by the time traj :  shape(batch_size, num_leg, 3) -> (batch_size, num_leg, 3, 1)
        cp1 = cp1.unsqueeze(-1)
        cp2 = cp2.unsqueeze(-1)
        cp3 = cp3.unsqueeze(-1)
        cp4 = cp4.unsqueeze(-1)

        # Prepare time traj to be multplied by cp_x : shape(batch_size, num_leg, decimation) -> (batch_size, num_leg, 1, decimation)
        t_traj = t_traj.unsqueeze(2)


        # Step 4. Compute the interpolation trajectory
        # shape (batch_size, num_legs, 3, decimation)
        desired_foot_pos_traj_lw = cp1*(1 - t_traj)**3 + 3*cp2*(t_traj)*(1 - t_traj)**2 + 3*cp3*((t_traj)**2)*(1 - t_traj) + cp4*(t_traj)**3
        desired_foot_vel_traj_lw = 3*(cp2 - cp1)*(1 - t_traj)**2 + 6*(cp3 - cp2)*(1 - t_traj)*(t_traj) + 3*(cp4 - cp3)*(t_traj)**2
        desired_foot_acc_traj_lw = 6*(1 - t_traj) * (cp3 - 2*cp2 + cp1) + 6 * (t_traj) * (cp4 - 2*cp3 + cp2)

        # shape (batch_size, num_legs, 9, decimation) (9 = xyz_pos, xzy_vel, xyz_acc)
        pt_lw = torch.cat((desired_foot_pos_traj_lw, desired_foot_vel_traj_lw, desired_foot_acc_traj_lw), dim=2)

        return pt_lw


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

        # Get the swing torque from the swing controller : swing torque has already been filered by C0* (ie. T_swing = T * ~c0*)
        # T_swing Shape (batch_size, num_legs, num_joints_per_leg)
        T_swing = self.swing_leg_controller(c0_star=c0_star, pt_i_star_lw=pt_i_star_lw, p_lw=p_lw, p_dot_lw=p_dot_lw, q_dot=q_dot, jacobian_lw=jacobian_lw, jacobian_dot_lw=jacobian_dot_lw, mass_matrix=mass_matrix, h=h)

        # Get the stance torque from the stance controller : stance toeque has already been filtered by c0* (ie. T_stance = T * c0*)
        # T_stance Shape (batch_size, num_legs, num_joints_per_leg)
        T_stance = self.stance_leg_controller(F0_star_lw=F0_star_lw, c0_star=c0_star, jacobian_lw=jacobian_lw)

        # T shape = (batch_size, num_legs, num_joints_per_leg)
        T = T_swing + T_stance 

        # Save variables
        self.p_lw_sim_prev = p_lw # Used in genereate trajectory

        return T


    def swing_leg_controller(self, c0_star: torch.Tensor, pt_i_star_lw: torch.Tensor, p_lw:torch.Tensor, p_dot_lw:torch.Tensor, q_dot: torch.Tensor,
                             jacobian_lw: torch.Tensor, jacobian_dot_lw: torch.Tensor, mass_matrix: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """ Given feet contact, and desired feet trajectory : compute joint torque with feedback linearization control
        T = M(q)*J⁻¹[p_dot_dot - J(q)*q_dot] + C(q,q_dot) + G(q)
        Note :
            The variable are in the 'local' world frame _wl. This notation is introduced to avoid confusion with the 'global' world frame, where all the batches coexists.

        Args:
            - c0*   (torch.bool): Optimized foot contact sequence           of shape(batch_size, num_legs)
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
        # Shape is (batch_size, num_legs, num_joints_per_leg, num_joints_per_leg)
        M_J_inv_p_dot_dot_min_J_dot_x_q_dot = torch.matmul(mass_matrix, J_inv_p_dot_dot_min_J_dot_x_q_dot.unsqueeze(-1)).squeeze(-1)

        # Final step        : # Shape is (batch_size, num_legs, num_joints_per_leg, num_joints_per_leg)
        T = torch.add(M_J_inv_p_dot_dot_min_J_dot_x_q_dot, h)

        # Keep torques only for leg in swing (~ operator inverse c0*)
        # C0_star must be expanded to perform operation : shape(batch_size, num_legs) -> shape(batch_size, num_legs, num_joints_per_leg)
        T_swing = T * (~c0_star.unsqueeze(-1).expand(*[-1] * len(c0_star.shape), T.shape[-1]))

        return T_swing
    

    def stance_leg_controller(self, F0_star_lw: torch.Tensor, c0_star: torch.Tensor, jacobian_lw: torch.Tensor) -> torch.Tensor:
        """ Given GRF and contact sequence -> compute joint torques using the jacobian : T = -J*F
        1. compute the jacobian using the simulation tool : end effector jacobian wrt to robot base
        2. compute the stance torque : T = -J*F
        Note :
            The variable are in the 'local' world frame _wl. This notation is introduced to avoid confusion with the 'global' world frame, where all the batches coexists.

        Args:
            - F0*_lw (th.Tensor): Opt. Ground Reac. Forces (GRF) in w fr.   of shape(batch_size, num_legs, 3)
            - c0*   (torch.bool): Optimized foot contact 1st el. seq.       of shape(batch_size, num_legs)
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
        T = T_unsqueezed.squeeze(-1)

        # Keep torques only for leg in contact
        # C0_star must be expanded to perform operation : shape(batch_size, num_legs) -> shape(batch_size, num_legs, num_joints_per_leg)
        T_stance = T * c0_star.unsqueeze(-1).expand(*[-1] * len(c0_star.shape), T.shape[-1])

        return T_stance


        