from abc import ABC
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

    Method :
        - late_init(device, num_envs, num_legs) : save environment variable and allow for lazy initialisation of variables
        - optimize_latent_variable(f, d, p, F) -> p*, F*, c*, pt*
        - compute_control_output(F0*, c0*, pt01*) -> T
        - gait_generator(f, d, phase) -> c, new_phase

    """

    def __init__(self):
        super().__init__()


    def late_init(self, device, num_envs, num_legs, time_horizon, dt_out):
        self._num_envs = num_envs
        self._device = device
        self._num_legs = num_legs
        self._time_horizon = time_horizon
        self._dt_out = dt_out


    def optimize_latent_variable(self, f: torch.Tensor, d: torch.Tensor, p: torch.Tensor, F: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Given the latent variable z=[f,d,p,F], return the optimized latent variable p*, F*, c*, pt*

        Args:
            - f   (torch.Tensor): Prior leg frequency                   of shape (batch_size, num_legs)
            - d   (torch.Tensor): Prior stepping duty cycle             of shape (batch_size, num_legs)
            - p   (torch.Tensor): Prior foot pos. sequence              of shape (batch_size, num_legs, 3, time_horizon)
            - F   (torch.Tensor): Prior Ground Reac. Forces (GRF) seq.  of shape (batch_size, num_legs, 3, time_horizon)

        Returns:
            - p*  (torch.Tensor): Optimized foot position sequence      of shape (batch_size, num_legs, 3, time_horizon)
            - F*  (torch.Tensor): Opt. Ground Reac. Forces (GRF) seq.   of shape (batch_size, num_legs, 3, time_horizon)
            - c*  (torch.Tensor): Optimized foot contact sequence       of shape (batch_size, num_legs, time_horizon)
            - pt* (torch.Tensor): Optimized foot swing trajectory       of shape (batch_size, num_legs, 3, decimation)
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
        - _device
        - _num_envs
        - _num_legs
        - _time_horizon : Outer Loop prediction time horizon
        - _dt_out       : Outer Loop time step 
        - phase (Tensor): Leg phase                                     of shape (batch_size, num_legs)
        - p0 (th.Tensor): Lift-off position                             of shape (batch_size, num_legs, 3)
        - c_prev (Tnsor): Previous contact value                        of shape (batch_size, num_legs)
        - swing_time (T): time progression of the leg in swing phase    of shape (batch_size, num_legs)  

    Method :
        - late_init(device, num_envs, num_legs) : save environment variable and allow for lazy initialisation of variables # Inherited
        - optimize_latent_variable(f, d, p, F) -> p*, F*, c*, pt*   # Inherited
        - compute_control_output(F0*, c0*, pt01*) -> T              # Inherited
        - gait_generator(f, d, phase) -> c, new_phase               # Inherited
        - swing_trajectory_generator(p, c, decimation) -> pt
        - swing_leg_controller(c0*, pt01*) -> T_swing
        - stance_leg_controller(F0*, c0*) -> T_stance
    """

    def __init__(self, swing_ctrl_pos_gain_fb = 1, swing_ctrl_vel_gain_fb=1):
        super().__init__()

        self.swing_ctrl_pos_gain_fb = swing_ctrl_pos_gain_fb
        self.swing_ctrl_vel_gain_fb = swing_ctrl_vel_gain_fb

        # Late init
        self.phase = None
        self.p0 = None
        self.c_prev = None
        self.swing_time = None


    def late_init(self, device, num_envs, num_legs, time_horizon, dt_out):
        super().late_init(device, num_envs, num_legs, time_horizon, dt_out)
        self.phase = torch.zeros(num_envs, num_legs, device=device)
        self.p0 = torch.zeros(num_envs, num_legs, 3, device=device) # TODO Should initialise with the reset foot position
        self.c_prev = torch.zeros(num_envs, num_legs, device=device)
        self.swing_time = torch.zeros(num_envs, num_legs, device=device)

# ----------------------------------- Outer Loop ------------------------------
    def optimize_latent_variable(self, f: torch.Tensor, d: torch.Tensor, p: torch.Tensor, F: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Given the latent variable z=[f,d,p,F], return the optimized latent variable p*, F*, c*, pt*

        Args:
            - f   (torch.Tensor): Prior leg frequency                   of shape (batch_size, num_legs)
            - d   (torch.Tensor): Prior stepping duty cycle             of shape (batch_size, num_legs)
            - p   (torch.Tensor): Prior foot pos. sequence              of shape (batch_size, num_legs, 3, time_horizon)
            - F   (torch.Tensor): Prior Ground Reac. Forces (GRF) seq.  of shape (batch_size, num_legs, 3, time_horizon)

        Returns:
            - p*  (torch.Tensor): Optimized foot position sequence      of shape (batch_size, num_legs, 3, time_horizon)
            - F*  (torch.Tensor): Opt. Ground Reac. Forces (GRF) seq.   of shape (batch_size, num_legs, 3, time_horizon)
            - c*  (torch.Tensor): Optimized foot contact sequence       of shape (batch_size, num_legs, time_horizon)
            - pt* (torch.Tensor): Optimized foot swing trajectory       of shape (batch_size, num_legs, 9, decimation)  (9 = pos, vel, acc)
        """

        # Compute the contact sequence and update the phase
        c, self.phase = self.gait_generator(f=f, d=d, phase=self.phase, time_horizon=self._time_horizon, dt=self._dt_out)

        pt = self.swing_trajectory_generator(p=p, c=c, decimation=10, d=d, f=f)

        p_star = p
        F_star = F
        c_star = c
        pt_star = pt

        return p_star, F_star, c_star, pt_star
    

    def gait_generator(self, f: torch.Tensor, d: torch.Tensor, phase: torch.tensor, time_horizon: int, dt) -> tuple[torch.Tensor, torch.Tensor]:
        """ Implement a gait generator that return a contact sequence given a leg frequency and a leg duty cycle
        Increment phase by dt*f 
        restart if needed
        return contact : 1 if phase < duty cyle, 0 otherwise 

        Note:
            No properties used, no for loop : purely functional -> made to be jitted

        Args:
            - f   (torch.Tensor): Leg frequency                         of shape(batch_size, num_legs, parallel_rollout)
            - d   (torch.Tensor): Stepping duty cycle                   of shape(batch_size, num_legs, parallel_rollout)
            - phase (tch.Tensor): phase of leg                          of shape(batch_size, num_legs, parallel_rollout)
            - time_horizon (int): Time horizon for the contact sequence

        Returns:
            - c     (torch.bool): Foot contact sequence                 of shape(batch_size, num_legs, parallel_rollout, time_horizon)
            - phase (tch.Tensor): The phase updated by one time steps   of shape(batch_size, num_legs, parallel_rollout)
        """
        
        # Increment phase of f*dt: new_phases[0] : incremented of 1 step, new_phases[1] incremented of 2 steps, etc. without a for loop.
        # new_phases = phase + f*dt*[1,2,...,time_horizon]
        # phase and f must be exanded from (batch_size, num_legs, parallel_rollout) to (batch_size, num_legs, parallel_rollout, time_horizon) in order to perform the operations
        new_phases = phase.unsqueeze(-1).expand(*[-1] * len(phase.shape),time_horizon) + f.unsqueeze(-1).expand(*[-1] * len(f.shape),time_horizon)*torch.linspace(start=1, end=time_horizon, steps=time_horizon, device=self._device)*dt

        # Make the phases circular (like sine) (% is modulo operation)
        new_phases = new_phases%1

        # Save first phase
        new_phase = new_phases[..., 0]

        # Make comparaison to return discret contat sequence
        c = new_phases < d.unsqueeze(-1).expand(*[-1] * len(d.shape), time_horizon)

        return c, new_phase
    

    def swing_trajectory_generator(self, p: torch.Tensor, c: torch.Tensor, f: torch.Tensor, d: torch.Tensor, decimation: int) -> torch.Tensor:
        """ Given feet position sequence and contact sequence -> compute swing trajectories by fitting a cubic spline between
        the lift-off and the touch down define in the contact sequence. 
        - Swing frequency and duty cycle are used to compute the swing period
        - A middle point is used for the interpolation : which is heuristically defined. It defines the step height
        - p1 (middle point) and p2 (touch-down) are updated each time, while p0 is conserved (always the same lift off position)

        Args:
            - p   (torch.Tensor): Foot position sequence                of shape(batch_size, num_legs, 3, time_horizon)
            - c   (torch.Tensor): Foot contact sequence                 of shape(batch_size, num_legs, time_horizon)
            - f   (torch.Tensor): Leg frequency                         of shape(batch_size, num_legs)
            - d   (torch.Tensor): Stepping duty cycle                   of shape(batch_size, num_legs)
            - decimation   (int): Number of timestep for the traj.

        Returns:
            - pt  (torch.Tensor): Desired Swing Leg trajectories        of shape(batch_size, num_legs, 9, decimation)   (9 = xyz_pos, xzy_vel, xyz_acc)
        """

        # Step 0. Define and Compute usefull variables

        # Heuristic TODO Save that on the right place, could also be a RL variable
        step_height = 0.05

        # Time during wich the leg is in swing. TODO Why +0.07 ? Is it an heuristic also ?
        # Shape (batch_size, num_legs)
        swing_period = ((1-d) / f) + 0.07

        half_swing_period = swing_period / 2
        time_fac = 1 / (swing_period / 2) #bezier_time_factor


        # Step 1. Retrieve the three interpolation points : p0, p1, p2 (lift-off, middle point, touch down)

        # Retrieve p0 : If c(0)=0 and c(-1)=1 : The leg lift-off -> p0 = p(0) # TODO p(0) or must it be from simulation data ? TODO Must it be p(0) or p(-1)
        # Update only the p0 that are new lift off positions
        lifting_off = (c[:,:,0]==0) * (self.c_prev == 1)
        self.p0 = (p[:,:,:,0] * lifting_off) + (self.p0 * ~lifting_off)  

        # Retrieve p2 : Retrieve the index of the touch down in the contact sequence : First Non-zero Index
        # Set the last value of c as ONE to avoid the case of only 0 in the contact sequence, wich return the first element (make more sense to retrun the last)
        # With the touch_down index, retrieve the touch down foot position : p2
        # shape (batch_size, num_legs, 3) 
        c[:,:,-1] = 1 # TODO Does it modify c also outside this function ?
        first_non_zero_indx = torch.argmax((c!=0).float(), dim=-1)
        p2 = torch.gather(p, -1, first_non_zero_indx.unsqueeze(-1)).squeeze(-1)

        # Retrieve p1 : (x,y) position are define as the middle point between p0 and p1 (lift-off and touch-down). z is heuristcally define
        # shape (batch_size, num_legs, 3)
        p1 = (self.p0[:,:,:2] + p2[:,:,:2]) / 2     # p1(x,y) is in the middle of p0 and p2
        p1 = torch.cat((p1, step_height*torch.ones_like(p1[:,:,:1])), dim=2) # Append a third dimension z : defined as step_height

        # Step 2. Compute the parameters for the interpolation

        # Swing time : reset if lifting off, then increment by one time step (outer loop)
        # then compute t in [0, Delta_t/2], which would be use for the spline interpolation
        # shape (batch_size, num_legs)
        self.swing_time = (self.swing_time * ~lifting_off) + self._dt_out
        t = self.swing_time % half_swing_period  # Swing time (half)

        # Compute the a,b,c,d polynimial coefficient for the cubic interpolation S(t) = a*t^3 + b*t^2 + c*t + d
        # If swing_time < swing period/2 -> S_0(t) (ie. first interpolation), otherwise -> S_1(t - delta_t/2) (ie. second interpolation)
        # cp_x shape (batch_size, num_legs, 3)
        is_S0 = (self.swing_time <=  half_swing_period).unsqueeze(-1).expand(*[-1] * len(self.swing_time.shape), 3)  # shape (batch_size, num_legs, 3)
        cp1 = (self.p0 * is_S0)                                         + (p1 * ~is_S0)
        cp2 = (self.p0 * is_S0)                                         + (torch.cat((p2[:,:,:2], p1[:,:,2:]), dim=2)* ~is_S0)
        cp3 = (torch.cat((self.p0[:,:,:2], p1[:,:,2:]), dim=2) * is_S0) + (p2 * ~is_S0)
        cp4 = (p1 * is_S0)                                              + (p2 * ~is_S0)

        # Step 3. Compute the interpolation trajectory
        desired_foot_pos_traj = cp1*(1 - time_fac*t)**3 + 3*cp2*(time_fac*t)*(1 - time_fac*t)**2 + 3*cp3*((time_fac*t)**2)*(1 - time_fac*t) + cp4*(time_fac*t)**3
        desired_foot_vel_traj = 3*(cp2 - cp1)*(1 - time_fac*t)**2 + 6*(cp3 - cp2)*(1 - time_fac*t)*(time_fac*t) + 3*(cp4 - cp3)*(time_fac*t)**2
        desired_foot_acc_traj = 6*(1 - time_fac*t) * (cp3 - 2*cp2 + cp1) + 6 * (time_fac*t) * (cp4 - 2*cp3 + cp2)
        pt = torch.cat((desired_foot_pos_traj, desired_foot_vel_traj, desired_foot_acc_traj), dim=2)

        return torch.zeros(self._num_envs, self._num_legs, 9, 10, device=self._device)


# ----------------------------------- Inner Loop ------------------------------
    def compute_control_output(self, F0_star: torch.Tensor, c0_star: torch.Tensor, pt_i_star: torch.Tensor, p:torch.Tensor, p_dot:torch.Tensor, q_dot: torch.Tensor, jacobian: torch.Tensor, jacobian_dot: torch.Tensor, mass_matrix: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
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
            - p   (torch.Tensor): Feet Position  (latest from sim)      of shape(batch_size, num_legs, 3)
            - p_dot (tch.Tensor): Feet velocity  (latest from sim)      of shape(batch_size, num_legs, 3)
            - q_dot (tch.Tensor): Joint velocity (latest from sim)      of shape(batch_size, num_legs, num_joints_per_leg)
            - jacobian  (Tensor): Jacobian -> joint frame to foot frame of shape(batch_size, num_legs, 3, num_joints_per_leg)
            - TODO Geet it from sim or compute it here ? jacobian_dot (Tsr): Jacobian derivative (forward euler)   of shape(batch_size, num_legs, 3, num_joints_per_leg)
            - mass_matrix (Tsor): Mass Matrix in joint space            of shape(batch_size, num_legs, num_joints_per_leg, num_joints_per_leg)
            - h   (torch.Tensor): C(q,q_dot) + G(q) (corr. and grav F.) of shape(batch_size, num_legs, num_joints_per_leg)

        Returns:
            - T   (torch.Tensor): control output (ie. Joint Torques)    of shape(batch_size, num_legs, num_joints_per_leg)
        """

        # Get the swing torque from the swing controller : swing torque has already been filered by C0* (ie. T_swing = T * ~c0*)
        # T_swing Shape (batch_size, num_legs, num_joints_per_leg)
        T_swing = self.swing_leg_controller(c0_star, pt_i_star, p, p_dot, q_dot, jacobian, jacobian_dot, mass_matrix, h)

        # Get the stance torque from the stance controller : stance toeque has already been filtered by c0* (ie. T_stance = T * c0*)
        # T_stance Shape (batch_size, num_legs, num_joints_per_leg)
        T_stance = self.stance_leg_controller(F0_star, c0_star, jacobian)

        # T shape = (batch_size, num_legs, num_joints_per_leg)
        T = T_swing + T_stance 

        return T


    def swing_leg_controller(self, c0_star: torch.Tensor, pt_i_star: torch.Tensor, p:torch.Tensor, p_dot:torch.Tensor, q_dot: torch.Tensor,
                             jacobian: torch.Tensor, jacobian_dot: torch.Tensor, mass_matrix: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """ Given feet contact, and desired feet trajectory : compute joint torque with feedback linearization control
        T = M(q)*J⁻¹[p_dot_dot - J(q)*q_dot] + C(q,q_dot) + G(q)

        Args:
            - c0*   (torch.bool): Optimized foot contact sequence       of shape(batch_size, num_legs)
            - pt_i* (tch.Tensor): Opt. Foot point in swing phase        of shape(batch_size, num_legs, 9) (9 = pos, vel, acc)
            - p   (torch.Tensor): Feet Position                         of shape(batch_size, num_legs, 3)
            - p_dot (tch.Tensor): Feet velocity                         of shape(batch_size, num_legs, 3)
            - q_dot (tch.Tensor): Joint velocity                        of shape(batch_size, num_legs, num_joints_per_leg)
            - jacobian  (Tensor): Jacobian -> joint frame to foot frame of shape(batch_size, num_legs, 3, num_joints_per_leg)
            - jacobian_dot (Tsr): Jacobian derivative (forward euler)   of shape(batch_size, num_legs, 3, num_joints_per_leg)
            - mass_matrix (Tsor): Mass Matrix in joint space            of shape(batch_size, num_legs, num_joints_per_leg, num_joints_per_leg)
            - h   (torch.Tensor): C(q,q_dot) + G(q) (corr. and grav F.) of shape(batch_size, num_legs, num_joints_per_leg)

        Returns:
            - T_swing (t.Tensor): Swing Leg joint torques               of shape(batch_size, num_legs, num_joints_per_leg)
        """

        # Intermediate variables
        pos_err = pt_i_star[:,:,0:3] - p
        vel_err = pt_i_star[:,:,3:6] - p_dot
        des_foot_acc = pt_i_star[:,:,6:9]

        # Intermediary step : p_dot_dot
        # Compute the desired acceleration : with a PD controller thanks to the feedback linearization
        # Shape (batch_size, num_legs, 3)
        p_dot_dot = des_foot_acc + self.swing_ctrl_pos_gain_fb * (pos_err) + self.swing_ctrl_vel_gain_fb * (vel_err)

        # Compute  the inverse jacobian. This synchronise CPU and GPU
        # Compute pseudo-inverse -> to be resilient to any number of joint per legs (not restricted to square matrix)
        # Changed shape from (batch_size, num_legs, 3, num_joints_per_leg) to -> (batch_size, num_legs, num_joints_per_leg, 3)
        jacobian_inv = torch.linalg.pinv(jacobian)
        
        # Intermediary step : J(q)*q_dot            (batch_size, num_legs, 3, num_joints_per_leg) * (batch_size, num_legs, num_joints_per_leg)
        # Must unsqueeze q_dot to perform matmul (ie add a singleton dimension on last position)
        # change q_dot shape from (batch_size, num_legs, num_joints_per_leg) to (batch_size, num_legs, num_joints_per_leg, 1)
        # J_dot_x_q_dot is of shape (batch_size, num_legs, 3) (The singleton dim is dropped by the squeeze operation)
        J_dot_x_q_dot = torch.matmul(jacobian_dot, q_dot.unsqueeze(-1)).squeeze(-1)

        # Intermediary step : [p_dot_dot - J(q)*q_dot]          : Shape (batch_size, num_legs, 3)
        p_dot_dot_min_J_dot_x_q_dot = p_dot_dot - J_dot_x_q_dot

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
    

    def stance_leg_controller(self, F0_star: torch.Tensor, c0_star: torch.Tensor, jacobian: torch.Tensor) -> torch.Tensor:
        """ Given GRF and contact sequence -> compute joint torques using the jacobian : T = -J*F
        1. compute the jacobian using the simulation tool : end effector jacobian wrt to robot base
        2. compute the stance torque : T = -J*F

        Args:
            - F0* (torch.Tensor): Opt. Ground Reac. Forces (GRF)        of shape(batch_size, num_legs, 3)
            - c0*   (torch.bool): Optimized foot contact 1st el. seq.   of shape(batch_size, num_legs)
            - jacobian  (Tensor): Jacobian -> joint frame to foot frame of shape(batch_size, num_legs, 3, num_joints_per_leg)

        Returns:
            - T_stance(t.Tensor): Stance Leg joint Torques              of shape(batch_size, num_legs, num_joints_per_leg)
        """
        
        # Transpose the jacobian -> In batch operation : permut the last two dimensions 
        # shape(batch_size, num_legs, 3, num_joints_per_leg) -> shape(batch_size, num_legs, num_joints_per_leg, 3)
        jacobian_T = jacobian.transpose(-1,-2)

        # Add a singleton dimension on last position to enable matmul operation 
        # shape(batch_size, num_legs, 3) -> shape(batch_size, num_legs, 3, 1)
        F0_star_unsqueezed = F0_star.unsqueeze(-1)

        # Perform the matrix multiplication T = - J^T * F
        # shape(batch_size, num_legs, num_joints_per_leg, 1)
        T_unsqueezed= -torch.matmul(jacobian_T, F0_star_unsqueezed)

        # Supress the singleton dimension added for the matmul operation
        # shape(batch_size, num_legs, num_joints_per_leg, 1) -> shape(batch_size, num_legs, num_joints_per_leg)
        T = T_unsqueezed.squeeze(-1)

        # Keep torques only for leg in contact
        # C0_star must be expanded to perform operation : shape(batch_size, num_legs) -> shape(batch_size, num_legs, num_joints_per_leg)
        T_stance = T * c0_star.unsqueeze(-1).expand(*[-1] * len(c0_star.shape), T.shape[-1])

        return T_stance


        