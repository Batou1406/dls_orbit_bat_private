from abc import ABC
import torch


class modelBaseController(ABC):
    """
    Abstract controller class for model base control implementation
    
    Properties : 
        - 

    Method :
        - optimize_latent_variable(f, d, p, F) -> p*, F*, c*, pt*
        - compute_control_output(F0*, c0*, pt01*) -> T
        - gait_generator(f, d, phase) -> c, new_phase

    """

    def __init__(self):
        pass


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


    def compute_control_output(self, F0_star: torch.Tensor, c0_star: torch.Tensor, pt01_star: torch.Tensor) -> torch.Tensor:
        """ Compute the output torque to be applied to the system
        typically, it would compute :
            - T_stance_phase = stance_leg_controller(GRF, q, c) # Update the jacobian with the new joint position.
            - T_swing_phase = swing_leg_controller(trajectory, q, q_dot, c) # Feedback lineraization control - trajectory computed with a spline to be followed - new updated joint controller.
            - T = (T_stance_phase * c_star) + (T_swing_phase * (~c_star))
        and return T
        
        Args:
            - F0* (torch.Tensor): Opt. Ground Reac. Forces (GRF)        of shape(batch_size, num_legs, 3)
            - c0* (torch.bool)  : Optimized foot contact sequence       of shape(batch_size, num_legs)
            - pt01* (tch.Tensor): Opt. Foot trajectory in swing phase   of shape(batch_size, num_legs, 3, decimation)
            - TODO p0  (torch.Tensor): last foot position (from sim.)        of shape(batch_size, num_legs, 3)

        Returns:
            - T   (torch.Tensor): control output (ie. Joint Torques)    of shape(batch_size, num_joints)
        """
        raise NotImplementedError
    

    def gait_generator(self, f: torch.Tensor, d: torch.Tensor, phase: torch.tensor, time_horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
        """ Implement a gait generator that return a contact sequence given a leg frequency and a leg duty cycle
        Increment phase by dt*f 
        restart if needed
        return contact : 1 if phase > duty cyle, 0 otherwise 

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
        - 

    Method :
        - optimize_latent_variable(f, d, p, F) -> p*, F*, c*, pt*   # Inherited
        - compute_control_output(F0*, c0*, pt01*) -> T              # Inherited
        - gait_generator(f, d, phase) -> c, new_phase               # Inherited
        - swing_trajectory_generator(p, c, decimation) -> pt
        - swing_leg_controller(c0*, pt01*) -> T_swing
        - stance_leg_controller(F0*, c0*) -> T_stance
    """

    def __init__(self):
        super().__init__()

    def gait_generator(self, f: torch.Tensor, d: torch.Tensor, phase: torch.tensor, time_horizon: int, dt) -> tuple[torch.Tensor, torch.Tensor]:
        """ Implement a gait generator that return a contact sequence given a leg frequency and a leg duty cycle
        Increment phase by dt*f 
        restart if needed
        return contact : 1 if phase > duty cyle, 0 otherwise 

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
        new_phases = phase.unsqueeze(-1).expand(*[-1] * len(phase.shape),time_horizon) + f.unsqueeze(-1).expand(*[-1] * len(f.shape),time_horizon)*torch.linspace(start=1, end=time_horizon, steps=time_horizon)*dt

        
        # Make the phases circular (like sine) (% is modulo operation)
        new_phases = new_phases%1

        # Save first phase
        new_phase = new_phases[..., 0]

        # Make comparaison to return discret contat sequence
        c = new_phases > d.unsqueeze(-1).expand(*[-1] * len(d.shape), time_horizon)

        return c, new_phase

    def swing_trajectory_generator(self, p: torch.Tensor, c: torch.Tensor, decimation: int) -> torch.Tensor:
        """ Given feet position and contact sequence -> compute swing trajectories

        Args:
            - p   (torch.Tensor): Foot position sequence                of shape(batch_size, num_legs, 3, parallel_rollout, time_horizon)
            - c   (torch.Tensor): Foot contact sequence                 of shape(batch_size, num_legs, parallel_rollout, time_horizon)
            - decimation   (int): Number of timestep for the traj.

        Returns:
            - pt  (torch.Tensor): Swing Leg trajectories                of shape(batch_size, num_legs, 3, decimation)
        """
        # Utiliser une convolution sur c (contact sequence) pour trouver le point de départ et d'arriver du pied.
        # Avec un filtre genre f = [0, 1], pour ne garder que les flancs montants
        # Imaginons p = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10]
        #           c = [ 1,  1,  0,  0,  0,  1,  1,  0,  0,   1]
        # Les points de départ serait p1, p6 et p10 les points d'arrivé p6 et p10
        # Il faudrait retourner qqch comme 
        #        key =  [1,   0,  0,  0,  0,  1,  0,  0,  0,   1]
        # Qui permetrait d'extraire facilement [p1, p6, p10] avec p[key]
        raise NotImplementedError


    def swing_leg_controller(self, c0_star: torch.Tensor, pt01_star: torch.Tensor) -> torch.Tensor:
        """ Given feet contact, and desired feet trajectory : compute joint torque with feedback linearization control
        Args:
            - c0*   (torch.bool): Optimized foot contact sequence       of shape(batch_size, num_legs)
            - pt01* (tch.Tensor): Opt. Foot trajectory in swing phase   of shape(batch_size, num_legs, 3, decimation)

        Returns:
            - T_swing (t.Tensor): Swing Leg joint torques               of shape(batch_size, num_joints)
        """
        raise NotImplementedError
    

    def stance_leg_controller(self, F0_star: torch.Tensor, c0_star: torch.Tensor) -> torch.Tensor:
        """ Given GRF and contact sequence -> compute joint torques using the jacobian : T = J*F

        Args:
            - F0* (torch.Tensor): Opt. Ground Reac. Forces (GRF)        of shape(batch_size, num_legs, 3)
            - c0* (torch.bool)  : Optimized foot contact sequence       of shape(batch_size, num_legs)

        Returns:
            - T_stance(t.Tensor): Stance Leg joint Torques              of shape(batch_size, num_joints)
        """
        raise NotImplementedError


    def optimize_gait(self):
        raise NotImplementedError


        