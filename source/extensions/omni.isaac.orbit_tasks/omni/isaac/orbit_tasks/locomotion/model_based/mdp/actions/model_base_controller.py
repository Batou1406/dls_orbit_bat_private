from abc import ABC
import torch


class modelBaseController(ABC):
    """
    Abstract controller class for model base control implementation
    
    Properties : 
        - 

    Method :
        - optimize_control_output(f, d, p, F) -> F*, p*, c*
        - gait_generator(f, d) -> c

    """

    def __init__(self):
        pass


    def optimize_control_output(self, f: torch.Tensor, d: torch.Tensor, p: torch.Tensor, F: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Given the latent variable z=[f,d,p,F], return the optimized the control output

        Args:
            - f   (torch.Tensor): Prior leg frequency                   of shape (batch_size, num_legs)
            - d   (torch.Tensor): Prior stepping duty cycle             of shape (batch_size, num_legs)
            - p   (torch.Tensor): Prior foot pos. sequence              of shape (batch_size, num_legs, 3, time_horizon)
            - F   (torch.Tensor): Prior Ground Reac. Forces (GRF) seq.  of shape (batch_size, num_legs, 3, time_horizon)

        Returns:
            - F*  (torch.Tensor): Opt. Ground Reac. Forces (GRF) seq.   of shape (batch_size, num_legs, 3, time_horizon)
            - p*  (torch.Tensor): Optimized foot position sequence      of shape (batch_size, num_legs, 3, time_horizon)
            - c*  (torch.Tensor): Optimized foot contact sequence       of shape (batch_size, num_legs, time_horizon)
        """

        raise NotImplementedError


    def gait_generator(self, f: torch.Tensor, d: torch.Tensor, phase: torch.tensor, time_horizon: int) -> torch.Tensor:
        """ ...

        Args:
            - f   (torch.Tensor): Leg frequency                         of shape(batch_size, num_legs, parallel_rollout)
            - d   (torch.Tensor): Stepping duty cycle                   of shape(batch_size, num_legs, parallel_rollout)
            - phase (torch.Tensor): phase of leg                        of shape(batch_size, num_legs, parallel_rollout)
            - time_horizon (int): Time horizon for the contact sequence

        Returns:
            - c   (torch.bool)  : Foot contact sequence                 of shape(batch_size, num_legs, time_horizon, parallel_rollout)
        """
        raise NotImplementedError


    def compute_control_output(self, F_star: torch.Tensor, p_star: torch.Tensor, c_star: torch.Tensor) -> torch.Tensor:
        """ Compute the output torque to be applied to the system
        typically, it would compute :
            - T_stance_phase = stance_leg_controller(...)
            - T_swing_phase = swing_leg_controller(...)
            - T = (T_stance_phase * c_star) + (T_swing_phase * (~c_star))
        and return T
        
        Args:
            - F0* (torch.Tensor): Opt. Ground Reac. Forces (GRF)        of shape(batch_size, num_legs, 3)
            - pt* (torch.Tensor): Opt. Foot trajectory in swing phase   of shape(batch_size, num_legs, 3, decimation)
            - c0* (torch.bool)  : Optimized foot contact sequence       of shape(batch_size, num_legs)
            - p0  (torch.Tensor): last foot position (from sim.)        of shape(batch_size, num_legs, 3)

        Returns:
            - T   (torch.Tensor): control output (ie. Joint Torques)    of shape(batch_size, num_joints)
        """
        raise NotImplementedError


    def swing_trajectory_controller(self):
        raise NotImplementedError


    def stance_leg_controller(self):
        raise NotImplementedError


    def swing_leg_controller(self):
        raise NotImplementedError


    def optimize_gait(self):
        # Est-ce que ça doit aller là ? ou plutôt dans la classe spcécialisé du sampling controller
        return None