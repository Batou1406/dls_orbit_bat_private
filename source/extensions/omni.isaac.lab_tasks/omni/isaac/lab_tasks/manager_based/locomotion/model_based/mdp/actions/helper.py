# Helper file witch mathematical functions
import torch

def inverse_conjugate_euler_xyz_rate_matrix(euler_xyz_angle: torch.Tensor) -> torch.Tensor:
    """
    Given euler angles in  the XYZ convention (ie. roll pitch yaw), return the inverse conjugate euler rate matrix.

    Note
        Follow the convention of 'Representing Attitude: Euler Angles, Unit Quaternions, and Rotation Vectors'
        The inverse_conjugate_euler_xyz_rate_matrix is given by eq. 79
        https://www.astro.rug.nl/software/kapteyn-beta/_downloads/attitude.pdf

    
    Args
        euler_xyz_angle (torch.tensor): XYZ euler angle of shape(bacth, 3)

    Return
        inverse_conjugate_euler_xyz_rate_matrix (torch.tensor): inverse conjugate XYZ euler rate matrix of shape(batch, 3, 3)
    """
    # Extract Roll Pitch Yaw
    roll  = euler_xyz_angle[:, 0] # shape(batch)
    pitch = euler_xyz_angle[:, 1] # shape(batch)
    yaw   = euler_xyz_angle[:, 2] # shape(batch)

    # Compute intermediary variables
    cos_roll  = torch.cos(roll)   # shape(batch)
    sin_roll  = torch.sin(roll)   # shape(batch)
    cos_pitch = torch.cos(pitch)  # shape(batch)
    sin_pitch = torch.sin(pitch)  # shape(batch)
    tan_pitch = torch.tan(pitch)  # shape(batch)

    # Check for singularities: pitch close to +/- 90 degrees (or +/- pi/2 radians)
    assert not torch.any(torch.abs(cos_pitch) < 1e-6), "Numerical instability likely due to pitch angle near +/- 90 degrees."

    # Create the matrix of # shape(batch, 3, 3)
    inverse_conjugate_euler_xyz_rate_matrix = torch.zeros((euler_xyz_angle.shape[0], 3, 3), dtype=euler_xyz_angle.dtype, device=euler_xyz_angle.device)

    # Fill the matrix with element given in eq. 79
    inverse_conjugate_euler_xyz_rate_matrix[:, 0, 0] = 1                    # shape(batch)
    inverse_conjugate_euler_xyz_rate_matrix[:, 0, 1] = sin_roll * tan_pitch # shape(batch)
    inverse_conjugate_euler_xyz_rate_matrix[:, 0, 2] = cos_roll * tan_pitch # shape(batch)

    inverse_conjugate_euler_xyz_rate_matrix[:, 1, 0] = 0                    # shape(batch)
    inverse_conjugate_euler_xyz_rate_matrix[:, 1, 1] = cos_roll             # shape(batch)
    inverse_conjugate_euler_xyz_rate_matrix[:, 1, 2] = -sin_roll            # shape(batch)

    inverse_conjugate_euler_xyz_rate_matrix[:, 2, 0] = 0                    # shape(batch)
    inverse_conjugate_euler_xyz_rate_matrix[:, 2, 1] = sin_roll / cos_pitch # shape(batch)
    inverse_conjugate_euler_xyz_rate_matrix[:, 2, 2] = cos_roll / cos_pitch # shape(batch)

    return inverse_conjugate_euler_xyz_rate_matrix


def rotation_matrix_from_w_to_b(euler_xyz_angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrix to transform value from wolrd frame orientation to base frame oriention
    given euler angles (Roll, Pitch, Yaw) in the XYZ convention : [roll, pitch, yaw].T -> SO(3)

    Apply the three successive rotation : 
    R_xyz(roll, pitch, yaw) = R_x(roll)*R_y(pitch)*R_z(yaw)

    Note 
        Follow the convention of 'Representing Attitude: Euler Angles, Unit Quaternions, and Rotation Vectors'
        The rotation matrix is given by eq. 67
        https://www.astro.rug.nl/software/kapteyn-beta/_downloads/attitude.pdf       

    Arg
        euler_xyz_angle (torch.tensor): XYZ euler angle of shape(bacth, 3)

    Return
        rotation_matrix_from_w_to_b (torch.Tensor): Rotation matrix that rotate from w to b of shape(batch, 3, 3)
    """

    # Extract Roll Pitch Yaw
    roll  = euler_xyz_angle[:, 0] # shape(batch)
    pitch = euler_xyz_angle[:, 1] # shape(batch)
    yaw   = euler_xyz_angle[:, 2] # shape(batch)

    # Compute intermediary variables
    cos_roll  = torch.cos(roll)   # shape(batch)
    sin_roll  = torch.sin(roll)   # shape(batch)
    cos_pitch = torch.cos(pitch)  # shape(batch)
    sin_pitch = torch.sin(pitch)  # shape(batch)
    cos_yaw   = torch.cos(yaw)  # shape(batch)
    sin_yaw   = torch.sin(yaw)  # shape(batch)

    # Create the matrix of # shape(batch, 3, 3)
    rotation_matrix_from_w_to_b = torch.zeros((euler_xyz_angle.shape[0], 3, 3), dtype=euler_xyz_angle.dtype, device=euler_xyz_angle.device)

    # Fill the matrix with element
    rotation_matrix_from_w_to_b[:, 0, 0] = cos_pitch*cos_yaw                                # shape(batch)
    rotation_matrix_from_w_to_b[:, 0, 1] = cos_pitch*sin_yaw                                # shape(batch)
    rotation_matrix_from_w_to_b[:, 0, 2] = -sin_pitch                                       # shape(batch

    rotation_matrix_from_w_to_b[:, 1, 0] = sin_roll*sin_pitch*cos_yaw - cos_roll*sin_yaw    # shape(batch)
    rotation_matrix_from_w_to_b[:, 1, 1] = sin_roll*sin_pitch*sin_yaw + cos_roll*cos_yaw    # shape(batch)
    rotation_matrix_from_w_to_b[:, 1, 2] = sin_roll*cos_pitch                               # shape(batch

    rotation_matrix_from_w_to_b[:, 2, 0] = cos_roll*sin_pitch*cos_yaw + sin_roll*sin_yaw    # shape(batch)
    rotation_matrix_from_w_to_b[:, 2, 1] = cos_roll*sin_pitch*sin_yaw - sin_roll*cos_yaw    # shape(batch)
    rotation_matrix_from_w_to_b[:, 2, 2] = cos_roll*cos_pitch                               # shape(batch)

    return rotation_matrix_from_w_to_b


@torch.jit.script
def gait_generator(f: torch.Tensor, d: torch.Tensor, phase: torch.Tensor, horizon: int, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
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
        - f     (Tensor): Leg frequency samples                 of shape(batch, num_legs)
        - d     (Tensor): Stepping duty cycle samples in [0,1]  of shape(batch, num_legs)
        - phase (Tensor): phase of leg samples in [0,1]         of shape((optionnally batch), num_legs)
        - horizon  (int): Time horizon for the contact sequence

    Returns:
        - c     (t.bool): Foot contact sequence samples         of shape(batch, num_legs, sampling_horizon)
        - phase (Tensor): The phase samples updated by 1 dt     of shape(batch, num_legs)
    """
    
    # Increment phase of f*dt: new_phases[0] : incremented of 1 step, new_phases[1] incremented of 2 steps, etc. without a for loop.
    # new_phases = phase + f*dt*[1,2,...,sampling_horizon]
    #                    (1 or n, num_legs, 1)                 + (samples, legs, 1)       * (1, 1, sampling_horizon) -> shape(samples, legs, sampling_horizon)
    if phase.dim() == 1 : 
        phase = phase.unsqueeze(0)
    new_phases = phase.unsqueeze(-1) + (f.unsqueeze(-1) * torch.linspace(start=1, end=horizon, steps=horizon, device=f.device).unsqueeze(0).unsqueeze(1)*dt)

    # Make the phases circular (like sine) (% is modulo operation)
    new_phases = new_phases%1

    # Save first phase -> shape(num_samples, num_legs)
    new_phase = new_phases[..., 0]

    # Make comparaison to return discret contat sequence : c = 1 if phase < d, 0 otherwise
    # (samples, legs, sampling_horizon) <= (samples, legs, 1) -> shape(num_samples, num_legs, sampling_horizon)
    c = new_phases <= d.unsqueeze(-1)

    return c, new_phase





