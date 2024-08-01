# Helper file witch mathematical functions
import torch

import matplotlib
matplotlib.use('GTK4Agg')
import matplotlib.pyplot as plt

@torch.jit.script
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
    # assert not torch.any(torch.abs(cos_pitch) < 1e-6), "Numerical instability likely due to pitch angle near +/- 90 degrees."

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


@torch.jit.script
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


@torch.jit.script
def compute_cubic_spline(parameters: torch.Tensor, step: int, horizon: int):
    """ Given a set of spline parameters, and the point in the trajectory return the function value 
    
    Args :
        parameters (Tensor): Spline action parameter      of shape(batch, num_legs, 3, spline_param)              
        step          (int): The point in the curve in [0, horizon]
        horizon       (int): The length of the curve
        
    Returns : 
        actions    (Tensor): Discrete action              of shape(batch, num_legs, 3)
    """
    # Find the point in the curve q in [0,1]
    tau = step/(horizon)        
    q = (tau - 0.0)/(1.0-0.0)
    
    # Compute the spline interpolation parameters
    a =  2*q*q*q - 3*q*q     + 1
    b =    q*q*q - 2*q*q + q
    c = -2*q*q*q + 3*q*q
    d =    q*q*q -   q*q

    # Compute intermediary parameters 
    phi_1 = 0.5*(parameters[...,2]  - (10*parameters[...,0])) # shape (batch, num_legs, 3)
    phi_2 = 0.5*((10*parameters[...,3])  - parameters[...,1]) # shape (batch, num_legs, 3)

    # Compute the spline
    actions = a*parameters[...,1] + b*phi_1 + c*parameters[...,2]  + d*phi_2 # shape (batch, num_legs, 3)

    return actions


@torch.jit.script
def fit_cubic(y: torch.Tensor) -> torch.Tensor:
    """ Minimize the sum of squared error between datapoints y_i and a cubic function parametrized with a,b,c,d 
    ie. -> Fit parameters a,b,c,d  that minimize : min(a,b,c,d) : sum( (y_i - (ax_i^3 + bx_i^2 + cx_i + d) )^2 )
    This problem is solved in closed form with exact solution
    
    Moreover, there is a constraint that d=y_0 -> which correspond to theta_0 = y_0 ie. the spline is exact for the first datapoint

    Then compute theta_-1, theta_0, theta_1, theta_2 parameters that are the cubic Hermite spline parameters 
    (Predictive Sampling : Real-time Behaviour Synthesis with MuJoCo - https://arxiv.org/pdf/2212.00541)

    Time has been imposed to be between [0, 1], with first x_0 = 0

    Args :
        x     (torch.tensor): time of the datapoints (if None linear in [0 1])           of shape (horizon)
        y     (torch.tensor): value of the datapoints           of shape (batch, num_legs, dim_3D, horizon)

    Returns :
        theta (torch.tensor): cubic Hermite spline parameters   of shape (batch, num_legs, 3, 4)
    """
    batch_size, num_legs, dim_3D, horizon = y.shape

    x = torch.linspace(0, 1, steps=horizon, device=y.device)  # shape (horizon)

    # Construct the design matrix X for the remaining points
    # This formulation directly gives theta, not a,b,c,d
    X = 0.5*torch.stack([-(x**3) + 2*(x**2) - x, 3*(x**3) - 5*(x**2) + 2, -3*(x**3) + 4*(x**2) + x, (x**3) - (x**2)], dim=-1)  # shape: (horizon, num_param
    num_param = X.shape[-1] # = 4 ie. a,b,c, d
    
    # Construct the target matrix Y 
    Y = y  # shape: (batch_size, num_legs, dim_3D, horizon)

    # Compute the normal equations components for the remaining points
    XtX = torch.einsum('nk,nm->km', X, X)      # (horizon-1, num_param) x (horizon, num_param) -> shape: (num_param, num_param)
    XtY = torch.einsum('nk,bijn->bijk', X, Y)  # (horizon-1, num_param) x (batch_size, num_legs, dim_3D, horizon) -> shape: (batch_size, num_legs, dim_3D, num_param)

    # Expand XtX to match the dimension of XtY for the solver
    XtX.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(batch_size, num_legs, dim_3D, num_param, num_param) # shape (batch_size, num_legs, dim_3D, num_param, num_param)

    # Solve for the remaining coefficients a, b, c
    beta = torch.linalg.solve(XtX, XtY.transpose(-1,-2)).transpose(-1,-2) # shape: (batch_size, num_legs, dim_3D, num_param)

    # Retrieve coefficients a, b, c, d for each batch, legs, dim_3D
    theta_n1 = beta[..., 0]    # shape (batch_size, num_legs, dim_3D)
    theta_0 = beta[..., 1]    # shape (batch_size, num_legs, dim_3D)
    theta_1 = beta[..., 2]    # shape (batch_size, num_legs, dim_3D)
    theta_2 = beta[..., 3]    # shape (batch_size, num_legs, dim_3D)

    # Concatenate the coefficient
    theta = torch.cat((theta_n1.unsqueeze(-1), theta_0.unsqueeze(-1),theta_1.unsqueeze(-1), theta_2.unsqueeze(-1)), dim=-1) # shape (batch_size, num_legs, dim_3D, 4)


    # env_idx = 0
    # leg_idx = 0
    # t = torch.arange(0, 101, device=y.device)
    # F = torch.empty((y.shape[0], y.shape[1], y.shape[2], 101), device=y.device)
    # for i in range(101):
    #     F[:,:,:,i] = compute_cubic_spline(parameters=theta, step=int(t[i]), horizon=100)
    # plt.plot(t.cpu().numpy()/100,F[env_idx,leg_idx,-1,:].cpu().numpy())
    # plt.scatter(x=x.cpu().numpy(), y=y[env_idx,leg_idx,-1,:].cpu().numpy(), c='red')
    # plt.scatter(x=torch.tensor([[-1, 0, 1, 2]]), y=theta[env_idx,leg_idx,-1,:].cpu().numpy())
    # plt.show()

    return theta  # Coefficients a, b, c, d for each batch, legs, dim_3D


@torch.jit.script
def compute_discrete(parameters: torch.Tensor, step: int, horizon: int):
    """ If actions are discrete actions, no interpolation are required.
    This function simply return the action at the right time step

    Args :
        parameters (Tensor): Discrete action parameter    of shape(batch, num_legs, 3, sampling_horizon)
        step          (int): The current step index along horizon
        horizon       (int): Not used : here for compatibility

    Returns :
        actions    (Tensor): Discrete action              of shape(batch, num_legs, 3)
    """

    actions = parameters[:,:,:,step]
    return actions


# Actually faster without the JIT as part of the class
@torch.jit.script
def normal_sampling(num_samples:int, mean:torch.Tensor, std:torch.Tensor|None=None, seed:int=-1, clip:bool=False) -> torch.Tensor:
    """ Normal sampling law given mean and std -> return a samples
    
    Args :
        mean     (Tensor): Mean of normal sampling law          of shape(num_dim1, num_dim2, etc.)
        std      (Tensor): Standard dev of normal sampling law  of shape(num_dim1, num_dim2, etc.)
        num_samples (int): Number of samples to generate
        seed        (int): seed to generate random numbers (-1 for no seed)

    Return :
        samples  (Tensor): Samples generated with mean and std  of shape(num_sammple, num_dim1, num_dim2, etc.)
    """

    # Seed if provided
    if seed == -1: 
        torch.manual_seed(seed)

    if std is None :
        std = torch.ones_like(mean)

    # Sample from a normal law with the provided parameters
    if clip == True :
        samples = mean + (std * torch.randn((num_samples,) + mean.shape, device=mean.device)).clamp(min=-2*std, max=2*std)
    else :
        samples = mean + (std * torch.randn((num_samples,) + mean.shape, device=mean.device))

    return samples


@torch.jit.script
def uniform_sampling(num_samples:int, mean:torch.Tensor, std:torch.Tensor|None=None, seed:int=-1, clip:bool=False) -> torch.Tensor:
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
    if seed == -1: 
        torch.manual_seed(seed)

    if std is None :
        std = torch.ones_like(mean)

    # Sample from a uniform law with the provided parameters
    samples = mean + (std * torch.empty((num_samples,) + mean.shape, device=mean.device).uniform_(-1.0, 1.0))

    return samples


@torch.jit.script
def enforce_friction_cone_constraints_torch(F:torch.Tensor, mu:float, F_z_min:float, F_z_max:float) -> torch.Tensor:
    """ Enforce the friction cone constraints
    ||F_xy|| < F_z*mu
    Args :
        F (torch.Tensor): The GRF                                    of shape(num_samples, num_legs, 3,(optinally F_param))

    Returns :
        F (torch.Tensor): The GRF with enforced friction constraints of shape(num_samples, num_legs, 3,(optinally F_param))
    """

    F_x = F[:,:,0].unsqueeze(2)
    F_y = F[:,:,1].unsqueeze(2)
    F_z = F[:,:,2].unsqueeze(2).clamp(min=F_z_min, max=F_z_max)

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


@torch.jit.script
def from_zero_twopi_to_minuspi_pluspi(roll:torch.Tensor, pitch:torch.Tensor, yaw:torch.Tensor):
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









