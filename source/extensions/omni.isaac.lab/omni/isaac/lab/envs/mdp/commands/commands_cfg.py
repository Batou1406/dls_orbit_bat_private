# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

from omni.isaac.lab.managers import CommandTermCfg
from omni.isaac.lab.utils import configclass

from .null_command import NullCommand
from .pose_2d_command import TerrainBasedPose2dCommand, UniformPose2dCommand
from .pose_command import UniformPoseCommand
from .velocity_command import NormalVelocityCommand, UniformVelocityCommand, CurriculumUniformVelocityCommand, CurriculumNormalVelocityCommand


@configclass
class NullCommandCfg(CommandTermCfg):
    """Configuration for the null command generator."""

    class_type: type = NullCommand

    def __post_init__(self):
        """Post initialization."""
        # set the resampling time range to infinity to avoid resampling
        self.resampling_time_range = (math.inf, math.inf)


@configclass
class UniformVelocityCommandCfg(CommandTermCfg):
    """Configuration for the uniform velocity command generator."""

    class_type: type = UniformVelocityCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    heading_command: bool = MISSING
    """Whether to use heading command or angular velocity command.

    If True, the angular velocity command is computed from the heading error, where the
    target heading is sampled uniformly from provided range. Otherwise, the angular velocity
    command is sampled uniformly from provided range.
    """
    heading_control_stiffness: float = MISSING
    """Scale factor to convert the heading error to angular velocity command."""
    rel_standing_envs: float = MISSING
    """Probability threshold for environments where the robots that are standing still."""
    rel_heading_envs: float = MISSING
    """Probability threshold for environments where the robots follow the heading-based angular velocity command
    (the others follow the sampled angular velocity command)."""

    @configclass
    class Ranges:
        """Uniform distribution ranges for the velocity commands."""

        lin_vel_x: tuple[float, float] = MISSING  # min max [m/s]
        lin_vel_y: tuple[float, float] = MISSING  # min max [m/s]
        ang_vel_z: tuple[float, float] = MISSING  # min max [rad/s]
        heading: tuple[float, float] = MISSING  # min max [rad]

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands."""


@configclass
class NormalVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for the normal velocity command generator."""

    class_type: type = NormalVelocityCommand
    heading_command: bool = False  # --> we don't use heading command for normal velocity command.

    @configclass
    class Ranges:
        """Normal distribution ranges for the velocity commands."""

        mean_vel: tuple[float, float, float] = MISSING
        """Mean velocity for the normal distribution.

        The tuple contains the mean linear-x, linear-y, and angular-z velocity.
        """
        std_vel: tuple[float, float, float] = MISSING
        """Standard deviation for the normal distribution.

        The tuple contains the standard deviation linear-x, linear-y, and angular-z velocity.
        """
        zero_prob: tuple[float, float, float] = MISSING
        """Probability of zero velocity for the normal distribution.

        The tuple contains the probability of zero linear-x, linear-y, and angular-z velocity.
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands."""


@configclass
class CurriculumUniformVelocityCommandCfg(CommandTermCfg):
    """Configuration for the curriculum velocity command generator."""

    class_type: type = CurriculumUniformVelocityCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    
    heading_control_stiffness: float = MISSING
    """Scale factor to convert the heading error to angular velocity command."""

    speed_threshold: float = 0.1
    """ Threshold under which a sample velocity (in xy plane) is clamp to zero"""

    initial_difficulty: float = 0.0
    """The initial difficulty for the sampled speed in [0,1]"""

    minmum_difficulty: float = 0.2
    """The minimum difficulty for the sampled speed in [0,1], if set to 1.0 -> no curriculum"""

    difficulty_scaling: float = 0.1
    """Speed of progression when a environment progress (or regress) in the curriculum (in R+) if set to 0.0 -> no curriculum
    (1/difficulty_scaling) is the number of time every environment must progress in the curriculum to increase the
    the difficulty from 0 to 1. """

    @configclass
    class Ranges:
        """Uniform distribution ranges for the velocity commands.
        Initial heading allow the robots to turn faster at the begining when the heading error is large.

        Properties :
            for_vel_b           : Commanded forward velocity of the robot (ie x direction in base frame)                : (min, max) [m/s] 
            lat_vel_b           : Commanded lateral velocity of the robot (ie y direction in base frame)                : (min, max) [m/s] 
            ang_vel_b           : Commanded angular velocity (in the z direction) the robot should follow  (omega_z)    : (min, max) [rad/s] 
            initial_heading_err : The initial heading error (ie. misalignment between heading and target heading)       : (min, max) [rad] in [0, 2pi]
        """
        for_vel_b: tuple[float, float] = MISSING           # (min, max) [m/s] 
        lat_vel_b: tuple[float, float] = MISSING           # (min, max) [m/s] 
        ang_vel_b: tuple[float, float] = MISSING           # (min, max) [rad/s] 
        initial_heading_err: tuple[float, float] = MISSING # (min, max) [rad] in [0, 2pi] 

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands."""

@configclass
class CurriculumNormalVelocityCommandCfg(CurriculumUniformVelocityCommandCfg):
    class_type: type = CurriculumNormalVelocityCommand

    std: float = 0.5
    """Standard Deviation for the Normal Sampling Law"""


@configclass
class UniformPoseCommandCfg(CommandTermCfg):
    """Configuration for uniform pose command generator."""

    class_type: type = UniformPoseCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    body_name: str = MISSING
    """Name of the body in the asset for which the commands are generated."""

    make_quat_unique: bool = False
    """Whether to make the quaternion unique or not. Defaults to False.

    If True, the quaternion is made unique by ensuring the real part is positive.
    """

    @configclass
    class Ranges:
        """Uniform distribution ranges for the pose commands."""

        pos_x: tuple[float, float] = MISSING  # min max [m]
        pos_y: tuple[float, float] = MISSING  # min max [m]
        pos_z: tuple[float, float] = MISSING  # min max [m]
        roll: tuple[float, float] = MISSING  # min max [rad]
        pitch: tuple[float, float] = MISSING  # min max [rad]
        yaw: tuple[float, float] = MISSING  # min max [rad]

    ranges: Ranges = MISSING
    """Ranges for the commands."""


@configclass
class UniformPose2dCommandCfg(CommandTermCfg):
    """Configuration for the uniform 2D-pose command generator."""

    class_type: type = UniformPose2dCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    simple_heading: bool = MISSING
    """Whether to use simple heading or not.

    If True, the heading is in the direction of the target position.
    """

    @configclass
    class Ranges:
        """Uniform distribution ranges for the position commands."""

        pos_x: tuple[float, float] = MISSING
        """Range for the x position (in m)."""
        pos_y: tuple[float, float] = MISSING
        """Range for the y position (in m)."""
        heading: tuple[float, float] = MISSING
        """Heading range for the position commands (in rad).

        Used only if :attr:`simple_heading` is False.
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the position commands."""


@configclass
class TerrainBasedPose2dCommandCfg(UniformPose2dCommandCfg):
    """Configuration for the terrain-based position command generator."""

    class_type = TerrainBasedPose2dCommand

    @configclass
    class Ranges:
        """Uniform distribution ranges for the position commands."""

        heading: tuple[float, float] = MISSING
        """Heading range for the position commands (in rad).

        Used only if :attr:`simple_heading` is False.
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the sampled commands."""
