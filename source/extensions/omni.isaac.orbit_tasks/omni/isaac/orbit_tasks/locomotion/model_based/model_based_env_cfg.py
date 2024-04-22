# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
""" Create amd initialise managers for the ModelBasedEnvCfg class
    --- Managers ---
    - MySceneCfg
    - CommandsCfg
    - ActionsCfg
    - ObservationsCfg
    - EventCfg
    - RewardsCfg
    - TerminationsCfg
    - CurriculumCfg

    --- Environment ---
    - LocomotionModelBasedEnvCfg
"""

from __future__ import annotations

import math
from dataclasses import MISSING

from omni.isaac.orbit.controllers.differential_ik_cfg import DifferentialIKControllerCfg
import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.orbit.envs import RLTaskEnvCfg
from omni.isaac.orbit.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.orbit.managers import EventTermCfg as EventTerm
from omni.isaac.orbit.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.orbit.managers import ObservationTermCfg as ObsTerm
from omni.isaac.orbit.managers import RewardTermCfg as RewTerm
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.managers import TerminationTermCfg as DoneTerm
from omni.isaac.orbit.scene import InteractiveSceneCfg
from omni.isaac.orbit.sensors import ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.orbit.terrains import TerrainImporterCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.noise import AdditiveUniformNoiseCfg as Unoise

# TODO re-Implements these
import omni.isaac.orbit_tasks.locomotion.model_based.mdp as mdp

# Local MDP
from .mdp.actions import model_base_controller
# import .mdp.observations as local_mdp
from .mdp.observations import leg_phase


##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane", # "generator"
        terrain_generator=None, # ROUGH_TERRAINS_CFG
        max_init_terrain_level=None, #5
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )

    # robots
    robot: ArticulationCfg = MISSING

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP.
    - base_velocity tuple[float, float, float, float] : (lin_vel_x, lin_vel_y, angl_vel_z, heading)
    """

    # train the robot to follow a velocity command with arbitrary velocity, direction, yaw rate and heading
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP.
    - Robot joint position - dim=12
    """
    model_base_variable = mdp.ModelBaseActionCfg(asset_name="robot", joint_names=[".*"], controller=model_base_controller.samplingController())

    # joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)
    # RR_foot = mdp.DifferentialInverseKinematicsActionCfg(asset_name="robot", joint_names=["RR.*"], body_name="RR_foot", controller=DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls"))
    # RL_foot = mdp.DifferentialInverseKinematicsActionCfg(asset_name="robot", joint_names=["RL.*"], body_name="RL_foot", controller=DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls"))
    # FR_foot = mdp.DifferentialInverseKinematicsActionCfg(asset_name="robot", joint_names=["FR.*"], body_name="FR_foot", controller=DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls"))
    # FL_foot = mdp.DifferentialInverseKinematicsActionCfg(asset_name="robot", joint_names=["FL.*"], body_name="FL_foot", controller=DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls"))



@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group.
        - Robot base linear velocity    - uniform noise ±0.1    - dim=3 
        - Robot base angular velocity   - uniform noise ±0.2    - dim=3
        - Gravity proj. on robot base   - uniform noise ±0.05   - dim=3
        - Robot base velocity commands  - no noise              - dim=4
        - Robot joint position          - uniform noise ±0.01   - dim=12
        - Robot joint velocity          - uniform noise ±1.5    - dim=12
        - Last action term (joint pos)  - no noise              - dim=12
        - height scan                   - uniform noise ±0.1    - dim=...
        """

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)#, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)#, noise=Unoise(n_min=-0.2, n_max=0.2))
        root_quat_w = ObsTerm(func=mdp.root_quat_w)
        # projected_gravity = ObsTerm(
        #     func=mdp.projected_gravity,
        #     noise=Unoise(n_min=-0.05, n_max=0.05),
        # )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)#, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)#, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        # height_scan = ObsTerm(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        #     clip=(-1.0, 1.0),
        # )

        # Model Based internal variable
        # leg_phase = ObsTerm(func=leg_phase, params={"action_name": "model_base_variable"})

        def __post_init__(self):
            # self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events.
    --- Startup ---
    - physics_material : assign random static and dynamic friction coef. to sim. materials - default no noise
    - add_base_mass : add a random mass to robot's body - uniform ±5

    --- Reset ---
    - base_external_force_torque : Set of random F(3) and T(3) to apply to robot base - default 0  
    - reset_base : reset robot body pose and velociy (x,y,yaw only) - uniform noise
    - reset_robot_joints : Reset robot joint position by rescalling position (vel not enable) given range

    --- Interval ---
    - push_robot : assign base a velocity at random interval - sample uniform vel ±0.5, time 10-15s
    """

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.add_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base"), "mass_range": (-5.0, 5.0)},
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP.
    - track_lin_vel_xy_exp  - track command xy vel      - weight=1.0
    - track_ang_vel_z_exp   - track command z vel       - weight=0.5
    - lin_vel_z_l2          - penalize z vel            - weight=2.0
    - ang_vel_xy_l2         - penalize xz vel           - weight=0.05
    - dof_torques_l2        - penalize joint torque     - weight=0.00001
    - dof_acc_l2            - penalize joint acc.       - weight=0.00000025
    - action_rate_l2        - penalize action diff      - weight=0.01
    - feet_air_time         - reward for long air time  - weight=0.125
    - undesired_contacts    - penalize contact F > tresh- weight=1.0
    - flat_orientation_l2   - penalize non flat pose    - weight=0.0
    - dof_pos_limits        - pen. joint pos > soft lim - weight=0.0
    """

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # track_ang_vel_z_exp = RewTerm(
    #     func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    # )
    # -- penalties
    # lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    # ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    # dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    # action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    # feet_air_time = RewTerm(
    #     func=mdp.feet_air_time,
    #     weight=0.125,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
    #         "command_name": "base_velocity",
    #         "threshold": 0.5,
    #     },
    # )
    # undesired_contacts = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=-1.0,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    # )
    # -- optional penalties
    # flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    # dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)

    # -- Reward for alive time
    is_alive = RewTerm(func=mdp.is_alive, weight=10.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP.
    - Time out - terminate when episode lentgh > max episode length
    - Illegal contact - terminate when contact force sensor > treshhold
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    bad_orientaion = DoneTerm(func=mdp.bad_orientation, params={"limit_angle":30*(3.14/180)})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP.
    - terrain_levels - adapt the terrain difficulty to the performance - default off
    """

    terrain_levels = None   # CurrTerm(func=mdp.terrain_levels_vel)


##
# Environment configuration
##


@configclass
class LocomotionModelBasedEnvCfg(RLTaskEnvCfg):
    """Configuration for the locomotion velocity-tracking environment with model based control.
    - num_envs          : default 4096
    - env_spacing       : default 2.5
    - decimation        : default 4
    - episode_length_s  : default 20.0
    - dt                : default 0.005
    """

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
