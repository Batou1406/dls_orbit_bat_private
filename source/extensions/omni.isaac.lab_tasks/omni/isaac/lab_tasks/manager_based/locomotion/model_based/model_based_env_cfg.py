# Copyright (c) 2022-2024, The lab Project Developers.
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

from omni.isaac.lab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import omni.isaac.lab_tasks.manager_based.locomotion.model_based.mdp as mdp

from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

from omni.isaac.lab.utils.assets import LOCAL_NUCLEUS_DIR,ISAAC_NUCLEUS_DIR

##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator", #"plane",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=0, # Initial difficulty of terrain, between [0, number of rows in the terrain (usually 10)]
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
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)), #pos=(0.0, 0.0, 20.0)
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.1, 
            size=(1.6, 1.0),
            direction=(0, 0, -1), # Raycaster is pointing down
            ordering="xy",        # Ordering of the index
            ),
        max_distance=1e6, #[m]
        drift_range=(0.0,0.0),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=False)

    # lights
    # light = AssetBaseCfg(
    #     prim_path="/World/light",
    #     spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    # )
    # sky_light = AssetBaseCfg(
    #     prim_path="/World/skyLight",
    #     spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    # )

    # Spot Light
    sky_light = AssetBaseCfg(
        prim_path="/World/spotLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            # color=(0.81081, 0.44141, 0.44141),
            color=(0.99, 0.8, 0.8),
            # texture_file=f"{LOCAL_NUCLEUS_DIR}/NVIDIA/Assets/Skies/Clear/evening_road_01_4k.hdr",
            texture_file=f"{LOCAL_NUCLEUS_DIR}/Library/skies/kloofendal_43d_clear_puresky_4k.hdr",
            # texture_file=f"{LOCAL_NUCLEUS_DIR}/NVIDIA/Assets/Skies/Clear/evening_road_01_4k.hdr",
        ),
    )

    # colored light
    colored_distant_light = AssetBaseCfg(
        prim_path="/World/coloredLight/distantLight",
        spawn=sim_utils.DistantLightCfg(color=(0.33692, 0.76232, 0.89961), intensity=3000.0, color_temperature=6500, exposure=0.2, angle=5.0),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0),rot=(0.97, 0.26, 0, 0)), #rot=(-0.4581756, 0.8888617, 0, 0)
    )
    colored_distant_light1 = AssetBaseCfg(
        prim_path="/World/coloredLight/distantLight1",
        spawn=sim_utils.DistantLightCfg(color=(0.81081, 0.44141, 0.44141), intensity=3000.0, color_temperature=6500, exposure=0.0, angle=10.0),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0),rot=(0.97, 0.00, 0.26, 0.0)) #rot=(-0.1044025, -0.0466693, -0.9069498, -0.4054185)
    )
    colored_distant_light2 = AssetBaseCfg(
        prim_path="/World/coloredLight/distantLight2",
        spawn=sim_utils.DistantLightCfg(color=(0.89189, 0.55451, 0.28926), intensity=3000.0, color_temperature=6500, exposure=0.0, angle=10.0),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0),rot=(0.93, -0.25, -0.25, 0.0)) #rot=(-0.0457169, 0.0886908, 0.4558891, -0.8844258)
    )

    # # Grey Studio
    # grey_distant_light = AssetBaseCfg(
    #     prim_path="/World/greyStudio/distantLight",
    #     spawn=sim_utils.DistantLightCfg(color=(1.0, 1.0, 1.0), intensity=300.0, color_temperature=6500, angle=34.3, exposure=0.0), #intensity = 3000
    # )
    # grey_dome_light = AssetBaseCfg(
    #     prim_path="/World/greyStudio/domeLight",
    #     spawn=sim_utils.DomeLightCfg(color=(1.0, 1.0, 1.0), intensity=100.0, exposure=0.4, texture_format="latlong"), #intensity = 1000
    # )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP.
    - base_velocity tuple[float, float, float, float] : (for_vel_b, lat_vel_b, ang_vel_b and initial_heading_err)
        It is given in the robot base frame
    """

    # train the robot to follow a velocity command with arbitrary velocity, direction, yaw rate and heading
    base_velocity = mdp.CurriculumUniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(1000.0, 1000.0),
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=mdp.CurriculumUniformVelocityCommandCfg.Ranges(
            for_vel_b=(-0.5,0.5), lat_vel_b=(-0.5, 0.5), ang_vel_b=(-0.5,0.5), initial_heading_err=(-math.pi,math.pi),
        ),
        # These parameters supress the curriculum -> become fairly equivalent to UniformVelocityCommand
        initial_difficulty=1.0,
        minmum_difficulty=1.0,
        difficulty_scaling=0.0,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP.
    - Robot joint position - dim=12
    """
    model_base_variable = mdp.ModelBaseActionCfg(
        asset_name="robot",
        joint_names=[".*"], 
        controller=mdp.samplingController,
        optimizerCfg=mdp.ModelBaseActionCfg.OptimizerCfg(),
        # controller=mdp.modelBaseController,
        )
    
    # joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations Term for policy group (order preserved).
        - Robot base linear velocity   - base frame     - uniform noise ±0.1    - dim=3 
        - Robot base angular velocity  - base frame     - uniform noise ±0.2    - dim=3
        - Robot base height            - world frame    -                       - dim=1
        - Gravity proj. on robot base  - base frame     - uniform noise ±0.05   - dim=3
        - Robot base velocity commands              - no noise              - dim=4
        - Robot joint position                      - uniform noise ±0.01   - dim=12
        - Robot joint velocity                      - uniform noise ±1.5    - dim=12
        - Last action term (joint pos)              - no noise              - dim=12
        - height scan                               - uniform noise ±0.1    - dim=...
        """

        # ---- Robot's pose ----
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))    # Base frame
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))    # Base frame
        # robot_height = ObsTerm(func=mdp.base_pos_z) # World Frame   : works poorly with other terrains than 'plane'
        # root_quat_w = ObsTerm(func=mdp.root_quat_w)
        # root_pos_w = ObsTerm(func=mdp.root_pos_w)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        # ---- Robot's joint variable ----
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))

        # ---- Policy Memory ----
        # actions = ObsTerm(func=mdp.last_action)
        actions = ObsTerm(func=mdp.last_model_base_action,  params={"action_name": "model_base_variable" })

        # ---- Policy Commands ----
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        
        # ---- Exteroceptive sensors
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )
        
        # ---- Model-Base internal variable ----
        leg_phase = ObsTerm(func=mdp.leg_phase, params={"action_name": "model_base_variable"})
        leg_in_contact = ObsTerm(func=mdp.leg_contact, params={"action_name": "model_base_variable"})

        def __post_init__(self):
            # Enable the noise if specified in the ObsTerm
            self.enable_corruption = True

            # Concatenate the obersvations along the last dimension, otherwise kept separeted as a disct
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

    # ----- startup -----
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
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base"), "operation": 'add',"mass_distribution_params": (-0.0, 0.0)},
    )

    # ---- reset ----
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
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # ----- interval -----
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 12.0),
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
    track_lin_vel_xy_exp    = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)})
    track_soft_vel_xy_exp   = RewTerm(func=mdp.soft_track_lin_vel_xy_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.1)})
    track_ang_vel_z_exp     = RewTerm(func=mdp.track_ang_vel_z_exp, weight=0.75, params={"command_name": "base_velocity", "std": math.sqrt(0.25)})
    track_robot_height_exp  = RewTerm(func=mdp.track_proprioceptive_height_exp, weight=0.1, params={"target_height": 0.38, "std": 0.1}) #0.38

    # -- Additionnal penalties : Need a negative weight
    penalty_lin_vel_z_l2    = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    penalty_ang_vel_xy_l2   = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    penalty_dof_torques_l2  = RewTerm(func=mdp.joint_torques_l2, weight=-0.0001)
    penalty_dof_acc_l2      = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-8)
    penalty_action_rate_l2  = RewTerm(func=mdp.action_rate_l2, weight=-0.001)
    undesired_contacts      = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*thigh"), "threshold": 1.0},)
    flat_orientation_l2     = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    dof_pos_limits          = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    penalty_friction        = RewTerm(
        func=mdp.friction_constraint,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot"),
            "mu": 0.55
        }
    )
    penalty_stance_foot_vel = RewTerm(func=mdp.penalize_foot_in_contact_displacement_l2, weight=-1.0)
    penalty_CoT             = RewTerm(func=mdp.penalize_cost_of_transport, weight=-0.1)
    penalty_close_feet      = RewTerm(func=mdp.penalize_close_feet, weight=-1e-3, params={"threshold": 0.05})


    # -- Model based penalty : Positive weight -> penalty is already negative
    penalty_leg_frequency        = RewTerm(func=mdp.penalize_large_leg_frequency_L1,  weight=1.0,  params={"action_name": "model_base_variable", "bound": (0.2,2.0)})
    penalty_leg_duty_cycle       = RewTerm(func=mdp.penalize_large_leg_duty_cycle_L1, weight=2.0,  params={"action_name": "model_base_variable", "bound": (0.3,0.7)})
    penalty_large_force          = RewTerm(func=mdp.penalize_large_Forces_L1,         weight=0.1,  params={"action_name": "model_base_variable", "bound": (0.0,160.0)}) # [N]
    penalty_large_step           = RewTerm(func=mdp.penalize_large_steps_L1,          weight=1.0,  params={"action_name": "model_base_variable", "bound_x": (0.20,-0.05), "bound_y": (0.05,-0.05)})
    penalty_frequency_variation  = RewTerm(func=mdp.penalize_frequency_variation_L2,  weight=1.0,  params={"action_name": "model_base_variable" })
    penatly_duty_cycle_variation = RewTerm(func=mdp.penalize_duty_cycle_variation_L2, weight=2.5,  params={"action_name": "model_base_variable" })
    penalty_step_variation       = RewTerm(func=mdp.penalize_steps_variation_L2,      weight=2.5,  params={"action_name": "model_base_variable" })
    penatly_force_variation      = RewTerm(func=mdp.penalize_Forces_variation_L2,     weight=1e-4, params={"action_name": "model_base_variable" })

    # -- Additionnal Reward : Need a positive weight
    reward_is_alive        = RewTerm(func=mdp.is_alive, weight=0.25)
    penalty_failed         = RewTerm(func=mdp.is_terminated, weight=1.0)


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
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle":45*(3.14/180)})
    # bad_height = DoneTerm(func=mdp.base_height, params={'minimum_height': 0.20})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP.
    - terrain_levels - adapt the terrain difficulty to the performance - default off
    """

    # --- Terrains Curriculum
    terrain_levels = CurrTerm(func=mdp.improved_terrain_levels_vel) # None

    # --- Rewards Curriculum
    # penalty_large_step_curr = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "penalty_large_step", "weight": 1.0, "num_steps": (400*24)})

    # --- Commands Curriculum
    speed_levels = CurrTerm(func=mdp.speed_command_levels_fast_walked_distance, params={'commandTermName': 'base_velocity'})


##
# Environment configuration
##

from omni.isaac.lab.envs.ui.manager_based_rl_env_window import BatManagerBasedRLEnvWindow

@configclass
class LocomotionModelBasedEnvCfg(ManagerBasedRLEnvCfg):
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

    ui_window_class_type: type = BatManagerBasedRLEnvWindow

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 15.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.render_interval = 4 #50Hz render
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

        self.viewer.eye             = (1.5, 1.5, 0.9)
        self.viewer.lookat          = (0.0, 0.0, 0.0)
        self.viewer.cam_prim_path   = "/OmniverseKit_Persp"
        self.viewer.resolution      = (1280, 720)     # 720p
        # self.viewer.resolution      = (1920, 1080)    # 1080p
        # self.viewer.resolution      = (2560, 1440)      # 2k
        # self.viewer.resolution      = (3840, 2160)      # 4k
        # self.viewer.resolution      = (1024, 1024)      # Square
        # self.viewer.resolution      = (2048, 2048)      # 2K Square
        self.viewer.origin_type     = "asset_root"
        self.viewer.env_index       = 0
        self.viewer.asset_name      = "robot"


