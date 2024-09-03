import gymnasium as gym

from . import agents, aliengo_base_env_cfg, aliengo_rough_env_cfg, aliengo_speed_env_cfg, aliengo_climb_env_cfg, aliengo_test_env_cfg 


##
# Register Gym environments.
##
gym.register(
    id="Isaac-Model-Based-Base-Aliengo-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True, #True
    kwargs={
        "env_cfg_entry_point": aliengo_base_env_cfg.UnitreeAliengoBaseEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.UnitreeAliengoBasePPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Model-Based-Rough-Aliengo-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True, #True
    kwargs={
        "env_cfg_entry_point": aliengo_rough_env_cfg.UnitreeAliengoRoughEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.UnitreeAliengoRoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Model-Based-Speed-Aliengo-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True, #True
    kwargs={
        "env_cfg_entry_point": aliengo_speed_env_cfg.UnitreeAliengoSpeedEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.UnitreeAliengoSpeedPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Model-Based-Climb-Aliengo-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True, #True
    kwargs={
        "env_cfg_entry_point": aliengo_climb_env_cfg.UnitreeAliengoClimbEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.UnitreeAliengoClimbPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Model-Based-Test-Aliengo-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True, #True
    kwargs={
        "env_cfg_entry_point": aliengo_test_env_cfg.UnitreeAliengoTestEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.UnitreeAliengoBasePPORunnerCfg,
    },
)