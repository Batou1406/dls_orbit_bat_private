import gymnasium as gym

from . import agents, aliengo_base_env_cfg


##
# Register Gym environments.
##
gym.register(
    id="Isaac-Model-Based-Base-Aliengo-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True, #True
    kwargs={
        "env_cfg_entry_point": aliengo_base_env_cfg.UnitreeAliengoBaseEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.UnitreeAliengoBasePPORunnerCfg,
    },
)