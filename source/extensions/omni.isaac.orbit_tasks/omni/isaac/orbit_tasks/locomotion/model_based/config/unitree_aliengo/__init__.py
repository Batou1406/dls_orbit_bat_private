import gymnasium as gym

# from . import agents, flat_env_cfg, rough_env_cfg


##
# Register Gym environments.
##

# gym.register(
#     id="Isaac-Velocity-Flat-Unitree-Aliengo-v0",
#     entry_point="omni.isaac.orbit.envs:RLTaskEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": flat_env_cfg.UnitreeAliengoFlatEnvCfg,
#         "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.UnitreeAliengoFlatPPORunnerCfg,
#     },
# )