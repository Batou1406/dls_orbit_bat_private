import subprocess
import gymnasium as gym

from omni.isaac.lab_tasks.manager_based.locomotion.model_based.config.unitree_aliengo import agents
from omni.isaac.lab.utils import configclass
from omni.isaac.lab_tasks.manager_based.locomotion.model_based.model_based_env_cfg import LocomotionModelBasedEnvCfg
from omni.isaac.lab_assets.unitree import UNITREE_ALIENGO_SELF_COLLISION_TORQUE_CONTROL_CFG  # isort: skip
from omni.isaac.lab.terrains.config.niceFlat import COBBLESTONE_ROAD_CFG, COBBLESTONE_FLAT_CFG
from omni.isaac.lab.terrains.config.climb import STAIRS_TERRAINS_CFG
from omni.isaac.lab.terrains.config.speed import SPEED_TERRAINS_CFG
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG
from omni.isaac.lab_tasks.manager_based.locomotion.model_based.mdp import modify_reward_weight
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.terrains import randomTerrainImporter
import omni.isaac.lab_tasks.manager_based.locomotion.model_based.mdp as mdp



# Define your arguments as a dictionary
args_dict = {
    '--task': 'Isaac-Model-Based-Base-Aliengo-v0',
    '--num_envs': '32',
    '--headless': None,  # For flags or options without values
    '--num_steps': '1000',
    '--multipolicies_folder': 'test_eval',
    '--experiment_folder': 'eval_2',
    '--experiment': 'alo',
}

list_of_experiment = ['one', 'two']
list_of_encoding = ['discrete', 'cubic_spline']








for i in range(len(list_of_experiment)):
    args_dict['--experiment'] = list_of_experiment[i]

    @configclass
    class ActionsCfg:
        """Action specifications for the MDP.
        - Robot joint position - dim=12
        """
        model_base_variable = mdp.ModelBaseActionCfg(
            asset_name="robot",
            joint_names=[".*"], 
            controller=mdp.samplingController,
            optimizerCfg=mdp.ModelBaseActionCfg.OptimizerCfg(
                multipolicy=1,
                prevision_horizon=15,
                discretization_time=0.02,
                parametrization_p='first',
                parametrization_F=list_of_encoding[i]
                ),
            )

    @configclass
    class env_cfg(LocomotionModelBasedEnvCfg):
        actions = ActionsCfg()
        
        def __post_init__(self):

            """ ----- Scene Settings ----- """
            self.scene.robot = UNITREE_ALIENGO_SELF_COLLISION_TORQUE_CONTROL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"  

            self.scene.terrain.terrain_generator = COBBLESTONE_FLAT_CFG # very Flat
            # self.scene.terrain.terrain_generator = COBBLESTONE_ROAD_CFG # Flat
            # self.scene.terrain.terrain_generator = STAIRS_TERRAINS_CFG
            # self.scene.terrain.terrain_generator = SPEED_TERRAINS_CFG
            # self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG    
            self.scene.terrain.class_type = randomTerrainImporter      

            # post init of parent
            super().__post_init__()

    gym.register(
        id=list_of_experiment[i],
        entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
        disable_env_checker=True, #True
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.UnitreeAliengoBasePPORunnerCfg,
        },
    )

    args_dict['--task'] = list_of_experiment[i]



    # Convert the dictionary to a list of arguments
    args_list = []
    for key, value in args_dict.items():
        args_list.append(key)
        if value is not None:
            args_list.append(str(value))

    # Run the experiment
    subprocess.run(['python3', './source/standalone/workflows/supervised_learning/play_eval.py'] + args_list)
