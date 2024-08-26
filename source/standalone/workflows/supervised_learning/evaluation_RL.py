import subprocess
import os

import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Process some folder paths.")
parser.add_argument('--num_envs', type=str, required=False, default=4096)
parser.add_argument('--model_folder', type=str, required=True, help='Path to the evaluation folder')
parser.add_argument('--result_folder', type=str, required=True, help='Path to the evaluation folder')

# Parse arguments
args = parser.parse_args()

num_envs = args.num_envs
model_folder = args.model_folder
result_folder = args.result_folder

args_dict = {
    '--num_envs': num_envs,
    '--headless': None,
    # '--num_trajectory': '8192',
    '--num_trajectory': '24000',
    '--multipolicies_folder': 'test_eval',
    '--result_folder': result_folder,
    '--eval_task':'eval_task',
    '--model_name': 'alo'
}

# eval_task_list = ['debug','omnidirectionnal_test', 'stair_test', 'base_test', 'speed_test','survival_test']
eval_task_list = ['speed_test', 'survival_test', 'base_test']

list_of_policy_folder = [f"{model_folder}/{name}" for name in os.listdir(f"model/{model_folder}") if os.path.isdir(f"model/{model_folder}/{name}")]
list_of_policy_name   = [f"{name}" for name in os.listdir(f"model/{model_folder}") if os.path.isdir(f"model/{model_folder}/{name}")]

print('Path to policy : ', f"model/{model_folder}")
print('Different policy :', list_of_policy_folder)

# speed_list =['fast', 'medium', 'slow']
# # freq_list = ['no_frequency_optimization', 'frequency_optimization']
# # duty_cycle_list = ['no_duty_cycle_optimization', 'duty_cycle_optimization']
# freq_list = ['no_frequency_optimization']
# duty_cycle_list = ['no_duty_cycle_optimization']

iter=0
for t in range(len(eval_task_list)):
    # for k in range(len(speed_list)):
    #     for j in range(len(freq_list)):
    #         for l in range(len(duty_cycle_list)):
                        for i in range(len(list_of_policy_folder)):

                            iter+=1
                            print(f"\nEvaluation {iter} / {len(list_of_policy_folder)*len(eval_task_list)} - Policy {list_of_policy_folder[i]}")

                            args_dict['--multipolicies_folder'] = list_of_policy_folder[i]         
                            args_dict['--eval_task'] = eval_task_list[t]
                            args_dict['--model_name'] = list_of_policy_name[i]

                            # args_dict['--speed'] = speed_list[k]
                            # args_dict['--f_opt'] = freq_list[j]
                            # args_dict['--d_opt'] = duty_cycle_list[l]
                            # if 'RL' in list_of_policy_folder[i] and freq_list[j]       == 'frequency_optimization' : continue
                            # if 'RL' in list_of_policy_folder[i] and duty_cycle_list[l] == 'duty_cycle_optimization': continue

                            print(args_dict)

                            # Convert the dictionary to a list of arguments
                            args_list = []
                            for key, value in args_dict.items():
                                args_list.append(key)
                                if value is not None:
                                    args_list.append(str(value))

                            # Run the experiment
                            result = subprocess.run(['python3', './source/standalone/workflows/supervised_learning/play_eval_RL.py'] + args_list)

                            if result.returncode != 0:
                                print('something went wrong')
                                breakpoint()


