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
    '--num_trajectory': '20000',
    # '--num_trajectory': '50000',
    '--multipolicies_folder': 'test_eval',
    '--result_folder': result_folder,
    '--eval_task':'eval_task',
    '--model_name': 'alo'
}

eval_task_list = ['debug'] #'omnidirectionnal_test', 'stair_test', 'base_test', 'speed_test','survival_test']

list_of_policy_folder = [f"{model_folder}/{name}" for name in os.listdir(f"model/{model_folder}") if os.path.isdir(f"model/{model_folder}/{name}")]
list_of_policy_name   = [f"{name}" for name in os.listdir(f"model/{model_folder}") if os.path.isdir(f"model/{model_folder}/{name}")]

print('Path to policy : ', f"model/{model_folder}")
print('Different policy :', list_of_policy_folder)

iter=0
for t in range(len(eval_task_list)):
                    for j in range(len(['fast', 'medium', 'slow'])):
                        for i in range(len(list_of_policy_folder)):

                            iter+=1
                            print(f"\nEvaluation {iter} / {len(list_of_policy_folder)*len(eval_task_list)} - Policy {list_of_policy_folder[i]}")

                            args_dict['--multipolicies_folder'] = list_of_policy_folder[i]         
                            args_dict['--eval_task'] = eval_task_list[t]
                            args_dict['--model_name'] = list_of_policy_name[i]

                            args_dict['--speed'] = ['fast', 'medium', 'slow'][j]

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


