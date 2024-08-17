import subprocess
import os

import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Process some folder paths.")
parser.add_argument('--model_folder', type=str, required=True, help='Path to the evaluation folder')
parser.add_argument('--result_folder', type=str, required=True, help='Path to the evaluation folder')

# Parse arguments
args = parser.parse_args()

model_folder = args.model_folder
result_folder = args.result_folder


args_dict = {
    '--num_envs': '1',
    '--headless': None,  # For flags or options without values
    '--num_steps': '30000',
    '--multipolicies_folder': 'test_eval',
    '--experiment_folder': result_folder,
    '--num_samples': 0,
    '--controller': 'samplingController',
    '--leg_freq': 'no_opt',
    '--duty_cycle': 'no_opt',
    '--speed':'fast',
}

controller_list = ['samplingController', 'samplingController_no_warm_start']
# controller_list = ['samplingController_no_warm_start']
# num_samples_list = [4000, 10000, 25000]
num_samples_list = [10000]

f_list = ['frequency_optimization', 'no_opt']
d_list = ['duty_cycle_optimization', 'no_opt']
speed_list = ['fast', 'medium', 'slow']

list_of_policy_folder = [f"{model_folder}/{name}" for name in os.listdir(f"model/{model_folder}") if os.path.isdir(f"model/{model_folder}/{name}")]

print('Path to policy : ', f"model/{model_folder}")
print('Different policy :', list_of_policy_folder)

iter=0
for s in range(len(speed_list)) :
    for d in range(len(d_list)) :
        for f in range(len(f_list)) :
            for k in range(len(controller_list)):
                for j in range(len(num_samples_list)):
                    for i in range(len(list_of_policy_folder)):
                        iter+=1
                        print(f"\nEvaluation {iter} / {len(list_of_policy_folder)*len(num_samples_list)*len(controller_list)*len(d_list)*len(f_list)*len(speed_list)} - Policy {list_of_policy_folder[i]}")

                        args_dict['--multipolicies_folder'] = list_of_policy_folder[i]
                        args_dict['--num_samples'] = num_samples_list[j]
                        args_dict['--controller'] = controller_list[k]

                        args_dict['--leg_freq'] = f_list[f]
                        args_dict['--duty_cylce'] = d_list[d]
                        args_dict['--speed'] = speed_list[s]

                        # Convert the dictionary to a list of arguments
                        args_list = []
                        for key, value in args_dict.items():
                            args_list.append(key)
                            if value is not None:
                                args_list.append(str(value))

                        # Run the experiment
                        subprocess.run(['python3', './source/standalone/workflows/supervised_learning/play_eval.py'] + args_list)




