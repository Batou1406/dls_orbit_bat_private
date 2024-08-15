import subprocess
import os

import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Process some folder paths.")
parser.add_argument('--eval_folder', type=str, required=True, help='Path to the evaluation folder')

# Parse arguments
args = parser.parse_args()

# Use the eval_folder argument
eval_folder = args.eval_folder


# Define your arguments as a dictionary
args_dict = {
    # '--task': 'Isaac-Model-Based-Base-Aliengo-v0',
    '--num_envs': '512',
    '--headless': None,  # For flags or options without values
    '--num_steps': '1500',
    '--multipolicies_folder': 'test_eval',
    '--experiment_folder': 'dagger_eval_contact_aligned_full_eval',
    # '--experiment': 'alo',
}


# Where the subfolder with the policy are located
# eval_folder = 'Isaac-Model-Based-Base-Aliengo-v0/dagger_eval_contact_aligned_full_eval2'


list_of_policy_folder = [f"{eval_folder}/{name}" for name in os.listdir(f"model/{eval_folder}") if os.path.isdir(f"model/{eval_folder}/{name}")]

print('alo :', list_of_policy_folder)



for i in range(len(list_of_policy_folder)):
    args_dict['--multipolicies_folder'] = list_of_policy_folder[i]

    # Convert the dictionary to a list of arguments
    args_list = []
    for key, value in args_dict.items():
        args_list.append(key)
        if value is not None:
            args_list.append(str(value))

    # Run the experiment
    subprocess.run(['python3', './source/standalone/workflows/supervised_learning/play_eval.py'] + args_list)




