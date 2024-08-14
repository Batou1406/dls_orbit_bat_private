import subprocess

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

# Convert the dictionary to a list of arguments
args_list = []
for key, value in args_dict.items():
    args_list.append(key)
    if value is not None:
        args_list.append(str(value))

# Run the second script with the arguments
subprocess.run(['python3', 'play_eval.py'] + args_list)
