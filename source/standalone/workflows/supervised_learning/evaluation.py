import subprocess

# Define your arguments as a dictionary
args_dict = {
    # '--task': 'Isaac-Model-Based-Base-Aliengo-v0',
    '--num_envs': '32',
    '--headless': None,  # For flags or options without values
    '--num_steps': '1000',
    '--multipolicies_folder': 'test_eval',
    '--experiment_folder': 'eval_3',
    # '--experiment': 'alo',
}

list_of_policy_folder = ['eval/experiment1', 'eval/experiment2']






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
