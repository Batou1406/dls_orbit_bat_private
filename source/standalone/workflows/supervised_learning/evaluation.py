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

list_of_experiment = ['one', 'two']

for experiment in list_of_experiment:
    args_dict['--experiment'] = experiment

    # Convert the dictionary to a list of arguments
    args_list = []
    for key, value in args_dict.items():
        args_list.append(key)
        if value is not None:
            args_list.append(str(value))

    # Run the experiment
    subprocess.run(['python3', './source/standalone/workflows/supervised_learning/play_eval.py'] + args_list)
