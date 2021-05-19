import glob
import random
import pathlib
from datetime import datetime
import pickle as pkl

def build_dataset(path_str):
    agent_folders = glob.glob(path_str)
    out_dict = {}
    total_f = 0
    for agent_folder in agent_folders:
        agent_name = agent_folder.split('/')[-1]
        agent_id = int(agent_name.split('_')[0]) - 1
        files = glob.glob(f'{agent_folder}/*')
        num_files = 0
        #if not agent_id in dict_class_agent:
        #    dict_class[agent_id] = []
        files = [f for f in files if 'result' not in f.split('/')[-1]] 
        #dict_class[agent_id] = files
        for it in range(len(files)):
            file = files[it]
            other_files = random.sample(files, 3)
            out_dict[file] = [agent_id, [file] + other_files]
            num_files += 1

        total_f += num_files
        print(f'{agent_id}: {num_files}')
    print(f'Total: {total_f}')
    print('----')
    return out_dict
    
if __name__ == '__main__':
    # arguments = get_args_pref_agent()
    # with open(arguments.config, 'r') as f:
    #     config = yaml.load(f)


    now = datetime.now() # current date and time
    dir_script = pathlib.Path(__file__).parent.absolute()

    dataset_name = 'dataset_agent_belief_v2_paired'
    train_path = f'{dir_script}/../../data_scratch/large_data_touch_v2/train_env_task_set_20_full_reduced_tasks1to3/*v2_modeinfo'
    test_path = f'{dir_script}/../../data_scratch/large_data_touch_v2/test_env_task_set_10_full_reduced_tasks1to3/*v2_modeinfo'

    dict_train = build_dataset(train_path)
    print("Building...")
    dict_test = build_dataset(test_path)
    dataset_dict = {}
    dataset_dict['train'] = dict_train
    dataset_dict['test'] = dict_test
    dataset_dict['generated'] = date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    with open(f'{dir_script}/../../dataset/{dataset_name}_train.pkl', 'wb+') as f:
        pkl.dump(dict_train, f)

    with open(f'{dir_script}/../../dataset/{dataset_name}_test.pkl', 'wb+') as f:
        pkl.dump(dict_test, f)

    ipdb.set_trace()
