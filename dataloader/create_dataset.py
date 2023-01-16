import glob
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
        for file in files:
            if 'result' not in file.split('/')[-1]:
                out_dict[file] = agent_id
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

    dataset_name = 'dataset_graph_pred'
    
    train_path = f'{dir_script}/../dataset_episodes/large_data_toy/train_env_task_set_100_full/*'
    test_path =  f'{dir_script}/../dataset_episodes/large_data_toy/test_env_task_set_10_full/*'

    dict_train = build_dataset(train_path)
    print("Building...")
    dict_test = build_dataset(test_path)
    dataset_dict = {}
    dataset_dict['train'] = dict_train
    dataset_dict['test'] = dict_test
    dataset_dict['generated'] = date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    with open(f'{dir_script}/../dataset/{dataset_name}_train.pkl', 'wb+') as f:
       pkl.dump(dict_train, f)

    with open(f'{dir_script}/../dataset/{dataset_name}_test.pkl', 'wb+') as f:
       pkl.dump(dict_test, f)

    ipdb.set_trace()
