import glob
from tqdm import tqdm
import pickle as pkl
from matplotlib import pyplot as plt
from multiprocessing import Pool
from collections import Counter

def get_prog_info(prog_info):
    html_str = ''
    goal_str = ', '.join(['{}: {}'.format(goal_name, count) for goal_name, count in prog_info['goals'].items() if count > 0])
    total_goal = sum(prog_info['goals'].values())
    html_str += "<a> Goal {}</a><br>".format(goal_str)
    html_str += "<a> Total {}</a><br>".format(total_goal)
    html_str += '<ol>'
    for action in prog_info['script']:
        instr_str = action.replace('<', '&lt').replace('>', '&gt')
        html_str += '<li> {} </li>'.format(instr_str)
    html_str += '</ol>'
    return [html_str]

def get_info_file(file_name):
    with open(file_name, 'rb') as f:
        aux = pkl.load(f)
    
    if 'action' not in aux:
        print('error in', file_name)
        return {}
    num_goal_finished = sum([len(val) for key, val in aux['goals_finished'][-1].items()])
    num_goal_total = sum([val for key, val in aux['goals'][0].items()])

    length_action = len(aux['action'][0])
    data_info = {
        'goals': aux['goals'][0],
        'length': length_action,
        'script': aux['action'][0],
        'finished': num_goal_finished,
        'total': num_goal_total
    }
    return data_info

def make_html(folder, html_file):
    data_folder = folder
    agents = glob.glob('{}/*'.format(data_folder))
    agent_files = {}
    all_files = []
    print(len(agents))
    for agent in agents:
        agent_name = agent.split('/')[-1]
        if agent_name[0] != '4':
            continue
        files = [filen for filen in glob.glob('{}/*'.format(agent)) if 'results_' not in filen]
        agent_files[agent_name] = files
        all_files += files
    
    
    data_info = {}
    with Pool(40) as p:
        info = p.map(get_info_file, all_files)
        for file_name, info_item in zip(all_files, info):
            data_info[file_name] = info_item

    html_str = []
    html_str.append('<html>')
    for file_c in tqdm(data_info):
        html_str += '<a> {}  </a>'.format(file_c)
        prog_file = get_prog_info(data_info[file_c])
        html_str += prog_file

    html_str.append('</html>')
    with open(html_file, 'w+') as f:
        f.writelines(html_str)
    

if __name__ == '__main__':
    root = '../../data_scratch/large_data/'
    #path_train = root + 'train_env_task_set_10_full_reduced_tasks/' 
    #path_test = root + 'test_env_task_set_10_full_reduced_tasks/'
    path_test = root + 'train_env_task_set_20_full_reduced_tasks_single/'
    #make_html(path_train+'1_full_opencost0_closecostFalse_walkcost0.05_forgetrate0v9_particles_v2', 'train_data.html')
    make_html(path_test, 'train_data_red_agentpobs.html')

