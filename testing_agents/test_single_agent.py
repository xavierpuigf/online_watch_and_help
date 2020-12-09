import sys
import os
import ipdb
import pickle
import json
import random
import numpy as np
from pathlib import Path

from envs.unity_environment import UnityEnvironment
from agents import MCTS_agent, MCTS_agent_particle
from arguments import get_args
from algos.arena_mp2 import ArenaMP
from utils import utils_goals



def get_class_mode(agent_args):
    mode_str = '{}_opencost{}_closecost{}_forgetrate{}'.format(
        agent_args['obs_type'], 
        agent_args['should_close'], 
        agent_args['open_cost'], 
        agent_args['belief']['forget_rate'])
    return mode_str

if __name__ == '__main__':
    args = get_args()

    num_tries = 1
    args.max_episode_length = 250
    args.num_per_apartment = 20
    args.dataset_path = './dataset/env_task_set_10_full.pik'


    args.obs_type = 'partial'
    open_cost = 0
    should_close = False
    forget_rate = 0.
    datafile = args.dataset_path.split('/')[-1].replace('.pik', '')
    agent_args = {
        'obs_type': args.obs_type,
        'open_cost': open_cost,
        'should_close': should_close,
        'belief': {'forget_rate': forget_rate}
    }
    args.mode = get_class_mode(agent_args)
    args.mode += 'v3'

    
    env_task_set = pickle.load(open(args.dataset_path, 'rb'))
    print(len(env_task_set))

    for env in env_task_set:
        init_gr = env['init_graph']
        gbg_can = [node['id'] for node in init_gr['nodes'] if node['class_name'] in ['garbagecan', 'clothespile']]
        init_gr['nodes'] = [node for node in init_gr['nodes'] if node['id'] not in gbg_can]
        init_gr['edges'] = [edge for edge in init_gr['edges'] if edge['from_id'] not in gbg_can and edge['to_id'] not in gbg_can]

    args.record_dir = '../test_results_single_episode/{}/{}'.format(datafile, args.mode)
    if not os.path.exists(args.record_dir):
        os.makedirs(args.record_dir)

    executable_args = {
                    'file_name': args.executable_file,
                    'x_display': 0,
                    'no_graphics': True
    }

    id_run = 0
    random.seed(id_run)
    episode_ids = list(range(len(env_task_set)))
    episode_ids = sorted(episode_ids)

    S = [[] for _ in range(len(episode_ids))]
    L = [[] for _ in range(len(episode_ids))]
    
    test_results = {}
    episode_ids = [episode_ids[0]]
 
    def env_fn(env_id):
        return UnityEnvironment(num_agents=1,
                                max_episode_length=args.max_episode_length,
                                port_id=env_id,
                                env_task_set=env_task_set,
                                observation_types=[args.obs_type],
                                use_editor=args.use_editor,
                                executable_args=executable_args,
                                base_port=args.base_port)


    args_common = dict(recursive=False,
                         max_episode_length=20,
                         num_simulation=100,
                         max_rollout_steps=20,
                         c_init=0.1,
                         c_base=1000000,
                         num_samples=1,
                         num_processes=1,
                         logging=True,
                         logging_graphs=True)

    args_agent1 = {'agent_id': 1, 'char_index': 0}
    args_agent1.update(args_common)
    args_agent1['agent_params'] = agent_args
    agents = [lambda x, y: MCTS_agent_particle(**args_agent1)]
    arena = ArenaMP(args.max_episode_length, id_run, env_fn, agents)

    for iter_id in range(num_tries):
        #if iter_id > 0:

        cnt = 0
        steps_list, failed_tasks = [], []
        current_tried = iter_id

        if not os.path.isfile(args.record_dir + '/results_{}.pik'.format(0)):
            test_results = {}
        else:
            test_results = pickle.load(open(args.record_dir + '/results_{}.pik'.format(0), 'rb'))
        
        for episode_id in episode_ids:
            if episode_id in [2, 6, 7, 12, 17, 20]:
                continue
            curr_log_file_name = args.record_dir + '/logs_agent_{}_{}_{}.pik'.format(
            env_task_set[episode_id]['task_id'],
            env_task_set[episode_id]['task_name'],
            iter_id)


            print('episode:', episode_id)

            for it_agent, agent in enumerate(arena.agents):
                agent.seed = it_agent + current_tried * 2

            
            if True:
                
                arena.reset(episode_id)
                success, steps, saved_info = arena.run()
                print('-------------------------------------')
                print('success' if success else 'failure')
                print('steps:', steps)
                print('-------------------------------------')
                if not success:
                    failed_tasks.append(episode_id)
                else:
                    steps_list.append(steps)
                is_finished = 1 if success else 0

                Path(args.record_dir).mkdir(parents=True, exist_ok=True)
                log_file_name = args.record_dir + '/logs_episode.{}_iter.{}.pik'.format(episode_id, iter_id)
                if len(saved_info['obs']) > 0:
                    pickle.dump(saved_info, open(log_file_name, 'wb'))
                else:
                    with open(log_file_name, 'w+') as f:
                        f.write(json.dumps(saved_info, indent=4))
            else:
                ipdb.set_trace()
                arena.reset_env()

            S[episode_id].append(is_finished)
            L[episode_id].append(steps)
            test_results[episode_id] = {'S': S[episode_id],
                                        'L': L[episode_id]}
                                        
        pickle.dump(test_results, open(args.record_dir + '/results_{}.pik'.format(0), 'wb'))
        print('average steps (finishing the tasks):', np.array(steps_list).mean() if len(steps_list) > 0 else None)
        print('failed_tasks:', failed_tasks)
        pickle.dump(test_results, open(args.record_dir + '/results_{}.pik'.format(0), 'wb'))

