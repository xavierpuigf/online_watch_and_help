import sys
import shutil
import os
import logging
import traceback
import os
import ipdb
import pickle
import json
import random
import numpy as np
from pathlib import Path

from envs.unity_environment import UnityEnvironment
from agents import MCTS_agent, MCTS_agent_particle_v2, MCTS_agent_particle
from arguments import get_args
from algos.arena_mp2 import ArenaMP
from utils import utils_goals
from utils import utils_exception



def get_class_mode(agent_args):
    mode_str = '{}_opencost{}_closecost{}_walkcost{}_forgetrate{}'.format(
        agent_args['obs_type'],  
        agent_args['open_cost'],
        agent_args['should_close'], 
        agent_args['walk_cost'],
        agent_args['belief']['forget_rate'])
    return mode_str

if __name__ == '__main__':
    args = get_args()
    num_proc = 0

    num_tries = 5
    args.executable_file = '../path_sim_dev/linux_exec.x86_64'
    args.max_episode_length = 250
    args.num_per_apartment = 20
    args.dataset_path = './dataset/test_env_task_set_10_full_reduced_tasks.pik'

    agent_types = [
            ['full', 0, 0.05, False, 0, "uniform"], # 0
            ['full', 0.5, 0.01, False, 0, "uniform"], # 1
            ['full', -500, 0.05, False, 0, "uniform"], # 2
            ['partial', 0, 0.05, False, 0, "uniform"], # 3
            ['partial', 0, 0.05, False, 0, "spiked"], # 4
            ['partial', 0, 0.05, False, 0.2, "uniform"], # 5
            ['partial', -500, 0.01, False, 0.01, "spiked"], # 6
            ['partial', -500, 0.05, False, 0.2, "uniform"], # 7
            ['partial', 0.5, 0.05, False, 0.2, "uniform"], # 8
    ]
    random_start = random.Random()
    agent_types_index = list(range(9))
    #random.shuffle(agent_types_index)
    if args.agenttype != 'all':
        agent_types_index = [int(x) for x in args.agenttype.split(',')]
    for agent_id in agent_types_index: #len(agent_types)):
        if agent_id in [4]:
            continue
        args.obs_type, open_cost, walk_cost, should_close, forget_rate, belief_type = agent_types[agent_id]
        datafile = args.dataset_path.split('/')[-1].replace('.pik', '')
        agent_args = {
            'obs_type': args.obs_type,
            'open_cost': open_cost,
            'should_close': should_close,
            'walk_cost': walk_cost,
            'belief': {'forget_rate': forget_rate, 'belief_type': belief_type}
        }
        args.mode = '{}_'.format(agent_id+1) + get_class_mode(agent_args)
        args.mode += 'v9_particles_v2'

        
        env_task_set = pickle.load(open(args.dataset_path, 'rb'))
        print(len(env_task_set))

        for env in env_task_set:
            init_gr = env['init_graph']
            gbg_can = [node['id'] for node in init_gr['nodes'] if node['class_name'] in ['garbagecan', 'clothespile']]
            init_gr['nodes'] = [node for node in init_gr['nodes'] if node['id'] not in gbg_can]
            init_gr['edges'] = [edge for edge in init_gr['edges'] if edge['from_id'] not in gbg_can and edge['to_id'] not in gbg_can]
            for node in init_gr['nodes']:
                if node['class_name'] == 'cutleryfork':
                    node['obj_transform']['position'][1] += 0.1

        args.record_dir = '../data_scratch/large_data/{}/{}'.format(datafile, args.mode)
        error_dir = '../data_scratch/large_data/logging/{}_{}'.format(datafile, args.mode)
        if not os.path.exists(args.record_dir):
            os.makedirs(args.record_dir)

        if not os.path.exists(error_dir):
            os.makedirs(error_dir)

        executable_args = {
                        'file_name': args.executable_file,
                        'x_display': 0,
                        'no_graphics': True
        }

        id_run = 0
        #random.seed(id_run)
        episode_ids = list(range(len(env_task_set)))
        episode_ids = sorted(episode_ids)
        random_start.shuffle(episode_ids)
        # episode_ids = episode_ids[10:]

        S = [[] for _ in range(len(episode_ids))]
        L = [[] for _ in range(len(episode_ids))]
        
        test_results = {}
        #episode_ids = [episode_ids[0]]
     
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
                             num_simulation=200,
                             max_rollout_steps=5,
                             c_init=0.1,
                             c_base=1000000,
                             num_samples=1,
                             num_processes=num_proc, 
                             num_particles=1,
                             logging=True,
                             logging_graphs=True)

        args_agent1 = {'agent_id': 1, 'char_index': 0}
        args_agent1.update(args_common)
        args_agent1['agent_params'] = agent_args
        agents = [lambda x, y: MCTS_agent_particle_v2(**args_agent1)]
        arena = ArenaMP(args.max_episode_length, id_run, env_fn, agents)
        episode_ids = episode_ids
        episode_ids = [24]
        for iter_id in range(4, num_tries):
            #if iter_id > 0:

            cnt = 0
            steps_list, failed_tasks = [], []
            current_tried = iter_id

            if not os.path.isfile(args.record_dir + '/results_{}.pik'.format(0)):
                test_results = {}
            else:
                test_results = pickle.load(open(args.record_dir + '/results_{}.pik'.format(0), 'rb'))
            
            logger = logging.getLogger() 
            logger.setLevel(logging.INFO)
            for episode_id in episode_ids:
                #if episode_id == 0:
                #    continue
                #if episode_id in [2, 6, 7, 12, 17, 20]:
                #    continue
                #curr_log_file_name = args.record_dir + '/logs_agent_{}_{}_{}.pik'.format(
                #env_task_set[episode_id]['task_id'],
                #env_task_set[episode_id]['task_name'],
                #iter_id)

                log_file_name = args.record_dir + '/logs_episode.{}_iter.{}.pik'.format(episode_id, iter_id)
                failure_file = '{}/{}_{}.txt'.format(error_dir, episode_id, iter_id)

                # if os.path.isfile(log_file_name):# or os.path.isfile(failure_file):
                    # continue
                if os.path.isfile(failure_file):
                    os.remove(failure_file)
                fileh = logging.FileHandler(failure_file, 'a')
                fileh.setLevel(logging.DEBUG)
                logger.addHandler(fileh)


                print('episode:', episode_id)

                for it_agent, agent in enumerate(arena.agents):
                    agent.seed = (it_agent + current_tried * 2) * 5

                
                try:
                    
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
                    if len(saved_info['obs']) > 0:
                        pickle.dump(saved_info, open(log_file_name, 'wb'))
                    else:
                        with open(log_file_name, 'w+') as f:
                            f.write(json.dumps(saved_info, indent=4))

                    logger.removeHandler(logger.handlers[0])
                    os.remove(failure_file)
                
                except utils_exception.UnityException as e:
                    traceback.print_exc()

                    print("Unity exception")
                    arena.reset_env()
                    #ipdb.set_trace()
                    continue

                except utils_exception.ManyFailureException as e:
                    traceback.print_exc()

                    print("ERRO HERE")
                    logging.exception("Many failure Error")
                    # print("OTHER ERROR")
                    logger.removeHandler(logger.handlers[0])
                    #exit()
                    #arena.reset_env()
                    print("Dione")
                    #ipdb.set_trace()
                    arena.reset_env()
                    continue

                except Exception as e:
                    #with open(failure_file, 'w+') as f:
                    #    error_str = 'Failure'
                    #    error_str += '\n'
                    #    stack_form = ''.join(traceback.format_stack())
                    #    error_str += stack_form

                    #    f.write(error_str)
                    traceback.print_exc()

                    logging.exception("Error")
                    print("OTHER ERROR")
                    logger.removeHandler(logger.handlers[0])
                    #exit()
                    arena.reset_env()
                    # ipdb.set_trace()
                    # ipdb.set_trace()
                    # pdb.set_trace()
                    continue
                S[episode_id].append(is_finished)
                L[episode_id].append(steps)
                test_results[episode_id] = {'S': S[episode_id],
                                            'L': L[episode_id]}
                                            
            ipdb.set_trace()
            # pickle.dump(test_results, open(args.record_dir + '/results_{}.pik'.format(0), 'wb'))
            print('average steps (finishing the tasks):', np.array(steps_list).mean() if len(steps_list) > 0 else None)
            print('failed_tasks:', failed_tasks)
            # pickle.dump(test_results, open(args.record_dir + '/results_{}.pik'.format(0), 'wb'))

