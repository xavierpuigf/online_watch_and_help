import sys
from termcolor import colored
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

    num_tries = 1
    args.executable_file = '../path_sim_dev/linux_exec.x86_64'
    args.max_episode_length = 150
    args.num_per_apartment = 20
    
    #args.dataset_path = './dataset/test_env_task_set_10_full_reduced_tasks_single.pik'
    #args.dataset_path = './dataset/train_env_task_set_20_full_reduced_tasks1to3.pik'
    args.dataset_path = './dataset/test_env_task_set_10_full_reduced_tasks1to3.pik'

    # Beliefs
    # spiked: object is in cabinet
    
    agent_types = [
            ['full', 0, 0.05, False, 0, "uniform"], # 0
            ['full', 0.5, 0.01, False, 0, "uniform"], # 1
            ['full', -5, 0.05, False, 0, "uniform"], # 2
            ['partial', 0, 0.05, False, 0, "uniform"], # 3
            ['partial', 0, 0.05, False, 0, "spiked"], # 4. kitchen and cabinet
            ['partial', 0, 0.05, False, 0.2, "uniform"], # 5
            ['partial', -500, 0.01, False, 0.01, "spiked"], # 6
            ['partial', -500, 0.05, False, 0.2, "uniform"], # 7
            ['partial', 0.5, 0.05, False, 0.2, "uniform"], # 8
            ['cone', 0, 0.05, False, 0, "uniform"], # 9
            ['partial', 0, 0.05, False, 0, "spiked2"], # 10 High prior for not inside
            ['partial', 0, 0.05, False, 0, "spiked3"], # 11 For sure not in bathroom
            ['partial', 0, 0.05, False, 0, "spiked4"], # 12 All things kithcen
            ['partial', 0, 0.05, False, 0.1, "spiked"], # 13
            ['partial', 0, 0.05, False, 0.1, "spiked2"] # 14
    ]
    random_start = random.Random()
    agent_types_index = list(range(9))
    agent_types_index =  [0, 3, 4, 10, 12, 13, 14]
    random_start.shuffle(agent_types_index)
    if args.agenttype != 'all':
        agent_types_index = [int(x) for x in args.agenttype.split(',')]
    
    agent_types_index = [10]
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
        args.mode += 'v9_particles_v2_modeinfo'

        
        env_task_set = pickle.load(open(args.dataset_path, 'rb'))
        print(len(env_task_set))

        for env in env_task_set:
            # Remove one of the goals
            new_dict_goal = {}
            for goal_pred in env['task_goal'][0]:
                if 'sit' in goal_pred:
                    env['task_goal'][0][goal_pred] = 0
                numpred = env['task_goal'][0][goal_pred]
                if goal_pred.split('_')[0] not in ['on', 'in', 'inside']:
                    continue
                goal_pred_new = 'touch_' + goal_pred.split('_')[1]
                new_dict_goal[goal_pred_new] = numpred
            env['task_goal'][0] = new_dict_goal
            #ipdb.set_trace()
            init_gr = env['init_graph']
            gbg_can = [node['id'] for node in init_gr['nodes'] if node['class_name'] in ['garbagecan', 'clothespile']]
            init_gr['nodes'] = [node for node in init_gr['nodes'] if node['id'] not in gbg_can]
            init_gr['edges'] = [edge for edge in init_gr['edges'] if edge['from_id'] not in gbg_can and edge['to_id'] not in gbg_can]
            for node in init_gr['nodes']:
                if node['class_name'] == 'cutleryfork':
                    node['obj_transform']['position'][1] += 0.1

        args.record_dir = '../data_scratch/large_data_v2/{}/{}'.format(datafile, args.mode)
        error_dir = '../data_scratch/large_data_v2/logging/{}_{}'.format(datafile, args.mode)
        if not os.path.exists(args.record_dir):
            os.makedirs(args.record_dir)

        if not os.path.exists(error_dir):
            os.makedirs(error_dir)

        executable_args = {
                        'file_name': args.executable_file,
                        'x_display': None,
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
        #episode_ids = [185]
        episode_ids = [9]

        
        file_failures = 'failures_{}.txt'.format(args.base_port)
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
                             c_base=10000,
                             num_samples=1,
                             num_processes=num_proc, 
                             num_particles=20,
                             logging=True,
                             logging_graphs=True)

        args_agent1 = {'agent_id': 1, 'char_index': 0}
        args_agent1.update(args_common)
        args_agent1['agent_params'] = agent_args
        agents = [lambda x, y: MCTS_agent_particle_v2(**args_agent1)]
        arena = ArenaMP(args.max_episode_length, id_run, env_fn, agents)
        
        # episode_ids = [20] #episode_ids
        num_tries = 1
        episode_ids = episode_ids

        for iter_id in range(num_tries):
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
            for episode_id in episode_ids: #46
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

                if os.path.isfile(log_file_name):# or os.path.isfile(failure_file):
                    # pass
                    continue
                if os.path.isfile(failure_file):
                    os.remove(failure_file)
                fileh = logging.FileHandler(failure_file, 'a')
                fileh.setLevel(logging.DEBUG)
                logger.addHandler(fileh)


                print('episode:', episode_id)

                for it_agent, agent in enumerate(arena.agents):
                    agent.seed = (it_agent + current_tried * 2) * 5

                failure = False
                try:
                    
                    arena.reset(episode_id)
                    if args.saveimg:
                        img_arg = os.path.join(args.record_dir, 'img', 'logs_episode.{}_iter.{}'.format(episode_id, iter_id)) 
                        Path(img_arg).mkdir(parents=True, exist_ok=True)
                    else:
                        img_arg = None
                    success, steps, saved_info = arena.run(save_img=img_arg)

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
                        print(colored("Saving..", log_file_name, 'green'))
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
                    failure = True
                    failure_str = 'unity'

                except utils_exception.ManyFailureException as e:
                    traceback.print_exc()

                    print("ERRO HERE")
                    logging.exception("Many failure Error")
                    # print("OTHER ERROR")
                    logger.removeHandler(logger.handlers[0])
                    #exit()
                    #arena.reset_env()
                    print("Dione")

                    #Path(args.record_dir).mkdir(parents=True, exist_ok=True)
                    saved_info = arena.saved_info
                    failure = True
                    #print("ACTIONS")
                    #for action in arena.saved_info['action'][0]:
                    #    print(action)
                    #print('---')
                    #if len(saved_info['obs']) > 0:
                    #    print(colored("Saving.."), 'green')
                    #    pickle.dump(saved_info, open(log_file_name, 'wb'))
                    arena.reset_env()
                    failure_str = "many_actions"
                    continue

                except Exception as e:
                    #with open(failure_file, 'w+') as f:
                    #    error_str = 'Failure'
                    #    error_str += '\n'
                    #    stack_form = ''.join(traceback.format_stack())
                    #    error_str += stack_form

                    #    f.write(error_str)
                    traceback.print_exc()

                    #Path(args.record_dir).mkdir(parents=True, exist_ok=True)
                    #saved_info = arena.saved_info
                    #if len(saved_info['obs']) > 0:
                    #    print(colored("Saving.."), 'green')
                    #    pickle.dump(saved_info, open(log_file_name, 'wb'))
                    #ipdb.set_trace()

                    logging.exception("Error")
                    print("OTHER ERROR")
                    logger.removeHandler(logger.handlers[0])
                    #exit()
                    arena.reset_env()
                    # ipdb.set_trace()
                    # ipdb.set_trace()
                    # pdb.set_trace()
                    failure = True
                    failure_str = "other"
                    continue

                if failure:
                    pass
                    #with open(file_failures, 'a+'):
                    #    str_file = 'Episode: {}. Try: {}. Agent: {}. Failure: {}\n'.format(episode_id, iter_id, agent_id, failure_str)
                    #    f.write(str_file)


                #S[episode_id].append(is_finished)
                #L[episode_id].append(steps)
                #test_results[episode_id] = {'S': S[episode_id],
                #                            'L': L[episode_id]}
                                            
            print('average steps (finishing the tasks):', np.array(steps_list).mean() if len(steps_list) > 0 else None)
            print('failed_tasks:', failed_tasks)
            #pickle.dump(test_results, open(args.record_dir + '/results_{}.pik'.format(0), 'wb'))

