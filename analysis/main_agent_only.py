import sys
import shutil
import os
import logging
import traceback
import random
import pickle
import copy
from pathlib import Path
import numpy as np
import pdb
import ipdb
import hydra
import time
from omegaconf import DictConfig, OmegaConf
from dataloader.dataloader_v2 import AgentTypeDataset
from dataloader import dataloader_v2 as dataloader_v2
from models import agent_pref_policy
from hydra.utils import get_original_cwd, to_absolute_path
from utils import utils_models_wb, utils_rl_agent

sys.path.append('.')
from envs.unity_environment import UnityEnvironment
from agents import MCTS_agent, MCTS_agent_particle_v2, MCTS_agent_particle

# from arguments import get_args
from algos.arena_mp2 import ArenaMP
from utils import utils_goals
from utils import utils_exception
import torch


def get_class_mode(agent_args):
    mode_str = '{}_opencost{}_closecost{}_walkcost{}_forgetrate{}'.format(
        agent_args['obs_type'],
        agent_args['open_cost'],
        agent_args['should_close'],
        agent_args['walk_cost'],
        agent_args['belief']['forget_rate'],
    )
    return mode_str


@hydra.main(config_path="../config/", config_name="config_default_toy_excl_plan")
def main(cfg: DictConfig):
    config = cfg
    print("Config")
    print(OmegaConf.to_yaml(cfg))
    args = config
    args_pred = args.agent_pred_graph
    num_proc = 0

    num_tries = 5
    args.executable_file = '/data/vision/torralba/frames/data_acquisition/SyntheticStories/website/release/simulator/v2.0/v2.2.5_beta/linux_exec.v2.2.5_beta.x86_64'
    args.max_episode_length = 250
    args.num_per_apartment = 20
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    # home_path = '../'
    rootdir = curr_dir + '/../'

    # args.dataset_path = f'{rootdir}/dataset/train_env_task_set_100_full.pik'
    args.dataset_path = f'/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/agent_preferences/dataset/test_env_task_set_10_full.pik'
    # args.dataset_path = './dataset/train_env_task_set_20_full_reduced_tasks_single.pik'

    cachedir = f'{get_original_cwd()}/outputs/main_agent_only'
    # cachedir = f'{rootdir}/dataset_episodes/main_agent_only'

    agent_types = [
        ['full', 0, 0.05, False, 0, "uniform"],  # 0
        ['full', 0.5, 0.01, False, 0, "uniform"],  # 1
        ['full', -5, 0.05, False, 0, "uniform"],  # 2
        ['partial', 0, 0.05, False, 0, "uniform"],  # 3
        ['partial', 0, 0.05, False, 0, "spiked"],  # 4
        ['partial', 0, 0.05, False, 0.2, "uniform"],  # 5
        ['partial', 0, 0.01, False, 0.01, "spiked"],  # 6
        ['partial', -5, 0.05, False, 0.2, "uniform"],  # 7
        ['partial', 0.5, 0.05, False, 0.2, "uniform"],  # 8
    ]
    random_start = random.Random()
    agent_id = 0
    (
        args.obs_type,
        open_cost,
        walk_cost,
        should_close,
        forget_rate,
        belief_type,
    ) = agent_types[0]
    datafile = args.dataset_path.split('/')[-1].replace('.pik', '')
    agent_args = {
        'obs_type': args.obs_type,
        'open_cost': open_cost,
        'should_close': should_close,
        'walk_cost': walk_cost,
        'belief': {'forget_rate': forget_rate, 'belief_type': belief_type},
    }
    # TODO: add num_samples to the argument
    # args.mode = '{}_'.format(agent_id + 1)
    # args.mode += 'v9_particles_v2'

    env_task_set = pickle.load(open(args.dataset_path, 'rb'))
    # print(env_task_set)
    print(len(env_task_set))

    for env in env_task_set:
        init_gr = env['init_graph']
        gbg_can = [
            node['id']
            for node in init_gr['nodes']
            if node['class_name'] in ['garbagecan', 'clothespile']
        ]
        init_gr['nodes'] = [
            node for node in init_gr['nodes'] if node['id'] not in gbg_can
        ]
        init_gr['edges'] = [
            edge
            for edge in init_gr['edges']
            if edge['from_id'] not in gbg_can and edge['to_id'] not in gbg_can
        ]
        for node in init_gr['nodes']:
            if node['class_name'] == 'cutleryfork':
                node['obj_transform']['position'][1] += 0.1

    args.record_dir = '{}/{}/'.format(cachedir, datafile)
    error_dir = '{}/logging/{}'.format(cachedir, datafile)
    if not os.path.exists(args.record_dir):
        os.makedirs(args.record_dir)

    if not os.path.exists(error_dir):
        os.makedirs(error_dir)

    executable_args = {
        'file_name': args.executable_file,
        'x_display': 0,
        'no_graphics': True,
    }

    id_run = 0
    # random.seed(id_run)
    episode_ids = list(range(len(env_task_set)))
    episode_ids = sorted(episode_ids)
    random_start.shuffle(episode_ids)
    # episode_ids = episode_ids[10:]

    S = [[] for _ in range(len(episode_ids))]
    L = [[] for _ in range(len(episode_ids))]

    test_results = {}
    # episode_ids = [episode_ids[0]]

    def env_fn(env_id):
        return UnityEnvironment(
            num_agents=1,
            max_episode_length=args.max_episode_length,
            port_id=env_id,
            env_task_set=env_task_set,
            observation_types=[args.obs_type, args.obs_type],
            use_editor=args.use_editor,
            executable_args=executable_args,
            base_port=args.base_port,
        )

    args_common = dict(
        recursive=False,
        max_episode_length=20,
        num_simulation=200,
        max_rollout_steps=5,
        c_init=0.1,
        c_base=100,
        num_samples=1,
        num_processes=num_proc,
        num_particles=20,
        logging=True,
        logging_graphs=True,
        get_plan_states=True,
    )
    if args.obs_type == 'full':
        args_common['num_particles'] = 1
    else:
        args_common['num_particles'] = 20

    args_agent1 = {'agent_id': 1, 'char_index': 0}
    args_agent1.update(args_common)
    args_agent1['agent_params'] = agent_args

    agents = [lambda x, y: MCTS_agent_particle_v2(**args_agent1)]
    arena = ArenaMP(args.max_episode_length, id_run, env_fn, agents)

    # # episode_ids = [20] #episode_ids
    # # num_tries = 1
    # episode_ids = [0]
    # ndict = {'on_book_329': 1}
    # env_task_set[91]['init_rooms'] = ['bedroom', 'bedroom']
    # env_task_set[91]['task_goal'] = {0: ndict, 1: ndict}

    for iter_id in range(0, num_tries):
        # if iter_id > 0:
        # iter_id = 1

        cnt = 0
        steps_list, failed_tasks = [], []
        current_tried = iter_id

        if not os.path.isfile(args.record_dir + '/results_{}.pik'.format(iter_id)):
            test_results = {}
        else:
            test_results = pickle.load(
                open(args.record_dir + '/results_{}.pik'.format(iter_id), 'rb')
            )

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # gt_dir = "/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/agent_preferences/dataset_episodes/large_data_toy/test_env_task_set_10_full/1_full_opencost0_closecostFalse_walkcost0.05_forgetrate0"
        # # pred_dir = "/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/results/predict_graph/train_data.dataset_graph_pred_train.pkl-agentsall/time_model.LSTM-stateenc.TF-edgepred.concat-lr0.0001-bs.16-goalenc.False_extended._costclose.1.0_costgoal.1.0_agentembed.False/test_env_task_set_10_full/1_full_opencost0_closecostFalse_walkcost0.05_forgetrate0"
        # # pred_dir = "/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/results/predict_graph/train_data.dataset_graph_pred_30step_train.pkl-agentsall/time_model.LSTM-stateenc.TF-globalrepr.pool-edgepred.concat-lr0.0001-bs.8-goalenc.False_extended._costclose.1.0_costgoal.1.0_agentembed.False_predchangeedge.True_inputgoal.True/test_env_task_set_10_full/1_full_opencost0_closecostFalse_walkcost0.05_forgetrate0"

        # root = "/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/results/predict_graph/train_data.dataset_graph_pred_30step_train.pkl-agentsall/"
        # pred_dir = (
        #     root
        #     + "time_model.LSTM-stateenc.TF-globalrepr.pool-edgepred.concat-lr0.0001-bs.8-goalenc.False_extended._costclose.1.0_costgoal.1.0_agentembed.False_predchange.edge_inputgoal.False_excledge.False/test_env_task_set_10_full/1_full_opencost0_closecostFalse_walkcost0.05_forgetrate0"
        #     # + "time_model.LSTM-stateenc.TF-globalrepr.pool-edgepred.concat-lr0.0001-bs.32-goalenc.False_extended._costclose.1.0_costgoal.1.0_agentembed.False_predchange.node_inputgoal.False_excledge.True/test_env_task_set_10_full/1_full_opencost0_closecostFalse_walkcost0.05_forgetrate0"
        # )
        # gt_p = Path(gt_dir).glob("*.pik")

        max_steps = args.max_episode_length

        for env_task in env_task_set:

            cnt = 0
            steps_list, failed_tasks = [], []
            current_tried = iter_id

            gt_goal = env_task['task_goal'][0]
            print('gt goal:', gt_goal)

            episode_id = env_task['task_id']

            # if episode_id != 12:
            #     continue

            log_file_name = args.record_dir + '/logs_episode.{}_iter.{}.pik'.format(
                episode_id, iter_id
            )
            failure_file = '{}/{}_{}.txt'.format(error_dir, episode_id, iter_id)

            # if os.path.isfile(log_file_name):  # or os.path.isfile(failure_file):
            #     print(log_file_name)
            #     continue

            if os.path.isfile(failure_file):
                os.remove(failure_file)
            fileh = logging.FileHandler(failure_file, 'a')
            fileh.setLevel(logging.DEBUG)
            logger.addHandler(fileh)

            print('episode:', episode_id)

            for it_agent, agent in enumerate(arena.agents):
                agent.seed = (it_agent + current_tried * 2) * 5

            try:
                obs = arena.reset(episode_id)
                arena.task_goal = None
                print(arena.env.task_goal, arena.env.agent_goals)
                # print(
                #     [edge for edge in obs[0]['edges'] if edge['from_id'] == 351]
                #     or edge['to_id'] == 351
                # )
                # for t, action in enumerate(actions):

                saved_info = {
                    'task_id': arena.env.task_id,
                    'env_id': arena.env.env_id,
                    'task_name': arena.env.task_name,
                    'gt_goals': arena.env.task_goal[0],
                    'goals': arena.task_goal,
                    'action': {0: [], 1: []},
                    'plan': {0: [], 1: []},
                    'finished': None,
                    'init_unity_graph': arena.env.init_graph,
                    'goals_finished': [],
                    'belief': {0: [], 1: []},
                    'belief_room': {0: [], 1: []},
                    'belief_graph': {0: [], 1: []},
                    'graph': [arena.env.init_unity_graph],
                    'obs': [],
                }

                steps = 2
                actions, curr_info = arena.get_actions(
                    obs, length_plan=10, must_replan={0: False, 1: False}, agent_id=0
                )
                (prev_obs, reward, done, infos) = arena.step_given_action(
                    {0: actions[0]}
                )
                print(curr_info[0]['subgoals'])
                print(curr_info[0]['plan'])
                ipdb.set_trace()
                prev_graph = infos['graph']

                if 'satisfied_goals' in infos:
                    saved_info['goals_finished'].append(infos['satisfied_goals'])
                for agent_id, action in actions.items():
                    saved_info['action'][agent_id].append(action)
                if 'graph' in infos:
                    saved_info['graph'].append(infos['graph'])
                for agent_id, info in curr_info.items():
                    if 'belief_room' in info:
                        saved_info['belief_room'][agent_id].append(info['belief_room'])
                    if 'belief' in info:
                        saved_info['belief'][agent_id].append(info['belief'])
                    if 'plan' in info:
                        saved_info['plan'][agent_id].append(info['plan'][:3])
                    if 'obs' in info:
                        saved_info['obs'].append([node['id'] for node in info['obs']])

                actions, curr_info = arena.get_actions(
                    prev_obs,
                    length_plan=10,
                    must_replan={0: False, 1: False},
                    agent_id=0,
                )
                prev_action = actions[0]

                (curr_obs, reward, done, infos) = arena.step_given_action(
                    {0: actions[0]}
                )
                curr_graph = infos['graph']
                print(curr_info[0]['subgoals'])
                print(curr_info[0]['plan'])
                ipdb.set_trace()

                if 'satisfied_goals' in infos:
                    saved_info['goals_finished'].append(infos['satisfied_goals'])
                for agent_id, action in actions.items():
                    saved_info['action'][agent_id].append(action)
                if 'graph' in infos:
                    saved_info['graph'].append(infos['graph'])
                for agent_id, info in curr_info.items():
                    if 'belief_room' in info:
                        saved_info['belief_room'][agent_id].append(info['belief_room'])
                    if 'belief' in info:
                        saved_info['belief'][agent_id].append(info['belief'])
                    if 'plan' in info:
                        saved_info['plan'][agent_id].append(info['plan'][:3])
                    if 'obs' in info:
                        saved_info['obs'].append([node['id'] for node in info['obs']])

                success = False
                while steps < max_steps:
                    steps += 1

                    # get main agent's action
                    # arena.task_goal = None
                    print('planning for the main agent')
                    selected_actions, curr_info = arena.get_actions(
                        curr_obs,
                        length_plan=10,
                        must_replan={0: False, 1: False},
                        agent_id=0,
                    )
                    print(curr_info[0]['subgoals'])
                    print(curr_info[0]['plan'])

                    (curr_obs, reward, done, infos) = arena.step_given_action(
                        selected_actions
                    )

                    print("agents' positions")
                    print(
                        [
                            (node['id'], node['bounding_box']['center'])
                            for node in curr_obs[0]['nodes']
                            if node['id'] < 3
                        ]
                    )
                    ipdb.set_trace()

                    curr_graph = infos['graph']

                    if 'satisfied_goals' in infos:
                        saved_info['goals_finished'].append(infos['satisfied_goals'])
                    for agent_id, action in actions.items():
                        saved_info['action'][agent_id].append(action)
                    if 'graph' in infos:
                        saved_info['graph'].append(infos['graph'])
                    for agent_id, info in curr_info.items():
                        if 'belief_room' in info:
                            saved_info['belief_room'][agent_id].append(
                                info['belief_room']
                            )
                        if 'belief' in info:
                            saved_info['belief'][agent_id].append(info['belief'])
                        if 'plan' in info:
                            saved_info['plan'][agent_id].append(info['plan'][:3])
                        if 'obs' in info:
                            saved_info['obs'].append(
                                [node['id'] for node in info['obs']]
                            )

                    print('success:', infos['finished'])
                    # pdb.set_trace()
                    if infos['finished']:
                        success = True
                        break

                    print('success:', infos['finished'])
                    # pdb.set_trace()
                    if infos['finished']:
                        success = True
                        break

                print('-------------------------------------')
                print('success' if success else 'failure')
                print('steps:', steps)
                print('-------------------------------------')
                if not success:
                    failed_tasks.append(episode_id)
                else:
                    steps_list.append(steps)
                is_finished = 1 if success else 0

                saved_info['obs'].append([node['id'] for node in curr_obs[0]['nodes']])
                saved_info['finished'] = success

                Path(args.record_dir).mkdir(parents=True, exist_ok=True)
                if len(saved_info['obs']) > 0:
                    pickle.dump(saved_info, open(log_file_name, 'wb'))
                else:
                    with open(log_file_name, 'w+') as f:
                        f.write(json.dumps(saved_info, indent=4))
                ipdb.set_trace()

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
                # ipdb.set_trace()
                continue

            except utils_exception.ManyFailureException as e:
                traceback.print_exc()

                print("ERRO HERE")
                logging.exception("Many failure Error")
                # print("OTHER ERROR")
                logger.removeHandler(logger.handlers[0])
                # exit()
                # arena.reset_env()
                print("Dione")
                # ipdb.set_trace()
                arena.reset_env()
                continue

            except Exception as e:
                with open(failure_file, 'w+') as f:
                    error_str = 'Failure'
                    error_str += '\n'
                    stack_form = ''.join(traceback.format_stack())
                    error_str += stack_form

                    f.write(error_str)
                traceback.print_exc()

                logging.exception("Error")
                print("OTHER ERROR")
                logger.removeHandler(logger.handlers[0])
                # exit()
                arena.reset_env()
                # ipdb.set_trace()
                # ipdb.set_trace()
                # pdb.set_trace()
                continue
            S[episode_id].append(is_finished)
            L[episode_id].append(steps)
            test_results[episode_id] = {'S': S[episode_id], 'L': L[episode_id]}
            # pdb.set_trace()

        print(test_results)

        pickle.dump(
            test_results,
            open(args.record_dir + '/results_{}.pik'.format(iter_id), 'wb'),
        )
        print(
            'average steps (finishing the tasks):',
            np.array(steps_list).mean() if len(steps_list) > 0 else None,
        )
        print('failed_tasks:', failed_tasks)
        # pickle.dump(
        #     test_results,
        #     open(args.record_dir + '/results_{}.pik'.format(iter_id), 'wb'),
        # )


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("running time: %s sec" % (time.time() - start_time))
