import sys
from termcolor import colored
import shutil
import os
import logging
import traceback
import ipdb
import pickle
import json
import random
import numpy as np
import multiprocessing as mp
from numpy import linalg as LA
import matplotlib.pyplot as plt
from pathlib import Path
try:
    from dtw import dtw
except ImportError:
    pass

from arguments import get_args
from algos.arena_mp2 import ArenaMP
from utils import utils_goals


def get_class_mode(agent_args):
    mode_str = '{}_opencost{}_closecost{}_walkcost{}_forgetrate{}'.format(
        agent_args['obs_type'],  
        agent_args['open_cost'],
        agent_args['should_close'], 
        agent_args['walk_cost'],
        agent_args['belief']['forget_rate'])
    return mode_str


def dist_l2(pos1, pos2):
    """get distance between two points"""
    return LA.norm(np.array(pos1) - np.array(pos2))


def dist_code(code1, code2):
    """get distance between two codes"""
    return int(code1 != code2)


def dtw_dist(traj1, traj2, dist):
    """distance between two trajectories using temporal dynamic wrapping"""
    # traj1 = traj10.reshape(-1, 3)
    # traj2 = traj20.reshape(-1, 3)
    d, _, _, _ = dtw(traj1, traj2, dist=dist)
    return d


def action_code(action):
    """action command -> action class"""
    if 'walk' in action:
        code = 0
    elif 'open' in action:
        code = 1
    elif 'close' in action:
        code = 2
    elif 'grab' in action:
        code = 3
    elif 'put' in action:
        code = 4
    else:
        code = 5
    # one_hot = [0] * 6
    # one_hot[code] = 1
    # return one_hot
    return code


def read_episodes(agent_id, agent_types, args, num_episodes, num_seeds, agent_paths, agent_actions):
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

    args.record_dir = '/data/vision/torralba/frames/data_acquisition/SyntheticStories/agent_preferences/data_scratch/large_data_touch_v2/{}/{}'.format(datafile, args.mode)
    print(args.record_dir)

    paths = {}
    actions = {}
    for episode_id in range(0, num_episodes, 10):
        print(agent_id, episode_id)
        paths[episode_id] = []
        actions[episode_id] = []
        for seed_id in range(num_seeds):
            file_path = '{}/logs_episode.{}_iter.{}.pik'.format(args.record_dir, episode_id, seed_id)
            if not os.path.exists(file_path): continue
            data = pickle.load(open(file_path, 'rb'))
            # print(data.keys())
            # print(data['env_id'], data['task_name'], data['goals'])
            T = len(data['graph'])
            agent_path = [None] * T
            for t in range(T):
                agent_pos = [n['bounding_box']['center'] for n in data['graph'][t]['nodes'] if n['class_name'] == 'character']
                assert len(agent_pos) == 1
                agent_path[t] = agent_pos[0]
            paths[episode_id].append(np.array(agent_path))
            T = len(data['action'][0])
            agent_action = [None] * T
            for t in range(T):
                agent_action[t] = action_code(data['action'][0][t])
            actions[episode_id].append(np.array(agent_action))
    agent_paths[agent_id] = paths
    agent_actions[agent_id] = actions


if __name__ == '__main__':
    args = get_args()
    num_proc = 10

    num_tries = 1
    args.executable_file = '../path_sim_dev/linux_exec.x86_64'
    args.max_episode_length = 150
    args.num_per_apartment = 20

    num_episodes = 439
    num_seeds = 5
    
    # args.dataset_path = './dataset/test_env_task_set_10_full_reduced_tasks1to3.pik'
    args.dataset_path = './dataset/train_env_task_set_20_full_reduced_tasks1to3.pik'

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
    agent_types_index =  [0, 3, 4, 10, 12]#, 13, 14]
    # random_start.shuffle(agent_types_index)
    if args.agenttype != 'all':
        agent_types_index = [int(x) for x in args.agenttype.split(',')]


    # agent_paths = {}
    # agent_actions = {}

    manager = mp.Manager()
    agent_paths = manager.dict()
    agent_actions = manager.dict()
    num_processes = 10
        
    for start_root_id in range(0, len(agent_types_index), num_processes):
        end_root_id = min(start_root_id + num_processes, len(agent_types_index))
        jobs = []
        for process_id in range(start_root_id, end_root_id):
            agent_id = agent_types_index[process_id]
            print(process_id, agent_id)
            p = mp.Process(target=read_episodes,
                           args=(agent_id, agent_types, args, num_episodes, num_seeds, 
                                 agent_paths, agent_actions))
            jobs.append(p)
            p.start()
        for p in jobs:
            p.join()

    D_agent_pos = np.empty([len(agent_types_index), len(agent_types_index)])
    for id_1 in range(len(agent_types_index) - 1):
        agent_id_1 = agent_types_index[id_1]
        for id_2 in range(id_1, len(agent_types_index)):
            agent_id_2 = agent_types_index[id_2]
            dist_lists = []
            for episode_id, curr_episode_paths_1 in agent_paths[agent_id_1].items():
                N1 = len(curr_episode_paths_1)
                curr_episode_paths_2 = agent_paths[agent_id_2][episode_id]
                N2 = len(curr_episode_paths_2)
                for i in range(N1):
                    for j in range(N2):
                        curr_dist = dtw_dist(curr_episode_paths_1[i], curr_episode_paths_2[j], dist=dist_l2)
                        dist_lists.append(curr_dist)
            D_agent_pos[id_1, id_2] = np.mean(dist_lists)
            D_agent_pos[id_2, id_1] = D_agent_pos[id_1, id_2]
    print(D_agent_pos)
    fig, ax = plt.subplots(1,1)
    img = ax.imshow(D_agent_pos)
    ax.set_xticklabels([0] + agent_types_index)
    ax.set_yticklabels([0] + agent_types_index)
    fig.colorbar(img)
    fig.tight_layout()
    fig.savefig('./testing_agents/D_agent_pos_{}.pdf'.format(num_episodes))

    D_agent_action = np.empty([len(agent_types_index), len(agent_types_index)])
    for id_1 in range(len(agent_types_index) - 1):
        agent_id_1 = agent_types_index[id_1]
        for id_2 in range(id_1, len(agent_types_index)):
            agent_id_2 = agent_types_index[id_2]
            dist_lists = []
            for episode_id, curr_episode_actions_1 in agent_actions[agent_id_1].items():
                N1 = len(curr_episode_actions_1)
                curr_episode_actions_2 = agent_actions[agent_id_2][episode_id]
                N2 = len(curr_episode_actions_2)
                for i in range(N1):
                    for j in range(N2):
                        curr_dist = dtw_dist(curr_episode_actions_1[i], curr_episode_actions_2[j], dist=dist_l2)
                        dist_lists.append(curr_dist)
            D_agent_action[id_1, id_2] = np.mean(dist_lists)
            D_agent_action[id_2, id_1] = D_agent_action[id_1, id_2]
    print(D_agent_action)
    plt.clf()
    fig, ax = plt.subplots(1,1)
    img = ax.imshow(D_agent_action)
    ax.set_xticklabels([0] + agent_types_index)
    ax.set_yticklabels([0] + agent_types_index)
    fig.colorbar(img)
    fig.tight_layout()
    plt.savefig('./testing_agents/D_agent_action_{}.pdf'.format(num_episodes))
                

