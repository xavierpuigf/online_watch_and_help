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
from numpy import linalg as LA
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
    agent_types_index =  [0, 3, 4, 10, 12, 13, 14]
    # random_start.shuffle(agent_types_index)
    if args.agenttype != 'all':
        agent_types_index = [int(x) for x in args.agenttype.split(',')]


    agent_paths = {}
    agent_actions = {}

    for agent_id in agent_types_index: #len(agent_types)):
        agent_paths[agent_id] = {}
        agent_actions[agent_id] = {}
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

        #args.record_dir = '/data/vision/torralba/frames/data_acquisition/SyntheticStories/agent_preferences/data_scratch/large_data_v2/{}/{}'.format(datafile, args.mode)
        args.record_dir = '/data/vision/torralba/frames/data_acquisition/SyntheticStories/agent_preferences/data_scratch/large_data_touch_v2/{}/{}'.format(datafile, args.mode)
        print(args.record_dir)

        for episode_id in range(1):#range(num_episodes):
            agent_paths[agent_id][episode_id] = []
            agent_actions[agent_id][episode_id] = []
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
                agent_paths[agent_id][episode_id].append(np.array(agent_path))
                T = len(data['action'][0])
                agent_action = [None] * T
                for t in range(T):
                    agent_action[t] = action_code(data['action'][0][t])
                agent_actions[agent_id][episode_id].append(np.array(agent_action))

    for agent_id in agent_types_index:
        dist_lists = []
        for episode_id, curr_episode_paths in agent_paths[agent_id].items():
            N = len(curr_episode_paths)
            for i in range(N - 1):
                for j in range(i + 1, N):
                    curr_dist = dtw_dist(curr_episode_paths[i], curr_episode_paths[j], dist=dist_l2)
                    dist_lists.append(curr_dist)

        print(agent_id, np.mean(dist_lists), np.std(dist_lists))

    for agent_id in agent_types_index:
        dist_lists = []
        for episode_id, curr_episode_actions in agent_actions[agent_id].items():
            N = len(curr_episode_actions)
            for i in range(N - 1):
                for j in range(i + 1, N):
                    curr_dist = dtw_dist(curr_episode_actions[i], curr_episode_actions[j], dist=dist_code)
                    dist_lists.append(curr_dist)

        print(agent_id, np.mean(dist_lists), np.std(dist_lists))
                

