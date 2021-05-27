import sys
import shutil
import os
import logging
import traceback
import os
import ipdb
import functools
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


def get_agent(agent_args, num_proc=0, num_particles=1):
    names = ['obs_type', 'open_cost', 'walk_cost', 'should_close', 'forget_rate', 'belief_type']
    obs_type, open_cost, walk_cost, should_close, forget_rate, belief_type = [agent_args[name] for name in names]
    args_common = dict(recursive=False,
                         max_episode_length=20,
                         num_simulation=200,
                         max_rollout_steps=5,
                         c_init=0.1,
                         c_base=100,
                         num_samples=1,
                         num_processes=agent_args['num_proc'], 
                         num_particles=agent_args['num_particles'],
                         logging=True,
                         logging_graphs=True)

    args_agent1 = {'agent_id': 1, 'char_index': 0}
    args_agent1.update(args_common)
    agent_args = {
        'obs_type': obs_type,
        'open_cost': open_cost,
        'should_close': should_close,
        'walk_cost': walk_cost,
        'belief': {'forget_rate': forget_rate, 'belief_type': belief_type}
    }
    args_agent1['agent_params'] = agent_args
    agents = [lambda x, y: MCTS_agent_particle_v2(**args_agent1)]
    return agents



def get_environment(obs_type, executable_file, max_episode_length=250, base_port=8080, use_editor=False):
    executable_args = {
                    'file_name': executable_file,
                    'x_display': 0,
                    'no_graphics': True
    }

    def env_fn(env_id):
        return UnityEnvironment(num_agents=1,
                                max_episode_length=max_episode_length,
                                port_id=env_id,
                                env_task_set=env_task_set,
                                observation_types=[obs_type],
                                use_editor=use_editor,
                                executable_args=executable_args,
                                base_port=base_port)
    return env_fn

def get_arena(agent_args, executable_file='.', max_episode_length=250, base_port=8080, use_editor=False):
    seed_agent = agent_args['seed']
    agents = get_agent(agent_args)
    obs_type = agent_args['obs_type']
    env_fn = get_environment(obs_type, executable_file, max_episode_length, base_port, use_editor) 
    arena = ArenaMP(args.max_episode_length, 0, env_fn, agents)
    if seed_agent:
        for it_agent, agent in enumerate(arena.agents):
            agent.seed = (it_agent + seed_agent * 2) * 5
    return arena


def simulate_agent(agent_args, episode_id, executable_file='../path_sim_dev/linux_exec.x86_64', max_episode_length=250, base_port=8080, use_editor=False):
    arena = get_arena(agent_args, executable_file, max_episode_length, base_port, use_editor)


    print(f"Starting Episode: {episode_id}")
    print(f"------------------------------")
    arena.reset(episode_id)
    
    done = False
    obs = arena.env.get_observations()
    actions = []
    while not done:
        # print([node['id'] for node in obs[0]['nodes']])
        action_space = arena.env.get_action_space()
        dict_actions, dict_info = arena.get_actions(obs, action_space)
        plan = dict_info[0]['plan']
        obs, reward, done, env_info = arena.env.step(dict_actions)
        actions.append(dict_actions[0])
        finished = env_info['finished']

    if finished:
        print("Succeeded")
        print("---------")
        print(actions)
    else:
        print("Failed")



if __name__ == '__main__':
    args = get_args()

    args.executable_file = '../path_sim_dev/linux_exec.x86_64'
    args.max_episode_length = 250
    args.num_per_apartment = 20
    
    #args.dataset_path = './dataset/test_env_task_set_10_full_reduced_tasks_single.pik'
    #args.dataset_path = './dataset/train_env_task_set_20_full_reduced_tasks1to3.pik'
    args.dataset_path = './dataset/test_env_task_set_10_full_reduced_tasks1to3.pik'

    # Change a bit the environment, moving a fork to a good position
    env_task_set = pickle.load(open(args.dataset_path, 'rb'))
    print(len(env_task_set))
    to_delete = []
    for item, env in enumerate(env_task_set):
        new_dict_goal = {}
        for goal_pred in env['task_goal'][0]:
            if 'sit' in goal_pred:
                env['task_goal'][0][goal_pred] = 0
            numpred = env['task_goal'][0][goal_pred]
            if goal_pred.split('_')[0] not in ['on', 'in', 'inside']:
                continue
            goal_pred_new = 'touch_' + goal_pred.split('_')[1]
            if numpred > 0:
                new_dict_goal[goal_pred_new] = numpred
        if len(new_dict_goal) == 0:
            to_delete.append(item)
        env['task_goal'][0] = new_dict_goal

        init_gr = env['init_graph']
        gbg_can = [node['id'] for node in init_gr['nodes'] if node['class_name'] in ['garbagecan', 'clothespile']]
        init_gr['nodes'] = [node for node in init_gr['nodes'] if node['id'] not in gbg_can]
        init_gr['edges'] = [edge for edge in init_gr['edges'] if edge['from_id'] not in gbg_can and edge['to_id'] not in gbg_can]
        for node in init_gr['nodes']:
            if node['class_name'] == 'cutleryfork':
                node['obj_transform']['position'][1] += 0.1

    env_task_set = [env_task_set[idi] for idi in range(len(env_task_set)) if idi not in to_delete]
    agent_types = [
            ['full', 0, 0.05, False, 0, "uniform"], # 0
            ['full', 0.5, 0.01, False, 0, "uniform"], # 1
            ['full', -5, 0.05, False, 0, "uniform"], # 2
            ['partial', 0, 0.05, False, 0, "uniform"], # 3
            ['partial', 0, 0.05, False, 0, "spiked"], # 4
            ['partial', 0, 0.05, False, 0.2, "uniform"], # 5
            ['partial', -500, 0.01, False, 0.01, "spiked"], # 6
            ['partial', -500, 0.05, False, 0.2, "uniform"], # 7
            ['partial', 0.5, 0.05, False, 0.2, "uniform"], # 8
            ['cone', 0, 0.05, False, 0, "uniform"], # 9
            ['partial', 0, 0.05, False, 0, "spiked2"], # 10 High prior for not inside
            ['partial', 0, 0.05, False, 0, "spiked3"], # 11 For sure not in bathroom
            ['partial', 0, 0.05, False, 0, "spiked4"], # 12 All things kithcen
            ['partial', 0, 0.05, False, 0.1, "spiked"], # 13
            ['partial', 0, 0.05, False, 0.1, "spiked2"], # 14
            ['partial', 0, 0.00, False, 0.1, "spiked2"] # 15
    ]
    names = ['obs_type', 'open_cost', 'walk_cost', 'should_close', 'forget_rate', 'belief_type']
    agent_args = {}
    type_id = 10
    for idi, name in enumerate(names):
        agent_args[name] = agent_types[type_id][idi]

    agent_args['num_proc'] = 0
    agent_args['num_particles'] = 1
    agent_args['seed'] = 2

    sim_agent = functools.partial(
            simulate_agent,
            executable_file=args.executable_file, 
            max_episode_length=args.max_episode_length, 
            base_port=args.base_port)
    sim_agent(agent_args=agent_args, episode_id=9)

    #pdb.set_trace()

