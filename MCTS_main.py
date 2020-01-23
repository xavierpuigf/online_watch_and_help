import gym
import ipdb
import sys
sys.path.append('../vh_mdp')
sys.path.append('../virtualhome')
import vh_graph
from vh_graph.envs import belief
import utils_viz
import utils
import json
import random
import numpy as np
import cProfile
import ipdb
from simulation.evolving_graph.utils import load_graph_dict
from simulation.unity_simulator import comm_unity as comm_unity

import pickle
sys.argv = ['-f']

from agents import MCTS_agent, PG_agent

import timeit

# Options, should go as argparse arguments
agent_type = 'MCTS' # PG/MCTS
simulator_type = 'unity' # unity/python
dataset_path = '../dataset_toy4/init_envs/'


def rollout_from_json(info):
    num_entries = len(info)
    count = 0

    while count < num_entries:

        info_entry = info[count]

        count += 1
        scene_index, _, graph_index = info_entry['env_path'].split('.')[0][len('TrimmedTestScene'):].split('_')
        path_init_env = '{}/{}'.format(dataset_path, info_entry['env_path'])
        state = load_graph_dict(path_init_env)['init_graph']
        goals = info_entry['goal']
        goal_index = info_entry['goal_index']


        env.reset(state, goals)

        env.to_pomdp()
        gt_state = env.vh_state.to_dict()

        agent_id = [x['id'] for x in gt_state['nodes'] if x['class_name'] == 'character'][0]

        print("{} / {}    ###(Goal: {} in scene{}_{})".format(count, num_entries, goals, scene_index, graph_index))

        if agent_type == 'PG':
            agent = PG_agent(env,
                             max_episode_length=9,
                             num_simulation=1000.,
                             max_rollout_steps=5)

            start = timeit.default_timer()
            agent.rollout(state, goals)
            end = timeit.default_timer()
            print(end - start)

        elif agent_type == 'MCTS':
            agent = MCTS_agent(env=env,
                               agent_id=agent_id,
                               max_episode_length=5,
                               num_simulation=100,
                               max_rollout_steps=5,
                               c_init=0.1,
                               c_base=1000000,
                               num_samples=1,
                               num_processes=1)
        else:
            print('Agent {} not implemented'.format(agent_type))

        start = timeit.default_timer()
        history = agent.rollout(state, goals)

        end = timeit.default_timer()

def interactive_rollout():
    env = gym.make('vh_graph-v0')

    comm = comm_unity.UnityCommunication()
    comm.reset(0)
    comm.add_character()
    
    node_id_new = 2007
    s, graph = comm.environment_graph()
    container_id = [node['id'] for node in graph['nodes'] if node['class_name'] in ['fridge', 'freezer']][0]
    new_node = {'id': node_id_new, 'class_name': 'glass', 'states': [], 'properties': ['GRABBABLE']}
    new_edge = {'from_id': node_id_new, 'relation_type': 'INSIDE', 'to_id': container_id}
    graph['nodes'].append(new_node)
    graph['edges'].append(new_edge)
    success = comm.expand_scene(graph)
    
    s, graph = comm.environment_graph()
    agent_id =  [x['id'] for x in graph['nodes'] if x['class_name'] == 'character'][0]
    agent = MCTS_agent(env=env,
                       agent_id=agent_id,
                       max_episode_length=5,
                       num_simulation=100,
                       max_rollout_steps=3,
                       c_init=0.1,
                       c_base=1000000,
                       num_samples=1,
                       num_processes=1)

   
    glasses_id = [node['id'] for node in graph['nodes'] if 'glass' in node['class_name']]
    table_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'kitchentable'][0]

    goals = ['put_{}_{}'.format(glass_id, table_id) for glass_id in glasses_id]
    task_goal = {0: goals}
    agent.reset(graph, task_goal)
    while True:
        s, graph = comm.environment_graph()
        id2node = {node['id']: node for node in graph['nodes']}
        

        env.reset(graph , task_goal)
        agent.sample_belief(env.get_observations(char_index=0))
        agent.sim_env.reset(agent.previous_belief_graph, task_goal)

        print('Get action...')
        action, info = agent.get_action(task_goal[0])

        print(action, 'Plan: ', info['plan'][:4])
        script = ['<char0> {}'.format(action)]
        success, message = comm.render_script(script, image_synthesis=[])
        print(success)
        ipdb.set_trace()

if __name__ == '__main__':


    # Non interactive rollout
    if simulator_type == 'python':
        env = gym.make('vh_graph-v0')
        print('Env created')


        info = [
            {
                'env_path': 'TrimmedTestScene1_graph_10.json',
                'goal':  {0: ['findnode_2007']},
                'goal_index': 0
            }]

        rollout_from_json(info)
    else:
        interactive_rollout()


