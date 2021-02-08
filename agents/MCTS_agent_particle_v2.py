import numpy as np
import traceback
import logging
import random
import time
import math
import copy
import importlib
import json
import multiprocessing as mp
from functools import partial
import ipdb
import pdb
import pickle


from . import belief
from envs.graph_env import VhGraphEnv
#
import pdb
from MCTS import *

import sys
sys.path.append('..')
from utils import utils_environment as utils_env
from utils import utils_exception


def find_heuristic(agent_id, char_index, unsatisfied, env_graph, simulator, object_target):
    observations = simulator.get_observations(env_graph, char_index=char_index)
    id2node = {node['id']: node for node in env_graph['nodes']}
    containerdict = {edge['from_id']: edge['to_id'] for edge in env_graph['edges'] if edge['relation_type'] == 'INSIDE'}
    target = int(object_target.split('_')[-1])
    observation_ids = [x['id'] for x in observations['nodes']]
    try:
        room_char = [edge['to_id'] for edge in env_graph['edges'] if edge['from_id'] == agent_id and edge['relation_type'] == 'INSIDE'][0]
    except:
        print('Error')
        #ipdb.set_trace()

    action_list = []
    cost_list = []
    # if target == 478:
    #     )ipdb.set_trace()
    while target not in observation_ids:
        try:
            container = containerdict[target]
        except:
            print(id2node[target])
            raise Exception
        # If the object is a room, we have to walk to what is insde

        if id2node[container]['category'] == 'Rooms':
            action_list = [('walk', (id2node[target]['class_name'], target), None)] + action_list 
            cost_list = [0.5] + cost_list
        
        elif 'CLOSED' in id2node[container]['states'] or ('OPEN' not in id2node[container]['states']):
            action = ('open', (id2node[container]['class_name'], container), None)
            action_list = [action] + action_list
            cost_list = [0.05] + cost_list

        target = container
    
    ids_character = [x['to_id'] for x in observations['edges'] if
                     x['from_id'] == agent_id and x['relation_type'] == 'CLOSE'] + \
                    [x['from_id'] for x in observations['edges'] if
                     x['to_id'] == agent_id and x['relation_type'] == 'CLOSE']

    if target not in ids_character:
        # If character is not next to the object, walk there
        action_list = [('walk', (id2node[target]['class_name'], target), None)]+ action_list
        cost_list = [1] + cost_list

    return action_list, cost_list

def grab_heuristic(agent_id, char_index, unsatisfied, env_graph, simulator, object_target):
    observations = simulator.get_observations(env_graph, char_index=char_index)
    target_id = int(object_target.split('_')[-1])

    observed_ids = [node['id'] for node in observations['nodes']]
    agent_close = [edge for edge in env_graph['edges'] if ((edge['from_id'] == agent_id and edge['to_id'] == target_id) or (edge['from_id'] == target_id and edge['to_id'] == agent_id) and edge['relation_type'] == 'CLOSE')]
    grabbed_obj_ids = [edge['to_id'] for edge in env_graph['edges'] if (edge['from_id'] == agent_id and 'HOLDS' in edge['relation_type'])]

    target_node = [node for node in env_graph['nodes'] if node['id'] == target_id][0]

    if target_id not in grabbed_obj_ids:
        target_action = [('grab', (target_node['class_name'], target_id), None)]
        cost = [0.05]
    else:
        target_action = []
        cost = []

    if len(agent_close) > 0 and target_id in observed_ids:
        return target_action, cost
    else:
        find_actions, find_costs = find_heuristic(agent_id, char_index, unsatisfied, env_graph, simulator, object_target)
        return find_actions, find_costs

def turnOn_heuristic(agent_id, char_index, unsatisfied, env_graph, simulator, object_target):
    observations = simulator.get_observations(env_graph, char_index=char_index)
    target_id = int(object_target.split('_')[-1])

    observed_ids = [node['id'] for node in observations['nodes']]
    agent_close = [edge for edge in env_graph['edges'] if ((edge['from_id'] == agent_id and edge['to_id'] == target_id) or (edge['from_id'] == target_id and edge['to_id'] == agent_id) and edge['relation_type'] == 'CLOSE')]
    grabbed_obj_ids = [edge['to_id'] for edge in env_graph['edges'] if (edge['from_id'] == agent_id and 'HOLDS' in edge['relation_type'])]

    target_node = [node for node in env_graph['nodes'] if node['id'] == target_id][0]

    if target_id not in grabbed_obj_ids:
        target_action = [('switchon', (target_node['class_name'], target_id), None)]
        cost = [0.05]
    else:
        target_action = []
        cost = []

    if len(agent_close) > 0 and target_id in observed_ids:
        return target_action, cost
    else:
        find_actions, find_costs = find_heuristic(agent_id, char_index, unsatisfied, env_graph, simulator, object_target)
        return find_actions + target_action, find_costs + cost

def sit_heuristic(agent_id, char_index, unsatisfied, env_graph, simulator, object_target):
    observations = simulator.get_observations(env_graph, char_index=char_index)
    target_id = int(object_target.split('_')[-1])

    observed_ids = [node['id'] for node in observations['nodes']]
    agent_close = [edge for edge in env_graph['edges'] if ((edge['from_id'] == agent_id and edge['to_id'] == target_id) or (edge['from_id'] == target_id and edge['to_id'] == agent_id) and edge['relation_type'] == 'CLOSE')]
    on_ids = [edge['to_id'] for edge in env_graph['edges'] if (edge['from_id'] == agent_id and 'ON' in edge['relation_type'])]

    target_node = [node for node in env_graph['nodes'] if node['id'] == target_id][0]

    if target_id not in on_ids:
        target_action = [('sit', (target_node['class_name'], target_id), None)]
        cost = [0.05]
    else:
        target_action = []
        cost = []

    if len(agent_close) > 0 and target_id in observed_ids:
        return target_action, cost
    else:
        find_actions, find_costs = find_heuristic(agent_id, char_index, unsatisfied, env_graph, simulator, object_target)
        return find_actions + target_action, find_costs + cost

def put_heuristic(agent_id, char_index, unsatisfied, env_graph, simulator, target):
    # Modif, now put heristic is only the immaediate after action
    observations = simulator.get_observations(env_graph, char_index=char_index)

    target_grab, target_put = [int(x) for x in target.split('_')[-2:]]

    if sum([1 for edge in observations['edges'] if edge['from_id'] == target_grab and edge['to_id'] == target_put and edge['relation_type'] == 'ON']) > 0:
        # Object has been placed
        return [], []

    if sum([1 for edge in observations['edges'] if edge['to_id'] == target_grab and edge['from_id'] != agent_id and 'HOLD' in edge['relation_type']]) > 0:
        # Object has been placed
        return [], []

    target_node = [node for node in env_graph['nodes'] if node['id'] == target_grab][0]
    target_node2 = [node for node in env_graph['nodes'] if node['id'] == target_put][0]
    id2node = {node['id']: node for node in env_graph['nodes']}
    target_grabbed = len([edge for edge in env_graph['edges'] if edge['from_id'] == agent_id and 'HOLDS' in edge['relation_type'] and edge['to_id'] == target_grab]) > 0


    object_diff_room = None
    if not target_grabbed:
        grab_obj1, cost_grab_obj1 = grab_heuristic(agent_id, char_index, unsatisfied, env_graph, simulator, 'grab_' + str(target_node['id']))
        if len(grab_obj1) > 0:
            if grab_obj1[0][0] == 'walk':
                id_room = grab_obj1[0][1][1]
                if id2node[id_room]['category'] == 'Rooms':
                    object_diff_room = id_room
        
        return grab_obj1, cost_grab_obj1  
    else:
        env_graph_new = env_graph
        grab_obj1 = []
        cost_grab_obj1 = []
        find_obj2, cost_find_obj2 = find_heuristic(agent_id, char_index, unsatisfied, env_graph_new, simulator, 'find_' + str(target_node2['id']))
    
    action = [('putback', (target_node['class_name'], target_grab), (target_node2['class_name'], target_put))]
    cost = [0.05]
    res = grab_obj1 + find_obj2 + action
    cost_list = cost_grab_obj1 + cost_find_obj2 + cost

    #print(res, target)
    return res, cost_list

def putIn_heuristic(agent_id, char_index, unsatisfied, env_graph, simulator, target):
    # TODO: change this as well
    observations = simulator.get_observations(env_graph, char_index=char_index)

    target_grab, target_put = [int(x) for x in target.split('_')[-2:]]

    if sum([1 for edge in observations['edges'] if edge['from_id'] == target_grab and edge['to_id'] == target_put and edge['relation_type'] == 'ON']) > 0:
        # Object has been placed
        return [], []

    if sum([1 for edge in observations['edges'] if edge['to_id'] == target_grab and edge['from_id'] != agent_id and 'HOLD' in edge['relation_type']]) > 0:
        # Object has been placed
        return None, None

    target_node = [node for node in env_graph['nodes'] if node['id'] == target_grab][0]
    target_node2 = [node for node in env_graph['nodes'] if node['id'] == target_put][0]
    id2node = {node['id']: node for node in env_graph['nodes']}
    target_grabbed = len([edge for edge in env_graph['edges'] if edge['from_id'] == agent_id and 'HOLDS' in edge['relation_type'] and edge['to_id'] == target_grab]) > 0


    object_diff_room = None
    if not target_grabbed:
        grab_obj1, cost_grab_obj1 = grab_heuristic(agent_id, char_index, unsatisfied, env_graph, simulator, 'grab_' + str(target_node['id']))
        if len(grab_obj1) > 0:
            if grab_obj1[0][0] == 'walk':
                id_room = grab_obj1[0][1][1]
                if id2node[id_room]['category'] == 'Rooms':
                    object_diff_room = id_room
        
        env_graph_new = copy.deepcopy(env_graph)
        
        if object_diff_room:
            env_graph_new['edges'] = [edge for edge in env_graph_new['edges'] if edge['to_id'] != agent_id and edge['from_id'] != agent_id]
            env_graph_new['edges'].append({'from_id': agent_id, 'to_id': object_diff_room, 'relation_type': 'INSIDE'})
        
        else:
            env_graph_new['edges'] = [edge for edge in env_graph_new['edges'] if (edge['to_id'] != agent_id and edge['from_id'] != agent_id) or edge['relation_type'] == 'INSIDE']
    else:
        env_graph_new = env_graph
        grab_obj1 = []
        cost_grab_obj1 = []
    find_obj2, cost_find_obj2 = find_heuristic(agent_id, char_index, unsatisfied, env_graph_new, simulator, 'find_' + str(target_node2['id']))
    target_put_state = target_node2['states']
    action_open = [('open', (target_node2['class_name'], target_put))]
    action_put = [('putin', (target_node['class_name'], target_grab), (target_node2['class_name'], target_put))]
    cost_open = [0.05]
    cost_put = [0.05]
    

    remained_to_put = 0
    for predicate, count in unsatisfied.items():
        if predicate.startswith('inside'):
            remained_to_put += count
    if remained_to_put == 1: # or agent_id > 1:
        action_close= []
        cost_close = []
    else:
        action_close = [('close', (target_node2['class_name'], target_put))]
        cost_close = [0.05]

    if 'CLOSED' in target_put_state or 'OPEN' not in target_put_state:
        res = grab_obj1 + find_obj2 + action_open + action_put + action_close
        cost_list = cost_grab_obj1 + cost_find_obj2 + cost_open + cost_put + cost_close
    else:
        res = grab_obj1 + find_obj2 + action_put + action_close
        cost_list = cost_grab_obj1 + cost_find_obj2 + cost_put + cost_close

    #print(res, target)
    return res, cost_list

def clean_graph(state, goal_spec, last_opened):
    new_graph = {}
    # get all ids
    ids_interaction = []
    nodes_missing = []
    for predicate in goal_spec:
        elements = predicate.split('_')
        nodes_missing += [int(x) for x in elements if x.isdigit()]
        for x in elements[1:]:
            if x.isdigit():
                nodes_missing += [int(x)]
            else:
                nodes_missing += [node['id'] for node in state['nodes'] if node['class_name'] == x]
    nodes_missing += [node['id'] for node in state['nodes'] if node['class_name'] == 'character' or node['category'] in ['Rooms', 'Doors']]
    def clean_node(curr_node):
        return {
            'id': curr_node['id'],
            'class_name': curr_node['class_name'],
            'category': curr_node['category'],
            'states': curr_node['states'],
            'properties': curr_node['properties']
        }

    id2node = {node['id']: clean_node(node) for node in state['nodes']}
    # print([node for node in state['nodes'] if node['class_name'] == 'kitchentable'])
    # print(id2node[235])
    # ipdb.set_trace()
    inside = {}
    for edge in state['edges']:
        if edge['relation_type'] == 'INSIDE':
            if edge['from_id'] not in inside.keys():
                inside[edge['from_id']] = []
            inside[edge['from_id']].append(edge['to_id'])
    
    while (len(nodes_missing) > 0):
        new_nodes_missing = []
        for node_missing in nodes_missing:
            if node_missing in inside:
                new_nodes_missing += [node_in for node_in in inside[node_missing] if node_in not in ids_interaction]
            ids_interaction.append(node_missing)
        nodes_missing = list(set(new_nodes_missing))

    if last_opened is not None:
        obj_id = int(last_opened[1][1:-1])
        if obj_id not in ids_interaction:
            ids_interaction.append(obj_id)

    # for clean up tasks, add places to put objects to
    augmented_class_names = []
    for key in goal_spec:
        elements = key.split('_')
        if elements[0] == 'off':
            if id2node[int(elements[2])]['class_name'] in ['dishwasher', 'kitchentable']:
                augmented_class_names += ['kitchencabinets', 'kitchencounterdrawer', 'kitchencounter']
                break
    for key in goal_spec:
        elements = key.split('_')
        if elements[0] == 'off':
            if id2node[int(elements[2])]['class_name'] in ['sofa', 'chair']:
                augmented_class_names += ['coffeetable']
                break
    containers = [[node['id'], node['class_name']] for node in state['nodes'] if node['class_name'] in augmented_class_names]
    for obj_id in containers:
        if obj_id not in ids_interaction:
            ids_interaction.append(obj_id)


    new_graph = {
            "edges": [edge for edge in state['edges'] if edge['from_id'] in ids_interaction and edge['to_id'] in ids_interaction],
            "nodes": [id2node[id_node] for id_node in ids_interaction]
    }

    return new_graph

def mp_run_mcts(root_node, mcts, nb_steps, last_subgoal, opponent_subgoal):
    heuristic_dict = {
        'find': find_heuristic,
        'grab': grab_heuristic,
        'put': put_heuristic,
        'putIn': putIn_heuristic,
        'sit': sit_heuristic,
        'turnOn': turnOn_heuristic
    }
    # res = root_node * 2
    try:
        new_mcts = copy.deepcopy(mcts)
        res = new_mcts.run(root_node, nb_steps, heuristic_dict, last_subgoal, opponent_subgoal)
    except Exception as e:
        # print("plan fail in index", root_node.particle_id)
        # traceback.print_stack()
        # print("raising")
        # print("Exception...")
        # print(utils_exception.ExceptionWrapper(e))
        # print('---')
        return utils_exception.ExceptionWrapper(e)
    return res


def get_plan(mcts, particles, env, nb_steps, goal_spec, last_subgoal, last_action, opponent_subgoal=None, num_process=10, verbose=True):    
    heuristic_dict = {
        'find': find_heuristic,
        'grab': grab_heuristic,
        'put': put_heuristic,
        'putIn': putIn_heuristic,
        'sit': sit_heuristic,
        'turnOn': turnOn_heuristic
    }
    root_nodes = []
    for particle_id in range(len(particles)):
        root_action = None
        root_node = Node(id=(root_action, [copy.deepcopy(goal_spec), 0, []]),
                         particle_id=particle_id,
                         state=copy.deepcopy(particles[particle_id]),
                         num_visited=0,
                         sum_value=0,
                         is_expanded=False)
        root_nodes.append(root_node)
    

    length_plan = 5
    # root_nodes = list(range(10))
    mp_run = partial(mp_run_mcts, 
        mcts=mcts, 
        nb_steps=nb_steps, 
        last_subgoal=last_subgoal, 
        opponent_subgoal=opponent_subgoal)

    if len(root_nodes) == 0:
        print("No root nodes")
        raise Exception
    if num_process > 0:
        # try:
        with mp.Pool(min(num_process, len(root_nodes))) as p:
            info = p.map(mp_run, root_nodes)
        for info_item in info:
            if info_item is None:
                print("ITEM IS NONE")
                raise utils_exception.PlannerException
            # print(info_item)       
            if  isinstance(info_item, utils_exception.ExceptionWrapper):
                # print("rasiing")
                print("raising")
                info_item.re_raise()
                print("done")

        # print("Done planner")
        # except Exception as e:
        #     print("EXcept", e)
        #     print("ERROR IN the MCTS planner")

        #     raise utils_exception.PlannerException(e)
    else:
        info = [mp_run(rn) for rn in root_nodes]
        for info_item in info:
            if  isinstance(info_item, utils_exception.ExceptionWrapper):
                print("rasiing")
                info_item.re_raise()



    if num_process > 0:
        print("Plan Done")
    rewards_all = [inf[-1] for inf in info] 
    plans_all = [inf[1] for inf in info]
    index_action = 0
    length_plan = 5
    prev_index_particles = list(range(len(info)))
    
    final_actions = []
    lambd = 0.5
    while index_action < length_plan:
        max_action = None
        max_score = None
        action_count_dict = {}
        action_reward_dict = {}
        # Which particles we select now
        index_particles = [p_id for p_id in prev_index_particles if len(plans_all[p_id]) > index_action]
        # print(index_particles)
        if len(index_particles) == 0:
            index_action += 1
            continue
        for ind in index_particles:
            action = plans_all[ind][index_action]
            if action is None:
                continue
            reward = rewards_all[ind][index_action]
            if not action in action_count_dict:
                action_count_dict[action] = []
                action_reward_dict[action] = 0
            action_count_dict[action].append(ind)
            action_reward_dict[action] += reward

        for action in action_count_dict:
            # Average reward of this action
            average_reward = action_reward_dict[action] * 1.0 / len(action_count_dict[action])
            # Average proportion of particles
            average_visit =  len(action_count_dict[action]) * 1.0 / len(index_particles)
            score = average_reward * lambd + average_visit

            if max_score is None or max_score < score:
                max_score = score
                max_action = action

        index_action += 1
        prev_index_particles = action_count_dict[max_action]
        # print(max_action, prev_index_particles)
        final_actions.append(max_action)

    # If there is no action predicted but there were goals missing...
    if len(final_actions) == 0:
        print("No actions")
        ipdb.set_trace()

    plan = final_actions
    subgoals = [[None, None, None], [None, None, None]]
    # next_root, plan, subgoals = mp_run_mcts(root_nodes[0])
    next_root = None

 

    if verbose:
        print('plan', plan)
        print('subgoal', subgoals)
    sample_id = None
    
    if sample_id is not None:
        res[sample_id] = plan
    else:

        return plan, next_root, subgoals


class MCTS_agent_particle_v2:
    """
    MCTS for a single agent
    """
    def __init__(self, agent_id, char_index,
                 max_episode_length, num_simulation, max_rollout_steps, c_init, c_base, num_particles=20, recursive=False,
                 num_samples=1, num_processes=1, comm=None, logging=False, logging_graphs=False, 
                 agent_params={}, seed=None):
        self.agent_type = 'MCTS'
        self.verbose = False
        self.recursive = recursive

        #self.env = unity_env.env
        if seed is None:
            seed = random.randint(0,100)
        self.seed = seed
        self.logging = logging
        self.logging_graphs = logging_graphs

        self.last_obs = None
        self.last_plan = None

        self.agent_id = agent_id
        self.char_index = char_index
        self.sim_env = VhGraphEnv()
        self.sim_env.pomdp = True
        self.belief = None

        self.belief_params = agent_params['belief']
        self.agent_params = agent_params
        self.max_episode_length = max_episode_length
        self.num_simulation = num_simulation
        self.max_rollout_steps = max_rollout_steps
        self.c_init = c_init
        self.c_base = c_base
        self.num_samples = num_samples
        self.num_processes = num_processes
        self.num_particles = num_particles

        self.previous_belief_graph = None
        self.verbose = False

        # self.should_close = True
        # if self.planner_params:
        #     if 'should_close' in self.planner_params:
        #         self.should_close = self.planner_params['should_close']

        self.mcts = None 
        #MCTS_particles_v2(self.agent_id, self.char_index, self.max_episode_length,
        #                 self.num_simulation, self.max_rollout_steps,
        #                 self.c_init, self.c_base, agent_params=self.agent_params)
        
        self.particles = [None for _ in range(self.num_particles)]
        self.particles_full = [None for _ in range(self.num_particles)]

        #if self.mcts is None:
        #    raise Exception

        # Indicates whether there is a unity simulation
        self.comm = comm


    def filtering_graph(self, graph):
        new_edges = []
        edge_dict = {}
        for edge in graph['edges']:
            key = (edge['from_id'], edge['to_id'])
            if key not in edge_dict:
                edge_dict[key] = [edge['relation_type']]
                new_edges.append(edge)
            else:
                if edge['relation_type'] not in edge_dict[key]:
                    edge_dict[key] += [edge['relation_type']]
                    new_edges.append(edge)

        graph['edges'] = new_edges
        return graph


    def sample_belief(self, obs_graph):
        new_graph = self.belief.update_graph_from_gt_graph(obs_graph)
        self.previous_belief_graph = self.filtering_graph(new_graph)
        return new_graph


    def get_relations_char(self, graph):
        # TODO: move this in vh_mdp
        char_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'character'][0]
        edges = [edge for edge in graph['edges'] if edge['from_id'] == char_id]
        print('Character:')
        print(edges)
        print('---')

    def new_obs(obs, ignore_ids=None):
        curr_visible_ids = [node['id'] for node in obs['nodes']]
        relations = {
            'ON': 0,
            'INSIDE': 1,
        }
        num_relations = len(relations)
        if set(curr_visible_ids) != set(self.last_obs['ids']):
            new_obs = True
        else:
            state_ids = np.zeros((len(curr_visible_ids), 4))
            edge_ids = np.zeros((len(curr_visible_ids), len(curr_visible_ids), num_relations))
            idnode2id = []
            for idnode, node in enumerate(nodes):
                idnode2id[node['id']] = idnode
                state_ids[idnode, 0] = 'OPEN' in node['states']
                state_ids[idnode, 1] = 'CLOSED' in node['states']
                state_ids[idnode, 0] = 'ON' in node['states']
                state_ids[idnode, 1] = 'OFF' in node['states']
            for edge in node['edges']:
                if edge['relation_type'] in relations.keys():
                    edge_id = relations[edge['relation_type']]
                    from_id, to_id = idnode2id[edge['from_id']], idnode2id[edge['to_id']]
                    edge_ids[from_id, to_id, edge_id] = 1

            if ignore_ids != None:
                # We will ignore some edges, for instance if we are grabbing an object
                self.last_obs['edges'][ignore_ids, :] = edge_ids[ignore_ids, :]
                self.last_obs['edges'][:, ignore_ids] = edge_ids[:, ignore_ids]

            if state_ids != self.last_obs['state'] or edge_ids != self.last_obs['edges']:
                new_obs = True
                self.last_obs['state'] = state_ids
                self.last_obs['edges'] = edge_ids
        return new_obs

    def get_action(self, obs, goal_spec, opponent_subgoal=None):


        # Create the particles
        # pdb.set_trace()

        self.belief.update_belief(obs)



        # TODO: maybe we will want to keep the previous belief graph to avoid replanning
        #self.sim_env.reset(self.previous_belief_graph, {0: goal_spec, 1: goal_spec})

        last_action = self.last_action
        last_subgoal = self.last_subgoal
        last_plan = self.last_plan

        # TODO: is this correct?
        nb_steps = 0
        root_action = None
        root_node = None
        verbose = self.verbose

        # If the current obs is the same as the last obs
        ignore_id = None

        should_replan = True

        goal_names = [goal_name.split('_')[1] for goal_name, ct in goal_spec.items() if ct[0] > 0]
        goal_ids = [nodeobs['id'] for nodeobs in obs['nodes'] if nodeobs['class_name'] in goal_names]
        close_ids = [edge['to_id'] for edge in obs['edges'] if edge['from_id'] == self.agent_id and edge['relation_type'] in ['CLOSE', 'INSIDE']]
        plan = []

        subgoals = self.last_subgoal
        if last_plan is not None and len(last_plan) > 0:
            should_replan = False

            # If there is a goal object that was not there before
            next_id_interaction = []
            if len(last_plan) > 1:
                next_id_interaction.append(int(last_plan[1].split('(')[1].split(')')[0]))
            
            new_observed_objects = set(goal_ids) - set(self.last_obs['goal_objs']) - set(next_id_interaction)
            # self.last_obs = {'goal_objs': goal_ids}
            if len(new_observed_objects) > 0:
                # New goal, need to replan 
                should_replan = True
            else:

                visible_ids = {node['id']: node for node in obs['nodes']}
                curr_plan = last_plan

                first_action_non_walk = [act for act in last_plan[1:] if 'walk' not in act]

                # If the first action other than walk is OPEN/CLOSE and the object is already open/closed...
                if len(first_action_non_walk):
                    first_action_non_walk = first_action_non_walk[0]
                    if 'open' in first_action_non_walk:
                        obj_id = int(curr_plan[0].split('(')[1].split(')')[0])
                        if obj_id in visible_ids:
                            if 'OPEN' in visible_ids[obj_id]['states']:
                                should_replan = True
                                print("IS OPEN")
                    elif 'close' in first_action_non_walk:
                        obj_id = int(curr_plan[0].split('(')[1].split(')')[0])
                        if obj_id in visible_ids:

                            if 'CLOSED' in visible_ids[obj_id]['states']:
                                should_replan = True
                                print("IS CLOSED")

                
                if 'open' in last_plan[0] or 'close' in last_plan[0] or 'put' in last_plan[0] or 'grab' in last_plan[0]:
                    if len(last_plan) == 1:
                        should_replan = True
                    else:
                        curr_plan = last_plan[1:]


                if 'open' in curr_plan[0] or 'close' in curr_plan[0] or 'put' in curr_plan[0] or 'grab' in curr_plan[0]:
                    obj_id = int(curr_plan[0].split('(')[1].split(')')[0])
                    if not obj_id in close_ids or not obj_id in visible_ids:
                        should_replan = True

                next_action = not should_replan
                while next_action and 'walk' in curr_plan[0]:

                    obj_id = int(curr_plan[0].split('(')[1].split(')')[0])

                
                    # If object is not visible, replan
                    if obj_id not in visible_ids:
                        should_replan = True
                        next_action = False
                    else:
                        if obj_id in close_ids:
                            if len(curr_plan) == 1:
                                should_replan = True
                                next_action = False
                            else:
                                curr_plan = curr_plan[1:]
                                
                        else:
                            # Keep with previous action
                            next_action = False
                
                if not should_replan:
                    plan = curr_plan

        self.last_obs = {'goal_objs': goal_ids}
        
        time1 = time.time()
        
        if should_replan:
            # ipdb.set_trace()
            for particle_id, particle in enumerate(self.particles):
                belief_states = []
                obs_ids = [node['id'] for node in obs['nodes']]

                if True: #particle is None:
                    new_graph = self.belief.update_graph_from_gt_graph(obs, resample_unseen_nodes=True, update_belief=False)
                    # ipdb.set_trace()
                    init_state = clean_graph(new_graph, goal_spec, self.mcts.last_opened)
                    satisfied, unsatisfied = utils_env.check_progress(init_state, goal_spec)
                    init_vh_state = self.sim_env.get_vh_state(init_state)

                    self.particles[particle_id] = (init_vh_state, init_state, satisfied, unsatisfied)

                else:
                    prev_graph = self.particles_full[particle_id]
                    new_graph = self.belief.update_graph_from_gt_graph(
                        obs, sampled_graph=prev_graph, resample_unseen_nodes=True, update_belief=False)
                    init_state = clean_graph(new_graph, goal_spec, self.mcts.last_opened)
                    init_vh_state = self.sim_env.get_vh_state(init_state)
                    if 'waterglass' not in [node['class_name'] for node in init_state['nodes']] and goal_spec['on_waterglass_232'][0] > 0:
                        raise Exception

                    satisfied, unsatisfied = utils_env.check_progress(init_state, goal_spec)
                    self.particles[particle_id] = (init_vh_state, init_state, satisfied, unsatisfied)
                

                self.particles_full[particle_id] = new_graph
            # print('-----')

            plan, root_node, subgoals = get_plan(self.mcts, self.particles, self.sim_env, nb_steps, goal_spec, last_plan, last_action, opponent_subgoal, verbose=verbose, num_process=self.num_processes)
            
            print(colored(plan[:min(len(plan), 10)], 'cyan'))
        else:
            subgoals = [[None, None, None], [None, None, None]]
        if len(plan) == 0:
            print("Plan empty")
            raise Exception
            
        
        if len(plan) > 0:
            action = plan[0]
            action = action.replace('[walk]', '[walktowards]')
        else:
            action = None
        if self.logging:
            info = {
                'plan': plan,
                'subgoals': subgoals,
                'belief': copy.deepcopy(self.belief.edge_belief),
            }
            if self.logging_graphs:
                info.update(
                    {'obs': obs['nodes']})
        else:
            info = {}

        self.last_action = action
        # self.last_subgoal = subgoals[0] if len(subgoals) > 0 else None
        self.last_plan = plan
        # print(info['subgoals'])
        print(action)
        time2 = time.time()
        print("Time: ", time2 - time1)
        return action, info

    def reset(self, observed_graph, gt_graph, task_goal, seed=0, simulator_type='python', is_alice=False):

        self.last_action = None
        self.last_subgoal = None
        self.init_gt_graph = gt_graph
        """TODO: do no need this?"""
        #if 'waterglass' not in [node['class_name'] for node in self.init_gt_graph['nodes']]:
        #    ipdb.set_trace()
        self.belief = belief.Belief(gt_graph, agent_id=self.agent_id, seed=seed, belief_params=self.belief_params)
        self.sim_env.reset(gt_graph)

        self.mcts = MCTS_particles_v2(gt_graph, self.agent_id, self.char_index, self.max_episode_length,
                         self.num_simulation, self.max_rollout_steps,
                         self.c_init, self.c_base, seed=seed, agent_params=self.agent_params)

        # self.mcts.should_close = self.should_close

