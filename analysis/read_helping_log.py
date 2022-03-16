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


def get_class_mode(agent_args):
    mode_str = '{}_opencost{}_closecost{}_walkcost{}_forgetrate{}'.format(
        agent_args['obs_type'],
        agent_args['open_cost'],
        agent_args['should_close'],
        agent_args['walk_cost'],
        agent_args['belief']['forget_rate'],
    )
    return mode_str


def get_edge_class(pred, t, source='pred'):
    # pred_edge_prob = pred['edge_prob']
    # print(len(pred['edge_input'][t]), len(pred['edge_pred'][t]))
    t = min(t, len(pred['edge_pred']) - 1)
    edge_pred = pred['edge_pred'][t] if source == 'pred' else pred['edge_input'][t]
    pred_edge_names = pred['edge_names']
    pred_nodes = pred['nodes']
    pred_from_ids = pred['from_id'] if source == 'pred' else pred['from_id_input']
    pred_to_ids = pred['to_id'] if source == 'pred' else pred['to_id_input']

    # edge_prob = pred_edge_prob[t]
    # edge_pred = np.argmax(edge_prob, 1)

    edge_pred_class = {}

    num_edges = len(edge_pred)
    # print(pred_from_ids[t], num_edges)
    for edge_id in range(num_edges):
        from_id = pred_from_ids[t][edge_id]
        to_id = pred_to_ids[t][edge_id]
        from_node_name = pred_nodes[from_id]
        to_node_name = pred_nodes[to_id]
        # if object_name in from_node_name or object_name in to_node_name:
        edge_name = pred_edge_names[edge_pred[edge_id]]
        if to_node_name.split('.')[1] == '-1':
            continue
        if edge_name in ['inside', 'on']:  # disregard room locations + plate
            if to_node_name.split('.')[0] in [
                'kitchen',
                'livingroom',
                'bedroom',
                'bathroom',
                'plate',
            ]:
                continue
            if from_node_name.split('.')[0] in [
                'kitchen',
                'livingroom',
                'bedroom',
                'bathroom',
                'character',
            ]:
                continue
        else:
            continue
        # if from_node_name.split('.')[0] not in [
        #     'apple',
        #     'cupcake',
        #     'plate',
        #     'waterglass',
        # ]:
        #     continue

        # if from_node_name.split('.')[0]

        # # TODO: need to infer the correct edge class
        # if 'table' in to_node_name.split('.')[0]:
        #     ipdb.set_trace()
        #     edge_name = 'on'

        edge_class = '{}_{}_{}'.format(
            edge_name, from_node_name.split('.')[0], to_node_name.split('.')[1]
        )
        # print(from_node_name, to_node_name, edge_name)
        if edge_class not in edge_pred_class:
            edge_pred_class[edge_class] = 1
        else:
            edge_pred_class[edge_class] += 1
    return edge_pred_class


def get_edge_instance(pred, t, source='pred'):
    # pred_edge_prob = pred['edge_prob']
    # print(len(pred['edge_input'][t]), len(pred['edge_pred'][t]))
    t = min(t, len(pred['edge_pred']) - 1)
    edge_pred = pred['edge_pred'][t] if source == 'pred' else pred['edge_input'][t]
    pred_edge_names = pred['edge_names']
    pred_nodes = pred['nodes']
    pred_from_ids = pred['from_id'] if source == 'pred' else pred['from_id_input']
    pred_to_ids = pred['to_id'] if source == 'pred' else pred['to_id_input']

    # edge_prob = pred_edge_prob[t]
    # edge_pred = np.argmax(edge_prob, 1)

    edge_pred_ins = {}

    num_edges = len(edge_pred)
    # print(pred_from_ids[t], num_edges)
    for edge_id in range(num_edges):
        from_id = pred_from_ids[t][edge_id]
        to_id = pred_to_ids[t][edge_id]
        from_node_name = pred_nodes[from_id]
        to_node_name = pred_nodes[to_id]
        # if object_name in from_node_name or object_name in to_node_name:
        edge_name = pred_edge_names[edge_pred[edge_id]]
        if to_node_name.split('.')[1] == '-1':
            continue
        if edge_name in ['inside', 'on']:  # disregard room locations + plate
            if to_node_name.split('.')[0] in [
                'kitchen',
                'livingroom',
                'bedroom',
                'bathroom',
                'plate',
            ]:
                continue
            if from_node_name.split('.')[0] in [
                'kitchen',
                'livingroom',
                'bedroom',
                'bathroom',
                'character',
            ]:
                continue
        else:
            continue
        # if from_node_name.split('.')[0] not in [
        #     'apple',
        #     'cupcake',
        #     'plate',
        #     'waterglass',
        # ]:
        #     continue

        edge_class = '{}_{}_{}'.format(
            edge_name, from_node_name.split('.')[0], to_node_name.split('.')[1]
        )

        # print(from_node_name, to_node_name, edge_name)
        if edge_class not in edge_pred_ins:
            edge_pred_ins[edge_class] = {
                'count': 0,
                'grab_obj_ids': [],
                'container_ids': [int(to_node_name.split('.')[1])],
            }
        edge_pred_ins[edge_class]['count'] += 1
        edge_pred_ins[edge_class]['grab_obj_ids'].append(
            int(from_node_name.split('.')[1])
        )
    return edge_pred_ins


def get_edge_instance_from_pred(pred):
    # pred_edge_prob = pred['edge_prob']
    # print(len(pred['edge_input'][t]), len(pred['edge_pred'][t]))

    edge_pred = pred['edge_pred'][-1]
    pred_edge_names = pred['edge_names']
    pred_nodes = pred['nodes']
    pred_from_ids = pred['from_id']
    pred_to_ids = pred['to_id']

    # edge_prob = pred_edge_prob[t]
    # edge_pred = np.argmax(edge_prob, 1)

    edge_pred_ins = {}
    edge_list = []

    num_edges = len(edge_pred)
    # print(pred_from_ids[t], num_edges)
    for edge_id in range(num_edges):
        from_id = pred_from_ids[-1][edge_id]
        to_id = pred_to_ids[-1][edge_id]
        from_node_name = pred_nodes[from_id].split('.')[0]
        to_node_name = pred_nodes[to_id].split('.')[0]
        from_node_id = int(pred_nodes[from_id].split('.')[1])
        to_node_id = int(pred_nodes[to_id].split('.')[1])
        edge_name = pred_edge_names[edge_pred[edge_id]]

        if 'hold' in edge_name:  # ignore left or right hand
            edge_name = 'offer'
            # ipdb.set_trace()
            # continue  # TODO: add handing over plan

        if edge_name in ['inside', 'on']:  # disregard room locations + plate
            if to_node_name in [
                'kitchen',
                'livingroom',
                'bedroom',
                'bathroom',
                'plate',
            ]:
                continue

        if from_node_name not in [
            'plate',
            'cutleryfork',
            'waterglass',
            'cupcake',
            'salmon',
            'apple',
            'remotecontrol',
            'chips',
            'condimentbottle',
            'condimentshaker',
            'wineglass',
            'pudding',
        ]:
            continue
        elif edge_name in ['close']:
            continue
        # if from_node_name not in ['apple', 'cupcake', 'plate', 'waterglass']:
        #     continue

        edge_class = "{}_{}_{}".format(edge_name, from_node_name, to_node_id)
        if edge_name == 'offer':
            if edge_class not in edge_pred_ins:
                edge_pred_ins[edge_class] = {
                    'count': 0,
                    'grab_obj_ids': [],
                    'container_ids': [from_id],
                }
            edge_pred_ins[edge_class]['count'] += 1
            edge_pred_ins[edge_class]['grab_obj_ids'].append(to_node_id)
        else:
            if edge_class not in edge_pred_ins:
                edge_pred_ins[edge_class] = {
                    'count': 0,
                    'grab_obj_ids': [],
                    'container_ids': [to_id],
                }
            edge_pred_ins[edge_class]['count'] += 1
            edge_pred_ins[edge_class]['grab_obj_ids'].append(from_node_id)
        edge_list.append(
            "{}_{}.{}_{}.{}".format(
                edge_name, from_node_name, from_node_id, to_node_name, to_node_id
            )
        )
    # print(edge_list)
    # ipdb.set_trace()
    return edge_pred_ins, edge_list


def get_edge_instance_from_state(state):
    id2node = {node['id']: node['class_name'] for node in state['nodes']}
    # print(id2node)
    edge_pred_ins = {}
    edge_list = []
    for edge in state['edges']:
        edge_name, from_id, to_id = (
            edge['relation_type'].lower(),
            edge['from_id'],
            edge['to_id'],
        )
        if 'hold' in edge_name:  # ignore left or right hand
            edge_name = 'offer'
            # ipdb.set_trace()
            # continue  # TODO: add handing over plan
        from_node_name = id2node[from_id]
        to_node_name = id2node[to_id]
        if edge_name in ['inside', 'on']:  # disregard room locations + plate
            if to_node_name in [
                'kitchen',
                'livingroom',
                'bedroom',
                'bathroom',
                'plate',
            ]:
                continue
            if from_node_name.split('.')[0] in [
                'kitchen',
                'livingroom',
                'bedroom',
                'bathroom',
                'character',
            ]:
                continue
        elif edge_name in ['close']:
            continue
        # if from_node_name not in ['apple', 'cupcake', 'plate', 'waterglass']:
        #     continue

        edge_class = "{}_{}_{}".format(edge_name, from_node_name, to_id)
        if edge_name == 'offer':
            if edge_class not in edge_pred_ins:
                edge_pred_ins[edge_class] = {
                    'count': 0,
                    'grab_obj_ids': [],
                    'container_ids': [from_id],
                }
            edge_pred_ins[edge_class]['count'] += 1
            edge_pred_ins[edge_class]['grab_obj_ids'].append(to_id)
        else:
            if edge_class not in edge_pred_ins:
                edge_pred_ins[edge_class] = {
                    'count': 0,
                    'grab_obj_ids': [],
                    'container_ids': [to_id],
                }
            edge_pred_ins[edge_class]['count'] += 1
            edge_pred_ins[edge_class]['grab_obj_ids'].append(from_id)
        edge_list.append(
            "{}_{}.{}_{}.{}".format(
                edge_name, from_node_name, from_id, to_node_name, to_id
            )
        )
    return edge_pred_ins, edge_list


def aggregate_multiple_pred(preds, t, change=False):
    t = min(t, len(preds[0]['edge_pred']) - 1)
    edge_classes = []
    edge_pred_class_all = {}
    N_preds = len(preds)
    for pred in preds:
        edge_pred_class = get_edge_class(pred, t)
        edge_classes += list(edge_pred_class.keys())
        for edge_class, count in edge_pred_class.items():
            if edge_class not in edge_pred_class_all:
                edge_pred_class_all[edge_class] = [count]
            else:
                edge_pred_class_all[edge_class] += [count]
    if change:
        edge_input_class = get_edge_class(preds[0], t, 'input')
        edge_classes += list(edge_input_class.keys())

    edge_classes = sorted(list(set(edge_classes)))
    edge_pred_class_estimated = {}
    for edge_class in edge_classes:
        if edge_class not in edge_pred_class_all:
            edge_pred_class_estimated[edge_class] = (-edge_input_class[edge_class], 0)
            continue
        curr_len = len(edge_pred_class_all[edge_class])
        if curr_len < N_preds:
            edge_pred_class_all[edge_class] += [0] * (N_preds - curr_len)
        if change:
            c = (
                np.mean(edge_pred_class_all[edge_class]) - edge_input_class[edge_class]
                if edge_class in edge_input_class
                else np.mean(edge_pred_class_all[edge_class])
            )
        else:
            c = np.mean(edge_pred_class_all[edge_class])
        edge_pred_class_estimated[edge_class] = (
            c,
            np.std(edge_pred_class_all[edge_class]),
        )
        # print(edge_class, edge_pred_class_estimated[edge_class])
    return edge_pred_class_estimated


def edge2goal(edge):
    # edge format: edgeName_fromNodeName.id_toNodeName.id
    elements = edge.split('_')
    edge_name = elements[0]
    from_node_id = int(elements[1].split('.')[-1])
    from_node_name = elements[1].split('.')[0]
    to_node_id = int(elements[2].split('.')[-1])

    goal_name = '{}_{}_{}'.format(edge_name, from_node_name, to_node_id)
    if edge_name == 'offer':
        goal = {
            goal_name: {
                'count': 1,
                'grab_obj_ids': [to_node_id],
                'container_ids': [from_node_id],
            }
        }
    else:
        goal = {
            goal_name: {
                'count': 1,
                'grab_obj_ids': [from_node_id],
                'container_ids': [to_node_id],
            }
        }
    return goal


def edge2name(edge):
    elements = edge.split('_')
    edge_name = elements[0]
    from_node_id = int(elements[1].split('.')[-1])
    from_node_name = elements[1].split('.')[0]
    to_node_id = int(elements[2].split('.')[-1])
    to_node_name = elements[2].split('.')[0]
    if edge_name == 'offer':
        goal_name = '{}_{}_{}'.format(edge_name, to_node_name, from_node_id)
    else:
        goal_name = '{}_{}_{}'.format(edge_name, from_node_name, to_node_id)
    return goal_name


log_dir = "/data/vision/torralba/frames/data_acquisition/SyntheticStories/agent_preferences/tshu/agent_preferences/outputs/helping_states_nohold_20_1.0_1.0/test_env_task_set_60_full_task.all"
# log_dir = "/data/vision/torralba/frames/data_acquisition/SyntheticStories/agent_preferences/tshu/agent_preferences/outputs/helping_states_20_1.0_1.0/test_env_task_set_60_full_task.all"
log_dir = "/data/vision/torralba/frames/data_acquisition/SyntheticStories/agent_preferences/tshu/agent_preferences/outputs/helping_action_freq_v2_20/test_env_task_set_60_full_task.all"
log_dir = "/data/vision/torralba/frames/data_acquisition/SyntheticStories/agent_preferences/tshu/agent_preferences/outputs/heling_gt/test_env_task_set_60_full_task.all"


results = pickle.load(open(log_dir + '/results_4.pik', 'rb'))

# print(results)

episode_ids = [3]  # 3, 290, 453, 600]

for episode_id in episode_ids:
    saved_info = pickle.load(
        open(log_dir + '/logs_episode.{}_iter.0.pik'.format(episode_id), 'rb')
    )
    task_name = saved_info['task_name']
    gt_goals = saved_info['gt_goals']
    actions = saved_info['action']
    graph_results = saved_info['graph_results']
    print(task_name)
    # print(gt_goals)
    for edge_class, ids in gt_goals.items():
        print(edge_class, ids['count'])
    print(len(actions[1]), len(graph_results))
    T = len(actions[1])

    print(actions[0][0])
    print(actions[0][1])
    for steps in range(0, T):
        # if steps < 25:
        #     continue
        edge_pred_class_estimated = aggregate_multiple_pred(
            graph_results[steps], steps, change=True
        )
        # print("step ", steps + 2)
        # print('change pred:')
        # for edge_class, count in edge_pred_class_estimated.items():
        #     if (
        #         edge_pred_class_estimated[edge_class][0] < 1e-6
        #         and edge_pred_class_estimated[edge_class][1] < 1e-6
        #     ):
        #         continue
        #     print(edge_class,edge_pred_class_estimated[edge_class])
        print(actions[0][2 + steps], actions[1][steps])
