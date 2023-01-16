import torch
from dgl import DGLGraph
import numpy as np
import os
import ipdb
import json
import pdb

def condense_walking(program):
    action_split = [instr.split(' ') for instr in program]
    indices = [0]
    for i in range(1, len(program)):
        do_skip = False
        if action_split[i][0] == '[walktowards]' and action_split[i-1][0] == '[walktowards]' and action_split[i][2] == action_split[i-1][2]:
            do_skip = True
        if not do_skip:
            indices.append(i)
    # ipdb.set_trace()
    return indices



class DictObjId:
    def __init__(self, elements=None, include_other=True):
        self.el2id = {}
        self.id2el = []
        self.include_other = include_other
        if include_other:
            self.el2id = {'other': 0}
            self.id2el = ['other']
        if elements:
            for element in elements:
                self.add(element)

    def get_el(self, id):
        if self.include_other and id >= len(self.id2el):
            return self.id2el[0]
        else:
            return self.id2el[id]

    def valid_el(self, el):
        return el in self.el2id.keys()

    def get_id(self, el):
        el = el.lower()
        if el in self.el2id.keys():
            return self.el2id[el]
        else:
            if self.include_other:
                return 0
            else:
                raise Exception

    def add(self, el):
        el = el.lower()
        if el not in self.el2id.keys():
            num_elems = len(self.id2el)
            self.el2id[el] = num_elems
            self.id2el.append(el)

    def __len__(self):
        return len(self.id2el)


class GraphHelper:
    def __init__(
        self,
        max_num_objects=100,
        max_num_edges=200,
        current_task=None,
        simulator_type='unity',
        include_touch=False,
        toy_dataset=False,
    ):
        self.toy_dataset = toy_dataset
        self.states = ['on', 'open', 'off', 'closed']
        self.relations = ['inside', 'on', 'hold']
        self.simulaor_type = simulator_type
        self.objects = self.get_objects()
        self.rooms = ['bathroom', 'bedroom', 'kitchen', 'livingroom']
        self.removed_categories = [
            'foor',
            'wall',
            'ceiling',
            'window',
            'lamp',
            'walllamp',
        ]

        if simulator_type == 'unity':
            self.actions = [
                'turnleft',
                'walkforward',
                'turnright',
                'walktowards',
                'open',
                'close',
                'put',
                'grab',
                'no_action',
            ]
            if include_touch:
                self.actions = self.actions[:-1] + ['touch', 'no_action']
            self.actions_no_args = ['turnleft', 'walkforward', 'turnright']
        else:
            self.actions = ['walk', 'open', 'close', 'put', 'grab', 'no_action']
        self.object_dict = DictObjId(
            self.objects + ['character'] + self.rooms + ['no_obj']
        )
        self.relation_dict = DictObjId(self.relations)
        self.state_dict = DictObjId(self.states)
        self.action_dict = DictObjId(self.actions, include_other=False)

        self.num_actions = len(self.actions)
        self.num_objects = max_num_objects
        self.num_edges = max_num_edges
        self.num_edge_types = len(self.relation_dict)
        self.num_classes = len(self.object_dict)
        self.num_states = len(self.state_dict)
        self.num_states = len(self.states)

        self.obj1_affordance = None
        self.get_action_affordance_map(current_task=current_task)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(f'{dir_path}/../dataset/object_info_small.json', 'r') as f:
            content = json.load(f)
        self.object_dict_types = content

        self.task_graph_dict, self.task_graph_list = self.build_task_graph_dict()

    def actionstr2index(self, action_str):
        if action_str is None:
            return self.action_dict.get_id('no_action'), -1, -1
        action_split = action_str.split()
        action = action_split[0][1:-1]
        obj1, obj2 = -1, -1
        if 'put' in action:
            action = 'put'
        if len(action_split) > 1:
            obj1 = action_split[2][1:-1]
            if len(action_split) > 4:
                obj2 = action_split[4][1:-1]

        try:
            action_id = self.action_dict.get_id(action)
            obj1_id = int(obj1)
            obj2_id = int(obj2)
        except:
            print("Error getting action {}, {}".format(action_str, action))
            ipdb.set_trace()
            raise Exception
        return action_id, obj1_id, obj2_id

    def get_action_affordance_map(self, current_task=None, id2node=None):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(f'{dir_path}/../dataset/object_info_small.json', 'r') as f:
            content = json.load(f)

        n_actions = len(self.actions)
        n_objects = len(self.object_dict)
        self.obj1_affordance = np.zeros((n_actions, n_objects))

        id_no_obj = self.object_dict.get_id('no_obj')
        id_grab = np.array(
            [self.object_dict.get_id(obj_name) for obj_name in content['objects_grab']]
        )
        id_surface = np.array(
            [
                self.object_dict.get_id(obj_name)
                for obj_name in content['objects_surface']
            ]
        )
        id_containers = np.array(
            [
                self.object_dict.get_id(obj_name)
                for obj_name in content['objects_inside']
            ]
        )
        for action in self.actions:
            action_id = self.action_dict.get_id(action)
            if args_per_action(action) == 0:

                self.obj1_affordance[action_id, id_no_obj] = 1

            if args_per_action(action) == 1:

                if action in ['open', 'close']:
                    self.obj1_affordance[action_id, id_containers] = 1

                elif action in ['grab']:
                    self.obj1_affordance[action_id, id_grab] = 1

                    if current_task is not None:
                        self.obj1_affordance[action_id, :] = 0
                        obj_names = [t.split('_')[1] for t in current_task[0].keys()]
                        ids_goal = [
                            self.object_dict.get_id(obj_name) for obj_name in obj_names
                        ]
                        id_goal = np.array(ids_goal)
                        self.obj1_affordance[action_id, id_goal] = 1

                elif action in ['walktowards', 'walk']:
                    self.obj1_affordance[action_id, :] = 1
                    self.obj1_affordance[action_id, id_no_obj] = 0

                # putin, put
                elif action.startswith('put'):
                    id2 = id_containers if action == 'putin' else id_surface
                    self.obj1_affordance[action_id, id2] = 1

                    if current_task is not None:
                        self.obj1_affordance[action_id, :] = 0
                        obj_names2 = [
                            id2node[int(t.split('_')[2])]['class_name']
                            for t in current_task[0].keys()
                            if t.split('_')[0] not in ['holds', 'sit', 'turnOn']
                        ]
                        ids_goal2 = np.array(
                            [
                                self.object_dict.get_id(obj_name)
                                for obj_name in obj_names2
                            ]
                        )
                        self.obj1_affordance[action_id, ids_goal2] = 1

        # self.obj1_affordance[:,self.object_dict.get_id('kitchencounterdrawer')] = 0

        self.obj1_affordance[
            self.action_dict.get_id('open'),
            self.object_dict.get_id('kitchencounterdrawer'),
        ] = 0
        self.obj1_affordance[
            self.action_dict.get_id('close'),
            self.object_dict.get_id('kitchencounterdrawer'),
        ] = 0

        if self.simulaor_type == 'unity':
            self.obj1_affordance[
                self.action_dict.get_id('walktowards'),
                self.object_dict.get_id('kitchencounterdrawer'),
            ] = 0
            for action_no_args in self.actions_no_args:
                self.obj1_affordance[
                    self.action_dict.get_id(action_no_args), id_no_obj
                ] = 1
        else:
            self.obj1_affordance[
                self.action_dict.get_id('walk'),
                self.object_dict.get_id('kitchencounterdrawer'),
            ] = 0

        # if np.sum(self.obj1_affordance.sum(0) == 0) > 0:
        #     pdb.set_trace()

    def get_objects(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        if self.toy_dataset:
            object_info_fname = f'{dir_path}/../dataset/object_info_toy.json'
        else:
            #object_info_fname = f'{dir_path}/../dataset/object_info_small.json'
            object_info_fname = f'{dir_path}/../dataset/object_info_good.json'
        with open(object_info_fname, 'r') as f:
            content = json.load(f)
        objects = []
        for obj in content.values():
            objects += obj
        return objects

    def one_hot(self, states):
        # if len(states)  > 0:
        #    pdb.set_trace()
        one_hot = np.zeros(len(self.state_dict) - 1)
        for state in states:
            if self.state_dict.valid_el(state.lower()):
                one_hot[self.state_dict.get_id(state) - 1] = 1
        return one_hot

    def build_task_graph_dict(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))

        if self.toy_dataset:
            object_info_fname = f'{dir_path}/../dataset/object_info_toy.json'
        else:
            object_info_fname = f'{dir_path}/../dataset/object_info_good.json'
        with open(object_info_fname, 'r') as f:
            content = json.load(f)

        objects_container = list(set(content['objects_inside']+content['objects_surface']+['character']))
        task_graph_dict = {}
        task_graph_list = []
        index = 0
        
        # objects_container = ['kitchentable', 'kitchencabinet', 'coffeetable', 'fridge', 'microwave', 'stove', 'cabinet', '']
        for object_graph in sorted(list(set(content['objects_grab']))):
            for object_container in sorted(objects_container):
                tuple_key = (self.object_dict.get_id(object_graph), self.object_dict.get_id(object_container))
                tuple_name = (object_graph, object_container)
                if tuple_key in task_graph_dict:
                    print("Error, key {},{} already exists".format(tuple_key[0], tuple_key[1]))
                    raise Exception
                task_graph_dict[tuple_key] = (tuple_name, index)
                task_graph_list.append(tuple_key)
                index += 1
        for object_container in sorted(self.rooms):
            object_graph = 'character'
            tuple_key = (self.object_dict.get_id(object_graph), self.object_dict.get_id(object_container))
            tuple_name = (object_graph, object_container)
            if tuple_key in task_graph_dict:
                print("Error, key {},{} already exists".format(tuple_key[0], tuple_key[1]))
                raise Exception
            task_graph_dict[tuple_key] = (tuple_name, index)
            task_graph_list.append(tuple_key)
            index += 1

        return task_graph_dict, task_graph_list

    def get_task_graph(self, task_graph, mask=None, use_dict=False):
        results = []
        if use_dict:
            results = {}
        for i in range(int(task_graph.shape[0])):
            if task_graph[i] != 0:
                if mask is None or mask[i] != 0:
                    graph_tuple = self.task_graph_list[i]
                    graph_names = self.task_graph_dict[graph_tuple][0]
                    if use_dict:
                        results[(graph_names[0], graph_names[1])] = int(task_graph[i])
                    else:
                        results.append("{} - {}: {}".format(graph_names[0], graph_names[1], task_graph[i]))
        return results

    def print_task_graph(self, task_graph, mask=None):
        for i in range(task_graph.shape[0]):
            if task_graph[i] != 0:
                if mask is None or mask[i] != 0:
                    graph_tuple = self.task_graph_list[i]
                    graph_names = self.task_graph_dict[graph_tuple][0]
                    print("{} - {}: {}".format(graph_names[0], graph_names[1], task_graph[i]))
                
    def build_task_graph(self, graph_info):
        num_graph_keys = len(self.task_graph_dict.keys())
        task_graph = np.zeros(num_graph_keys)
        num_edges = int(graph_info['mask_edge'].sum())
        edge_tuples = graph_info['edge_tuples']
        class_objects = graph_info['class_objects']
        for i in range(num_edges):
            class_from, class_to = class_objects[edge_tuples[i, 0]], class_objects[edge_tuples[i, 1]]
            tuple_key = (class_from, class_to)
            if tuple_key in self.task_graph_dict:
                task_index = self.task_graph_dict[tuple_key][1]
                task_graph[task_index] += 1
                if task_graph[task_index] > 7:
                    print(self.task_graph_dict[self.task_graph_list[task_index]])
                    ipdb.set_trace()
            else:
                # TODO: check that here we have the right objects
                pass
        return task_graph

    def build_graph_for_task(
        self,
        graph,
        character_id,
        ids=None,
        include_edges=False,
        plot_graph=False,
        action_space_ids=None,
        obs_ids=None,
        level=1,
        relative_coords=True,
        unique_from=False,
    ):
        if ids is None:
            ids = [
                node['id']
                for node in graph['nodes']
                if self.object_dict.valid_el(node['class_name'])
            ]

        for node in graph['nodes']:
            if node['category'] == 'Rooms':
                assert node['class_name'] in self.rooms
        room_ids = [
            node['id'] for node in graph['nodes'] if node['category'] == 'Rooms'
        ]
        if level > 0:
            # Include other rooms
            ids = room_ids + ids

        ids = [idi for idi in ids if idi != character_id]
        ids = list(set(ids))
        id2node = {node['id']: node for node in graph['nodes']}

        nodes = [id2node[idi] for idi in ids]
        nodes.append({'id': -1, 'class_name': 'no_obj', 'states': []})
        node_ids = [node['id'] for node in nodes]

        # TODO: remove
        # ids =[idi for idi in ids if id2node[idi]['class_name'] not in self.rooms]
        # action_space_ids = [idi for idi in action_space_ids if id2node[idi]['class_name'] not in self.rooms]

        # Character is always the first one
        ids = [character_id] + ids
        max_nodes = self.num_objects
        max_edges = self.num_edges

        edges = [edge.copy() for edge in graph['edges']]

        # Add a holding edge
        holding_object = []
        inside_object = []
        for edge in edges:
            if 'hold' in edge['relation_type'].lower():
                edge['relation_type'] = 'hold'
                holding_object.append(edge['to_id'])
            if 'inside' in edge['relation_type'].lower():
                inside_object.append((edge['from_id'], edge['to_id']))

        # If holding an object, remove close edge
        edges = [
            edge
            for edge in edges
            if edge['from_id'] in ids
            and edge['to_id'] in ids
            and edge['relation_type'].lower() in self.relations
        ]
        edges = [
            edge
            for edge in edges
            if (
                edge['relation_type'].lower() != 'close'
                or edge['to_id'] not in holding_object
            )
            and not (edge['from_id'] < 10 and edge['relation_type'].lower() == 'on')
            and (
                edge['relation_type'].lower() != 'close'
                or edge['from_id'] not in holding_object
            )
        ]

        # an object cannot be inside and on a given object
        edges = [
            edge
            for edge in edges
            if not (
                edge['relation_type'].lower() == 'on'
                and (edge['from_id'], edge['to_id']) in inside_object
            )
        ]

        # reverse holding directions
        for it, edge in enumerate(edges):
            if edge['relation_type'] == 'hold':
                # print(edges[it])
                edges[it] = {
                    'from_id': edge['to_id'],
                    'to_id': edge['from_id'],
                    'relation_type': edge['relation_type'],
                }
                # print(edges[it])
        # print("UNIQUE")
        if unique_from:
            try:
                edges = make_edges_unique(edges, room_ids, id2node)
            except:
                "Failure in making edges unique"
                raise Exception
        # Check if there is more than one edge between two nodes
        edge_tup = [(edge['from_id'], edge['to_id']) for edge in edges]

        if unique_from:
            # print("CHECK")
            repeated = []
            edge_from = [edge['from_id'] for edge in edges]
            if len(set(edge_from)) != len(edge_from):

                for el in list(set(edge_from)):
                    if edge_from.count(el) > 1:
                        repeated.append(el)
                print("Repeated elems: ", repeated)
                print([edge for edge in edges if edge['from_id'] in repeated])
                # ipdb.set_trace()
                raise Exception
        # print("DONE")
        try:
            assert len(set(edge_tup)) == len(edge_tup)
        except:
            print("Fail edges")
            print(sorted(edge_tup))
            print('\n')
            print(sorted(set(edge_tup)))
            raise Exception("duplicated edges")

        nodes = [id2node[idi] for idi in ids]
        nodes.append({'id': -1, 'class_name': 'no_obj', 'states': []})


        id2index = {node['id']: it for it, node in enumerate(nodes)}

        class_names_str = [node['class_name'] for node in nodes]
        # print(set(class_names_str))
        node_ids = [node['id'] for node in nodes]
        # print(nodes)

        # The self agent is equal to no_obj
        # class_names_str[0] = 'no_obj'



        class_names = np.array(
            [self.object_dict.get_id(class_name) for class_name in class_names_str]
        )

        if len(class_names) > max_nodes:
            print(
                "Error, more nodes than allowed ({}): found {}".format(
                    max_nodes, len(class_names)
                )
            )
            print(class_names_str)
            print('----')


        

        if len(edges) > 0:
            # print([edge for edge in edges if edge['relation_type'] == 'CLOSE'])
            edge_ids = np.concatenate(
                [
                    np.array([id2index[edge['from_id']], id2index[edge['to_id']]])[
                        None, :
                    ]
                    for edge in edges
                ],
                axis=0,
            )

        # else:
        #     pdb.set_trace()

        if include_edges and len(edges) > max_edges:
            pdb.set_trace()

        mask_edges = np.zeros(max_edges)
        all_edge_ids = np.zeros((max_edges, 2)).astype(np.int32)
        
        all_class_names = np.zeros((max_nodes)).astype(np.int32)
        
        if len(edges) > 0 and include_edges:
            mask_edges[: len(edges)] = 1.0
            all_edge_ids[: len(edges), :] = edge_ids


        all_node_ids = np.zeros((max_nodes)).astype(np.int32)

        all_class_names[: len(nodes)] = class_names
        all_node_ids[: len(nodes)] = node_ids
        


        # pdb.set_trace()
        output = {
            'class_objects': all_class_names,
            'edge_tuples': all_edge_ids,
            'mask_edge': mask_edges,
            'node_ids': all_node_ids
        }

        return output, None

    def build_graph(
        self,
        graph,
        character_id,
        ids=None,
        include_edges=False,
        plot_graph=False,
        action_space_ids=None,
        obs_ids=None,
        level=1,
        relative_coords=True,
        unique_from=False,
    ):
        if ids is None:
            ids = [
                node['id']
                for node in graph['nodes']
                if self.object_dict.valid_el(node['class_name'])
            ]

        for node in graph['nodes']:
            if node['category'] == 'Rooms':
                assert node['class_name'] in self.rooms
        room_ids = [
            node['id'] for node in graph['nodes'] if node['category'] == 'Rooms'
        ]
        if level > 0:
            # Include other rooms
            ids = room_ids + ids

        ids = [idi for idi in ids if idi != character_id]
        ids = list(set(ids))
        id2node = {node['id']: node for node in graph['nodes']}

        # TODO: remove
        # ids =[idi for idi in ids if id2node[idi]['class_name'] not in self.rooms]
        # action_space_ids = [idi for idi in action_space_ids if id2node[idi]['class_name'] not in self.rooms]

        # Character is always the first one
        ids = [character_id] + ids
        max_nodes = self.num_objects
        max_edges = self.num_edges

        edges = [edge.copy() for edge in graph['edges']]

        # Add a holding edge
        holding_object = []
        inside_object = []
        for edge in edges:
            if 'hold' in edge['relation_type'].lower():
                edge['relation_type'] = 'hold'
                holding_object.append(edge['to_id'])
            if 'inside' in edge['relation_type'].lower():
                inside_object.append((edge['from_id'], edge['to_id']))

        # If holding an object, remove close edge
        edges = [
            edge
            for edge in edges
            if edge['from_id'] in ids
            and edge['to_id'] in ids
            and edge['relation_type'].lower() in self.relations
        ]
        edges = [
            edge
            for edge in edges
            if (
                edge['relation_type'].lower() != 'close'
                or edge['to_id'] not in holding_object
            )
            and not (edge['from_id'] < 10 and edge['relation_type'].lower() == 'on')
            and (
                edge['relation_type'].lower() != 'close'
                or edge['from_id'] not in holding_object
            )
        ]

        # an object cannot be inside and on a given object
        edges = [
            edge
            for edge in edges
            if not (
                edge['relation_type'].lower() == 'on'
                and (edge['from_id'], edge['to_id']) in inside_object
            )
        ]

        # reverse holding directions
        for it, edge in enumerate(edges):
            if edge['relation_type'] == 'hold':
                # print(edges[it])
                edges[it] = {
                    'from_id': edge['to_id'],
                    'to_id': edge['from_id'],
                    'relation_type': edge['relation_type'],
                }
                # print(edges[it])
        # print("UNIQUE")
        if unique_from:
            try:
                edges = make_edges_unique(edges, room_ids, id2node)
            except:
                "Failure in making edges unique"
                raise Exception
        # Check if there is more than one edge between two nodes
        edge_tup = [(edge['from_id'], edge['to_id']) for edge in edges]

        if unique_from:
            # print("CHECK")
            repeated = []
            edge_from = [edge['from_id'] for edge in edges]
            if len(set(edge_from)) != len(edge_from):

                for el in list(set(edge_from)):
                    if edge_from.count(el) > 1:
                        repeated.append(el)
                print("Repeated elems: ", repeated)
                print([edge for edge in edges if edge['from_id'] in repeated])
                # ipdb.set_trace()
                raise Exception
        # print("DONE")
        try:
            assert len(set(edge_tup)) == len(edge_tup)
        except:
            print("Fail edges")
            print(sorted(edge_tup))
            print('\n')
            print(sorted(set(edge_tup)))
            raise Exception("duplicated edges")

        nodes = [id2node[idi] for idi in ids]
        nodes.append({'id': -1, 'class_name': 'no_obj', 'states': []})

        bbox_available = (
            'bounding_box' in nodes[0].keys() and nodes[0]['bounding_box'] is not None
        )
        if bbox_available:
            char_coord = np.array(nodes[0]['bounding_box']['center'])
            if relative_coords:
                rel_coords = [
                    np.array([0, 0, 0])[None, :]
                    if 'bounding_box' not in node.keys()
                    else (np.array(node['bounding_box']['center']) - char_coord)[
                        None, :
                    ]
                    for node in nodes
                ]
            else:
                rel_coords = [
                    np.array([0, 0, 0])[None, :]
                    if 'bounding_box' not in node.keys()
                    else (np.array(node['bounding_box']['center']))[None, :]
                    for node in nodes
                ]

            # pdb.set_trace()
            # for node in nodes:
            #     if 'bounding_box' not in node:
            #         print(node['class_name'])

            bounds = [
                np.array([0, 0, 0])[None, :]
                if 'bounding_box' not in node.keys()
                else np.array(node['bounding_box']['size'])[None, :]
                for node in nodes
            ]
            rel_coords = np.concatenate([rel_coords, bounds], axis=2)
        id2index = {node['id']: it for it, node in enumerate(nodes)}

        class_names_str = [node['class_name'] for node in nodes]
        # print(set(class_names_str))
        node_ids = [node['id'] for node in nodes]
        # print(nodes)

        # The self agent is equal to no_obj
        # class_names_str[0] = 'no_obj'

        visible_nodes = [
            (class_name, node_id)
            for class_name, node_id in zip(class_names_str, node_ids)
        ]

        class_names = np.array(
            [self.object_dict.get_id(class_name) for class_name in class_names_str]
        )
        node_states = np.array([self.one_hot(node['states']) for node in nodes])

        if len(class_names) > max_nodes:
            print(
                "Error, more nodes than allowed ({}): found {}".format(
                    max_nodes, len(class_names)
                )
            )
            print(class_names_str)
            print('----')

        edge_types = np.array(
            [self.relation_dict.get_id(edge['relation_type']) for edge in edges]
        )

        close_ids = [
            edge['to_id']
            for edge in edges
            if edge['relation_type'] == 'CLOSE' and edge['from_id'] == 1
        ]

        if len(edges) > 0:
            # print([edge for edge in edges if edge['relation_type'] == 'CLOSE'])
            edge_ids = np.concatenate(
                [
                    np.array([id2index[edge['from_id']], id2index[edge['to_id']]])[
                        None, :
                    ]
                    for edge in edges
                ],
                axis=0,
            )

        # else:
        #     pdb.set_trace()

        if include_edges and len(edges) > max_edges:
            pdb.set_trace()

        mask_edges = np.zeros(max_edges)
        all_edge_ids = np.zeros((max_edges, 2)).astype(np.int32)
        all_edge_types = np.zeros((max_edges)).astype(np.int32)

        mask_nodes = np.zeros((max_nodes))
        close_nodes = np.zeros((max_nodes))
        mask_action_nodes = np.zeros((max_nodes))
        mask_obs_nodes = np.zeros((max_nodes))
        all_class_names = np.zeros((max_nodes)).astype(np.int32)
        all_node_states = np.zeros((max_nodes, len(self.states)))
        all_node_ids = np.zeros((max_nodes)).astype(np.int32)

        if len(edges) > 0 and include_edges:
            mask_edges[: len(edges)] = 1.0
            all_edge_ids[: len(edges), :] = edge_ids
            all_edge_types[: len(edges)] = edge_types

        if action_space_ids is not None:
            mask_action_nodes[: len(nodes)] = np.array(
                [1 if node_id in action_space_ids else 0 for node_id in node_ids]
            )
        else:
            mask_action_nodes[: len(nodes)] = 1.0
        mask_nodes[: len(nodes)] = 1.0

        if obs_ids is not None:
            mask_obs_nodes[: len(nodes)] = np.array(
                [1 if node_id in obs_ids else 0 for node_id in node_ids]
            )
        else:
            mask_obs_nodes = mask_nodes

        all_class_names[: len(nodes)] = class_names
        all_node_states[: len(nodes)] = node_states
        all_node_ids[: len(nodes)] = node_ids
        close_nodes[: len(nodes)] = [
            1 if node_id in close_ids else 0 for node_id in node_ids
        ]

        if self.simulaor_type == 'unity':
            obj_coords = np.zeros((max_nodes, 6))
            obj_coords[: len(nodes)] = np.concatenate(rel_coords, 0)

        if plot_graph:
            graph_viz = DGLGraph()
            graph_viz.add_nodes(len(nodes), {'names': class_names})
            labeldict = {it: class_str for it, class_str in enumerate(class_names_str)}
        else:
            labeldict = None
            graph_viz = None

        # pdb.set_trace()
        output = {
            'class_objects': all_class_names,
            'states_objects': all_node_states,
            'edge_tuples': all_edge_ids,
            'edge_classes': all_edge_types,
            'mask_object': mask_nodes,
            'mask_edge': mask_edges,
            'mask_action_node': mask_action_nodes,
            'mask_obs_node': mask_obs_nodes,
            'node_ids': all_node_ids,
            'gt_close': close_nodes,
        }

        if self.simulaor_type == 'unity':
            output['object_coords'] = obj_coords
        # print(node_ids[:len(nodes)])
        return output, (graph_viz, labeldict, action_space_ids, visible_nodes)


def make_edges_unique(edges, rooms, id2node):
    edge_from_dict = {}
    for edge in edges:
        if id2node[edge['to_id']]['class_name'] == 'plate':
            continue
        if edge['from_id'] not in edge_from_dict:
            edge_from_dict[edge['from_id']] = edge
        else:
            curr_edge = edge_from_dict[edge['from_id']]
            if curr_edge['to_id'] in rooms:
                # If the object is in a room, forget that it is in the room
                edge_from_dict[edge['from_id']] = edge
            elif edge['to_id'] in rooms:
                pass
            elif edge['relation_type'] == 'INSIDE':
                # prioritize inside
                edge_from_dict[edge['from_id']] = edge
            elif edge['relation_type'] == 'ON':
                if curr_edge['relation_type'] != 'INSIDE':
                    if 'GRABBABLE' in id2node[edge['to_id']]['states']:
                        edge_from_dict[edge['to_id']] = edge['from_id']

    for edge in edge_from_dict.values():
        idn = id2node[edge['to_id']]
        if 'GRABBABLE' in idn['states']:
            print(idn['class_name'])
    edges = list(edge_from_dict.values())
    return edges



def args_per_action(action):

    action_dict = {
        'turnleft': 0,
        'walkforward': 0,
        'turnright': 0,
        'walktowards': 1,
        'open': 1,
        'close': 1,
        'putback': 1,
        'putin': 1,
        'put': 1,
        'grab': 1,
        'touch': 1,
        'no_action': 0,
        'walk': 1,
    }
    return action_dict[action]


# class GraphSpace(spaces.Space):
#     def __init__(self):
#         self.shape = None
#         self.dtype = "graph"

#         pass


def update_probs(
    log_probs, i, actions, object_classes, mask_observations, obj1_affordance
):
    """
    :param log_probs: current log probs
    :param i: which action are we currently considering
    :param actions: actions already selected
    :param mask_observations: bs x max_nodes with the valid nodes
    :return:
    """

    inf_val = 1e9  # log(prob(obj_non_visible)) = -1e9
    if i == 1:
        # Deciding on the object
        mask_and_class = mask_observations * (object_classes > 0)
        log_probs = log_probs * mask_and_class + (1.0 - mask_and_class) * -inf_val
        # log_probs = clamp(log_probs) * mask_and_class + (1. - mask_and_class) * -inf_val
        # check if an object cannot in no class

        # b x num_classes
        # mask_object_class = obj1_affordance.sum(1) > 0
        # if np.sum(mask_object_class) != mask_object_class.shape[0]:
        #    pdb.set_trace()
        ## batch x nodes x object_class
        # one_hot = torch.LongTensor(object_classes.shape[0], object_classes.shape[1],
        #                           mask_object_class.shape[-1]).zero_().to(object_classes.device)
        # target_one_hot = one_hot.scatter_(2, object_classes.unsqueeze(-1).long(), 1)
        # mask_nodes = ((mask_object_class * target_one_hot).sum(-1) > 0)[:, :log_probs.shape[1]]
        # mask = mask_nodes.to(log_probs.device).float()
        # log_probs = log_probs * mask + (1. - mask) * -inf_val

        return log_probs

    elif i == 0:
        # Deciding on the action
        # pdb.set_trace()
        selected_obj1 = torch.gather(object_classes, 1, actions[1].long())

        mask = (
            torch.gather(
                obj1_affordance,
                2,
                selected_obj1.unsqueeze(-2)
                .repeat(1, obj1_affordance.shape[1], 1)
                .long(),
            )
            .squeeze(-1)
            .float()
            .to(log_probs.device)
        )
        # mask[action_dict.get_id('open'),object_dict.get_id('kitchencounterdrawer') ]= 0

        if mask.sum() == 0:
            pdb.set_trace()
        log_probs = log_probs * mask + (1.0 - mask) * -inf_val

        # print("CLASS OBJ")
        # print(object_classes[0, :5])
        # print(actions, 'CLASS', selected_obj1)
        # print(log_probs)
        # if log_probs[:, 8] > -inf_val:
        #    print(log_probs)
        #   pdb.set_trace()

        return log_probs
