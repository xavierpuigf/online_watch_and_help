from torch.utils.data import Dataset
import scipy
import torch
import ipdb
import glob
from tqdm import tqdm
import pickle as pkl
from utils import utils_rl_agent
from arguments import *
from agents import belief
import yaml
import torch.nn.functional as F
import multiprocessing as mp
import numpy as np

def set_init_belief(id2node, label_agent, room_ids, container_ids):
    label_agent_dict = {
            4: 'spiked',
            10: 'spiked2',
            11: 'spiked3',
            12: 'spiked4'
    }
    belief_type = label_agent_dict[label_agent]
    init_values_container = belief.get_container_prior(id2node, belief_type, container_ids)
    init_values_room = belief.get_rooms(id2node, belief_type, room_ids)
    return init_values_container, init_values_room


class AgentTypeDataset(Dataset):
    def __init__(self, path_init, args_config, split='train'):
        self.path_init = path_init
        self.get_edges = args_config['model']['state_encoder'] == 'GNN'
        self.graph_helper = utils_rl_agent.GraphHelper(
                max_num_objects=args_config['model']['max_nodes'], 
                include_touch=True)
        # Build the agent types

        with open(self.path_init, 'rb+') as f:
            agent_files = pkl.load(f)
            agent_files = agent_files

        agent_type_max = max([x[0] for x in agent_files.values()])
        
        # clean the agent folder
        pkl_files = []
        labels = []
        other_files = []
        agent_labels = [
            # full/partial, mem high, mem low, open high, open low, spiked/uniform
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 1, 1, 0, 0],
        ]
        if args_config['train']['agents'] == 'all':
            agents_use = list(range(agent_type_max+1))
        else:
            agents_use = [int(x) for x in args_config['train']['agents'].split(',')]

        agents_use = [x for x in agents_use if x != 0]
        for filename, [label_agent, otherfiles] in agent_files.items():
            if label_agent in agents_use:
                pkl_files.append(filename)
                labels.append(label_agent)
                other_files.append(otherfiles)


        self.max_labels = agent_type_max+1 
        self.labels = labels
        self.other_files = other_files
        self.pkl_files = pkl_files
        self.overfit = args_config['train']['overfit']
        self.max_tsteps = args_config['model']['max_tsteps']
        self.max_actions = args_config['model']['max_actions']
        self.failed_items = mp.Array('i', len(self.pkl_files))
        
        print("Loading data...")
        print("Filename: {}. Episodes: {}. Objects: {}".format(path_init, len(self.pkl_files), len(self.graph_helper.object_dict)))
        print("---------------")
        assert self.max_actions == len(self.graph_helper.action_dict)+1, '{} vs {}'.format(self.max_actions, len(self.graph_helper.action_dict))

    def __len__(self):
        return len(self.pkl_files)

    def failure(self, index):
        # print("Fail", index)
        if index not in self.failed_items:
            self.failed_items[index] = 1
        return self.__getitem__(0)

    def get_failures(self):
        cont = [item for item in self.failed_items]
        return sum(cont)

    def process_content(self, content):
        goals = content['goals'][0]
        target_obj_class = [self.graph_helper.object_dict.get_id('no_obj')] * 6
        target_loc_class = [self.graph_helper.object_dict.get_id('no_obj')] * 6
        mask_goal_pred = [0.0] * 6
        pre_id = 0
        obj_pred_names, loc_pred_names = [], []

        id2node = {node['id']: node for node in content['graph'][0]['nodes']}
        class2id = {}
        for node in content['graph'][0]['nodes']:
            if node['class_name'] not in class2id:
                class2id[node['class_name']] = []
            class2id[node['class_name']].append(node['id'])

        for predicate, info in content['goals'][0].items():
            count = info
            if count == 0:
                continue

            elements = predicate.split('_')
            obj_class_id = int(self.graph_helper.object_dict.get_id(elements[1]))
            if len(elements) < 3:
                loc_class_id = 1
            else:
                loc_class_id = int(self.graph_helper.object_dict.get_id(id2node[int(elements[2])]['class_name']))

            #obj_pred_names.append(elements[1])
            #loc_pred_names.append(id2node[int(elements[2])]['class_name'])
            for _ in range(count):
                try:
                    target_obj_class[pre_id] = obj_class_id
                    target_loc_class[pre_id] = loc_class_id
                    mask_goal_pred[pre_id] = 1.0
                    pre_id += 1
                except:
                    raise Exception
        goal = {
            'target_loc_class': torch.tensor(target_loc_class), 
            'target_obj_class': torch.tensor(target_obj_class), 
            'mask_goal_pred': torch.tensor(mask_goal_pred)
        }

        attributes_include = ['class_objects', 'states_objects', 'object_coords', 'mask_object', 'node_ids', 'mask_obs_node']
        if self.get_edges:
            attributes_include += ['edge_tuples', 'edge_classes', 'mask_edge']
        time_graph = {attr_name: [] for attr_name in attributes_include}
        # print(list(content.keys()))




        time_graph['mask_close'] = []
        time_graph['mask_goal'] = []

        for it, graph in enumerate(content['graph']):
            # if it == len(content['graph']) - 1:
            #     # Skip the last graph
            #     continue

            if it >= self.max_tsteps:
                break
            graph_info, _ = self.graph_helper.build_graph(graph, character_id=1, include_edges=True, obs_ids=content['obs'][it])
            prev_nodes = [node['id'] for node in graph['nodes']]
            
            #ipdb.set_trace()

            # class names
            for attribute_name in attributes_include:
                if attribute_name not in graph_info:
                    print(attribute_name, index, self.pkl_files[index])
                    return self.failure(index)
                time_graph[attribute_name].append(torch.tensor(graph_info[attribute_name]))

            # ipdb.set_trace()
            # Build closeness and goal mask
            close_rel_id = self.graph_helper.relation_dict.get_id('CLOSE')
            close_nodes = list(graph_info['edge_tuples'][graph_info['edge_classes'] == close_rel_id])
            
            mask_close = np.zeros(graph_info['class_objects'].shape)
            mask_goal = np.zeros(graph_info['class_objects'].shape) 

            # fill up the closeness mask
            if len(close_nodes) > 0:
                indexe = [int(edge[1]) for edge in close_nodes if edge[0] == 0]
                mask_close[np.array(indexe)] = 1.0

            # Fill up goal object mask
            goal_loc = [target_loc for it_pred, target_loc in enumerate(target_loc_class) if mask_goal_pred[it_pred] == 1]
            goal_obj = [target_obj for it_pred, target_obj in enumerate(target_obj_class) if mask_goal_pred[it_pred] == 1]
            goal_obs = list(set(goal_obj))
            for goal_id in goal_obs:
                mask_goal[graph_info['class_objects'] == goal_id] = 1.0

            time_graph['mask_close'].append(torch.tensor(mask_close))
            time_graph['mask_goal'].append(torch.tensor(mask_goal))
                
        num_tsteps = min(len(content['graph']), self.max_tsteps)

        # print('TSTEPS', num_tsteps)
        for attribute_name in time_graph.keys():
            unpadded_tensor = torch.cat([item[None, :] for item in time_graph[attribute_name]][:num_tsteps]).float()
            # Do padding
            padding_amount = self.max_tsteps - num_tsteps
            # if padding_amount < 0:
            #     num_tsteps2 = len(content['action'][0])
            #     print(num_tsteps, num_tsteps2)

            # ipdb.set_trace()
            padding = [0] * unpadded_tensor.dim() * 2
            padding[-1] = padding_amount
            tuple_pad = tuple(padding)
            time_graph[attribute_name] = F.pad(unpadded_tensor, pad=tuple_pad, mode='constant', value=0.)

        return graph_info, time_graph, class2id, goal, id2node

    def get_program_info(self, content, graph_info):
        # Match graph indices to index in the tensor
        node_ids = graph_info['node_ids']
        indexgraph2ind = {node_id: idi for idi, node_id in enumerate(node_ids)}

        program = content['action'][0]
        if len(program) == 0:
            print(index)

        # We will start with a No-OP action
        program_batch = {
            'action': [self.max_actions - 1],
            'obj1': [-1],
            'obj2': [-1],
            'indobj1': [indexgraph2ind[-1]],
            'indobj2': [indexgraph2ind[-1]],
        }
                # We start at 1 to skip the first instruction
        for it, instr in enumerate(program):
            
            # we want to add an ending action
            if it >= self.max_tsteps - 1:
                break
            instr_item = self.graph_helper.actionstr2index(instr)
            program_batch['action'].append(instr_item[0])
            program_batch['obj1'].append(instr_item[1])
            program_batch['obj2'].append(instr_item[2])
            try:
                program_batch['indobj1'].append(indexgraph2ind[instr_item[1]])
                program_batch['indobj2'].append(indexgraph2ind[instr_item[2]])
            except:
                # print("erorr in ", instr)
                raise Exception
        program_batch['action'].append(self.max_actions - 1)
        program_batch['obj1'].append(-1)
        program_batch['obj2'].append(-1)
        program_batch['indobj1'].append(indexgraph2ind[-1])
        program_batch['indobj2'].append(indexgraph2ind[-1])
        

        num_tsteps = min(len(program_batch['action']) - 1, self.max_tsteps)
        # print('TSTEP2S', num_tsteps)

        length_mask = torch.zeros(self.max_tsteps)
        length_mask[:num_tsteps] = 1.

        for key in program_batch.keys():
            unpadded_tensor = torch.tensor(program_batch[key])

            # The program has an extra step
            padding_amount = self.max_tsteps - num_tsteps
            padding = [0] * unpadded_tensor.dim() * 2
            padding[-1] = padding_amount
            tuple_pad = tuple(padding)
            program_batch[key] = F.pad(unpadded_tensor, pad=tuple_pad, mode='constant', value=0.)
            #print(program_batch[key].shape)

        return program_batch, length_mask, indexgraph2ind


    def __getitem__(self, index):
        if self.overfit:
            index = 0
        file_name = self.pkl_files[index]
        file_names = self.other_files[index] 
        # print(len(file_names))
        #print(file_name)
        seed_number = int(file_name.split('.')[-2]) 
        with open(file_names[0], 'rb') as f:
            content = pkl.load(f)

        other_content = []
        for it in range(1, len(file_names)):
            with open(file_names[it], 'rb') as f:
                other_content.append(pkl.load(f))

        # print("Loaded")
        ##############################
        #### Inputs high level policy
        ##############################
        # Encode goal
        if 'action' not in content:
            print("FAil", self.pkl_files[index]) 
            return self.failure(index)
        

        # try:
        # print("Lod1")
        try:

            graph_info, time_graph, class2id, goal, id2node = self.process_content(content)
            program_batch, length_mask, indexgraph2ind = self.get_program_info(content, graph_info)
        # print("CONTE")
        except:

            return self.failure(index)
        other_data = {
            'time_graph': [],
            'program_batch': [],
            'length_mask': [],
            'goal': [],
        }

        for itc, ct in enumerate(other_content):
            try:
                graph_info_o, time_graph_o, _, goal_o, _ = self.process_content(ct)
                program_batch_o, length_mask_o, _ = self.get_program_info(ct, graph_info_o)
            except:

                return self.failure(index)
            other_data['time_graph'].append(time_graph_o)
            other_data['program_batch'].append(program_batch_o)
            other_data['length_mask'].append(length_mask_o)
            other_data['goal'].append(goal_o)

        # Merge data from the episodes
        goal_data_batch = {}
        program_data_batch = {}
        time_graph_batch = {}
        for goalk in goal.keys():
            goal_data_batch[goalk] = torch.cat([g[goalk].unsqueeze(0) for g in other_data['goal']])
            # print(goalk, goal_data_batch[goalk].shape)

        for progk in program_batch.keys():
            program_data_batch[progk] = torch.cat([g[progk].unsqueeze(0) for g in other_data['program_batch']])
            # print(progk, program_data_batch[progk].shape)

        for graphk in time_graph.keys():
            time_graph_batch[graphk] = torch.cat([g[graphk].unsqueeze(0) for g in other_data['time_graph']])
            # print(graphk, time_graph_batch[graphk].shape)

        # ipdb.set_trace()
        other_data['time_graph'] = time_graph_batch
        other_data['goal'] = goal_data_batch
        other_data['program_batch'] = program_data_batch
        other_data['length_mask'] = torch.cat([lm.unsqueeze(0) for lm in other_data['length_mask']])

        label_one_hot = torch.tensor(self.labels[index])
        

        #############
        # Get Belief
        ############
        elements = list(content['goals'][0].keys())[0].split('_')
        obj_id =  class2id[elements[1]][0]

        initial_belief = content['belief'][0][0][obj_id]['INSIDE']
        initial_belief_room = content['belief_room'][0][0][obj_id]

        # Define initial belief
        room_ids = initial_belief_room[0]
        container_ids = initial_belief[0]
        initial_belief_values, initial_belief_room_values = set_init_belief(id2node, self.labels[index], room_ids, container_ids) 
        initial_belief_room[1] = initial_belief_room_values
        initial_belief[1] = initial_belief_values
        #print(initial_belief_room_values)

        node_ids = graph_info['node_ids']
        sm_room = scipy.special.softmax(initial_belief_room[1])
        sm_belief = scipy.special.softmax(initial_belief[1])
        mask_belief_room, mask_belief_container = [0]*len(node_ids), [0]*len(node_ids)

        belief_room, belief_container = [0]*len(node_ids), [0]*len(node_ids)
        for it, ind in enumerate(initial_belief_room[0]):
            try:
                index_belief_ind = indexgraph2ind[ind]
                mask_belief_room[index_belief_ind] = 1
                belief_room[index_belief_ind] = sm_room[it]
            except:
                print("Error loading belief")

        for it, ind in enumerate(initial_belief[0]):
            if ind is None:
                ind = -1
            try:
                index_belief_ind = indexgraph2ind[ind]
                mask_belief_container[index_belief_ind] = 1
                belief_container[index_belief_ind] = sm_belief[it]
            except:
                print("Error Loading belief")


        ####
        belief_info = {
            'mask_belief_container': mask_belief_container,
            'mask_belief_room': mask_belief_room,
            'belief_room': belief_room,
            'belief_container': belief_container,
            'index': index
        }
        
        for key in belief_info.keys():
            belief_info[key] = torch.tensor(belief_info[key]).float()
            # print(belief_info[key].shape)



        label_agent = seed_number + self.labels[index] * 5
        real_label = self.labels[index]
        # ipdb.set_trace()
        # print("ok", index)
        return time_graph, program_batch, label_one_hot, length_mask, goal, label_agent, real_label, belief_info, other_data

if __name__ == '__main__':
    arguments = get_args_pref_agent()
    config_file = 'config/agent_pref_v0/config_default_lowlr_belief.yaml'
    with open(config_file, 'r') as f:
        config = yaml.load(f)
    dataset = AgentTypeDataset(path_init='../dataset/dataset_agent_belief_v2_paired_train.pkl', args_config=config)
    data = dataset[0]
    # ipdb.set_trace()
