from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import dgl
import torch
import ipdb

from termcolor import colored
import glob
from tqdm import tqdm
import pickle as pkl
from utils import utils_rl_agent
from arguments import *
import yaml
import torch.nn.functional as F
import multiprocessing as mp
import numpy as np


class AgentTypeDataset(Dataset):
    def __init__(self, path_init, args_config, split='train', build_graphs_in_loader=False, first_last=False):
        self.path_init = path_init
        self.max_num_edges = 200
        self.first_last = first_last
        self.graph_helper = utils_rl_agent.GraphHelper(max_num_objects=args_config['model']['max_nodes'], toy_dataset=args_config['model']['reduced_graph'])
        self.get_edges = True # args_config['model']['state_encoder'] == 'GNN'
        # Build the agent types

        with open(self.path_init, 'rb+') as f:
            agent_files = pkl.load(f)

        agent_type_max = max(agent_files.values())
        
        # clean the agent folder
        pkl_files = []
        labels = []


        # Unused, this was when we had multiple agents
        if args_config['train']['agents'] == 'all':
            agents_use = list(range(agent_type_max+1))
        else:
            agents_use = [int(x) for x in args_config['train']['agents'].split(',')]

        for filename, label_agent in agent_files.items():
            if label_agent in agents_use:
                pkl_files.append(filename)
                labels.append(label_agent)


        self.max_labels = agent_type_max+1 
        self.labels = labels
        self.pkl_files = pkl_files

        if args_config['train']['overfit']:
            self.pkl_files = pkl_files[:1]

        self.overfit = args_config['train']['overfit']
        self.max_tsteps = args_config['model']['max_tsteps']
        self.max_actions = args_config['model']['max_actions']
        self.failed_items = mp.Array('i', len(self.pkl_files))
        self.condense_walking = args_config['model']['condense_walking']
        self.args_config = args_config
        
        print("Loading data...")
        print("Filename: {}. Episodes: {}. Objects: {}".format(path_init, len(self.pkl_files), len(self.graph_helper.object_dict)))
        print("---------------")
        assert self.max_actions == len(self.graph_helper.action_dict)+1, '{} vs {}'.format(self.max_actions, len(self.graph_helper.action_dict))

    def __len__(self):
        return len(self.pkl_files)

    def failure(self, index, print_index=False):

        if print_index:
            print(colored(f"Failure at {index}"))
        file_name = self.pkl_files[index]
        if index not in self.failed_items:
            self.failed_items[index] = 1
        return self.__getitem__(0)

    def get_failures(self):
        cont = [item for item in self.failed_items]
        return sum(cont)

    def __getitem__(self, index):
        include_time_graph = False

        if self.overfit:
            index = 0
        # index = 2349
        file_name = self.pkl_files[index]
        seed_number = int(file_name.split('.')[-2]) 
        ipdb.set_trace()
        with open(file_name, 'rb') as f:
            content = pkl.load(f)



        ##############################
        #### Inputs high level policy
        ##############################
        # Encode goal
        if 'action' not in content:
            print("FAil", self.pkl_files[index]) 
            return self.failure(index)
        


        goals = content['goals'][0]

        target_obj_class = [self.graph_helper.object_dict.get_id('no_obj')] * 6
        target_loc_class = [self.graph_helper.object_dict.get_id('no_obj')] * 6
        mask_goal_pred = [0.0] * 6

        pre_id = 0
        obj_pred_names, loc_pred_names = [], []

        id2node = {node['id']: node for node in content['graph'][0]['nodes']}
        for predicate, info in content['goals'][0].items():

            if type(info) == int:
                count = info
            else:
                count = info['count']
            if count == 0:
                continue

            # if not (predicate.startswith('on') or predicate.startswith('inside')):
            #     continue

            elements = predicate.split('_')
            obj_class_id = int(self.graph_helper.object_dict.get_id(elements[1]))
            loc_class_id = int(self.graph_helper.object_dict.get_id(id2node[int(elements[2])]['class_name']))

            obj_pred_names.append(elements[1])
            loc_pred_names.append(id2node[int(elements[2])]['class_name'])
            #for _ in range(count):
            #    try:
            #        target_obj_class[pre_id] = obj_class_id
            #        target_loc_class[pre_id] = loc_class_id
            #        mask_goal_pred[pre_id] = 1.0
            #        pre_id += 1
            #    except:
            #        pdb.set_trace()

        goal = {'target_loc_class': torch.tensor(target_loc_class), 
                'target_obj_class': torch.tensor(target_obj_class), 
                'mask_goal_pred': torch.tensor(mask_goal_pred)}



        label_one_hot = torch.tensor(self.labels[index])
        # print(content.keys())
        attributes_include = ['class_objects', 'states_objects', 'object_coords', 'mask_object', 'node_ids', 'mask_obs_node']
        if self.get_edges:
            attributes_include += ['edge_tuples', 'edge_classes', 'mask_edge']
        

        attributes_include = ['class_objects', 'node_ids']
        if include_time_graph:
            time_graph = {attr_name: [] for attr_name in attributes_include}
        # print(list(content.keys()))

        program = content['action'][0]
        if len(program) == 0:
            print(index)


        # time_graph['mask_close'] = []
        # time_graph['mask_goal'] = []
        
        # print("Building graph")
        # num_tsteps = len(program)


        num_tsteps = len(program)
        steps_keep = list(range(num_tsteps))

        if self.condense_walking:
            steps_keep = utils_rl_agent.condense_walking(program)

        if self.first_last:
            steps_keep = [len(program)-1]

        contit = 0
        task_graphs = []

        for it, graph in enumerate(content['graph']):
            
            if it > 0 and it-1 not in steps_keep:
                continue
            if contit >= self.max_tsteps:
                break
            try:
                # ipdb.set_trace()
                graph_info, _ = self.graph_helper.build_graph_for_task(
                    graph, character_id=1, include_edges=self.get_edges, 
                    obs_ids=content['obs'][it], relative_coords=self.args_config['model']['relative_coords'],
                    unique_from=self.args_config.model.exclusive_edge)
                # ipdb.set_trace()
                
                # Needs the following
                # num_edges = int(graph_info['mask_edge'].sum())
                # edge_tuples = graph_info['edge_tuples']
                # class_objects = graph_info['class_objects']
                
                task_graph = self.graph_helper.build_task_graph(graph_info)
                task_graphs.append(task_graph)
            except:
                print("Failure building grahp", file_name, it)
                if self.args_config.train.num_workers == 0:
                    raise Exception
                return self.failure(index)
                #raise Exception("Error building graph")

            # class names
            if include_time_graph:    
                for attribute_name in attributes_include:
                    if attribute_name not in graph_info:
                        print("Failure with attr name", attribute_name, index, self.pkl_files[index])
                        return self.failure(index)
                    time_graph[attribute_name].append(torch.tensor(graph_info[attribute_name]))


            
            contit += 1
            # ipdb.set_trace()
        # ipdb.set_trace()
        # Match graph indices to index in the tensor

        # task_graphs[0] = task_graphs[-1]



        node_ids = graph_info['node_ids']
        indexgraph2ind = {node_id: idi for idi, node_id in enumerate(node_ids)}


        task_graph_time = torch.cat([torch.tensor(tg, dtype=torch.int8)[None, :] for tg in task_graphs])
        final_task_graph = task_graph_time[-1, ...]
        if self.args_config['model']['predict_diff']:
            mask_task_graphs = (task_graph_time - task_graph_time[-1, :]) != 0
        else:
            mask_task_graphs = torch.ones_like(task_graph_time).bool()

        # ipdb.set_trace()
        # We will start with a No-OP action
        # program_batch = {
        #     'action': [self.max_actions - 1],
        #     'obj1': [-1],
        #     'obj2': [-1],
        #     'indobj1': [indexgraph2ind[-1]],

        #     'indobj2': [indexgraph2ind[-1]],
        # }

        
        # Do not encode program
        encode_program = True
        if encode_program:
            program_batch = {
                'action': [],
                'obj1': [],
                'obj2': [],
                'indobj1': [],
                'indobj2': [],
            }

            contit = 0
            # We start at 1 to skip the first instruction
            if self.first_last:
                steps_keep = [0]
            for it, instr in enumerate(program):
                if it not in steps_keep:
                    continue
                # we want to add an ending action
                if contit >= self.max_tsteps - 1:
                    print("PROGRAM TOO LONG")
                    # ipdb.set_trace()
                    return self.failure(index, print_index=False)
                instr_item = self.graph_helper.actionstr2index(instr)
                program_batch['action'].append(instr_item[0])
                program_batch['obj1'].append(instr_item[1])
                program_batch['obj2'].append(instr_item[2])
                try:
                    program_batch['indobj1'].append(indexgraph2ind[instr_item[1]])
                    program_batch['indobj2'].append(indexgraph2ind[instr_item[2]])
                except:
                    #print("Index", index, program, it)
                    # ipdb.set_trace()
                    # if self.args_config.train.num_workers == 0:
                    #     ipdb.set_trace()
                    # Open pinter ??
                    return self.failure(index, print_index=True)
                contit += 1

            num_tsteps = len(program_batch['action'])
            program_batch['action'].append(self.max_actions - 1)
            program_batch['obj1'].append(-1)
            program_batch['obj2'].append(-1)
            program_batch['indobj1'].append(indexgraph2ind[-1])
            program_batch['indobj2'].append(indexgraph2ind[-1])
            # if len(program_batch['action']) != len(time_graph['mask_close']):
            #     ipdb.set_trace()
            for key in program_batch.keys():
                unpadded_tensor = torch.tensor(program_batch[key])

                # The program has an extra step
                padding_amount = self.max_tsteps - num_tsteps
                padding = [0] * unpadded_tensor.dim() * 2
                padding[-1] = padding_amount
                tuple_pad = tuple(padding)
                try:
                    program_batch[key] = F.pad(unpadded_tensor, pad=tuple_pad, mode='constant', value=0.)
                except:
                    ipdb.set_trace()

        # num actions = num states - 1            
        num_tsteps = len(task_graphs)
        padding_amount = self.max_tsteps - num_tsteps
        padding = [0] * 2 * 2
        padding[-1] = padding_amount
        tuple_pad = tuple(padding)

        task_graph_time = F.pad(task_graph_time, pad=tuple_pad, mode='constant', value=0.)
        mask_task_graphs = F.pad(mask_task_graphs, pad=tuple_pad, mode='constant', value=0.)

        # task_graph_time = task_graph_time[None, :]
        # mask_task_graphs = mask_task_graphs[None, :]


        length_mask = torch.zeros(self.max_tsteps)
        length_mask[:num_tsteps] = 1.

        # Batch across time
        if include_time_graph:
            for attribute_name in time_graph.keys():
                # print(attribute_name, len(time_graph[attribute_name]))
                unpadded_tensor = torch.cat([item[None, :] for item in time_graph[attribute_name]]).float()
                # Do padding
                padding_amount = self.max_tsteps - num_tsteps
                # ipdb.set_trace()
                padding = [0] * unpadded_tensor.dim() * 2
                padding[-1] = padding_amount
                tuple_pad = tuple(padding)
                time_graph[attribute_name] = F.pad(unpadded_tensor, pad=tuple_pad, mode='constant', value=0.)
                # if time_graph[attribute_name].shape[0] > self.max_tsteps:
                #     print(self.max_tsteps, num_tsteps, len(content['graph']), unpadded_tensor.shape[0])
        
        # for attribute in program_graph.keys():
        #     print(attribute, program_graph[attribute].shape)
        # print('----')

        # Add one hot info

        if self.args_config.model.state_encoder == 'GNN' and include_time_graph:
                ne_ind2 = torch.arange(0, self.max_num_edges)[None, :].repeat(self.max_tsteps, 1)[..., None]
                ne_ind = torch.arange(0, self.max_tsteps)[:, None].repeat(1, self.max_num_edges)[..., None] # 0 0 0 0 
                ind_from = time_graph['edge_tuples'][ ..., 0, None]
                ind_to = time_graph['edge_tuples'][..., 1, None]     
                # ipdb.set_trace()

                from_indices_onehot = torch.cat([ne_ind, ne_ind2, ind_from], dim=-1).long()
                to_indices_onehot = torch.cat([ne_ind, ne_ind2, ind_to], dim=-1).long()

                time_graph['from_indices_onehot'] = from_indices_onehot
                time_graph['to_indices_onehot'] = to_indices_onehot


        label_agent = seed_number + self.labels[index] * 5
        real_label = self.labels[index]

        # if self.args_config['model']['state_encoder'] == 'GNN' and build_graphs_in_loader:
        #     time_graph['graph'] = build_graph(time_graph)

        # ipdb.set_trace()
        task_graph = {'task_graph': task_graph_time, 'mask_task_graph': mask_task_graphs, 'gt_task_graph': final_task_graph}
        # ipdb.set_trace()
        # print("Loaded")
        return program_batch, length_mask, goal, task_graph, index 
        # if encode_program:
        #     return time_graph, program_batch, label_one_hot, length_mask, goal, label_agent, real_label, task_graph, index
        # else:
        #     return time_graph, program_batch, label_one_hot, length_mask, goal, label_agent, real_label, task_graph, index

def build_graph(time_graph):
    graphs = []
    tsteps = len(time_graph['mask_object'])
    for t in range(tsteps):
        g = dgl.DGLGraph()
        num_nodes = time_graph['mask_object'][t].sum()
        num_edges = int(time_graph['mask_edge'][t].sum())
        edge_tuples = time_graph['edge_tuples'][t]
        edge_classes = time_graph['edge_classes'][t]
        g.add_nodes(num_nodes)
        g.add_edges(edge_tuples[:num_edges, 0].long(), edge_tuples[:num_edges, 1].long(), 
                {'rel_type': edge_classes[:num_edges]})
        graphs.append(g)
    return graphs


def collate_fn(inputs):
    special_collate_keys = ['from_indices_onehot', 'to_indices_onehot']
    time_graph = [inp[0] for inp in inputs]

    collate_timegraph = {}
    keys_timegraph = time_graph[0].keys()
    for key in keys_timegraph:
        if key not in special_collate_keys:
            collate_timegraph[key] = default_collate([tgraph[key] for tgraph in time_graph])
        
    # Special collate for timegraph
    index_t = lambda ind: torch.tensor([ind, 0, 0])
    tstep = time_graph[0]['from_indices_onehot'].shape[0]
    from_indices = [(time_graph[i]['from_indices_onehot']+(index_t(i)*tstep))[None, :] for i in range(len(time_graph))]
    to_indices = [(time_graph[i]['to_indices_onehot']+(index_t(i)*tstep))[None, :] for i in range(len(time_graph))]

    collate_timegraph['from_indices_onehot'] = torch.cat(from_indices, 0)
    collate_timegraph['to_indices_onehot'] = torch.cat(to_indices, 0)



    program_batch_l = [inp[1] for inp in inputs] 
    label_one_hot_l = [inp[2] for inp in inputs]
    length_mask_l = [inp[3] for inp in inputs]
    goal_l = [inp[4] for inp in inputs]
    label_agent_l = [inp[5] for inp in inputs] 
    real_label_l = [inp[6] for inp in inputs]
    task_graph_l = [inp[7] for inp in inputs] 
    index_l = [inp[8] for inp in inputs]

    

    program_batch = default_collate(program_batch_l)
    label_one_hot = default_collate(label_one_hot_l)
    length_mask = default_collate(length_mask_l)
    goal = default_collate(goal_l)
    label_agent = default_collate(label_agent_l)
    real_label = default_collate(real_label_l)
    task_graph = default_collate(task_graph_l)
    index = default_collate(index_l)
    # ipdb.set_trace()

    return collate_timegraph, program_batch, label_one_hot, length_mask, goal, label_agent, real_label, task_graph, index


# def collate_fn(inputs):
#     new_inputs = []
#     for i in range(1, len(inputs[0])):
#         new_inputs.append(default_collate([inp[i] for inp in inputs]))
    
#     first_inp = {}
#     for key in inputs[0][0].keys():
#         if key not in ['graph', 'edge_tuples', 'edge_classes', 'mask_edge']:
#             first_inp[key] = default_collate([inp[0][key] for inp in inputs])

#     #ipdb.set_trace()
#     graph_list = [inp[0]['graph'] for inp in inputs]
#     graph_list = [graph for graphs in graph_list for graph in graphs]
#     #print(type(graph_list[0]))
#     first_inp['graph'] = dgl.batch(graph_list)
#     new_inputs = [first_inp] + new_inputs
#     return new_inputs

if __name__ == '__main__':
    arguments = get_args_pref_agent()
    with open(arguments.config, 'r') as f:
        config = yaml.load(f)
    dataset = AgentTypeDataset(path_init='../dataset/dataset_agent_model_v0_train.pkl', args_config=config)
    data = dataset[0]
    # ipdb.set_trace()
