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
import random
import numpy as np


class AgentTypeDataset(Dataset):
    def __init__(self, path_init, args_config, split='train', build_graphs_in_loader=False, first_last=False):
        self.path_init = path_init
        self.predict_diff_preds = args_config.model.predict_diff_preds
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

        self.list_category_classes = ['put_dishwasher', 'put_fridge', 'watch_tv', 'prepare_food', 'setup_table']
        self.dict_category_classes = {class_name: class_id+1 for class_id, class_name in enumerate(self.list_category_classes)}
        self.num_categories = len(self.list_category_classes)

        print("Loading data...")
        print("Filename: {}. Episodes: {}. Objects: {}".format(path_init, len(self.pkl_files), len(self.graph_helper.object_dict)))
        print("---------------")
        assert self.max_actions == len(self.graph_helper.action_dict)+1, '{} vs {}'.format(self.max_actions, len(self.graph_helper.action_dict))

    def __len__(self):
        return len(self.pkl_files)

    def failure(self, index, print_index=False):
        # ipdb.set_trace()
        if print_index:
            print(colored(f"Failure at {index}"))
        file_name = self.pkl_files[index]
        if index not in self.failed_items:
            self.failed_items[index] = 1
        new_index = random.randint(0, len(self.pkl_files)-1)
        return self.__getitem__(new_index)

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
        with open(file_name.replace('.pik', '_reduced.pik'), 'rb') as f:
            content = pkl.load(f)

        # try:
        if not content['valid']:
            return self.failure(index)
        task_graph_time = content['task_graph_time']
        mask_task_graphs = content['mask_task_graphs']
        gt_task_graph = content['gt_task_graph']
        length_mask = content['length_mask']
        goal = content['goal']
        program_batch = content['program_batch']

        

        if self.predict_diff_preds:
            init_task_graph = task_graph_time[0].clone()
            # Compute the difference between current step and init, forget about negatives
            curr_len = int(length_mask.sum())
            # for ind_t in range(curr_len):
            task_graph_time[:curr_len] = torch.nn.functional.threshold(
                task_graph_time[:curr_len] - init_task_graph[None, :], 0, 0, inplace=False)
            # ipdb.set_trace()
            gt_task_graph = torch.nn.functional.threshold(gt_task_graph - init_task_graph, 0, 0, inplace=False)
        task_graph = {'task_graph': task_graph_time, 'mask_task_graph': mask_task_graphs, 'gt_task_graph': gt_task_graph}
        # except:
        #     self.failure(index)
        # ipdb.set_trace()
        # print("Loaded")
        if content['task_name'] in self.dict_category_classes:
            class_id = self.dict_category_classes[content['task_name']]
        else:
            class_id = 0
        return program_batch, length_mask, goal, task_graph, index, class_id 
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
