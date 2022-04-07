from torch.utils.data import Dataset
import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt

from torch.utils.data.dataloader import default_collate
import dgl
import torch
import ipdb
import sys
sys.path.append('.')
import os
import p_tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
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
from hydra.utils import get_original_cwd, to_absolute_path
from multiprocessing import Pool
from p_tqdm import p_map



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
        new_file_name = file_name.replace('.pik', '_reduced.pik')

        return 0, ""


    def __getitem__(self, index):
        include_time_graph = False

        if self.overfit:
            index = 0
        # index = 2349
        file_name = self.pkl_files[index]
        seed_number = int(file_name.split('.')[-2]) 
        new_file_name = file_name.replace('.pik', '_reduced.pik')
        with open(new_file_name, 'rb') as f:
            content = pkl.load(f)
        try:
            task_graph_time = content['task_graph_time']
            length = int(content['length_mask'].sum() - 1)
            task_graph = task_graph_time[length]
            task_graph_0 = task_graph_time[0]
        except:
            return None, None
        return task_graph, task_graph_0

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



def get_loaders(args):
    print("Loading dataset...")
    print("Train: {}".format(args['data']['train_data']))
    print("Test: {}".format(args['data']['test_data']))
    curr_file = os.path.dirname(get_original_cwd())
    first_last = False
    dataset = AgentTypeDataset(
        path_init='{}/agent_preferences/dataset/{}'.format(
            curr_file, args['data']['train_data'] 
        ),
        first_last=first_last,
        args_config=args,
    )
    if not args['train']['overfit']:
        dataset_test = AgentTypeDataset(
            path_init='{}/agent_preferences/dataset/{}'.format(
                curr_file, args['data']['test_data']
            ),
            first_last=first_last,
            args_config=args,
        )
    else:
        dataset_test = AgentTypeDataset(
            path_init='{}/agent_preferences/dataset/{}'.format(
                curr_file, args['data']['train_data']
            ),
            first_last=first_last,
            args_config=args,
        )


    return dataset, dataset_test

def get_elems(curr_dataset, i):
    a = curr_dataset[i]
    return a

def process(dataset):


    # manager = mp.Manager()ßß
    num_process = 16
    num_elems = len(dataset)
    num_preds = 136 
    t0, tlast, tdif = np.zeros((num_preds, 9)), np.zeros((num_preds, 9)), np.zeros((num_preds, 9))
    zero_tensor = np.zeros((num_preds))
    for i in tqdm(range(num_elems)):
        ctlast, ct0 = get_elems(dataset, i)
        if ctlast is None:
            continue

        ctdif = np.maximum(ctlast - ct0, zero_tensor)
        # ipdb.set_trace()
        t0[np.arange(num_preds), ct0.int()] += 1
        tlast[np.arange(num_preds), ctlast.int()] += 1
        tdif[np.arange(num_preds), ctdif.int()] += 1
        
    #elem_ids = [(dataset, i) for i in list(range(len(dataset)))]
    ## print(elem_ids)
    #with Pool(num_process) as p:
    #    res2 = p.starmap(get_elems, elem_ids)
    ## res = p_map(get_elems, elem_ids, num_cpus=num_process)
    #res = [r[0] for r in res2]
    #task_names = [r[1] for r in res2]
    #print(set(task_names))
    #print(len(res), sum(res))
    ## for process_id in range(0, num_elems, num_process):
    ##     p = mp.Process(target=get_elem, args=(process_id))
    ##     jobs.append(p)
    ##     p.start()
    return t0, tlast, tdif

@hydra.main(config_path="../config/agent_pred_graph", config_name="config_default_large_excl_task")
def main(cfg: DictConfig):
    config = cfg
    print("Config")
    print(OmegaConf.to_yaml(cfg))
    # ipdb.set_trace()

    # assert not (cfg.model.predict_edge_change)
    assert cfg['model']['exclusive_edge']

    cfg.model.input_goal = False

    # cfg.num_gpus = torch.cuda.device_count()
    
    train_dataset, test_dataset = get_loaders(config)

    t0_train, tlast_train, tdif_train = process(train_dataset)
    t0_test, tlast_test, tdif_test = process(test_dataset)    
    # ipdb.set_trace()
    t0_train *= 1./t0_train[0].sum()
    tlast_train *= 1./tlast_train[0].sum()

    t0_test *= 1./t0_test[0].sum()
    tlast_test *= 1./tlast_test[0].sum()
    names = ['t0_test.png', 'tl_test.png', 'dif_test.png',
             't0_train.png', 'tl_train.png', 'dif_train.png']
    plots = [t0_test, tlast_test, tdif_test, 
             t0_train, tlast_train, tdif_train]
    it = 1
    fig = plt.figure()

    for name, plot in zip(names, plots):
        ax = fig.add_subplot(1,6,it)
        ax.imshow(plot, interpolation='none', aspect='auto')
        # plt.delaxes(ax)
        ax.title.set_text(name.replace('.png', ''))
        print(name)

        # fig.savefig('result.png')
        # ipdb.set_trace()
        it += 1
    fig.savefig('result.png')

if __name__ == '__main__':
    main()

