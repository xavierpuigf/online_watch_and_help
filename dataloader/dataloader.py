from torch.utils.data import Dataset
import torch
import ipdb
import glob
from tqdm import tqdm
import pickle as pkl
from utils import utils_rl_agent
from arguments import *
import yaml
import torch.nn.functional as F
import multiprocessing as mp


class AgentTypeDataset(Dataset):
    def __init__(self, path_init, args_config):
        self.path_init = path_init
        self.graph_helper = utils_rl_agent.GraphHelper(max_num_objects=args_config['model']['max_nodes'])
        # Build the agent types

        print("Loading data...")
        print("Objects: {}".format(len(self.graph_helper.object_dict)))
        print("---------------")

        agent_folder = glob.glob('{}/*'.format(path_init))

        # clean the agent folder
        pkl_files = []
        labels = []
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
        for agent_path_name in tqdm(agent_folder):
            try:
                agent_id = int(agent_path_name.split('/')[-1].split('_')[0]) - 1
            except:
                continue
            curr_agent_label = agent_labels[agent_id]
            input_files = glob.glob('{}/*.pik'.format(agent_path_name))
            pkl_files += input_files
            # print(len(pkl_files), agent_path_name)
            labels += [curr_agent_label for _ in input_files]

        self.labels = labels
        self.pkl_files = pkl_files
        self.overfit = args_config['train']['overfit']
        self.max_tsteps = args_config['model']['max_tsteps']
        self.max_actions = args_config['model']['max_actions']
        self.failed_items = mp.Array('i', len(self.pkl_files))
        assert self.max_actions == len(self.graph_helper.action_dict)+1, '{} vs {}'.format(self.max_actions, len(self.graph_helper.action_dict))

    def __len__(self):
        return len(self.pkl_files)

    def failure(self, index):
        if index not in self.failed_items:
            self.failed_items[index] = 1
        return self.__getitem__(0)

    def get_failures(self):
        cont = [item for item in self.failed_items]
        return sum(cont)

    def __getitem__(self, index):
        if self.overfit:
            index = 0
        with open(self.pkl_files[index], 'rb') as f:
            content = pkl.load(f)

        label_one_hot = torch.tensor(self.labels[index])
        # print(content.keys())
        attributes_include = ['class_objects', 'states_objects', 'object_coords', 'mask_object', 'node_ids', 'mask_obs_node']
        time_graph = {attr_name: [] for attr_name in attributes_include}
        # print(list(content.keys()))
        program = content['action'][0]
        if len(program) == 0:
            print(index)





        for it, graph in enumerate(content['graph']):
            # if it == len(content['graph']) - 1:
            #     # Skip the last graph
            #     continue

            if it >= self.max_tsteps:
                break
            graph_info, _ = self.graph_helper.build_graph(graph, character_id=1, include_edges=True, obs_ids=content['obs'][it])

            # class names
            for attribute_name in attributes_include:
                if attribute_name not in graph_info:
                    print(attribute_name, index, self.pkl_files[index])
                    return self.failure(index)
                time_graph[attribute_name].append(torch.tensor(graph_info[attribute_name]))
        

        # Match graph indices to index in the tensor
        node_ids = graph_info['node_ids']
        id2node = {node['id']: node for node in graph['nodes']}
        indexgraph2ind = {node_id: idi for idi, node_id in enumerate(node_ids)}

        # We will start with a No-OP action
        program_batch = {
            'action': [self.max_actions - 1],
            'obj1': [-1],
            'obj2': [-1],
            'indobj1': [indexgraph2ind[-1]],
            'indobj2': [indexgraph2ind[-1]],
        }

        # We start at 1 to skip the first instruction
        for it, instr in enumerate(program[1:]):
            
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
                return self.failure(index)

        program_batch['action'].append(self.max_actions - 1)
        program_batch['obj1'].append(-1)
        program_batch['obj2'].append(-1)
        program_batch['indobj1'].append(indexgraph2ind[-1])
        program_batch['indobj2'].append(indexgraph2ind[-1])

        num_tsteps = len(program_batch['action']) - 1
        for key in program_batch.keys():
            unpadded_tensor = torch.tensor(program_batch[key])

            # The program has an extra step
            padding_amount = self.max_tsteps - num_tsteps
            padding = [0] * unpadded_tensor.dim() * 2
            padding[-1] = padding_amount
            tuple_pad = tuple(padding)
            program_batch[key] = F.pad(unpadded_tensor, pad=tuple_pad, mode='constant', value=0.)

        length_mask = torch.zeros(self.max_tsteps)
        length_mask[:num_tsteps] = 1.

        # Batch across time
        for attribute_name in time_graph.keys():
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

        # Encode goal
        goals = content['goals'][0]

        target_obj_class = [self.graph_helper.object_dict.get_id('no_obj')] * 6
        target_loc_class = [self.graph_helper.object_dict.get_id('no_obj')] * 6
        mask_goal_pred = [0.0] * 6

        pre_id = 0
        obj_pred_names, loc_pred_names = [], []

        ##############################
        #### Inputs high level policy
        ##############################

        for predicate, info in content['goals'][0].items():
            count = info
            if count == 0:
                continue

            # if not (predicate.startswith('on') or predicate.startswith('inside')):
            #     continue

            elements = predicate.split('_')
            obj_class_id = int(self.graph_helper.object_dict.get_id(elements[1]))
            loc_class_id = int(self.graph_helper.object_dict.get_id(id2node[int(elements[2])]['class_name']))

            obj_pred_names.append(elements[1])
            loc_pred_names.append(id2node[int(elements[2])]['class_name'])
            for _ in range(count):
                try:
                    target_obj_class[pre_id] = obj_class_id
                    target_loc_class[pre_id] = loc_class_id
                    mask_goal_pred[pre_id] = 1.0
                    pre_id += 1
                except:
                    pdb.set_trace()

        goal = {'target_loc_class': torch.tensor(target_loc_class), 
                'target_obj_class': torch.tensor(target_obj_class), 
                'mask_goal_pred': torch.tensor(mask_goal_pred)}
        return time_graph, program_batch, label_one_hot, length_mask, goal

if __name__ == '__main__':
    arguments = get_args_pref_agent()
    with open(arguments.config, 'r') as f:
        config = yaml.load(f)
    dataset = AgentTypeDataset(path_init='../data_scratch/train_env_task_set_20_full_reduced_tasks/', args_config=config)
    data = dataset[0]
