from torch.utils.data import Dataset
import torch
import ipdb
import glob
from tqdm import tqdm
import pickle as pkl
from utils import utils_rl_agent
import torch.nn.functional as F

class AgentTypeDataset(Dataset):
	def __init__(self, path_init, args):
		self.path_init = path_init
		self.graph_helper = utils_rl_agent.GraphHelper(max_num_objects=150)
		# Build the agent types

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
		self.max_tsteps = args.max_tsteps

	def __len__(self):
		return len(self.pkl_files)


	def __getitem__(self, index):
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

		program_batch = {
			'action': [],
			'obj1': [],
			'obj2': []
		}


		if 'grab' in program[0]:
			print(index, "Frist grab")
		
		for it, instr in enumerate(program):
			instr_item = self.graph_helper.actionstr2index(instr)
			program_batch['action'].append(instr_item[0])
			program_batch['obj1'].append(instr_item[1])
			program_batch['obj2'].append(instr_item[2])


		for key in program_batch.keys():
			unpadded_tensor = torch.tensor(program_batch[key])
			padding_amount = self.max_tsteps - len(program)
			padding = [0] * unpadded_tensor.dim() * 2
			padding[-1] = padding_amount
			tuple_pad = tuple(padding)
			program_batch[key] = F.pad(unpadded_tensor, pad=tuple_pad, mode='constant', value=0.)

		if set(content['obs'][1]) != set(content['obs'][2]) and 'open' not in program[1]:
			# print(sorted(content['obs'][0]))
			# print(sorted(content['obs'][1]))
			print(index, "BAD", program[1], program[2], set(content['obs'][0]).symmetric_difference(set(content['obs'][1])))
		for it, graph in enumerate(content['graph']):
			graph_info, _ = self.graph_helper.build_graph(graph, character_id=1, include_edges=True, obs_ids=content['obs'][it])

			# class names
			for attribute_name in attributes_include:
				if attribute_name not in graph_info:
					print(attribute_name, index, self.pkl_files[index])
					return self.__getitem__(index+1)
				time_graph[attribute_name].append(torch.tensor(graph_info[attribute_name]))
			
		# Batch across time
		for attribute_name in time_graph.keys():
			unpadded_tensor = torch.cat([item[None, :] for item in time_graph[attribute_name]])
			# Do padding
			padding_amount = self.max_tsteps - len(program)
			# ipdb.set_trace()
			padding = [0] * unpadded_tensor.dim() * 2
			padding[-1] = padding_amount
			tuple_pad = tuple(padding)
			time_graph[attribute_name] = F.pad(unpadded_tensor, pad=tuple_pad, mode='constant', value=0.)

		return time_graph, program_batch, label_one_hot