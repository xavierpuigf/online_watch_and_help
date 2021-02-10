from . import base_nets
import torch.nn.functional as F
import torch.nn as nn
import ipdb

class ActionPredNetwork(nn.Module):
    def __init__(self, args):
        super(ActionPredNetwork, self).__init__()
        self.max_actions = args.max_actions
        self.max_nodes = args.max_nodes
        self.max_timesteps = args.max_tsteps
        self.max_num_classes = args.max_class_objects
        self.hidden_size = args.hidden_size
        self.num_states = args.num_states
        args_tf = {
                'hidden_size': self.hidden_size,
                'max_nodes': self.max_nodes,
                'num_classes': self.max_num_classes,
                'num_states': self.num_states,
        }
        self.graph_encoder = base_nets.TransformerBase(**args_tf)
        self.action_embedding = nn.Embedding(self.max_actions, self.hidden_size)

    def forward(self, inputs):
        program = inputs['program']
        graph = inputs['graph']
        node_embeddings = self.graph_encoder(graph)
        action_embed = self.action_embedding(program['action'])
        ipdb.set_trace()
