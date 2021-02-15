from . import base_nets
import torch
import torch.nn.functional as F
import torch.nn as nn
import ipdb

class ActionPredNetwork(nn.Module):
    def __init__(self, args):
        super(ActionPredNetwork, self).__init__()
        args = args['model']
        self.max_actions = args['max_actions']
        self.max_nodes = args['max_nodes']
        self.max_timesteps = args['max_tsteps']
        self.max_num_classes = args['max_class_objects']
        self.hidden_size = args['hidden_size']
        self.num_states = args['num_states']
        args_tf = {
                'hidden_size': self.hidden_size,
                'max_nodes': self.max_nodes,
                'num_classes': self.max_num_classes,
                'num_states': self.num_states,
        }
        self.graph_encoder = base_nets.TransformerBase(**args_tf)
        self.action_embedding = nn.Embedding(self.max_actions, self.hidden_size)

        # Combine previous action and graph
        self.comb_layer = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.num_layer_lstm = 2

        self.RNN = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layer_lstm, batch_first=True)
        self.action_pred = nn.Linear(self.hidden_size, self.max_actions)
        self.object1_pred = nn.Linear(self.hidden_size*2, 1)
        self.object2_pred = nn.Linear(self.hidden_size*2, 1)


    def forward(self, inputs):
        program = inputs['program']
        graph = inputs['graph']
        mask_len = inputs['mask_len']
        mask_nodes = graph['mask_object']
        index_obj1 = program['indobj1']
        index_obj2 = program['indobj2']
        node_embeddings = self.graph_encoder(graph)
        # Is this ok?
        node_embeddings[node_embeddings.isnan()] = 1

        dims = list(node_embeddings.shape)
        action_embed = self.action_embedding(program['action'])
        
        assert torch.all(inputs['graph']['node_ids'][:,0,0] == 1).item()
        # Objects in the current obs

        # index_obj1 = index_obj1[:,: , None, None].repeat([1, 1, 1, dims[-1]])
        # index_obj2 = index_obj2[:,:, None, None].repeat([1, 1, 1, dims[-1]])

        # print("get object embedding")

        # print(node_embeddings.shape, index_obj1.max())
        # obj1_embed = torch.gather(node_embeddings, 1, index_obj1).squeeze(-2)
        # obj2_embed = torch.gather(node_embeddings, 1, index_obj2).squeeze(-2)

        # Graph representation, it is the representation of the character
        graph_repr = node_embeddings[:, :, 0]

        # Input previous action and current graph
        action_graph = torch.cat([action_embed[:, :-1, :], graph_repr], -1)
        input_embed = self.comb_layer(action_graph)
        
        # Input a combination of previous actions and graph 
        graph_output, (h_t, c_t) = self.RNN(input_embed)

        # skip the last graph

        ## Action logit
        action_logits = self.action_pred(graph_output)

        # Output of lstm, concatenate with output of graph
        graph_output_nodes = graph_output.unsqueeze(-2).repeat([1, 1, self.max_nodes, 1])

        graphs_at_output = node_embeddings
        # ipdb.set_trace() 

        output_and_lstm = torch.cat([graph_output_nodes, graphs_at_output], -1)

        obj1_logit = self.object1_pred(output_and_lstm).squeeze(-1)
        obj2_logit = self.object2_pred(output_and_lstm).squeeze(-1)

        # Mask out logits according to the nodes that exist in the graph
        
        obj1_logit = obj1_logit * mask_nodes + (1 - mask_nodes) * -1e9
        obj2_logit = obj2_logit * mask_nodes + (1 - mask_nodes) * -1e9
        # ipdb.set_trace()
        return {'action_logits': action_logits, 'o1_logits': obj1_logit, 'o2_logits': obj2_logit}
        
        # loss_action = nn.CrossEntropyLoss(action_logits, None, reduce=None)  
        # loss_o1 = None
        # loss_o2 = None 
