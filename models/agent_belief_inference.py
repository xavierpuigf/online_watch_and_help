from . import base_nets
import torch
import torch.nn.functional as F
import torch.nn as nn
import ipdb

class ModelAggregate(nn.Module):
    def __init__(self, model, args):
        super(ModelAggregate, self).__init__()
        self.model_single_ep = model

    def forward(self, inputs, other_inputs):
        # first, we platten the episodes
        other_episode_inputs = {}
        other_inputs_tg_flatten = {}
        other_inputs_program_flatten = {}


        other_episode_inputs['graph'] = {}
        other_episode_inputs['program'] = {}
        other_episode_inputs['goal'] = {}

        for key in other_inputs['time_graph']:
            dims = list(other_inputs['time_graph'][key].shape)
            bs, neps = dims[:2]
            other_episode_inputs['graph'][key] = torch.reshape(other_inputs['time_graph'][key], [-1]+dims[2:])

        for key in other_inputs['program_batch']:
            dims = list(other_inputs['program_batch'][key].shape)
            other_episode_inputs['program'][key] = torch.reshape(other_inputs['program_batch'][key], [-1]+dims[2:])

        for key in other_inputs['goal']:
            dims = list(other_inputs['goal'][key].shape)
            other_episode_inputs['goal'][key] = torch.reshape(other_inputs['goal'][key], [-1]+dims[2:])


        dims = list(other_inputs['length_mask'].shape)
        other_episode_inputs['mask_len'] = torch.reshape(other_inputs['length_mask'], [-1]+dims[2:])

        # ipdb.set_trace()

        outputs_eps = self.model_single_ep(other_episode_inputs, compute_belief=False)
        laststep = other_episode_inputs['mask_len'].sum(dim=-1).long() - 1
        outputs_eps = torch.gather(outputs_eps, 1, laststep[:, None, None].repeat(1, 1, outputs_eps.shape[-1]))[:, 0, :]
        output_eps = outputs_eps.reshape([bs, neps, -1]).mean(1)
        return self.model_single_ep(inputs, context_feat=output_eps)

class ActionGatedPredNetwork(nn.Module):

    def mlp2l(self, dim_in, dim_out):
        return nn.Sequential(nn.Linear(dim_in, dim_out), nn.ReLU(), nn.Linear(dim_out, dim_out))

    def __init__(self, args):
        super(ActionGatedPredNetwork, self).__init__()
        args = args['model']
        self.max_actions = args['max_actions']
        self.max_nodes = args['max_nodes']
        self.max_timesteps = args['max_tsteps']
        self.max_num_classes = args['max_class_objects']
        self.hidden_size = args['hidden_size']
        self.num_states = args['num_states']
        self.categorical_belief = args['categorical_belief']
        args_tf = {
                'hidden_size': self.hidden_size,
                'max_nodes': self.max_nodes,
                'num_classes': self.max_num_classes,
                'num_states': self.num_states,
        }
        if args['state_encoder'] == 'TF':
            self.graph_encoder = base_nets.TransformerBase(**args_tf)
        elif args['state_encoder'] == 'GNN':
            self.graph_encoder = base_nets.GNNBase(**args_tf)
        #self.graph_encoder = base_nets.TransformerBase(**args_tf)
        self.action_embedding = nn.Embedding(self.max_actions, self.hidden_size)


        self.agent_embedding = nn.Embedding(args['num_agents'], self.hidden_size)
        self.use_agent_embedding = args['agent_embed']


        # Combine previous action and graph
        multi = 2
        if self.use_agent_embedding:
            multi = 3

        self.fc_att_action = self.mlp2l(self.hidden_size , self.hidden_size)
        self.fc_att_object = self.mlp2l(self.hidden_size , self.hidden_size)
        self.fc_att_object2 = self.mlp2l(self.hidden_size, self.hidden_size)

        self.comb_layer = nn.Linear(self.hidden_size*multi, self.hidden_size)
        self.num_layer_lstm = 2
        self.time_aggregate = args['time_aggregate']
        
        if args['time_aggregate'] == 'LSTM':
            self.RNN = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layer_lstm, batch_first=True)
            self.RNN2 = nn.LSTM(self.hidden_size*2, self.hidden_size, self.num_layer_lstm, batch_first=True)
        elif args['time_aggregate'] == 'none':
            self.COMBTime = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                       nn.ReLU(),
                                       nn.Linear(self.hidden_size, self.hidden_size))
            self.COMBTime2 = nn.Sequential(nn.Linear(self.hidden_size*2, self.hidden_size),
                                       nn.ReLU(),
                                       nn.Linear(self.hidden_size, self.hidden_size))



        self.action_pred = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, self.max_actions))
        self.object1_pred = nn.Sequential(nn.Linear(self.hidden_size*2, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, 1))
        self.object2_pred = nn.Sequential(nn.Linear(self.hidden_size*2, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, 1))

        if not self.categorical_belief:
            self.belief_pred = nn.Sequential(nn.Linear(self.hidden_size*2, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, 1))
            self.belief_pred_rooms = nn.Sequential(nn.Linear(self.hidden_size*2, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, 1))
        else:
            self.num_containers =  args['ncontbelief']
            self.num_rooms = args['nroomsbelief']
            self.belief_pred = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, self.num_containers))
            self.belief_pred_rooms = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, self.num_rooms))

        self.pred_close_net = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden_size, 1))
        self.pred_goal_net = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                           nn.ReLU(),
                                           nn.Linear(self.hidden_size, 1))



        self.goal_inp = args['goal_inp']
        if args['goal_inp']:
            self.goal_encoder = base_nets.GoalEncoder(self.max_num_classes, self.hidden_size, obj_class_encoder=self.graph_encoder.object_class_encoding)

    def forward(self, inputs, cond=None, compute_belief=True, context_feat=None):
        # Cond is an embedding of the past, optionally used

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
        

        #print(program['action'].shape, node_embeddings.shape)
        #ipdb.set_trace()
        assert torch.all(inputs['graph']['node_ids'][:,0,0] == 1).item()
        

        # Graph representation, it is the representation of the character
        graph_repr = node_embeddings[:, :, 0]

        # Input previous action and current graph
        if not self.goal_inp:

            action_graph = torch.cat([action_embed[:, :-1, :], graph_repr], -1)
        else:
            # Goal encoding
            obj_class_name = inputs['goal']['target_obj_class']  # [:, 0].long()
            loc_class_name = inputs['goal']['target_loc_class']  # [:, 0].long()
            mask_goal = inputs['goal']['mask_goal_pred']
            # goal_enc = self.goal_encoder()
            # ipdb.set_trace()
            goal_encoding = self.goal_encoder(obj_class_name, loc_class_name, mask_goal)

            goal_mask_action = torch.sigmoid(self.fc_att_action(goal_encoding))
            goal_mask_object1 = torch.sigmoid(self.fc_att_object(goal_encoding))
            goal_mask_object2 = torch.sigmoid(self.fc_att_object2(goal_encoding))

            goal_encoding = goal_encoding[:, None, :].repeat(1, graph_repr.shape[1], 1)
            gated_goal = graph_repr * goal_mask_action[:, None, :]
            action_graph = torch.cat([action_embed[:, :-1, :], gated_goal], -1)


        if self.use_agent_embedding:
            tsteps = action_graph.shape[1]
            
            agent_embeddings = self.agent_embedding(inputs['label_agent'])
            agent_embeddings = agent_embeddings[:, None, :].repeat([1, tsteps, 1])
            action_graph = torch.cat([action_graph, agent_embeddings], -1)

        input_embed = self.comb_layer(action_graph)
        
        if cond is not None:
            cond_vec = cond
            ipdb.set_trace()
            input_embed = torch.cat([input_embed, cond_vec], -1)


        # Input a combination of previous actions and graph 
        if self.time_aggregate == 'LSTM':
            tstep, nnodes = list(input_embed.shape)[1:3]
            if context_feat is None:
                graph_output, (h_t, c_t) = self.RNN(input_embed)
            else:
                context_feat = context_feat[:, None, :].repeat(1, tstep, 1)
                input_embed = torch.cat([input_embed, context_feat], -1)
                graph_output, (h_t, c_t) = self.RNN2(input_embed)

        elif self.time_aggregate == 'none':
            if context_Feat is None:
                graph_output = self.COMBTime(input_embed)
            else:
                context_feat = context_feat[:, None, :].repeat(1, tstep, 1)
                input_embed = torch.cat([input_embed, context_feat], -1)
                graph_output = self.COMBTime2(input_embed)

        if not compute_belief:
            return graph_output

        # skip the last graph

        ## Action logit
        action_logits = self.action_pred(graph_output)

        # Output of lstm, concatenate with output of graph
        graph_output_nodes = graph_output.unsqueeze(-2).repeat([1, 1, self.max_nodes, 1])

        graphs_at_output = node_embeddings
        graphs_at_output_gate1 =  goal_mask_object1[:, None, None, :] * graphs_at_output
        graphs_at_output_gate2 = goal_mask_object2[:, None, None, :] * graphs_at_output
        
        output_and_lstm1 = torch.cat([graph_output_nodes, graphs_at_output_gate1], -1)
        output_and_lstm2 = torch.cat([graph_output_nodes, graphs_at_output_gate2], -1)

        # For now, the beliefs are independent of the goals
        output_and_lstm_belief = torch.cat([graph_output_nodes, node_embeddings], -1)

        obj1_logit = self.object1_pred(output_and_lstm1).squeeze(-1)
        obj2_logit = self.object2_pred(output_and_lstm2).squeeze(-1)


        # Belief prediction
        # Only interested in the last step belief
        laststep = mask_len.sum(dim=-1).long() - 1
        laststep = laststep[:, None, None]



        pred_close = self.pred_close_net(graphs_at_output).squeeze(-1)
        pred_goal = self.pred_goal_net(graphs_at_output_gate1).squeeze(-1)


        # Mask out logits according to the nodes that exist in the graph
        
        obj1_logit = obj1_logit * mask_nodes + (1 - mask_nodes) * -1e9
        obj2_logit = obj2_logit * mask_nodes + (1 - mask_nodes) * -1e9
        # ipdb.set_trace()


        # Mask out belief logit
        if not self.categorical_belief:
            belief_logit = torch.gather(self.belief_pred(output_and_lstm2).squeeze(-1), 1, laststep)[:, 0, :]
            belief_logit_room = torch.gather(self.belief_pred_rooms(output_and_lstm2).squeeze(-1), 1, laststep)[:, 0, :]
            mask_b = inputs['belief_info']['mask_belief_container']
            mask_b_room =  inputs['belief_info']['mask_belief_room']
            belief_logit = belief_logit * mask_b + (1 - mask_b) * -1e9
            belief_logit_room = belief_logit_room * mask_b_room + (1 - mask_b_room) * -1e9
        else:
            graph_output_laststep = torch.gather(graph_output, 1, laststep.repeat(1, 1, self.hidden_size))
            belief_logit = self.belief_pred(graph_output_laststep).squeeze(1)
            belief_logit_room = self.belief_pred_rooms(graph_output_laststep).squeeze(1)
            # ipdb.set_trace()
        res = {
            'action_logits': action_logits, 
            'o1_logits': obj1_logit, 
            'o2_logits': obj2_logit, 
            'pred_goal': pred_goal, 
            'pred_close': pred_close,
            'belief_logit': belief_logit, 
            'belief_logit_room': belief_logit_room
        }
            
        return res
        
        # loss_action = nn.CrossEntropyLoss(action_logits, None, reduce=None)  
        # loss_o1 = None
        # loss_o2 = None 


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
        multi = 2
        if args['goal_inp']:
            multi = 3 # input goal as well

        self.comb_layer = nn.Linear(self.hidden_size*multi, self.hidden_size)
        self.num_layer_lstm = 2

        self.RNN = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layer_lstm, batch_first=True)


        self.action_pred = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, self.max_actions))
        self.object1_pred = nn.Sequential(nn.Linear(self.hidden_size*2, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, 1))
        self.object2_pred = nn.Sequential(nn.Linear(self.hidden_size*2, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, 1))


        self.pred_close_net = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden_size, 1))
        self.pred_goal_net = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                           nn.ReLU(),
                                           nn.Linear(self.hidden_size, 1))

        self.goal_inp = args['goal_inp']
        if args['goal_inp']:
            self.goal_encoder = base_nets.GoalEncoder(self.max_num_classes, self.hidden_size, obj_class_encoder=self.graph_encoder.object_class_encoding)

    def forward(self, inputs, cond=None):
        # Cond is an embedding of the past, optionally used

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
        

        # Graph representation, it is the representation of the character
        graph_repr = node_embeddings[:, :, 0]

        # Input previous action and current graph
        if not self.goal_inp:

            action_graph = torch.cat([action_embed[:, :-1, :], graph_repr], -1)
        else:
            # Goal encoding
            obj_class_name = inputs['goal']['target_obj_class']  # [:, 0].long()
            loc_class_name = inputs['goal']['target_loc_class']  # [:, 0].long()
            mask_goal = inputs['goal']['mask_goal_pred']
            # goal_enc = self.goal_encoder()
            # ipdb.set_trace()
            goal_encoding = self.goal_encoder(obj_class_name, loc_class_name, mask_goal)
            goal_encoding = goal_encoding[:, None, :].repeat(1, graph_repr.shape[1], 1)
            action_graph = torch.cat([action_embed[:, :-1, :], graph_repr, goal_encoding], -1)

        input_embed = self.comb_layer(action_graph)
        
        if cond is not None:
            cond_vec = cond
            ipdb.set_trace()
            input_embed = torch.cat([input_embed, cond_vec], -1)


        ipdb.set_trace()
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


        pred_close = self.pred_close_net(graphs_at_output).squeeze(-1)
        pred_goal = self.pred_goal_net(graphs_at_output).squeeze(-1)


        # Mask out logits according to the nodes that exist in the graph
        
        obj1_logit = obj1_logit * mask_nodes + (1 - mask_nodes) * -1e9
        obj2_logit = obj2_logit * mask_nodes + (1 - mask_nodes) * -1e9
        # ipdb.set_trace()
        return {'action_logits': action_logits, 'o1_logits': obj1_logit, 'o2_logits': obj2_logit, 'pred_goal': pred_goal, 'pred_close': pred_close}
        
        # loss_action = nn.CrossEntropyLoss(action_logits, None, reduce=None)  
        # loss_o1 = None
        # loss_o2 = None 



class ActionCharNetwork(nn.Module):
    def __init__(self, args):
        super(ActionCharNetwork, self).__init__()
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
        multi = 2
        if args['goal_inp']:
            multi = 3 # input goal as well

        self.comb_layer = nn.Linear(self.hidden_size*multi, self.hidden_size)
        self.num_layer_lstm = 2

        self.RNN = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layer_lstm, batch_first=True)


        self.action_pred = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, self.max_actions))
        self.object1_pred = nn.Sequential(nn.Linear(self.hidden_size*2, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, 1))
        self.object2_pred = nn.Sequential(nn.Linear(self.hidden_size*2, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, 1))


        self.pred_close_net = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden_size, 1))
        self.pred_goal_net = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                           nn.ReLU(),
                                           nn.Linear(self.hidden_size, 1))

        self.goal_inp = args['goal_inp']
        if args['goal_inp']:
            self.goal_encoder = base_nets.GoalEncoder(self.max_num_classes, self.hidden_size, obj_class_encoder=self.graph_encoder.object_class_encoding)

    def forward(self, inputs, cond=None):
        # Cond is an embedding of the past, optionally used

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
        

        # Graph representation, it is the representation of the character
        graph_repr = node_embeddings[:, :, 0]

        # Input previous action and current graph
        if not self.goal_inp:

            action_graph = torch.cat([action_embed[:, :-1, :], graph_repr], -1)
        else:
            # Goal encoding
            obj_class_name = inputs['goal']['target_obj_class']  # [:, 0].long()
            loc_class_name = inputs['goal']['target_loc_class']  # [:, 0].long()
            mask_goal = inputs['goal']['mask_goal_pred']
            # goal_enc = self.goal_encoder()
            # ipdb.set_trace()
            goal_encoding = self.goal_encoder(obj_class_name, loc_class_name, mask_goal)
            goal_encoding = goal_encoding[:, None, :].repeat(1, graph_repr.shape[1], 1)
            action_graph = torch.cat([action_embed[:, :-1, :], graph_repr, goal_encoding], -1)

        input_embed = self.comb_layer(action_graph)
        
        if cond is not None:
            cond_vec = cond
            ipdb.set_trace()
            input_embed = torch.cat([input_embed, cond_vec], -1)

        # Input a combination of previous actions and graph 
        graph_output, (h_t, c_t) = self.RNN(input_embed)

        return graph_output
