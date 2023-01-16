from . import base_nets
import torch
import torch.nn.functional as F
import torch.nn as nn
import ipdb



class GraphPredNetworkVAE(nn.Module):
    def mlp2l(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out), nn.ReLU(), nn.Linear(dim_out, dim_out)
        )

    def __init__(self, args):
        super(GraphPredNetworkVAE, self).__init__()
        args = args['model']
        self.max_actions = args['max_actions']
        self.max_nodes = args['max_nodes']
        self.max_timesteps = args['max_tsteps']
        self.max_num_classes = args['max_class_objects']
        self.hidden_size = args['hidden_size']
        self.num_states = args['num_states']
        self.exclusive_edge = False

        if args['exclusive_edge']:
            self.exclusive_edge = True
            self.edge_types = 1
        else:
            self.edge_types = args['edge_types']
        self.global_repr = args['global_repr']
        args_tf = {
            'hidden_size': self.hidden_size,
            'max_nodes': self.max_nodes,
            'num_classes': self.max_num_classes,
            'num_states': self.num_states,
        }

        if args['state_encoder'] == 'TF':
            self.graph_encoder = base_nets.TransformerBase(**args_tf)
        elif args['state_encoder'] == 'GNN':
            self.graph_encoder = base_nets.GNNBase2(**args_tf)

        self.action_embedding = nn.Embedding(self.max_actions, self.hidden_size)
        self.agent_embedding = nn.Embedding(args['num_agents'], self.hidden_size)
        self.use_agent_embedding = args['agent_embed']

        # Combine previous action and graph
        multi = 2
        if self.use_agent_embedding:
            raise Exception
            multi = 3

        # Used to transform the goal encoding into features that we will use for action/object prediction
        '''
        self.fc_att_action = self.mlp2l(self.hidden_size , self.hidden_size)
        self.fc_att_object = self.mlp2l(self.hidden_size , self.hidden_size)
        self.fc_att_object2 = self.mlp2l(self.hidden_size, self.hidden_size)
        '''

        self.comb_layer = nn.Linear(self.hidden_size * multi, self.hidden_size)
        self.comb_out_layer = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.num_layer_lstm = 2
        self.time_aggregate = args['time_aggregate']

        if args['time_aggregate'] == 'LSTM' or 'VAE' in args['time_aggregate']:
            self.RNN = nn.LSTM(
                self.hidden_size,
                self.hidden_size,
                self.num_layer_lstm,
                batch_first=True,
            )
        elif args['time_aggregate'] == 'none':
            # use the current state
            self.COMBTime = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            )
        elif args['time_aggregate'] == 'firstcurr':
            # use the current state
            self.COMBTime = nn.Sequential(
                nn.Linear(self.hidden_size*2, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            )

        # VAE related
        self.cond_prior = False
        self.args = args
        if args['cond_prior']:
            self.cond_prior = True
            self.prior_net = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, 128 * 2)
            )
        self.posterior = nn.Sequential(
            nn.Linear(self.hidden_size*2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 128 * 2)
        )


        if self.args['cond_prior']:
            input_dim_zproj = 128
        else:
            input_dim_zproj = 128 + self.hidden_size
        self.z_projection = nn.Sequential(
            nn.Linear(input_dim_zproj, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )


        multi_edge = 1
        if args['edge_pred'] == 'concat':
            multi_edge = 2

        self.edge_pred = nn.Sequential(
            nn.Linear(self.hidden_size * multi_edge, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.edge_types),
        )

        self.edge_pred = nn.Sequential(
            nn.Linear(self.hidden_size * multi_edge, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.edge_types),
        )

        self.state_pred = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_states),
        )

        self.pred_change = args['predict_edge_change']
        self.node_change = args['predict_node_change']
        if self.pred_change:
            self.edge_change_pred = nn.Sequential(
                nn.Linear(self.hidden_size * multi_edge, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, 2),
            )
        if self.node_change:
            self.node_change_pred = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, 2),
            )

        # self.action_pred = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, self.max_actions))
        # self.object1_pred = nn.Sequential(nn.Linear(self.hidden_size*2, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, 1))
        # self.object2_pred = nn.Sequential(nn.Linear(self.hidden_size*2, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, 1))

        # self.pred_close_net = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
        #                            nn.ReLU(),
        #                            nn.Linear(self.hidden_size, 1))
        # self.pred_goal_net = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
        #                                    nn.ReLU(),
        #                                    nn.Linear(self.hidden_size, 1))

        self.goal_inp = args['goal_inp']
        self.edge_pred_mode = args['edge_pred']
        if args['goal_inp']:
            self.goal_encoder = base_nets.GoalEncoder(
                self.max_num_classes,
                self.hidden_size,
                obj_class_encoder=self.graph_encoder.object_class_encoding,
            )

    def pred_obj_states(self, inputs):
        # inputs: Batch x T x num_nodes x embed
        nnodes = inputs.shape[-2]
        states = self.state_pred(inputs)
        edges1 = inputs.repeat([1, 1, nnodes, 1])
        edges2 = inputs.repeat_interleave(nnodes, dim=2)

        if self.edge_pred_mode == 'concat':
            edge_embeds = torch.cat([edges1, edges2], dim=-1)

        elif self.edge_pred_mode == 'dot':
            edge_embeds = edges1 * edges2

        else:
            raise Exception

        edges = self.edge_pred(edge_embeds)
        change = None
        if self.pred_change:
            change = self.edge_change_pred(edge_embeds)
        if self.node_change:
            change = self.node_change_pred(inputs)
        return states, edges, change

    def sample_param(self, qvec):
        # ipdb.set_trace()
        mids = qvec.shape[-1] // 2
        mu = qvec[..., :mids]
        logvar = qvec[..., mids:]
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu
        return z

    def forward(self, inputs, cond=None, inference=False, seed=None):
        # ipdb.set_trace()
        # Cond is an embedding of the past, optionally used
        # ipdb.set_trace()
        program = inputs['program']
        graph = inputs['graph']
        mask_len = inputs['mask_len']
        mask_nodes = graph['mask_object']
        index_obj1 = program['indobj1']
        index_obj2 = program['indobj2']

        # Compute d_{i}
        node_embeddings = self.graph_encoder(graph)
        
        dims = list(node_embeddings.shape)
        B, T, num_nodes, embed_size = dims





        # Is this ok?
        # node_embeddings[node_embeddings.isnan()] = 1


        action_embed = self.action_embedding(program['action'])

        assert torch.all(inputs['graph']['node_ids'][:, 0, 0] == 1).item()

        # Graph representation, it is the representation of the character or pool
        if self.global_repr == 'pool':
            graph_repr = (
                node_embeddings
                * mask_nodes.unsqueeze(-1).expand(-1, -1, -1, node_embeddings.shape[-1])
            ).sum(-2)
        else:
            graph_repr = node_embeddings[:, :, 0]

        

        # last_tstep_embeddings, get the graph at the last step
        # ipdb.set_trace()
        tsteps = mask_len.sum(-1)[..., None, None].repeat(1, 1, embed_size).long() - 1
        last_tstep_embeddings = torch.gather(graph_repr, 1, tsteps)


        # Prepare input to the LSTM
        action_graph = torch.cat([action_embed[:, :, :], graph_repr], -1)
        input_embed = self.comb_layer(action_graph)

        graph_output, (h_t, c_t) = self.RNN(input_embed)

        # Compute posterior
        q_post = self.posterior(torch.cat([last_tstep_embeddings.repeat(1, T, 1), graph_output], -1))
        if self.cond_prior:
            p_prior = self.prior_net(graph_output) 
        else:
            mean_logvar = torch.zeros(q_post.shape)
            p_prior = mean_logvar.to(q_post.device)
            # p_prior = torch.cat([mean, log_var], -1)

        d = p_prior.shape[-1] // 2
        mu_prior, logvar_prior = p_prior[..., :d], p_prior[..., d:]
        mu_posterior, logvar_posterior =q_post[..., :d], q_post[..., d:]
        if not inference:
            
            z_vec = self.sample_param(q_post)
        else:
            # print("sampling...")
            z_vec = self.sample_param(p_prior) 
            # print('sampled')
        if self.time_aggregate in ['LSTM', 'none']:
            # Output of lstm, concatenate with output of graph
            graph_output_nodes = graph_output.unsqueeze(-2).repeat(
                [1, 1, self.max_nodes, 1]
            )  # Recurrent part, we may want to replace that by a z later?
        elif 'VAE' in self.time_aggregate:
            if not self.cond_prior:
                # ipdb.set_trace()
                z_and_lstm = torch.cat([graph_output, z_vec], -1)
                graph_output_nodes = self.z_projection(z_and_lstm)
                graph_output_nodes = graph_output_nodes[:, :, None, :].repeat([1, 1, num_nodes, 1])
            else:
                z_vec = self.z_projection(z_vec)
                graph_output_nodes = z_vec[:, :, None, :].repeat([1, 1, num_nodes, 1])

        elif self.time_aggregate == 'firstcurr':
            graph_output_nodes = node_embeddings[:, 0, :, :].repeat([1, T, 1, 1])

        graphs_at_output = node_embeddings  # Before the recurrent net

        output_and_lstm = torch.cat([graph_output_nodes, graphs_at_output], -1)
        output_and_lstm = self.comb_out_layer(output_and_lstm)


        pred_states, pred_edges, pred_changes = self.pred_obj_states(output_and_lstm)

        if self.exclusive_edge:
            N = mask_nodes.shape[-1]
            mask_nodes_edge = mask_nodes.repeat(1, 1, N)[..., None]
            pred_edges = -1e9 * (1 - mask_nodes_edge) + mask_nodes_edge * pred_edges
        # ipdb.set_trace()
        name_change = 'edge_change'
        if self.node_change:
            name_change = 'node_change'
        return {'states': pred_states, 'edges': pred_edges, name_change: pred_changes, 
                'vae_params': [mu_prior, logvar_prior, mu_posterior, logvar_posterior]}

        # loss_action = nn.CrossEntropyLoss(action_logits, None, reduce=None)
        # loss_o1 = None
        # loss_o2 = None





class GraphPredNetwork(nn.Module):
    def mlp2l(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out), nn.ReLU(), nn.Linear(dim_out, dim_out)
        )

    def __init__(self, args):
        super(GraphPredNetwork, self).__init__()
        args = args['model']
        self.max_actions = args['max_actions']
        self.max_nodes = args['max_nodes']
        self.max_timesteps = args['max_tsteps']
        self.max_num_classes = args['max_class_objects']
        self.hidden_size = args['hidden_size']
        self.num_states = args['num_states']
        self.exclusive_edge = False

        if args['exclusive_edge']:
            self.exclusive_edge = True
            self.edge_types = 1
        else:
            self.edge_types = args['edge_types']
        self.global_repr = args['global_repr']
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

        self.action_embedding = nn.Embedding(self.max_actions, self.hidden_size)
        self.agent_embedding = nn.Embedding(args['num_agents'], self.hidden_size)
        self.use_agent_embedding = args['agent_embed']

        # Combine previous action and graph
        multi = 2
        if self.use_agent_embedding:
            raise Exception
            multi = 3

        # Used to transform the goal encoding into features that we will use for action/object prediction
        '''
        self.fc_att_action = self.mlp2l(self.hidden_size , self.hidden_size)
        self.fc_att_object = self.mlp2l(self.hidden_size , self.hidden_size)
        self.fc_att_object2 = self.mlp2l(self.hidden_size, self.hidden_size)
        '''

        self.comb_layer = nn.Linear(self.hidden_size * multi, self.hidden_size)
        self.comb_out_layer = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.num_layer_lstm = 2
        self.time_aggregate = args['time_aggregate']

        if args['time_aggregate'] == 'LSTM':
            self.RNN = nn.LSTM(
                self.hidden_size,
                self.hidden_size,
                self.num_layer_lstm,
                batch_first=True,
            )
        elif args['time_aggregate'] == 'none':
            # use the current state
            self.COMBTime = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            )
        elif args['time_aggregate'] == 'firstcurr':
            # use the current state
            self.COMBTime = nn.Sequential(
                nn.Linear(self.hidden_size*2, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            )


        multi_edge = 1
        if args['edge_pred'] == 'concat':
            multi_edge = 2

        self.edge_pred = nn.Sequential(
            nn.Linear(self.hidden_size * multi_edge, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.edge_types),
        )

        self.edge_pred = nn.Sequential(
            nn.Linear(self.hidden_size * multi_edge, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.edge_types),
        )

        self.state_pred = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_states),
        )

        self.pred_change = args['predict_edge_change']
        self.node_change = args['predict_node_change']
        if self.pred_change:
            self.edge_change_pred = nn.Sequential(
                nn.Linear(self.hidden_size * multi_edge, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, 2),
            )
        if self.node_change:
            self.node_change_pred = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, 2),
            )

        # self.action_pred = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, self.max_actions))
        # self.object1_pred = nn.Sequential(nn.Linear(self.hidden_size*2, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, 1))
        # self.object2_pred = nn.Sequential(nn.Linear(self.hidden_size*2, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, 1))

        # self.pred_close_net = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
        #                            nn.ReLU(),
        #                            nn.Linear(self.hidden_size, 1))
        # self.pred_goal_net = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
        #                                    nn.ReLU(),
        #                                    nn.Linear(self.hidden_size, 1))

        self.goal_inp = args['goal_inp']
        self.edge_pred_mode = args['edge_pred']
        if args['goal_inp']:
            self.goal_encoder = base_nets.GoalEncoder(
                self.max_num_classes,
                self.hidden_size,
                obj_class_encoder=self.graph_encoder.object_class_encoding,
            )

    def pred_obj_states(self, inputs):
        # inputs: Batch x T x num_nodes x embed
        nnodes = inputs.shape[-2]
        states = self.state_pred(inputs)
        edges1 = inputs.repeat([1, 1, nnodes, 1])
        edges2 = inputs.repeat_interleave(nnodes, dim=2)

        if self.edge_pred_mode == 'concat':
            edge_embeds = torch.cat([edges1, edges2], dim=-1)

        elif self.edge_pred_mode == 'dot':
            edge_embeds = edges1 * edges2

        else:
            raise Exception

        edges = self.edge_pred(edge_embeds)
        change = None
        if self.pred_change:
            change = self.edge_change_pred(edge_embeds)
        if self.node_change:
            change = self.node_change_pred(inputs)
        return states, edges, change

    def forward(self, inputs, cond=None, inference=False):
        # Cond is an embedding of the past, optionally used
        # ipdb.set_trace()
        program = inputs['program']
        graph = inputs['graph']
        mask_len = inputs['mask_len']
        mask_nodes = graph['mask_object']
        index_obj1 = program['indobj1']
        index_obj2 = program['indobj2']
        print('inp')
        node_embeddings = self.graph_encoder(graph)
        print('out')

        # Is this ok?
        # node_embeddings[node_embeddings.isnan()] = 1

        dims = list(node_embeddings.shape)
        B, T, num_nodes, embed_size = dims
        action_embed = self.action_embedding(program['action'])

        assert torch.all(inputs['graph']['node_ids'][:, 0, 0] == 1).item()

        # Graph representation, it is the representation of the character
        if self.global_repr == 'pool':
            graph_repr = (
                node_embeddings
                * mask_nodes.unsqueeze(-1).expand(-1, -1, -1, node_embeddings.shape[-1])
            ).sum(-2)
        else:
            graph_repr = node_embeddings[:, :, 0]

        # Input previous action and current graph
        if not self.goal_inp:

            action_graph = torch.cat([action_embed[:, :, :], graph_repr], -1)
        else:
            raise Exception
            # Goal encoding
            # obj_class_name = inputs['goal']['target_obj_class']  # [:, 0].long()
            # loc_class_name = inputs['goal']['target_loc_class']  # [:, 0].long()
            # mask_goal = inputs['goal']['mask_goal_pred']
            # # goal_enc = self.goal_encoder()
            # # ipdb.set_trace()
            # goal_encoding = self.goal_encoder(obj_class_name, loc_class_name, mask_goal)

            # goal_mask_action = torch.sigmoid(self.fc_att_action(goal_encoding))
            # goal_mask_object1 = torch.sigmoid(self.fc_att_object(goal_encoding))
            # goal_mask_object2 = torch.sigmoid(self.fc_att_object2(goal_encoding))

            # goal_encoding = goal_encoding[:, None, :].repeat(1, graph_repr.shape[1], 1)
            # gated_goal = graph_repr * goal_mask_action[:, None, :]
            # action_graph = torch.cat([action_embed[:, :, :], gated_goal], -1)

        if self.use_agent_embedding:
            raise Exception
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
        # input_embed: B x T x Num_nodes x embed
        # graph_output is Batch x T x Dim

        if self.time_aggregate == 'LSTM':
            graph_output, (h_t, c_t) = self.RNN(input_embed)
        elif self.time_aggregate == 'none':
            graph_output = self.COMBTime(input_embed)
        

        if self.time_aggregate in ['LSTM', 'none']:
            # Output of lstm, concatenate with output of graph
            graph_output_nodes = graph_output.unsqueeze(-2).repeat(
                [1, 1, self.max_nodes, 1]
            )  # Recurrent part, we may want to replace that by a z later?
    
        elif self.time_aggregate == 'firstcurr':
            graph_output_nodes = node_embeddings[:, 0, :, :].repeat([1, T, 1, 1])

        graphs_at_output = node_embeddings  # Before the recurrent net

        output_and_lstm = torch.cat([graph_output_nodes, graphs_at_output], -1)

        output_and_lstm = self.comb_out_layer(output_and_lstm)
        pred_states, pred_edges, pred_changes = self.pred_obj_states(output_and_lstm)

        if self.exclusive_edge:
            N = mask_nodes.shape[-1]
            mask_nodes_edge = mask_nodes.repeat(1, 1, N)[..., None]
            pred_edges = -1e9 * (1 - mask_nodes_edge) + mask_nodes_edge * pred_edges
        # ipdb.set_trace()
        name_change = 'edge_change'
        if self.node_change:
            name_change = 'node_change'
        return {'states': pred_states, 'edges': pred_edges, name_change: pred_changes}

        # loss_action = nn.CrossEntropyLoss(action_logits, None, reduce=None)
        # loss_o1 = None
        # loss_o2 = None


class GoalConditionedGraphPredNetwork(nn.Module):
    def mlp2l(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out), nn.ReLU(), nn.Linear(dim_out, dim_out)
        )

    def __init__(self, args):
        super(GoalConditionedGraphPredNetwork, self).__init__()
        args = args['model']
        self.max_actions = args['max_actions']
        self.max_nodes = args['max_nodes']
        self.max_timesteps = args['max_tsteps']
        self.max_num_classes = args['max_class_objects']
        self.hidden_size = args['hidden_size']
        self.num_states = args['num_states']
        self.edge_types = args['edge_types']
        self.global_repr = args['global_repr']
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

        self.action_embedding = nn.Embedding(self.max_actions, self.hidden_size)
        self.agent_embedding = nn.Embedding(args['num_agents'], self.hidden_size)
        self.use_agent_embedding = args['agent_embed']

        # Combine previous action and graph
        multi = 2
        if self.use_agent_embedding:
            raise Exception
            multi = 3

        # Used to transform the goal encoding into features that we will use for action/object prediction
        '''
        self.fc_att_action = self.mlp2l(self.hidden_size , self.hidden_size)
        self.fc_att_object = self.mlp2l(self.hidden_size , self.hidden_size)
        self.fc_att_object2 = self.mlp2l(self.hidden_size, self.hidden_size)
        '''

        self.comb_layer = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.comb_out_layer = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.num_layer_lstm = 2
        self.time_aggregate = args['time_aggregate']

        if args['time_aggregate'] == 'LSTM':
            self.RNN = nn.LSTM(
                self.hidden_size,
                self.hidden_size,
                self.num_layer_lstm,
                batch_first=True,
            )
        elif args['time_aggregate'] == 'none':
            # use the current state
            self.COMBTime = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            )

        multi_edge = 1
        if args['edge_pred'] == 'concat':
            multi_edge = 2

        self.edge_pred = nn.Sequential(
            nn.Linear(self.hidden_size * multi_edge, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.edge_types),
        )

        self.edge_pred = nn.Sequential(
            nn.Linear(self.hidden_size * multi_edge, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.edge_types),
        )

        self.state_pred = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_states),
        )

        self.edge_change_pred = nn.Sequential(
            nn.Linear(self.hidden_size * multi_edge, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 2),
        )

        # self.action_pred = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, self.max_actions))
        # self.object1_pred = nn.Sequential(nn.Linear(self.hidden_size*2, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, 1))
        # self.object2_pred = nn.Sequential(nn.Linear(self.hidden_size*2, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, 1))

        # self.pred_close_net = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
        #                            nn.ReLU(),
        #                            nn.Linear(self.hidden_size, 1))
        # self.pred_goal_net = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
        #                                    nn.Linear(self.hidden_size, 1))

        self.goal_inp = args['goal_inp']
        self.edge_pred_mode = args['edge_pred']
        self.pred_change = args['predict_edge_change']
        if args['goal_inp']:
            self.goal_encoder = base_nets.GoalEncoder(
                self.max_num_classes,
                self.hidden_size,
                obj_class_encoder=self.graph_encoder.object_class_encoding,
            )

    def pred_obj_states(self, inputs):
        # inputs: Batch x T x num_nodes x embed
        nnodes = inputs.shape[-2]
        states = self.state_pred(inputs)
        edges1 = inputs.repeat([1, 1, nnodes, 1])
        edges2 = inputs.repeat_interleave(nnodes, dim=2)

        if self.edge_pred_mode == 'concat':
            edge_embeds = torch.cat([edges1, edges2], dim=-1)

        elif self.edge_pred_mode == 'dot':
            edge_embeds = edges1 * edges2

        else:
            raise Exception

        edges = self.edge_pred(edge_embeds)
        edge_change = None
        if self.pred_change:
            edge_change = self.edge_change_pred(edge_embeds)
        return states, edges, edge_change

    def forward(self, inputs, cond=None):
        # Cond is an embedding of the past, optionally used

        program = inputs['program']
        graph = inputs['graph']
        goal = inputs['goal_graph']
        # goal = inputs['goal']
        mask_len = inputs['mask_len']
        mask_nodes = graph['mask_object']
        index_obj1 = program['indobj1']
        index_obj2 = program['indobj2']
        node_embeddings = self.graph_encoder(graph)
        goal_embeddings = self.graph_encoder(goal)
        # Is this ok?
        node_embeddings[node_embeddings.isnan()] = 1
        goal_embeddings[goal_embeddings.isnan()] = 1

        dims = list(node_embeddings.shape)
        action_embed = self.action_embedding(program['action'])

        assert torch.all(inputs['graph']['node_ids'][:, 0, 0] == 1).item()

        # Graph representation
        if self.global_repr == 'pool':
            graph_repr = (
                node_embeddings
                * mask_nodes.unsqueeze(-1).expand(-1, -1, -1, node_embeddings.shape[-1])
            ).sum(-2)
        else:
            graph_repr = node_embeddings[:, :, 0]

        goal_repr = (
            goal_embeddings
            * mask_nodes.unsqueeze(-1).expand(-1, -1, -1, goal_embeddings.shape[-1])
        ).sum(-2)

        # Input previous action and current graph
        # if not self.goal_inp:

        #     action_graph = torch.cat([action_embed[:, :-1, :], graph_repr], -1)
        # else:
        #     raise Exception
        #     # Goal encoding
        #     obj_class_name = inputs['goal']['target_obj_class']  # [:, 0].long()
        #     loc_class_name = inputs['goal']['target_loc_class']  # [:, 0].long()
        #     mask_goal = inputs['goal']['mask_goal_pred']
        #     # goal_enc = self.goal_encoder()
        #     # ipdb.set_trace()
        #     goal_encoding = self.goal_encoder(obj_class_name, loc_class_name, mask_goal)

        #     goal_mask_action = torch.sigmoid(self.fc_att_action(goal_encoding))
        #     goal_mask_object1 = torch.sigmoid(self.fc_att_object(goal_encoding))
        #     goal_mask_object2 = torch.sigmoid(self.fc_att_object2(goal_encoding))

        #     goal_encoding = goal_encoding[:, None, :].repeat(1, graph_repr.shape[1], 1)
        #     gated_goal = graph_repr * goal_mask_action[:, None, :]
        #     action_graph = torch.cat([action_embed[:, :-1, :], gated_goal], -1)

        # if self.use_agent_embedding:
        #     raise Exception
        #     tsteps = action_graph.shape[1]

        #     agent_embeddings = self.agent_embedding(inputs['label_agent'])
        #     agent_embeddings = agent_embeddings[:, None, :].repeat([1, tsteps, 1])
        #     action_graph = torch.cat([action_graph, agent_embeddings], -1)

        goal_graph = torch.cat([graph_repr, goal_repr], -1)
        input_embed = self.comb_layer(goal_graph)

        if cond is not None:
            cond_vec = cond
            ipdb.set_trace()
            input_embed = torch.cat([input_embed, cond_vec], -1)

        # Input a combination of previous actions and graph
        if self.time_aggregate == 'LSTM':
            graph_output, (h_t, c_t) = self.RNN(input_embed)
        elif self.time_aggregate == 'none':
            graph_output = self.COMBTime(input_embed)

        # graph_output is Batch x T x Dim

        # Output of lstm, concatenate with output of graph
        graph_output_nodes = graph_output.unsqueeze(-2).repeat(
            [1, 1, self.max_nodes, 1]
        )  # Recurrent part, we may want to replace that by a z later?
        graphs_at_output = node_embeddings  # Before the recurrent net

        output_and_lstm = torch.cat([graph_output_nodes, graphs_at_output], -1)

        output_and_lstm = self.comb_out_layer(output_and_lstm)
        pred_states, pred_edges, pred_changes = self.pred_obj_states(output_and_lstm)
        return {'states': pred_states, 'edges': pred_edges, 'edge_change': pred_changes}

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
            multi = 3  # input goal as well

        self.comb_layer = nn.Linear(self.hidden_size * multi, self.hidden_size)
        self.num_layer_lstm = 2

        self.RNN = nn.LSTM(
            self.hidden_size, self.hidden_size, self.num_layer_lstm, batch_first=True
        )

        self.action_pred = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.max_actions),
        )
        self.object1_pred = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
        )
        self.object2_pred = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
        )

        self.pred_close_net = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
        )
        self.pred_goal_net = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
        )

        self.goal_inp = args['goal_inp']
        if args['goal_inp']:
            self.goal_encoder = base_nets.GoalEncoder(
                self.max_num_classes,
                self.hidden_size,
                obj_class_encoder=self.graph_encoder.object_class_encoding,
            )

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

        assert torch.all(inputs['graph']['node_ids'][:, 0, 0] == 1).item()

        # Graph representation, it is the representation of the character
        graph_repr = node_embeddings[:, :, 0]

        # Input previous action and current graph
        if not self.goal_inp:

            action_graph = torch.cat([action_embed[:, :, :], graph_repr], -1)
        else:
            # Goal encoding
            obj_class_name = inputs['goal']['target_obj_class']  # [:, 0].long()
            loc_class_name = inputs['goal']['target_loc_class']  # [:, 0].long()
            mask_goal = inputs['goal']['mask_goal_pred']
            # goal_enc = self.goal_encoder()
            # ipdb.set_trace()
            goal_encoding = self.goal_encoder(obj_class_name, loc_class_name, mask_goal)
            goal_encoding = goal_encoding[:, None, :].repeat(1, graph_repr.shape[1], 1)
            action_graph = torch.cat(
                [action_embed[:, :, :], graph_repr, goal_encoding], -1
            )

        input_embed = self.comb_layer(action_graph)

        if cond is not None:
            cond_vec = cond
            # ipdb.set_trace()
            input_embed = torch.cat([input_embed, cond_vec], -1)

        # ipdb.set_trace()
        # Input a combination of previous actions and graph
        graph_output, (h_t, c_t) = self.RNN(input_embed)

        # skip the last graph

        ## Action logit
        action_logits = self.action_pred(graph_output)

        # Output of lstm, concatenate with output of graph
        graph_output_nodes = graph_output.unsqueeze(-2).repeat(
            [1, 1, self.max_nodes, 1]
        )

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
        return {
            'action_logits': action_logits,
            'o1_logits': obj1_logit,
            'o2_logits': obj2_logit,
            'pred_goal': pred_goal,
            'pred_close': pred_close,
        }

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
            multi = 3  # input goal as well

        self.comb_layer = nn.Linear(self.hidden_size * multi, self.hidden_size)
        self.num_layer_lstm = 2

        self.RNN = nn.LSTM(
            self.hidden_size, self.hidden_size, self.num_layer_lstm, batch_first=True
        )

        self.action_pred = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.max_actions),
        )
        self.object1_pred = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
        )
        self.object2_pred = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
        )

        self.pred_close_net = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
        )
        self.pred_goal_net = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
        )

        self.goal_inp = args['goal_inp']
        if args['goal_inp']:
            self.goal_encoder = base_nets.GoalEncoder(
                self.max_num_classes,
                self.hidden_size,
                obj_class_encoder=self.graph_encoder.object_class_encoding,
            )

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

        assert torch.all(inputs['graph']['node_ids'][:, 0, 0] == 1).item()

        # Graph representation, it is the representation of the character
        graph_repr = node_embeddings[:, :, 0]

        # Input previous action and current graph
        if not self.goal_inp:

            action_graph = torch.cat([action_embed[:, :, :], graph_repr], -1)
        else:
            # Goal encoding
            obj_class_name = inputs['goal']['target_obj_class']  # [:, 0].long()
            loc_class_name = inputs['goal']['target_loc_class']  # [:, 0].long()
            mask_goal = inputs['goal']['mask_goal_pred']
            # goal_enc = self.goal_encoder()
            # ipdb.set_trace()
            goal_encoding = self.goal_encoder(obj_class_name, loc_class_name, mask_goal)
            goal_encoding = goal_encoding[:, None, :].repeat(1, graph_repr.shape[1], 1)
            action_graph = torch.cat(
                [action_embed[:, :, :], graph_repr, goal_encoding], -1
            )

        input_embed = self.comb_layer(action_graph)

        if cond is not None:
            cond_vec = cond
            ipdb.set_trace()
            input_embed = torch.cat([input_embed, cond_vec], -1)

        # Input a combination of previous actions and graph
        graph_output, (h_t, c_t) = self.RNN(input_embed)

        return graph_output
