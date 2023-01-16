import numpy as np
from . import base_nets
import torch
import torch.nn.functional as F
import torch.nn as nn
from .graph_nn import Transformer
from .base_nets import PositionalEncoding
import ipdb


class TaskEncoder(nn.Module):
    def mlp2l(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out), nn.ReLU(), nn.Linear(dim_out, dim_out), nn.ReLU()
        )
    def __init__(self, num_preds, num_counts, out_dim):
        super(TaskEncoder, self).__init__()
        self.bottleneck = out_dim
        self.quantity_encoder = self.mlp2l(num_counts, self.bottleneck)
        # self.aggregate = aggregate

        # if aggregate:
        self.rest_encoder = self.mlp2l(num_preds*self.bottleneck, out_dim)
        

    def forward(self, input_goal, aggregate=False):
        # B x num_preds x num_counts
        B = input_goal.shape[0]
        bottleneck_feats = self.quantity_encoder(input_goal)
        if not aggregate:
            return bottleneck_feats
        flatten_tensor = bottleneck_feats.reshape(B, -1) 
        embedding = self.rest_encoder(flatten_tensor)
        return embedding




class TaskFCEncoder(nn.Module):
    def __init__(self, num_preds, num_counts, out_dim, task_encoder):
        super(TaskFCEncoder, self).__init__()
        self.task_count_encoder = task_encoder
        self.hidden_size = out_dim
        self.fc_enc = nn.Sequential(
            nn.Linear(num_preds*self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

    def forward(self, task_graph_curr):
        # ipdb.set_trace()
        # B x T x num_preds x max_count
        B, T = task_graph_curr.shape[:2]
        out1 = self.task_count_encoder(task_graph_curr)
        # B x T x num_preds x dim
        out1 = out1.reshape(B, T, -1)
        output = self.fc_enc(out1)
        # ipdb.set_trace()

        # B x T x ndim
        return output


class TaskTransformerEncoder(nn.Module):
    def __init__(self, num_preds, num_counts, out_dim, task_encoder):
        super(TaskTransformerEncoder, self).__init__()
        self.task_count_encoder = task_encoder
        self.interm_embedding = out_dim
        self.pos_encoding = PositionalEncoding(d_model=self.interm_embedding, max_len=300)        
        in_feat = out_dim
        self.out_feat = out_dim 
        self.transformer_encoder = Transformer(None, None, in_feat, self.out_feat, nhead=1, num_layers=3)

    def forward(self, task_graph_curr, task_graph_init=None):
        if task_graph_init is not None:
            T = task_graph_curr.shape[1]
            task_graph_init = task_graph_init.repeat(1, T, 1, 1)
            # ipdb.set_trace()
            task_graph_curr = torch.cat([task_graph_init, task_graph_curr], 2)

        # B x T x num_preds x dim
        numpreds = task_graph_curr.shape[2]
        B, T = task_graph_curr.shape[:2]
        
        # B x T x 2*preds x dim
        input_embed = self.task_count_encoder(task_graph_curr)
        

        input_embed = input_embed.reshape(-1, numpreds, self.interm_embedding).permute(1, 0, 2)
        
        embed = self.pos_encoding(input_embed).transpose(0,1)


        one_mask = torch.ones((B*T, numpreds)).to(embed.device)
        output = self.transformer_encoder(embed, one_mask).reshape(B, T, numpreds, -1)
        # ipdb.set_trace()

        if task_graph_init is not None:
            d = task_graph_curr.shape[2]
            return output[:, :, d//2:, ...]
        else:
            return output

class GraphPredNetworkVAETask2(nn.Module):
    def mlp2l(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_in), nn.ReLU(), nn.Linear(dim_in, dim_out)
        )

    def __init__(self, args):
        super(GraphPredNetworkVAETask2, self).__init__()
        args = args['model']
        self.predict_diff = args['predict_diff']
        self.max_counts = args['num_counts']
        self.num_task_preds = args['num_task_preds']

        self.max_actions = args['max_actions']
        self.max_nodes = args['max_nodes']
        self.max_timesteps = args['max_tsteps']
        self.max_num_classes = args['max_class_objects']
        self.hidden_size = args['hidden_size']
        self.num_states = args['num_states']
        
        if args['task_encoder'] == 'TF':
            self.task_encoder_count = TaskEncoder(self.num_task_preds, self.max_counts, self.hidden_size)
            self.task_input_encoder = TaskTransformerEncoder(self.num_task_preds, self.max_counts, self.hidden_size, self.task_encoder_count)
            self.task_z_encoder = TaskTransformerEncoder(self.num_task_preds, self.max_counts, self.hidden_size, self.task_encoder_count)  
        else:
            self.task_encoder_count = TaskEncoder(self.num_task_preds, self.max_counts, self.hidden_size)
            self.task_input_encoder = TaskFCEncoder(self.num_task_preds, self.max_counts, self.hidden_size, self.task_encoder_count) 
            self.task_z_encoder = TaskFCEncoder(self.num_task_preds, self.max_counts, self.hidden_size, self.task_encoder_count) 

            

        self.task_pred = self.mlp2l(self.hidden_size, self.max_counts)
        self.mask_pred = self.mlp2l(self.hidden_size, 1)
        
        self.input_vae = args['input_vae']
        # if self.input_vae:

        self.global_repr = args['global_repr']
        

        # Combine previous action and graph
        multi = 2
        
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



        # VAE related
        self.cond_prior = False
        self.args = args
        self.use_vae = self.args.input_vae != 'none'
        if args['cond_prior']:
            self.cond_prior = True
            self.prior_net = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, 128 * 2)
            )
        self.posterior = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
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




    def sample_param(self, qvec):
        # ipdb.set_trace()
        mids = qvec.shape[-1] // 2
        mu = qvec[..., :mids]
        logvar = qvec[..., mids:]
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu
        return z

    def forward(self, inputs, cond=None, inference=False, seed=None, verbose=False):
        # ipdb.set_trace()
        # Cond is an embedding of the past, optionally used
        if verbose:
            ipdb.set_trace()

        inp_task_graph = F.one_hot(inputs['task_graph']['task_graph'].long(), self.max_counts)

        B, T, numpreds, nc = inp_task_graph.shape




        # ipdb.set_trace()
        # B x T x num_preds x dim
        embeddings_input_graph = self.task_input_encoder(inp_task_graph.float(), task_graph_init.float())
        ipdb.set_trace()
        end_task = F.one_hot(inputs['task_graph']['gt_task_graph'].long(), self.max_counts).float()
        # end_task_masked = end_task[:, None, ...].repeat(1, T, 1, 1) * inputs['task_graph']['mask_task_graph'][..., None]
        end_task_transformer = self.task_z_encoder(end_task[:, None, ...])[:, 0]

        # B x T x D we just take one embedding
        # ipdb.set_trace()
        encoded_goal_task = end_task_transformer[:, 0, :]
        
        q_post = self.posterior(encoded_goal_task)
        if self.cond_prior:
            p_prior = self.prior_net(graph_output) 
        else:
            mean_logvar = torch.zeros(q_post.shape)
            p_prior = mean_logvar.to(q_post.device)
            # p_prior = torch.cat([mean, log_var], -1)

        d = p_prior.shape[-1] // 2
        mu_prior, logvar_prior = p_prior[..., :d], p_prior[..., d:]
        mu_posterior, logvar_posterior =q_post[..., :d], q_post[..., d:]
        # ipdb.set_trace()
        if self.use_vae:
            if not inference:
                
                z_vec = self.sample_param(q_post)
            else:
                # print("sampling...")
                z_vec = self.sample_param(p_prior) 
                # print('sampled')
            # ipdb.set_trace()
            z_vec = z_vec[:, None, None, :].repeat(1, T, self.num_task_preds, 1)
            # ipdb.set_trace()
            z_and_input = torch.cat([z_vec, embeddings_input_graph], -1)
            z_and_input = self.z_projection(z_and_input)
        else:
            z_and_input = embeddings_input_graph
        # ipdb.set_trace()
        pred_graph = self.task_pred(z_and_input)
        pred_mask = self.mask_pred(z_and_input)[..., 0]

        # ipdb.set_trace()
        inp_mask = inputs['task_graph']['mask_task_graph'][..., None].float()

        # ipdb.set_trace()


        if self.predict_diff:
            pred_graph = inp_task_graph * (1 - inp_mask) + inp_mask * pred_graph
        return {'pred_mask': pred_mask, 'pred_graph': pred_graph, 
                'vae_params': [mu_prior, logvar_prior, mu_posterior, logvar_posterior]}

        # loss_action = nn.CrossEntropyLoss(action_logits, None, reduce=None)
        # loss_o1 = None
        # loss_o2 = None




class UniformModel():
    def __init__(self, repeat_every_tstep=False):
        self.every_tstep = repeat_every_tstep
        max_vec = ("0 0 0 2 0 5 0 3 0 0 2 0 0 0 1 0 0 0 0 0 "
                   "0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
                   "0 0 0 0 0 0 0 2 0 7 0 2 0 0 2 0 0 0 3 7 "
                   "0 0 3 0 0 0 0 0 0 3 7 0 0 3 0 0 0 0 0 0 "
                   "2 0 6 0 3 0 0 2 0 0 0 1 0 0 0 0 0 0 0 0 "
                   "0 0 2 0 7 0 3 0 0 2 0 0 0 2 6 0 0 3 0 0 "
                   "0 0 0 0 3 6 0 0 3 0 0 0 0 0 1 1")
        max_vec = np.array([int(x) for x in max_vec.split()])
        self.max_vec = max_vec
        self.use_vae = True

    def get_random(self, size, random_state):
        # returns a tensor of size [size, 136] with random samples
        
        new_size = list(size)+[self.max_vec.shape[0]]
        res = random_state.rand(*new_size)
        
        res *= self.max_vec
        res = res.astype(np.int32)
        return res

    def __call__(self, inputs, cond=None, inference=False, z_vec=None, seed=None, verbose=False):
        random_state = np.random.RandomState(seed) 
        size = inputs['task_graph']['task_graph'].shape[:-1]
        pred_graph = self.get_random(size, random_state)
        # pred_graph = 
        T = size[1]

        if self.every_tstep:
            # Make a prediction per tstep
            pred_graph = pred_graph[:, :1, ...].repeat(1, T, 1)
        prev_pred_graph = pred_graph
        output = {'pred_mask': None, 'pred_graph': pred_graph, 'pred_graph_total': prev_pred_graph}
        # ipdb.set_trace()
        return output

class GraphPredNetworkVAETask3(nn.Module):
    # Similar to VAE2, but the decoder decodes to the 136
    def mlp2l(self, dim_in, dim_out):

        return nn.Sequential(
            nn.Linear(dim_in, dim_in), nn.ReLU(), nn.Linear(dim_in, dim_out)
        )

    def __init__(self, args):
        super(GraphPredNetworkVAETask3, self).__init__()
        args = args['model']
        self.num_categories = 6
        self.predict_category = args['predict_category']
        self.use_only_input = args['use_only_input']
        self.predict_diff = args['predict_diff']
        
        # predict_diff_preds: We dont predict a mask explicitly, the data itself just represents the pred difference
        self.predict_diff_preds = args['predict_diff_preds'] 
        self.max_counts = args['num_counts']
        self.num_task_preds = args['num_task_preds']

        self.max_actions = args['max_actions']
        self.max_nodes = args['max_nodes']
        self.max_timesteps = args['max_tsteps']
        self.max_num_classes = args['max_class_objects']
        self.hidden_size = args['hidden_size']
        self.z_dim = args['z_dim']
        self.num_states = args['num_states']
        self.verbose = False
        self.latent_size = 128
        
        if self.predict_category:
            self.category_pred = nn.Linear(self.hidden_size, self.num_categories)


        if args['state_encoder'] == 'TF':
            self.task_encoder_count = TaskEncoder(self.num_task_preds, self.max_counts, self.hidden_size)
            self.task_input_encoder = TaskTransformerEncoder(self.num_task_preds, self.max_counts, self.hidden_size, self.task_encoder_count)
            self.task_z_encoder = TaskTransformerEncoder(self.num_task_preds, self.max_counts, self.hidden_size, self.task_encoder_count)  
        else:
            self.task_encoder_count = TaskEncoder(self.num_task_preds, self.max_counts, self.hidden_size)
            self.task_input_encoder = TaskFCEncoder(self.num_task_preds, self.max_counts, self.hidden_size, self.task_encoder_count) 
            self.task_z_encoder = TaskFCEncoder(self.num_task_preds, self.max_counts, self.hidden_size, self.task_encoder_count) 


        

        self.task_pred = self.mlp2l(self.hidden_size, self.max_counts)
        self.mask_pred = self.mlp2l(self.hidden_size, 1)
        
        self.input_vae = args['input_vae']
        # if self.input_vae:

        self.global_repr = args['global_repr']
        

        # Combine previous action and graph
        multi = 2
        
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



        # VAE related
        self.cond_prior = False
        self.args = args
        self.use_vae = self.args.input_vae != 'none'
        if args['cond_prior']:
            self.cond_prior = True
            self.prior_net = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.z_dim * 2)
            )
        self.posterior = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.z_dim * 2)
        )


        if self.args['cond_prior'] or self.args.autoencoder_type == 'pure_autoencoder' or self.use_only_input:
            input_dim_zproj = self.latent_size
        else:
            input_dim_zproj = self.latent_size + self.hidden_size
        self.z_projection = nn.Sequential(
                nn.Linear(128, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )


        multi_edge = 1
        if args['edge_pred'] == 'concat':
            multi_edge = 2

        input_dim = 128
        if self.use_only_input:
            if not self.use_vae:
                # Use the current step
                input_dim = 100
            else:
                input_dim = 128 + 100 # use both the last an ducrrent step
        else:
            input_dim = 128 + 100

        self.z_projection_out = self.mlp2l(input_dim, 128*self.num_task_preds)
        self.z_convert = self.mlp2l(128, 128)

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




    def sample_param(self, qvec):
        # ipdb.set_trace()
        mids = qvec.shape[-1] // 2
        mu = qvec[..., :mids]
        logvar = qvec[..., mids:]
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu
        # z = mu
        return z

    def z_vec_project(self, z_vec):
        # input, B x dim or B x T x dim
        # output B x num_task_preds x dim
        
        # This blows up
        time_included = len(z_vec.shape) > 2
        z_vec_flat = self.z_projection_out(z_vec)
        B = z_vec.shape[0]
        if time_included:
            T = z_vec.shape[1]
            z_vec_reshape = z_vec_flat.reshape(B, T, self.num_task_preds, 128)
        else:
            z_vec_reshape = z_vec_flat.reshape(B, self.num_task_preds, 128)
        return z_vec_reshape

    def forward(self, inputs, cond=None, inference=False, z_vec=None, seed=None, verbose=False):
        # ipdb.set_trace()
        # Cond is an embedding of the past, optionally used

        if self.verbose:
            ipdb.set_trace()
        inp_task_graph = F.one_hot(inputs['task_graph']['task_graph'].long(), self.max_counts)
        B, T, numpreds, nc = inp_task_graph.shape

        # Take the graph at step 0 + Graph at steps 0 -- N-1

        if self.args.autoencoder_type == 'pure_autoencoder':
            task_graph_init = None
            # Let's encode the last step
            inp_task_graph = inp_task_graph[:, -1:, ...]
        else:
            task_graph_init = inp_task_graph[:, :1, ...].float()
            # B x T x num_preds x dim
            # Note we will not use this in the pure decoder mode 
            if self.predict_diff_preds:
                # B x T x dim
                embeddings_input_graph = self.task_input_encoder(inp_task_graph.float())
            else:

                embeddings_input_graph = self.task_input_encoder(inp_task_graph.float(), task_graph_init)
        
        # ipdb.set_trace()

        if not inference:
            # ONLY USE THIS FOR NON INFERENCE
            # We need the GT graph to understand the embedding
            end_task = F.one_hot(inputs['task_graph']['gt_task_graph'].long(), self.max_counts).float()
            # end_task_masked = end_task[:, None, ...].repeat(1, T, 1, 1) * inputs['task_graph']['mask_task_graph'][..., None]
            

            if self.args['state_encoder'] == 'TF':
                end_task_transformer = self.task_z_encoder(end_task[:, None, ...])[:, 0]

                # B x T x D we just take one embedding
                encoded_goal_task = end_task_transformer[:, 0, :]
            else:
                # Index 0 in time
                end_task_transformer = self.task_z_encoder(end_task[:, None, ...])

                # B x T x D we just take one embedding
                encoded_goal_task = end_task_transformer[:, 0]

            q_post = self.posterior(encoded_goal_task)
            if self.predict_category:
                predicted_category = self.category_pred(encoded_goal_task)
        else:
            predicted_category = torch.zeros((B, self.num_categories)).to(inp_task_graph.device)
            
        if self.cond_prior:
            p_prior = self.prior_net(graph_output) 
        else:

            mean_logvar = torch.zeros([B, T, 128*2])[:, 0] # We dont take timesteps
            p_prior = mean_logvar.to(inp_task_graph.device)
            # p_prior = torch.cat([mean, log_var], -1)
        # print(p_prior.shape)
        # ipdb.set_trace()
        d = p_prior.shape[-1] // 2
        mu_prior, logvar_prior = p_prior[..., :d], p_prior[..., d:]
        if not inference:
            mu_posterior, logvar_posterior = q_post[..., :d], q_post[..., d:]
        else:
            mu_posterior, logvar_posterior = mu_prior, logvar_prior
        if self.use_vae:
            if not inference:
                
                z_vec = self.sample_param(q_post)
            else:
                # print("sampling...")
                if z_vec is None:
                    z_vec = self.sample_param(p_prior) 
                else:
                    z_vec = z_vec[:B, ...]
                # print('sampled')


            # Main change with VAE2
            # z_vec = z_vec[:, None, None, :].repeat(1, T, self.num_task_preds, 1)

            if self.use_only_input:
                # Whether we are not using the current step to predict, simply a VAE
                # z_and_input = torch.cat([z_vec, torch.zeros_like(embeddings_input_graph)], -1)

                z_vec = self.z_vec_project(z_vec)
                z_vec = z_vec[:, None, ...].repeat(1, T, 1, 1)
                z_and_input = z_vec
            else:
                z_vec_e = self.z_convert(z_vec)
                z_vec_e = z_vec_e[:, None, ...].repeat(1, T, 1)
                # ipdb.set_trace()
                
                z_vec_and_input = torch.cat([z_vec_e, embeddings_input_graph], -1)
                #print(z_vec_and_input.shape)
                z_and_input = self.z_vec_project(z_vec_and_input)
                #z_and_input = torch.cat([z_vec, embeddings_input_graph], -1)
                #print(z_and_input.shape)
            # ipdb.set_trace()
            z_and_input = self.z_projection(z_and_input)
        else:
            if self.args.autoencoder_type == 'pure_autoencoder': # autoencode output
                # from z to predicates
                # ipdb.set_trace()
                z_vec = self.z_vec_project(q_post[:, :d])[:, None, ...].repeat(1, T, 1, 1)
                # z_and_input = torch.cat([z_vec, torch.zeros_like(embeddings_input_graph)], -1)
                z_and_input = self.z_projection(z_vec) 
            else:
                # Use input to decode output, no VAE
                z_and_input = self.z_vec_project(embeddings_input_graph)
                z_and_input = self.z_projection(z_and_input) 
        pred_graph = self.task_pred(z_and_input)
        if self.predict_diff:
            if self.predict_diff_preds:
                raise Exception
            if self.use_only_input:
                # Use gt_mask as pred_mask, since it is impossibel to derive the mask
                pred_mask = inputs['task_graph']['mask_task_graph']
            else:
                pred_mask = self.mask_pred(z_and_input)[..., 0]
        else:
            pred_mask = torch.ones_like(inputs['task_graph']['mask_task_graph'])
        #print("MASK ", pred_mask.shape)

        # ipdb.set_trace()

        # ipdb.set_trace()

        prev_pred_graph = pred_graph
        if self.predict_diff:
            if not inference:
                inp_mask = inputs['task_graph']['mask_task_graph'][..., None].float()
                pred_graph = inp_task_graph * (1 - inp_mask) + inp_mask * pred_graph
            else:
                inp_mask = (pred_mask[..., None] > 0.).float()
                pred_graph = inp_task_graph * (1 - inp_mask) + inp_mask * pred_graph
        # ipdb.set_trace()
        output = {'pred_mask': pred_mask, 'pred_graph': pred_graph, 'pred_graph_total': prev_pred_graph, 
                'vae_params': [mu_prior, logvar_prior, mu_posterior, logvar_posterior]}
        output['zvec'] = z_vec
        if self.predict_category:
            output['predict_category'] = predicted_category
        #for outn, outp in output.items():
        #    if type(outp) == list:
        #        for pp in outp:
        #            print(outn, pp.get_device())
        #    else:
        #        print(outn, outp.get_device())
        return output

        # loss_action = nn.CrossEntropyLoss(action_logits, None, reduce=None)
        # loss_o1 = None
        # loss_o2 = None
















class GraphPredNetworkVAETask(nn.Module):
    def mlp2l(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out), nn.ReLU(), nn.Linear(dim_out, dim_out)
        )

    def __init__(self, args):
        super(GraphPredNetworkVAETask, self).__init__()
        args = args['model']
        self.max_counts = args['num_counts']
        self.num_task_preds = args['num_task_preds']

        self.max_actions = args['max_actions']
        self.max_nodes = args['max_nodes']
        self.max_timesteps = args['max_tsteps']
        self.max_num_classes = args['max_class_objects']
        self.hidden_size = args['hidden_size']
        self.num_states = args['num_states']
        

        self.task_pred = self.mlp2l(self.hidden_size, self.num_task_preds)
        self.mask_pred = self.mlp2l(self.hidden_size, 1)
        
        self.input_vae = args['input_vae']
        if self.input_vae:
            self.task_encoder = TaskEncoder(self.num_task_preds, self.max_counts, self.hidden_size)

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
                nn.Linear(self.hidden_size, self.z_dim * 2)
            )
        self.posterior = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.z_dim * 2)
        )


        if self.args['cond_prior']:
            input_dim_zproj = self.z_dim
        else:
            input_dim_zproj = self.z_dim + self.hidden_size
        self.z_projection = nn.Sequential(
            nn.Linear(input_dim_zproj, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )


        multi_edge = 1
        if args['edge_pred'] == 'concat':
            multi_edge = 2




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
        # TODO: this should change from the previous setting! Uncommend below
        end_task = F.one_hot(inputs['task_graph']['gt_task_graph'].long(), self.max_counts).float()
        end_task_masked = end_task[:, None, ...].repeat(1, T, 1, 1) * inputs['task_graph']['mask_task_graph'][..., None]
        # ipdb.set_trace()
        end_task_masked = end_task_masked.reshape(-1, self.num_task_preds, self.max_counts)
        encoded_goal_task = self.task_encoder(end_task_masked)
        encoded_goal_task = encoded_goal_task.reshape(B, T, -1)

        # ipdb.set_trace()

        q_post = self.posterior(encoded_goal_task)
        if self.cond_prior:
            p_prior = self.prior_net(graph_output) 
        else:
            mean_logvar = torch.zeros(q_post.shape)
            p_prior = mean_logvar.to(q_post.device)
            # p_prior = torch.cat([mean, log_var], -1)

        d = p_prior.shape[-1] // 2
        mu_prior, logvar_prior = p_prior[..., :d], p_prior[..., d:]
        mu_posterior, logvar_posterior = q_post[..., :d], q_post[..., d:]
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
                # graph_output_nodes = graph_output_nodes[:, :, None, :].repeat([1, 1, num_nodes, 1])
            else:
                z_vec = self.z_projection(z_vec)
                # graph_output_nodes = z_vec[:, :, None, :].repeat([1, 1, num_nodes, 1])
                graph_output = z_vec
        elif self.time_aggregate == 'firstcurr':
            graph_output_nodes = node_embeddings[:, 0, :, :].repeat([1, T, 1, 1])

        


        pred_graph = self.task_pred(graph_output_nodes)

        pred_mask = self.mask_pred(graph_output_nodes)

        # ipdb.set_trace()
        inp_task_graph = F.one_hot(inputs['task_graph']['task_graph'].long(), self.max_counts)
        inp_mask = inputs['task_graph']['mask_task_graph'][..., None].float()

        pred_graph = inp_task_graph * (1 - inp_mask) + inp_mask * pred_graph
        return {'pred_mask': pred_mask, 'pred_graph': pred_graph, 
                'vae_params': [mu_prior, logvar_prior, mu_posterior, logvar_posterior]}

        # loss_action = nn.CrossEntropyLoss(action_logits, None, reduce=None)
        # loss_o1 = None
        # loss_o2 = None






class GraphPredNetworkVAE(nn.Module):
    def mlp2l(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out), nn.ReLU(), nn.Linear(dim_out, dim_out)
        )

    def __init__(self, args):
        super(GraphPredNetworkVAE, self).__init__()
        args = args['model']
        self.max_counts = args['num_counts']
        self.num_task_preds = args['num_task_preds']

        self.max_actions = args['max_actions']
        self.max_nodes = args['max_nodes']
        self.max_timesteps = args['max_tsteps']
        self.max_num_classes = args['max_class_objects']
        self.hidden_size = args['hidden_size']
        self.num_states = args['num_states']
        

        self.mask_pred = MaskPredictor(self.hidden_size, self.num_task_preds)
        self.task_pred = TaskPredictor(self.hidden_size, self.num_task_preds, self.max_counts)
        
        self.input_vae = args['input_vae']
        # if self.input_vae:
        #     self.task_encoder = TaskEncoder(self.num_task_preds, self.max_counts, self.hidden_size)

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
        # TODO: this should change from the previous setting! Uncommend below
        q_post = self.posterior(torch.cat([last_tstep_embeddings.repeat(1, T, 1), graph_output], -1))
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
                # graph_output_nodes = graph_output_nodes[:, :, None, :].repeat([1, 1, num_nodes, 1])
            else:
                z_vec = self.z_projection(z_vec)
                # graph_output_nodes = z_vec[:, :, None, :].repeat([1, 1, num_nodes, 1])
                graph_output = z_vec
        elif self.time_aggregate == 'firstcurr':
            graph_output_nodes = node_embeddings[:, 0, :, :].repeat([1, T, 1, 1])

        


        pred_graph = self.task_pred(graph_output_nodes)

        pred_mask = self.mask_pred(graph_output_nodes)

        # ipdb.set_trace()
        inp_task_graph = F.one_hot(inputs['task_graph']['task_graph'].long(), self.max_counts)
        inp_mask = inputs['task_graph']['mask_task_graph'][..., None].float()

        pred_graph = inp_task_graph * (1 - inp_mask) + inp_mask * pred_graph
        return {'pred_mask': pred_mask, 'pred_graph': pred_graph, 
                'vae_params': [mu_prior, logvar_prior, mu_posterior, logvar_posterior]}

        # loss_action = nn.CrossEntropyLoss(action_logits, None, reduce=None)
        # loss_o1 = None
        # loss_o2 = None






class TaskPredictor(nn.Module):
    def mlp2l(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out), nn.ReLU(), nn.Linear(dim_out, dim_out)
        )
    def __init__(self, input_embed, num_tasks, num_task_count):
        super(TaskPredictor, self).__init__()
        self.mask_layers = nn.ModuleList([self.mlp2l(input_embed, num_task_count) for _ in range(num_tasks)])

    def forward(self, input_embedding):
        output_preds = []
        for layer in self.mask_layers:
            output_preds.append(layer(input_embedding)[..., None, :])

        output_preds = torch.cat(output_preds, -2)
        return output_preds

class MaskPredictor(nn.Module):
    def mlp2l(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out), nn.ReLU(), nn.Linear(dim_out, dim_out)
        )
    def __init__(self, input_embed, num_tasks):
        super(MaskPredictor, self).__init__()
        self.mask_layer = self.mlp2l(input_embed, num_tasks)

    def forward(self, input_embedding):
        return self.mask_layer(input_embedding)

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
