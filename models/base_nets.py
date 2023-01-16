from torch import nn
from .graph_nn import Transformer, GraphModel, GraphModelGGNN
import pdb
from utils.utils_models import init
import torch
import ipdb
import numpy as np
import math


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.lstm = nn.LSTM(recurrent_input_size, hidden_size)
            for name, param in self.lstm.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_lstm(self, x, hidden, masks):
        if x.size(0) == hidden[0].size(0):

            assert x.ndim == 2 and hidden[0].ndim == 2
            x, (h, c) = self.lstm(
                x.unsqueeze(0),
                ((hidden[0] * masks).unsqueeze(0), (hidden[1] * masks).unsqueeze(0)),
            )
            x = x.squeeze(0)
            h = h.squeeze(0)
            c = c.squeeze(0)
        else:
            raise Exception

        return x, (h, c)


class GoalEncoder(nn.Module):
    def __init__(self, num_classes, output_dim, obj_class_encoder=None):
        super(GoalEncoder, self).__init__()

        if obj_class_encoder is None:
            inp_dim = output_dim
            self.object_embedding = nn.Embedding(num_classes, inp_dim)
        else:
            self.object_embedding = obj_class_encoder
            inp_dim = self.object_embedding.embedding_dim

        self.combine_obj_loc = nn.Sequential(
            nn.Linear(inp_dim * 2, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, object_class_name, loc_class_name, mask_goal_pred):
        obj_embedding = self.object_embedding(object_class_name)
        loc_embedding = self.object_embedding(loc_class_name)
        obj_loc = torch.cat([obj_embedding, loc_embedding], axis=2)
        object_location = self.combine_obj_loc(obj_loc)

        num_preds = mask_goal_pred.sum(-1)
        # norm_mask = (mask_goal_pred/num_preds.unsqueeze(-1)).unsqueeze(-1)
        # Difference with low level policy
        norm_mask = mask_goal_pred.unsqueeze(-1)

        average_pred = (object_location * norm_mask).sum(1)

        if torch.isnan(average_pred).any():
            pdb.set_trace()
        return average_pred

    # def forward(self, object_class_name, loc_class_name, mask_goal_pred):
    #     obj_embedding = self.object_embedding(object_class_name)
    #     loc_embedding = self.object_embedding(loc_class_name)
    #     obj_loc = torch.cat([obj_embedding, loc_embedding], axis=2)
    #     object_location = self.combine_obj_loc(obj_loc)

    #     sum_pred = (object_location * mask_goal_pred).sum(1)

    #     if torch.isnan(sum_pred).any():
    #         pdb.set_trace()
    #     return sum_pred


# class GoalAttentionModel(NNBase):
#     def __init__(self, recurrent=False, hidden_size=128, num_classes=100, node_encoder=None, context_type='avg'):
#         super(GoalAttentionModel, self).__init__(recurrent, hidden_size, hidden_size)

#         init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
#                                constant_(x, 0), nn.init.calculate_gain('relu'))


#         self.main = node_encoder
#         self.context_size = hidden_size
#         self.critic_linear = init_(nn.Linear(hidden_size, 1))

#         self.object_context_combine = self.mlp2l(2 * hidden_size, hidden_size)

#         self.goal_encoder = GoalEncoder(num_classes, 2 * hidden_size, obj_class_encoder=self.main.object_class_encoding)
#         # self.goal_encoder = nn.EmbeddingBag(num_classes, hidden_size, mode='sum')
#         self.context_type = context_type

#         self.fc_att_action = self.mlp2l(hidden_size * 2, hidden_size)
#         self.fc_att_object = self.mlp2l(hidden_size * 2, hidden_size)
#         self.train()

#     def mlp2l(self, dim_in, dim_out):
#         return nn.Sequential(nn.Linear(dim_in, dim_out), nn.ReLU(), nn.Linear(dim_out, dim_out))

#     def forward(self, inputs, rnn_hxs, masks):
#         # Use transformer to get feats for every object
#         mask_visible = inputs['mask_object']

#         features_obj = self.main(inputs)
#         #pdb.set_trace()

#         # 1 x ndim. Avg pool the features for the context vec
#         mask_visible = mask_visible.unsqueeze(-1)

#         # Mean pool of transformer
#         if self.context_type == 'avg':
#             context_vec = (features_obj * mask_visible).sum(1) / (1e-9 + mask_visible.sum(1))
#         else:
#             context_vec = features_obj[:, 0, :]


#         # Goal embedding
#         obj_class_name = inputs['target_obj_class']  # [:, 0].long()
#         loc_class_name = inputs['target_loc_class']  # [:, 0].long()
#         mask_goal = inputs['mask_goal_pred']

#         goal_encoding = self.goal_encoder(obj_class_name, loc_class_name, mask_goal)

#         # goal_encoding_obj = self.goal_encoder(obj_class_name).squeeze(1)
#         # goal_encoding_loc = self.goal_encoder(loc_class_name).squeeze(1)
#         # goal_encoding = torch.cat([goal_encoding_obj, goal_encoding_loc], dim=-1)


#         goal_mask_action = torch.sigmoid(self.fc_att_action(goal_encoding))
#         goal_mask_object = torch.sigmoid(self.fc_att_object(goal_encoding))

#         # Recurrent context
#         if self.is_recurrent:
#             r_context_vec, rnn_hxs = self._forward_gru(context_vec, rnn_hxs, masks)
#         else:
#             r_context_vec = context_vec

#         # h' = GA . h [bs, h]
#         context_goal = goal_mask_action * r_context_vec

#         # Combine object representations with global representations
#         r_object_vec = torch.cat([features_obj, r_context_vec.unsqueeze(1).repeat(1, features_obj.shape[1], 1)], 2)
#         r_object_vec_comb = self.object_context_combine(r_object_vec)

#         # Sg' = GA . Sg [bs, N, h]
#         object_goal = goal_mask_object[:, None, :] * r_object_vec_comb

#         if torch.isnan(context_goal).any() or torch.isnan(object_goal).any():
#             pdb.set_trace()

#         return context_goal, object_goal, rnn_hxs


class GoalAttentionModel(NNBase):
    def __init__(
        self,
        recurrent=False,
        hidden_size=128,
        num_classes=100,
        node_encoder=None,
        context_type='avg',
    ):
        super(GoalAttentionModel, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'),
        )

        self.main = node_encoder
        self.context_size = hidden_size
        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.object_context_combine = self.mlp2l(2 * hidden_size, hidden_size)

        self.goal_encoder = GoalEncoder(
            num_classes, 2 * hidden_size
        )  # , obj_class_encoder=self.main.object_class_encoding)
        # self.goal_encoder = nn.EmbeddingBag(num_classes, hidden_size, mode='sum')
        self.context_type = context_type

        self.fc_att_action = self.mlp2l(hidden_size * 2, hidden_size)
        self.fc_att_object = self.mlp2l(hidden_size * 2, hidden_size)
        self.train()

    def mlp2l(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out), nn.ReLU(), nn.Linear(dim_out, dim_out)
        )

    def forward(self, inputs, rnn_hxs, masks):
        # Use transformer to get feats for every object
        mask_visible = inputs['mask_object']

        features_obj = self.main(inputs)
        # pdb.set_trace()

        # 1 x ndim. Avg pool the features for the context vec
        mask_visible = mask_visible.unsqueeze(-1)

        # Mean pool of transformer
        if self.context_type == 'avg':
            context_vec = (features_obj * mask_visible).sum(1) / (
                1e-9 + mask_visible.sum(1)
            )
        else:
            context_vec = features_obj[:, 0, :]

        # Goal embedding
        obj_class_name = inputs['target_obj_class']  # [:, 0].long()
        loc_class_name = inputs['target_loc_class']  # [:, 0].long()
        mask_goal = inputs['mask_goal_pred']

        goal_encoding = self.goal_encoder(obj_class_name, loc_class_name, mask_goal)

        # goal_encoding_obj = self.goal_encoder(obj_class_name).squeeze(1)
        # goal_encoding_loc = self.goal_encoder(loc_class_name).squeeze(1)
        # goal_encoding = torch.cat([goal_encoding_obj, goal_encoding_loc], dim=-1)

        goal_mask_action = torch.sigmoid(self.fc_att_action(goal_encoding))
        goal_mask_object = torch.sigmoid(self.fc_att_object(goal_encoding))

        # h' = GA . h [bs, h]
        context_goal = goal_mask_action * context_vec

        # Recurrent context
        if self.is_recurrent:
            r_context_vec, rnn_hxs = self._forward_lstm(context_goal, rnn_hxs, masks)
        else:
            r_context_vec = context_goal

        # Combine object representations with global representations
        r_object_vec = torch.cat(
            [
                goal_mask_object[:, None, :] * features_obj,
                r_context_vec.unsqueeze(1).repeat(1, features_obj.shape[1], 1),
            ],
            2,
        )
        r_object_vec_comb = self.object_context_combine(r_object_vec)

        if torch.isnan(r_context_vec).any() or torch.isnan(r_object_vec_comb).any():
            pdb.set_trace()

        return r_context_vec, r_object_vec_comb, rnn_hxs


class GNNBase(nn.Module):
    def __init__(
        self, hidden_size=128, max_nodes=150, num_rels=5, num_classes=100, num_states=4
    ):
        super(GNNBase, self).__init__()
        self.hidden_size = hidden_size
        self.graph_encoder = GraphModelGGNN(
            num_classes=num_classes,
            num_nodes=max_nodes,
            h_dim=hidden_size,
            out_dim=hidden_size,
            num_rels=num_rels,
            num_states=num_states,
        )
        self.object_class_encoding = self.graph_encoder.class_encoding

    def forward(self, inputs):
        # Build the graph

        # ipdb.set_trace()

        hidden_feats = self.graph_encoder(inputs)
        return hidden_feats

class GNNBase2(nn.Module):
    def __init__(
        self, hidden_size=128, max_nodes=150, num_rels=5, num_classes=100, num_states=4, num_prop=3
    ):
        super(GNNBase2, self).__init__()
        self.hidden_size = hidden_size
        self.graph_encoder = GraphModelIN(
            num_nodes=max_nodes,
            h_dim=hidden_size,
            
        )
        # self.object_class_encoding = self.graph_encoder.class_encoding
        self.single_object_encoding = ObjNameCoordStateEncode(
            output_dim=hidden_size, num_classes=num_classes, num_states=num_states
        )
        self.num_prop = num_prop

    def forward(self, inputs):
        # Build the graph

        # ipdb.set_trace()
        # print('1')
        edges = inputs['edge_tuples']
        mask_edges = inputs['mask_edge']
        input_node_embedding = self.single_object_encoding(
            inputs['class_objects'].long(),
            inputs['object_coords'],
            inputs['states_objects'],
        )

        ne = edges.shape[2]
        b, t, n, h = input_node_embedding.shape
        input_node_embedding = input_node_embedding.reshape([b*t, n, h])
        edges = edges.reshape([b*t, ne, 2])


        mask_edges = mask_edges.reshape([b*t, ne])
        node_embeddings = torch.zeros_like(input_node_embedding)

        # Flatten time and batch
        # ind1 = np.repeat(np.arange(b*t), ne)
        # ne_ind = np.tile(np.arange(ne), b*t)
        # ind_from = edges[..., 0].reshape(-1)
        # ind_to = edges[..., 1].reshape(-1)

        # indices_from = torch.LongTensor([ind1, ne_ind, ind_from]).to(mask_edges.device)
        # indices_to = torch.LongTensor([ind1, ne_ind, ind_to]).to(mask_edges.device)

        inputs['from_indices_onehot'][..., 0] = inputs['from_indices_onehot'][..., 0] - inputs['from_indices_onehot'][0,0,0,0]
        inputs['to_indices_onehot'][..., 0] = inputs['to_indices_onehot'][..., 0] - inputs['to_indices_onehot'][0,0,0,0]
        
        indices_from = inputs['from_indices_onehot'].reshape(-1, 3).transpose(0,1)
        indices_to = inputs['to_indices_onehot'].reshape(-1, 3).transpose(0,1)

        values = mask_edges.reshape(-1)
        # ipdb.set_trace()
        # print('5')
        # THE STUFF ABOVE SHOULD GO IN THE DATALOADER   
        edge_from = torch.sparse.FloatTensor(indices_from, values, (b*t, ne, n))
        edge_to = torch.sparse.FloatTensor(indices_to, values, (b*t, ne, n))
        edges = [edge_from, edge_to, mask_edges]

        # print('6')

        for i in range(self.num_prop):
            node_embeddings = self.graph_encoder(node_embeddings, input_node_embedding, edges)
        # print('osut-')
        node_embeddings = node_embeddings.reshape([b, t, n, h])
        # ipdb.set_trace()
        return node_embeddings

class GraphModelIN(nn.Module):
    def __init__(self, num_nodes, h_dim):
        super(GraphModelIN, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.particle_prop = nn.Sequential(nn.Linear(self.h_dim * 2, self.h_dim), nn.ReLU())
        self.edge_build = nn.Sequential(nn.Linear(self.h_dim * 2, self.h_dim), nn.ReLU())
                

    def forward(self, node_embeddings, input_node_embedding, edges):
        # edge propagate
        edge_from, edge_to, edge_mask = edges
        # ipdb.set_trace()
        from_info = edge_from.bmm(node_embeddings)
        to_info = edge_to.bmm(node_embeddings)
        edge_info = self.edge_build(torch.cat([to_info, from_info], 2))

        # We aggregate the incoming edges
        edge_to_t = edge_to.transpose(1,2)

        node_info = edge_to_t.bmm(edge_info)
        embeddings_out = self.particle_prop(torch.cat([node_embeddings, input_node_embedding], 2))

        return embeddings_out

class TransformerBase(nn.Module):
    def __init__(self, hidden_size=128, max_nodes=150, num_classes=100, num_states=4):
        super(TransformerBase, self).__init__()

        self.main = Transformer(
            num_classes=num_classes,
            num_nodes=max_nodes,
            in_feat=hidden_size,
            out_feat=hidden_size,
        )
        # self.single_object_encoding = ObjNameCoordEncode(output_dim=hidden_size, num_classes=num_classes)
        self.single_object_encoding = ObjNameCoordStateEncode(
            output_dim=hidden_size, num_classes=num_classes, num_states=num_states
        )
        self.object_class_encoding = self.single_object_encoding.class_embedding
        self.train()

    def forward(self, inputs):
        # Use transformer to get feats for every object
        mask_visible = inputs['mask_object']
        input_node_embedding = self.single_object_encoding(
            inputs['class_objects'].long(),
            inputs['object_coords'],
            inputs['states_objects'],
            )
        should_reshape = False
        if input_node_embedding.ndim > 3:
            should_reshape = True
            dims = list(input_node_embedding.shape)
            input_node_embedding = input_node_embedding.reshape([-1] + dims[-2:])
            mask_visible = mask_visible.reshape([-1] + dims[-2:-1])

        node_embedding = self.main(input_node_embedding, mask_visible)
        if should_reshape:
            new_dims = list(node_embedding.shape)
            node_embedding = input_node_embedding.reshape(dims[:-1] + new_dims[-1:])
        return node_embedding


class ObjNameCoordStateEncode(nn.Module):
    def __init__(self, output_dim=128, num_classes=50, num_states=4):
        super(ObjNameCoordStateEncode, self).__init__()
        assert output_dim % 2 == 0
        self.output_dim = output_dim
        self.num_classes = num_classes

        self.class_embedding = nn.Embedding(num_classes, int(output_dim / 2))
        self.state_embedding = nn.Linear(num_states, int(output_dim / 2))
        self.coord_embedding = nn.Sequential(
            nn.Linear(6, int(output_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(output_dim / 2), int(output_dim / 2)),
        )
        inp_dim = int(output_dim + output_dim / 2)
        self.combine = nn.Sequential(nn.ReLU(), nn.Linear(inp_dim, output_dim))

    def forward(self, class_ids, coords, state):
        # print(self.class_embedding, class_ids.max())

        state_embedding = self.state_embedding(state)
        class_embedding = self.class_embedding(class_ids)
        coord_embedding = self.coord_embedding(coords)
        inp = torch.cat([class_embedding, coord_embedding, state_embedding], dim=-1)

        return self.combine(inp)


class ObjNameCoordEncode(nn.Module):
    def __init__(self, output_dim=128, num_classes=50):
        super(ObjNameCoordEncode, self).__init__()
        self.output_dim = output_dim
        self.num_classes = num_classes

        self.class_embedding = nn.Embedding(num_classes, int(output_dim / 2))
        self.coord_embedding = nn.Sequential(
            nn.Linear(3, int(output_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(output_dim / 2), int(output_dim / 2)),
        )

    def forward(self, class_ids, coords):
        class_embedding = self.class_embedding(class_ids)
        coord_embedding = self.coord_embedding(coords)
        return torch.cat([class_embedding, coord_embedding], dim=2)
