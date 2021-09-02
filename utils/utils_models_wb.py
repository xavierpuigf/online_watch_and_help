import glob

from omegaconf import DictConfig, OmegaConf
import yaml
import os
import random
import wandb
import tensorflow as tf
import tensorflow as tf
from hydra.utils import get_original_cwd, to_absolute_path
import tensorboard as tb

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
from torch.utils.tensorboard import SummaryWriter
import datetime
import numpy as np
import torch
import wandb
import json
import ipdb
import pdb
import torch.nn as nn
import matplotlib.pyplot as plt
from .utils_plot import Plotter

plt.switch_backend('agg')

# Nice vectorized sampling function 
# https://stackoverflow.com/questions/34187130/fast-random-weighted-selection-across-all-rows-of-a-stochastic-matrix/34190035
def vectorized(prob_matrix):
    s = prob_matrix.cumsum(axis=-1)
    r = np.random.rand(*prob_matrix.shape[:-1])[..., None]
    # ipdb.set_trace()
    k = (s < r).sum(axis=-1)
    if k.max() > prob_matrix.shape[-1]:
        ipdb.set_trace()
    k[k == prob_matrix.shape[-1]] = prob_matrix.shape[-1] - 1
    return k


def obtain_graph(
    graph_helper,
    graph,
    edge_prob,
    state_prob,
    mask_edge,
    changed_edges,
    batch_item,
    len_mask,
    changed_nodes=None,
    samples=None
):
    
    all_samples = []
    changed_edges_new = [changed_edges[0], changed_edges[1]]
    do_sample = True

    if samples is None:
        samples = 1
        do_sample = False
        # edge_prob = edge_prob.cpu().numpy(
    else:
        pass
        # edge_prob = nn.functional.softmax(edge_prob, dim=-1).cpu().numpy()
    
    prev_step_edges = changed_edges_new[1].argmax(-1)
    for sample in range(samples):
        # Sample edge_prob
        if do_sample:
            edge_pred = vectorized(edge_prob)
        else:
            edge_pred = edge_prob.argmax(-1)
        if len(changed_edges) > 0:
            # Sample changed edges
            if do_sample:
                changed_edges_new[0] = vectorized(changed_edges[0])
            else:
                changed_edges_new[0] = changed_edges[0].argmax(-1)
            # The changed edges should be boolean at this point
            # ipdb.set_trace()
            try:
                assert (changed_edges_new[0] == 1).sum() == (changed_edges_new[0] != 0).sum()
            except:
                ipdb.set_trace()
            # ipdb.set_trace()
            try:
                edge_pred = (
                    changed_edges_new[0][...] * edge_pred + (1 - changed_edges_new[0][...]) * prev_step_edges
                )
            except:
                ipdb.set_trace()


        # We are predicting the next graph, so we sum
        num_tsteps = int(len_mask[batch_item].sum()) - 1
        offset = 0
        nedges = len(graph_helper.relation_dict)
        state_names = [(graph_helper.states[it],) for it in range(4)]
        edge_names = [graph_helper.relation_dict.get_el(it) for it in range(nedges)]
        info = {'results': [], 'state_names': state_names, 'edge_names': edge_names}
        all_edges, all_from, all_to, all_edges_input = [], [], [], []
        object_states = state_prob[batch_item, :num_tsteps].numpy()
        for step in range(num_tsteps):
            result = {}
            mask_object = int(graph['mask_object'][batch_item, step + offset].sum())
            object_names = graph['class_objects'][batch_item, step + offset]
            num_nodes = graph['mask_object'].shape[-1]

            node_ids = graph['node_ids'][batch_item, step + offset]

            print_node = False
            obj_names = []
            for nid in range(mask_object):

                class_name = graph_helper.object_dict.get_el(int(object_names[nid]))
                idi = int(node_ids[nid])
                obj_name_complete = f"{class_name}.{idi}"
                obj_names.append(obj_name_complete)

            current_mask_edge = mask_edge[batch_item, step]
            # current_edge = edge_info[batch_item, step]
            indices_valid = np.where(current_mask_edge == 1)[0]

            edge_pred_step = edge_pred[batch_item, step + offset, indices_valid]
            edge_input_step = prev_step_edges[batch_item, step + offset, indices_valid]
            # edge_probs = edge_prob[batch_item, step + offset, indices_valid]

            from_id = indices_valid // num_nodes
            to_id = indices_valid % num_nodes

            curr_res = {}
            # all_edges.append(edge_prob[None, :].numpy())
            all_edges.append(edge_pred_step[None, :])
            all_edges_input.append(edge_input_step[None, :])
            all_from.append(from_id[None, :])
            all_to.append(to_id[None, :])

        all_edges = np.concatenate(all_edges, 0)
        all_edges_input = np.concatenate(all_edges_input, 0)
        all_from = np.concatenate(all_from, 0)
        all_to = np.concatenate(all_to, 0)
        info['edge_pred'] = all_edges
        info['edge_input'] = all_edges_input
        info['from_id'] = all_from
        info['to_id'] = all_to
        info['states'] = object_states

        info['nodes'] = obj_names
        # ipdb.set_trace()
        if len(changed_edges) > 0:
            info['changed_edges'] = changed_edges[0]
        all_samples.append(info)
    return all_samples


def print_graph_2(
    graph_helper,
    graph,
    edge_info,
    mask_edge,
    state_info,
    changed_edges,
    batch_item,
    step,
    changed_nodes=None,
):

    # If we are only predicitng edge change, the edge is a combination of previous edge and new, modulagted by prediction
    if len(changed_edges) > 0:

        if changed_edges[0].shape[-1] != changed_edges[1].shape[-1]:
            # Changes as nodes
            num_nodes = changed_edges[0].shape[-1]
            changed_edges_build = changed_edges[0].repeat_interleave(num_nodes, dim=2)
            changed_edges = [changed_edges_build, changed_edges[1]]


        # The changed edges should be boolean at this point
        # ipdb.set_trace()
        assert (changed_edges[0] == 1).sum() == (changed_edges[0] != 0).sum()

        # ipdb.set_trace()
        try:
            edge_info = (
                changed_edges[0] * edge_info + (1 - changed_edges[0]) * changed_edges[1]
            )
        except:
            ipdb.set_trace()
    # We are predicting the next graph, so we sum 1
    offset = 1
    mask_object = int(graph['mask_object'][batch_item, step + offset].sum())
    object_names = graph['class_objects'][batch_item, step + offset]
    object_states = state_info[batch_item, step]
    num_nodes = graph['mask_object'].shape[-1]

    # object_coords = graph['object_coords'][batch_item, step+offset]

    # ipdb.set_trace()
    node_ids = graph['node_ids'][batch_item, step + offset]

    # ipdb.set_trace()
    print_node = False
    print("Graph")
    print("==========")
    if print_node:
        print("Nodes:")
    obj_names = []
    for nid in range(mask_object):

        state_names = [
            graph_helper.states[it]
            for it in range(4)
            if int(object_states[nid][it]) == 1
        ]
        state_names = ' '.join(state_names)
        class_name = graph_helper.object_dict.get_el(int(object_names[nid]))
        idi = int(node_ids[nid])
        # coords = list(object_coords[nid][:3])
        # coords_str = '{:.2f}, {:.2f}, {:.2f}'.format(coords[0], coords[1], coords[2])
        obj_name_complete = f"{class_name}.{idi}"
        obj_name_complete += ' ' * (20 - len(obj_name_complete))
        obj_names.append(obj_name_complete)
        if print_node:
            print(f"{obj_name_complete}. {state_names}")

    if print_node:
        print('\n')

    print('Edges')
    current_mask_edge = mask_edge[batch_item, step]
    current_edge = edge_info[batch_item, step]
    # Only store on and inside edges, and hold
    inside_of = {}
    id_inside = graph_helper.relation_dict.get_id('inside')
    id_on = graph_helper.relation_dict.get_id('on')
    id_hold = graph_helper.relation_dict.get_id('hold')

    inside = np.where(
        np.logical_and(current_edge == id_inside, current_mask_edge == 1)
    )[0]
    on = np.where(np.logical_and(current_edge == id_on, current_mask_edge == 1))[0]
    hold = np.where(np.logical_and(current_edge == id_hold, current_mask_edge == 1))[0]

    inside_from, inside_to = (inside // num_nodes), inside % num_nodes
    on_from, on_to = (on // num_nodes), on % num_nodes
    hold_from, hold_to = (hold // num_nodes), hold % num_nodes

    on = {}
    inside_of = {}
    for elem_from, elem_to in zip(inside_from.tolist(), inside_to.tolist()):
        if int(elem_to) not in inside_of:
            inside_of[int(elem_to)] = []
        inside_of[int(elem_to)].append(int(elem_from))

    for elem_from, elem_to in zip(on_from.tolist(), on_to.tolist()):
        if int(elem_to) not in on:
            on[int(elem_to)] = []
        on[int(elem_to)].append(int(elem_from))

    all_elems = sorted(list(set(list(on.keys()) + list(inside_of.keys()))))

    print("HOLDING:", list(zip(hold_from, hold_to)))
    for elem in all_elems:
        inside_curr, on_curr = [], []
        if elem in inside_of:
            inside_curr = inside_of[elem]
        if elem in on:
            on_curr = on[elem]
        # ipdb.set_trace()
        on_str = ' '.join([obj_names[itt].strip() for itt in on_curr])
        inside_str = ' '.join([obj_names[itt].strip() for itt in inside_curr])
        elem2 = obj_names[elem]
        print(f'{elem2}: ON: [{on_str}]   INSIDE: [{inside_str}]')
    print("==========")


def print_graph(graph_helper, graph, batch_item, step):
    mask_object = int(graph['mask_object'][batch_item, step].sum())
    object_names = graph['class_objects'][batch_item, step]
    object_states = graph['states_objects'][batch_item, step]
    object_coords = graph['object_coords'][batch_item, step]
    # ipdb.set_trace()
    node_ids = graph['node_ids'][batch_item, step]
    # ipdb.set_trace()
    print("Graph")
    print("==========")
    print("Nodes:")
    obj_names = []
    for nid in range(mask_object):

        state_names = [
            graph_helper.states[it]
            for it in range(4)
            if int(object_states[nid][it]) == 1
        ]
        state_names = ' '.join(state_names)
        class_name = graph_helper.object_dict.get_el(int(object_names[nid]))
        idi = int(node_ids[nid])
        coords = list(object_coords[nid][:3])
        coords_str = '{:.2f}, {:.2f}, {:.2f}'.format(coords[0], coords[1], coords[2])
        obj_name_complete = f"{class_name}.{idi}"
        obj_name_complete += ' ' * (20 - len(obj_name_complete))
        obj_names.append(obj_name_complete)

        print(f"{obj_name_complete} ({coords_str}). {state_names}")
    print('\nEdges')
    # Only store on and inside edges
    inside_of = {}
    id_inside = graph_helper.relation_dict.get_id('inside')
    id_on = graph_helper.relation_dict.get_id('on')
    inside = graph['edge_tuples'][batch_item, step][
        graph['edge_classes'][batch_item, step] == id_inside
    ]
    inside_from, inside_to = inside[:, 0], inside[:, 1]
    on = graph['edge_tuples'][batch_item, step][
        graph['edge_classes'][batch_item, step] == id_on
    ]
    on_from, on_to = on[:, 0], on[:, 1]

    on = {}
    inside_of = {}
    for elem_from, elem_to in zip(inside_from.tolist(), inside_to.tolist()):
        if int(elem_to) not in inside_of:
            inside_of[int(elem_to)] = []
        inside_of[int(elem_to)].append(int(elem_from))

    for elem_from, elem_to in zip(on_from.tolist(), on_to.tolist()):
        if int(elem_to) not in on:
            on[int(elem_to)] = []
        on[int(elem_to)].append(int(elem_from))

    all_elems = sorted(list(set(list(on.keys()) + list(inside_of.keys()))))
    for elem in all_elems:
        inside_curr, on_curr = [], []
        if elem in inside_of:
            inside_curr = inside_of[elem]
        if elem in on:
            on_curr = on[elem]
        # ipdb.set_trace()
        on_str = ' '.join([obj_names[itt].strip() for itt in on_curr])
        inside_str = ' '.join([obj_names[itt].strip() for itt in inside_curr])
        elem2 = obj_names[elem]
        print(f'{elem2}: ON: [{on_str}]   INSIDE: [{inside_str}]')
    print("==========")


def print_script(graph_helper, program):
    program_str = decode_program(graph_helper, program)
    ipdb.set_trace()


# Get a render function
def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None


def get_pred_results_str(graph_helper, prog_gt, prog_pred):
    program_gt = decode_program(graph_helper, prog_gt)
    program_pred = decode_program(graph_helper, prog_pred)
    res_str = ''
    res_str += "GT{}|Pred\n".format(' ' * 43)
    if len(program_pred) < len(program_gt):
        program_pred += [' '] * (len(program_gt) - len(program_pred))
    else:
        program_pred[: len(program_gt)]

    for instr_gt, instr_pred in zip(program_gt, program_pred):
        res_str += '{: <45}| {}\n'.format(instr_gt, instr_pred)

    res_str += '{}\n'.format('-' * 45)
    return res_str


def decode_program(graph_helper, program_info, index=0):
    # ipdb.set_trace()
    program_info_new = {}

    program_info['class_objects'] = program_info['graph']['class_objects']
    program_info['node_ids'] = program_info['graph']['node_ids']
    for key, val in program_info.items():
        if key != 'graph':
            program_info_new[key] = val[index, :].cpu().numpy()

    length = int(program_info_new['mask_len'].sum())

    action_ind = list(program_info_new['action'][:length])
    o1_ind = list(program_info_new['o1'][:length])
    o2_ind = list(program_info_new['o2'][:length])

    class_obj = list(program_info_new['class_objects'][index])
    node_ids = list(program_info_new['node_ids'][index])

    program_str = []
    for it in range(length):
        if action_ind[it] == len(graph_helper.action_dict):
            action_str = '[None]'
        else:
            action_str = '[{}]'.format(graph_helper.action_dict.get_el(action_ind[it]))
        o1_id = int(node_ids[o1_ind[it]])
        o2_id = int(node_ids[o2_ind[it]])
        # ipdb.set_trace()
        if o1_id != -1:
            action_str += ' <{}> ({})'.format(
                graph_helper.object_dict.get_el(int(class_obj[o1_ind[it]])), o1_id
            )
        if o2_id != -1:
            action_str += ' <{}> ({})'.format(
                graph_helper.object_dict.get_el(int(class_obj[o2_ind[it]])), o2_id
            )

        program_str.append(action_str)
    return program_str


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, do_print=True):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]

        if do_print:
            print('\t'.join(entries))
        else:
            return '\t'.join(entries)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)


def get_epsilon(init_eps, end_eps, num_steps, episode):
    return (init_eps - end_eps) * (
        1.0 - min(1.0, float(episode) / float(num_steps))
    ) + end_eps


class AggregatedStats:
    def __init__(self):
        self.success_per_apt = {}
        self.success_per_goal = {}

    def reset(self):
        success_per_apt = {}
        success_per_goal = {}

    def add(self, success, goal, apt):
        if goal not in self.success_per_goal:
            sg_count, g_count = 0, 0
        else:
            sg_count, g_count = self.success_per_goal[goal]
        if apt not in self.success_per_apt:
            sa_count, a_count = 0, 0
        else:
            sa_count, a_count = self.success_per_apt[apt]
        r = 1 if success else 0
        self.success_per_goal[goal] = [sg_count + r, g_count + 1]
        self.success_per_apt[apt] = [sa_count + r, a_count + 1]

    def add_list(self, info_list):
        for item_info in info_list:
            self.add(item_info['success'], item_info['goal'], item_info['apt'])

    def barplot(self, values, names=None, special_ids=None):
        fig, ax = plt.subplots()
        index = range(values.shape[0])

        a1 = plt.bar(index, values)

        if names is not None:
            ax.set_xticks(np.asarray([i for i in range(len(names))]))
            ax.set_xticklabels(names, rotation=65)
        if special_ids is not None:
            a1[special_ids].set_color('r')
        plt.tight_layout()
        return fig

    def create_histogram(self, cont_dict):
        keys = sorted(list(cont_dict.keys()))
        values = [cont_dict[x] for x in keys]
        success = np.array([x[0] for x in values])
        cont = np.array([x[1] for x in values])
        return self.barplot(success, keys), self.barplot(cont, keys)

    def print_hist(self, tb_writer):
        img_goal, cnt_goal = self.create_histogram(self.success_per_goal)
        img_apt, cnt_apt = self.create_histogram(self.success_per_apt)

        tb_writer.add_figure("histogram/success_per_goal", img_goal)
        tb_writer.add_figure("histogram/success_per_apt", img_apt)
        tb_writer.add_figure("histogram/count_per_goal", cnt_goal)
        tb_writer.add_figure("histogram/count_per_apt", cnt_apt)


def dict2md(content, tabs=0):
    mdown_str = ''
    for key, val in content.items():
        if type(val) == dict:
            actual_str = '  <br />' + dict2md(val, tabs=tabs + 2)
        else:
            actual_str = str(val)

        mdown_str += '{}**{}**: {}  <br />'.format('..' * tabs, key, actual_str)
    # print(tabs, mdown_str)
    return mdown_str


class LoggerSteps:
    def __init__(self, args, log_steps=True):
        self.args = args
        self.log_steps = log_steps
        self.experiment_name = self.get_experiment_name()
        self.wandb = None
        self.save_dir = os.path.dirname(get_original_cwd())

        self.name_log = None if len(args.name_log) == 0 else args.name_log+str(random.randint(0,100))

        self.ckpt_save_dir = os.path.join(self.save_dir, 'ckpts', self.experiment_name)
        self.results_path = os.path.join(self.save_dir, 'results', self.experiment_name)
        self.logs_logdir = os.path.join(
            self.save_dir, 'logs_model', self.experiment_name
        )

        now = datetime.datetime.now()
        self.tstmp = now.strftime('%Y-%m-%d_%H-%M-%S')
        self.set_tensorboard()
        self.first_log = False
        # self.stats = AggregatedStats()

        # self.plot = Plotter2(self.experiment_name, root_dir=self.logs_logdir)
        self.info_episodes = []
        self.file_name_log = '{}/{}/log.json'.format(
            self.logs_logdir, self.experiment_name
        )
        print("Saving to: {}".format(self.experiment_name))

        # f.writelines(json.dumps(dict_args, indent=4))

    def set_tensorboard(self):
        if self.log_steps:
            self.wandb = wandb.init(
                project="graph-prediction",
                name=self.name_log,
                entity='virtualhome',
                config=OmegaConf.to_container(self.args),
            )

    def get_experiment_name(self):
        args = self.args
        pred_change = 'none'

        if args['model']['predict_edge_change']:
            pred_change = 'edge'
        if args['model']['predict_node_change']:
            pred_change = 'node'
        experiment_name = (
            'predict_graph/train_data.{}-agents{}/'
            'time_model.{}-stateenc.{}-globalrepr.{}-edgepred.{}-lr{}-bs.{}-'
            'goalenc.{}_extended._costclose.{}_costgoal.{}_agentembed.{}_predchange.{}_inputgoal.{}_excledge.{}'
        ).format(
            args['data']['train_data'],
            args['train']['agents'],
            args['model']['time_aggregate'],
            args['model']['state_encoder'],
            args['model']['global_repr'],
            args['model']['edge_pred'],
            args['train']['lr'],
            args['train']['batch_size'],
            args['model']['goal_inp'],
            args['train']['loss_close'],
            args['train']['loss_goal'],
            args['model']['agent_embed'],
            pred_change,
            args['model']['input_goal'],
            args['model']['exclusive_edge']
        )
        if args['model']['gated']:
            experiment_name += '_gated'

        if 'exp_name' in self.args and self.args.exp_name != '':
            experiment_name += '_{}'.format(args['exp_name'])
        if 'debug' in args:
            experiment_name += 'debug'
        if args['train']['overfit']:
            experiment_name += 'overfit'
        return experiment_name

    def log_embeds(self, total_num_steps, embed_info):
        embeddings = embed_info.weight
        embedding_labels = []
        for i in range(embeddings.shape[0]):
            agentn = int(i / 5)
            seedn = i % 5
            embedding_labels.append("agent.{}_seed.{}".format(agentn, seedn))

        self.wandb.add_embedding(embeddings, metadata=embedding_labels)

    def log_data2(self, total_num_steps, info):
        if self.first_log:
            self.first_log = False
            if self.tensorboard_logdir is not None:
                self.set_tensorboard()

        if self.wandb is not None:
            for agent_id in info.keys():
                for acc_name, acc_item in info[agent_id]['accuracy'].items():
                    self.wandb.log(
                        "agents/accuracy/agent_{}/{}".format(agent_id, acc_name),
                        acc_item,
                        total_num_steps,
                    )

    def log_data(self, total_num_steps, info):
        if self.first_log:
            self.first_log = False
            if self.tensorboard_logdir is not None:
                self.set_tensorboard()

        res_dict = {}
        res_dict['total_num_steps'] = total_num_steps
        res_dict['epoch'] = info['misc']['epoch']

        if self.wandb is not None:
            for loss_name, loss_item in info['losses'].items():
                res_dict.update({"losses/{}".format(loss_name): loss_item})

            for acc_name, acc_item in info['accuracy'].items():

                res_dict.update({"accuracy/{}".format(acc_name): acc_item})

            if 'misc' in info.keys():
                for acc_name, acc_item in info['misc'].items():
                    res_dict.update({"misc/{}".format(acc_name): acc_item})
            # ipdb.set_trace()
            self.wandb.log(res_dict)

            # if 'plots' in info.keys():
            #     info_plot = info['plots']
            #     fig, ax = plt.subplots(4,2)
            #     bs = len(info_plot['gt_belief_room'])
            #     for i in range(min(4, bs)):
            #         it = 0
            #         for curr_str in ['gt_belief_room', 'gt_belief_container']:
            #             try:
            #                 names = info_plot[curr_str.replace('gt', 'names')][i]
            #             except:
            #                 ipdb.set_trace()
            #             x = np.arange(len(info_plot[curr_str][i]))
            #             ax[i,it].bar(x-0.15, info_plot[curr_str][i], width=0.3)
            #             ax[i,it].bar(x+0.15, info_plot[curr_str.replace('gt', 'pred')][i], width=0.3)
            #             ax[i, it].set_xticks(range(len(names)))
            #             ax[i, it].set_xticklabels(names)
            #             ax[i, it].set_ylim((0,1))
            #             ax[i, it].tick_params(axis='both', which='major', labelsize=8)
            #             ax[i, it].tick_params(axis='both', which='minor', labelsize=8)
            #             ax[i, it].grid(axis='y')
            #             it += 1
            #     self.wandb.add_figure(info['plots']['name'], fig, total_num_steps)

    def log_info(self, info_ep):
        try:
            os.makedirs(self.logs_logdir)

        except:
            pass

        log_txt_file = '{}/logs.txt'.format(self.logs_logdir)
        with open(log_txt_file, 'a+') as f:
            f.write(info_ep['str'])

    def save_model(self, j, model, optimizer):

        save_path = os.path.join(self.ckpt_save_dir)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.wandb.log({'misc/ckpt': wandb.Html('<a>{}</a>'.format(str(save_path)))})
        with open('{}/config.yaml'.format(self.ckpt_save_dir), 'w+') as f:
            f.write(OmegaConf.to_yaml(self.args))
        # ipdb.set_trace()
        torch.save(
            {'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
            os.path.join(save_path, "{}.pt".format(j)),
        )
