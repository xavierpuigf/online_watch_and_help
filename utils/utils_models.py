import glob

from omegaconf import DictConfig, OmegaConf
import yaml
import os
import wandb
import tensorflow as tf
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
from torch.utils.tensorboard import SummaryWriter
import datetime
import numpy as np
import torch
import json
import ipdb
import pdb
import torch.nn as nn
import matplotlib.pyplot as plt
from .utils_plot import Plotter
plt.switch_backend('agg')

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

        state_names = [graph_helper.states[it] for it in range(4) if int(object_states[nid][it]) == 1]
        state_names = ' '.join(state_names)
        class_name = graph_helper.object_dict.get_el(int(object_names[nid]))
        idi = int(node_ids[nid])
        coords = list(object_coords[nid][:3])      
        coords_str = '{:.2f}, {:.2f}, {:.2f}'.format(coords[0], coords[1], coords[2])
        obj_name_complete = f"{class_name}.{idi}"
        obj_name_complete += ' '*(20 - len(obj_name_complete))
        obj_names.append(obj_name_complete)


        print(f"{obj_name_complete} ({coords_str}). {state_names}")
    print('\nEdges')
    # Only store on and inside edges
    inside_of = {}
    id_inside = graph_helper.relation_dict.get_id('inside')
    id_on = graph_helper.relation_dict.get_id('on')
    inside = graph['edge_tuples'][batch_item, step][graph['edge_classes'][batch_item, step] == id_inside]
    inside_from, inside_to = inside[:, 0], inside[:, 1]
    on = graph['edge_tuples'][batch_item, step][graph['edge_classes'][batch_item, step] == id_on]
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
    res_str += "GT{}|Pred\n".format(' '*43)
    if len(program_pred) < len(program_gt):
        program_pred += [' ']*(len(program_gt) - len(program_pred))
    else:
        program_pred[:len(program_gt)]
    
    for instr_gt, instr_pred in zip(program_gt, program_pred):
        res_str += '{: <45}| {}\n'.format(instr_gt, instr_pred)

    res_str += '{}\n'.format('-'*45)
    return res_str

def decode_program(graph_helper, program_info):
    # ipdb.set_trace()
    program_info_new = {}
    
    program_info['class_objects'] = program_info['graph']['class_objects']
    program_info['node_ids'] = program_info['graph']['node_ids']
    for key, val in program_info.items():
        if key != 'graph':
            program_info_new[key] = val[0, :].cpu().numpy()
    
    length = int(program_info_new['mask_len'].sum())

    action_ind = list(program_info_new['action'][:length])
    o1_ind = list(program_info_new['o1'][:length])
    o2_ind = list(program_info_new['o2'][:length])

    class_obj = list(program_info_new['class_objects'][0])
    node_ids = list(program_info_new['node_ids'][0])

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
            action_str += ' <{}> ({})'.format(graph_helper.object_dict.get_el(int(class_obj[o1_ind[it]])), o1_id)
        if o2_id != -1:
            action_str += ' <{}> ({})'.format(graph_helper.object_dict.get_el(int(class_obj[o2_ind[it]])), o2_id)
        
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
    return (init_eps - end_eps) * (1.0 - min(1.0, float(episode) / float(num_steps))) + end_eps

class AggregatedStats():
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
            actual_str = '  <br />'+dict2md(val, tabs=tabs+2)
        else:
            actual_str = str(val)

        mdown_str += '{}**{}**: {}  <br />'.format('..'*tabs, key, actual_str)
    # print(tabs, mdown_str)
    return mdown_str

class LoggerSteps():
    def __init__(self, args):
        self.args = args
        self.experiment_name = self.get_experiment_name()
        self.tensorboard_writer = None
        self.save_dir = '.'

        self.ckpt_save_dir = os.path.join(self.save_dir, 'ckpts', self.experiment_name)
        self.tensorboard_logdir = os.path.join(self.save_dir, 'tensorboard', self.experiment_name)
        self.logs_logdir = os.path.join(self.save_dir, 'logs_model', self.experiment_name)


        now = datetime.datetime.now()
        self.tstmp = now.strftime('%Y-%m-%d_%H-%M-%S')
        self.set_tensorboard()
        self.first_log = False
        # self.stats = AggregatedStats()



        # self.plot = Plotter2(self.experiment_name, root_dir=self.logs_logdir)
        self.info_episodes = []


        self.file_name_log = '{}/{}/log.json'.format(self.logs_logdir, self.experiment_name)
        print("Saving to: {}".format(self.experiment_name))

            # f.writelines(json.dumps(dict_args, indent=4))


    def set_tensorboard(self):
        try:
            os.makedirs(self.tensorboard_logdir)
        except:
            pass
        now = datetime.datetime.now()
        self.tensorboard_writer = SummaryWriter(log_dir=self.tensorboard_logdir)
        if self.args['log']['delete_prior_logs']:
            files_rm = glob.glob('{}/*'.format(self.tensorboard_logdir))
            print("Deleting logs")
            for fr in files_rm:
                os.remove(fr)
        print("Logging in: {}".format(self.tensorboard_logdir))
        
        dict_args = self.args
        text_tboard = "\n**experiment_name:** {}  <br />".format(self.experiment_name)
        text_tboard += ""+dict2md(dict_args)+""
        self.tensorboard_writer.add_text("experiment_params", json.dumps(text_tboard, indent=4))

    def get_experiment_name(self):
        args = self.args
        experiment_name = ('predict_graph/train_data.{}-agents{}/reducedgraph.{}-'
                          'time_model.{}-stateenc.{}-edgepred.{}-lr{}-bs.{}-goalenc.{}_extended._costclose.{}_costgoal.{}_agentembed.{}').format(
            args['data']['train_data'],
            args['train']['agents'],
            args['model']['reduced_graph'],
            args['model']['time_aggregate'],
            args['model']['state_encoder'],
            args['model']['edge_pred'],
            args['train']['lr'],
            args['train']['batch_size'],
            args['model']['goal_inp'],
            args['train']['loss_close'],
            args['train']['loss_goal'],
            args['model']['agent_embed']
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

        self.tensorboard_writer.add_embedding(embeddings, metadata=embedding_labels)


    def log_data2(self, total_num_steps, info):
        if self.first_log:
            self.first_log = False
            if self.tensorboard_logdir is not None:
                self.set_tensorboard()

        if self.tensorboard_writer is not None:
            for agent_id in info.keys():
                for acc_name, acc_item in info[agent_id]['accuracy'].items():
                    self.tensorboard_writer.add_scalar("agents/accuracy/agent_{}/{}".format(agent_id, acc_name), acc_item, total_num_steps)
            

    def log_data(self, total_num_steps, info):
        if self.first_log:
            self.first_log = False
            if self.tensorboard_logdir is not None:
                self.set_tensorboard()

        if self.tensorboard_writer is not None:
            for loss_name, loss_item in info['losses'].items():
                self.tensorboard_writer.add_scalar("losses/{}".format(loss_name), loss_item, total_num_steps)
            
            for acc_name, acc_item in info['accuracy'].items():

                self.tensorboard_writer.add_scalar("accuracy/{}".format(acc_name), acc_item, total_num_steps)
            
            if 'misc' in info.keys():
                for acc_name, acc_item in info['misc'].items():
                    self.tensorboard_writer.add_scalar("misc/{}".format(acc_name), acc_item, total_num_steps)

            if 'plots' in info.keys():
                info_plot = info['plots']
                fig, ax = plt.subplots(4,2)
                bs = len(info_plot['gt_belief_room'])
                for i in range(min(4, bs)):
                    it = 0
                    for curr_str in ['gt_belief_room', 'gt_belief_container']:
                        try:
                            names = info_plot[curr_str.replace('gt', 'names')][i]
                        except:
                            ipdb.set_trace()
                        x = np.arange(len(info_plot[curr_str][i]))
                        ax[i,it].bar(x-0.15, info_plot[curr_str][i], width=0.3)
                        ax[i,it].bar(x+0.15, info_plot[curr_str.replace('gt', 'pred')][i], width=0.3)
                        ax[i, it].set_xticks(range(len(names)))
                        ax[i, it].set_xticklabels(names)
                        ax[i, it].set_ylim((0,1))
                        ax[i, it].tick_params(axis='both', which='major', labelsize=8)
                        ax[i, it].tick_params(axis='both', which='minor', labelsize=8)
                        ax[i, it].grid(axis='y')
                        it += 1
                self.tensorboard_writer.add_figure(info['plots']['name'], fig, total_num_steps)





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
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
            with open('{}/config.yaml'.format(self.ckpt_save_dir), 'w+') as f:
                f.write(OmegaConf.to_yaml(self.args))
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            }, os.path.join(save_path, "{}.pt".format(j)))




class Logger():
    def __init__(self, args):
        self.args = args
        self.experiment_name = self.get_experiment_name()
        self.tensorboard_writer = None
        self.save_dir = args.save_dir

        now = datetime.datetime.now()
        self.tstmp = now.strftime('%Y-%m-%d_%H-%M-%S')
        self.set_tensorboard()
        self.first_log = False
        self.stats = AggregatedStats()

        save_path = os.path.join(self.save_dir, self.experiment_name)
        root_dir = None
        if args.use_editor:
            root_dir = '/Users/xavierpuig/Desktop/experiment_viz/'
        self.plot = Plotter(self.experiment_name, root_dir=root_dir)
        self.info_episodes = []
        try:
            os.makedirs(save_path)
        except OSError:
            pass

        self.file_name_log = '{}/{}/log.json'.format(self.save_dir, self.experiment_name)
        with open('{}/{}/args.txt'.format(self.save_dir, self.experiment_name), 'w+') as f:
            dict_args = vars(args)
            f.writelines(json.dumps(dict_args, indent=4))


    def set_tensorboard(self):
        now = datetime.datetime.now()
        self.tensorboard_writer = SummaryWriter(log_dir=os.path.join(self.args.tensorboard_logdir,
                                                                     self.tstmp))
        dict_args = vars(self.args)
        self.tensorboard_writer.add_text("experiment_name", json.dumps(dict_args, indent=4))

    def get_experiment_name(self):
        args = self.args
        info_mcts = 'stepmcts.{}-lep.{}-teleport.{}-beliefgraph-forcepred'.format(args.num_steps_mcts, args.max_episode_length, args.teleport)
        experiment_name = 'env.{}/task.{}-numproc.{}-obstype.{}-sim.{}/taskset.{}/agent.{}_alice.{}/'\
                          'mode.{}-algo.{}-base.{}-gamma.{}-cclose.{}-cgoal.{}-lr{}-bs.{}{}_goodTF/{}'.format(
            args.env_name,
            args.task_type,
            args.num_processes,
            args.obs_type,
            args.simulator_type,
            args.task_set,
            args.agent_type,
            args.use_alice,
            args.train_mode,
            args.algo,
            args.base_net,
            args.gamma,
            args.c_loss_close,
            args.c_loss_goal,
            args.lr,
            args.batch_size,
            '' if len(args.load_model) == 0 else '_finetuned',
            info_mcts)

        if args.debug:
            experiment_name += 'debug'
        return experiment_name

    def log_data(self, j, total_num_steps, fps, episode_rewards, dist_entropy, epsilon, successes, num_steps, info_aux):
        if self.first_log:
            self.first_log = False
            if self.args.tensorboard_logdir is not None:
                self.set_tensorboard()

        if self.tensorboard_writer is not None:
            if 'accuracy_goal' in info_aux.keys():
                self.tensorboard_writer.add_scalar("aux_info/accuracy_goal", np.max(info_aux['accuracy_goal']), total_num_steps)
                self.tensorboard_writer.add_scalar("aux_info/precision_close", np.mean(info_aux['precision_close']), total_num_steps)
                self.tensorboard_writer.add_scalar("aux_info/recall_close", np.mean(info_aux['recall_close']), total_num_steps)

                self.tensorboard_writer.add_scalar("losses/loss_close", np.mean(info_aux['loss_close']), total_num_steps)
                self.tensorboard_writer.add_scalar("losses/loss_goal", np.mean(info_aux['loss_goal']), total_num_steps)
            #
            # self.tensorboard_writer.add_scalar("losses/loss_close", np.mean(info_aux['loss_close']), total_num_steps)
            # self.tensorboard_writer.add_scalar("losses/loss_goal", np.mean(info_aux['loss_goal']), total_num_steps)



            self.tensorboard_writer.add_scalar("info/max_reward", np.max(episode_rewards), total_num_steps)
            self.tensorboard_writer.add_scalar("info/mean_reward", np.mean(episode_rewards), total_num_steps)

            # tensorboard_writer.add_scalar("median_reward", np.median(episode_rewards), total_num_steps)
            # tensorboard_writer.add_scalar("min_reward", np.min(episode_rewards), total_num_steps)
            # tensorboard_writer.add_scalar("max_reward", np.max(episode_rewards), total_num_steps)
            self.tensorboard_writer.add_scalar("action_entropy/action", dist_entropy[0], total_num_steps)
            self.tensorboard_writer.add_scalar("action_entropy/object", dist_entropy[1], total_num_steps)
            # self.tensorboard_writer.add_scalar("losses/value_loss", value_loss, total_num_steps)
            # self.tensorboard_writer.add_scalar("losses/action_loss", action_loss, total_num_steps)
            self.tensorboard_writer.add_scalar("info/epsilon", epsilon, total_num_steps)
            self.tensorboard_writer.add_scalar("info/episode", j, total_num_steps)
            self.tensorboard_writer.add_scalar("info/success", np.mean(successes), total_num_steps)
            self.tensorboard_writer.add_scalar("info/numsteps", np.mean(num_steps), total_num_steps)
            self.tensorboard_writer.add_scalar("info/fps", fps, total_num_steps)

    def log_info(self, info_ep):
        info_ep = {key: val for key, val in info_ep.items() if key not in ['pred_close']}
        self.info_episodes.append(info_ep)
        self.plot.add_episode(info_ep)
        with open(self.file_name_log, 'w+') as f:
            f.write(json.dumps(info_ep, indent=4))
        print("Dumped in {}".format(self.file_name_log))
        self.plot.render()

    def save_model(self, j, actor_critic):
        save_path = os.path.join(self.save_dir, self.experiment_name)

        torch.save([
            actor_critic,
        ], os.path.join(save_path, "{}.pt".format(j)))
