import torch
import time
import os
import glob
import yaml
import pickle as pkl
from tqdm import tqdm
import ipdb
import numpy as np
from dataloader.dataloader_v2 import AgentTypeDataset
from dataloader import dataloader_v2 as dataloader_v2
from arguments import *
from torch import nn
import torch.optim as optim
from models import agent_pref_policy
from hydra.utils import get_original_cwd, to_absolute_path
from termcolor import colored
import utils.utils_models_wb as utils_models
from collections import Counter
import utils.utils_rl_agent as utils_agent
from utils.utils_models_wb import AverageMeter, ProgressMeter, LoggerSteps
import math
import p_tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import pathlib
from functools import partial
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)    

def cleanup():
    dist.destroy_process_group()

def decode_graphs(graph_helper, graph_info, pred_edge, pred_state, pred_change, mask_edges, input_edges, len_mask, index):
    result_graph = utils_models.obtain_graph_3(
        graph_helper, graph_info, predicted_edge, predicted_state, predicted_change, mask_edges, input_edges,
        len_mask, index
    )
    return result_graph



def build_goal_graph(graph_info, len_mask):
    num_nodes = graph_info['mask_object'].shape[2]
    T = graph_info['mask_object'].shape[1]
    object_coords_dim = graph_info['object_coords'].shape[3]
    states_objects_dim = graph_info['states_objects'].shape[3]
    goal_graph = {}

    tsteps = len_mask.sum(-1)[:, None, None].repeat(1, 1, num_nodes).long() - 1
    goal_graph['mask_object'] = (
        torch.gather(graph_info['mask_object'].cuda(), 1, tsteps.cuda())
        .repeat(1, T, 1)
        .cuda()
    )
    goal_graph['class_objects'] = (
        torch.gather(graph_info['class_objects'].cuda(), 1, tsteps.cuda())
        .repeat(1, T, 1)
        .cuda()
    )
    tsteps = (
        len_mask.sum(-1)[:, None, None, None]
        .repeat(1, 1, num_nodes, object_coords_dim)
        .long()
        - 1
    )
    goal_graph['object_coords'] = (
        torch.gather(graph_info['object_coords'].cuda(), 1, tsteps.cuda())
        .repeat(1, T, 1, 1)
        .cuda()
    )
    tsteps = (
        len_mask.sum(-1)[:, None, None, None]
        .repeat(1, 1, num_nodes, states_objects_dim)
        .long()
        - 1
    )
    goal_graph['states_objects'] = (
        torch.gather(graph_info['states_objects'].cuda(), 1, tsteps.cuda())
        .repeat(1, T, 1, 1)
        .cuda()
    )
    return goal_graph

def merge2d(tensor):
    dim = list(tensor.shape)
    return tensor.reshape([-1] + dim[2:])


def unmerge(tensor, firstdim):
    dim = list(tensor.shape)
    return tensor.reshape([firstdim, -1] + dim[1:])



# # Convert adjacency matrix to adjacency list
# def edge_matrix_to_edge_list(edge_matrix):
#     num_nodes_sq = edge_matrix.shape[-1]
#     num_nodes = int(math.sqrt(num_nodes_sq))
#     assert (num_nodes ** 2) == num_nodes_sq
#     edges_something = edges


def compute_forward_pass(args, data_item, data_loader, model, criterions, evaluation=False, posterior=False):
    (
        graph_info,
        program,
        label,
        len_mask,
        goal,
        label_agent,
        real_label_agent,
        ind,
    ) = data_item

    inputs = {
        'program': program,
        'graph': graph_info,
        'mask_len': len_mask,
        'goal': goal,
        'label_agent': label_agent,
    }
    T = graph_info['mask_object'].shape[1]
    object_coords_dim = graph_info['object_coords'].shape[3]
    states_objects_dim = graph_info['states_objects'].shape[3]
    mask_obs_node = graph_info['mask_obs_node']

    goal_graph = build_goal_graph(graph_info, len_mask)
    inputs['goal_graph'] = goal_graph


    ##################
    # Obtain GT
    ##################
    edge_dict = utils_models.build_gt_edge(
        graph_info, data_loader.dataset.graph_helper, 
        exclusive_edge=args.model.exclusive_edge)
    gt_state = goal_graph['states_objects'][:, 1:, ...]
    gt_edge = get_gt_edge(args, goal_graph, edge_dict, len_mask) 

    input_edges = edge_dict['gt_edges'][:, :-1, :]
    edge_interest = edge_dict['edge_interest']

    # Obtain GT Change
    gt_change = (gt_edge != input_edges.cuda()).long().cuda()
    gt_change *= mask_obs_node[:, 1:, ...].long().cuda()

    # inputs['input_edges'] = edge_dict['gt_edges']
    # try:
    print(len_mask.shape)
    if not evaluation:
        output = model(inputs)
        
    else:
        with torch.no_grad():
            output = model(inputs, inference=not posterior)
    # except:
    #     print("ERROR!!!")
    #     ipdb.set_trace()
    ##################
    # Get Predictions
    ##################
    pred_edge = output['edges'][:, :-1, ...]
    b, t, num_nodes, _ = output['states'].shape
    pred_edge = pred_edge.reshape([b, t-1, num_nodes, num_nodes])


    pred_state = output['states'][:, :-1, ...]
    mask_state = graph_info['mask_object'][:, 1:, :, None].cuda()
    mask_length = len_mask[:, 1:].cuda()

    pred_changes = output['node_change'][:, :-1, ...]


    mask_obs_node = graph_info['mask_obs_node']
    mask_edges = mask_obs_node.cuda()
    mask_edges = mask_edges[:, 1:, ...]

    ##################
    # Compute losses
    ##################
    loss = 0
    losses_dict = {}

    # Loss state
    loss_state = criterions['state'](
        pred_state,
        gt_state,
    )
    # ipdb.set_trace()
    loss_state = loss_state * gt_change[..., None] # graph_info['mask_object'][:, 1:, :, None].cuda()
    loss_state = loss_state.mean()
    loss_state *= 0.

    # Loss change
    loss_change = criterions['change'](
        pred_changes.permute(0, 3, 1, 2), gt_change
    )
    loss_change = loss_change * mask_obs_node[:, 1:, :].cuda()
    loss_change = loss_change.mean()

    losses_dict['losses_change'] = loss_change.item()
    loss += loss_change


    # loss edges edges in prediction are stored as a B x Time x N x N x num_edge_class tensor
    # GT is stored as B x Time x Num_edges, we need to convert
    # ipdb.set_trace()
    loss_edges = criterions['edge'](pred_edge.permute(0, 3, 1, 2), gt_edge)
    loss_edges = loss_edges * gt_change
    loss_edges = loss_edges.mean()
    loss += loss_edges + loss_state


    losses_dict.update({
        'losses': loss.item(),
        'losses_state': loss_state.item(),
        'losses_edge': loss_edges.item()
    })

    if 'VAE' in args.model.time_aggregate:
        loss_kl = compute_kl_loss(output, len_mask)
        losses_dict['kldiv'] = loss_kl.item()
        loss += loss_kl

    gt = {
        'gt_state': gt_state,
        'gt_edge': gt_edge,
        'gt_change': gt_change
    }

    predictions = {
        'pred_state': pred_state,
        'pred_edge': pred_edge,
        'pred_change': pred_changes
    }

    misc = {
        'input_edges': input_edges,
        'id_nothing': edge_dict['id_nothing'].cuda(),
        'mask_nodes': mask_state,
        'mask_length': mask_length,
        'edge_interest': edge_dict['edge_interest']
    }
    return gt, predictions, misc, losses_dict, inputs, loss


def plot_func(html_name, graph_helper, graph_info, len_mask, index, input_edges, pred_edge, pred_change, pred_state,
              gt_edge, gt_change, gt_state, metrics_item, metrics_item_tstep, prog_gt, mask_edges_orig):
    
    # ipdb.set_trace()
    program_gt = utils_models.decode_program(graph_helper, prog_gt, index=index)
    pred_graph_samples = []
    num_samples = len(pred_change)

    # Obtain GT Graph
    gt_graph = utils_models.obtain_graph_3(
        graph_helper,
        graph_info,
        gt_edge,
        gt_state,
        gt_change,
        mask_edges_orig,
        input_edges,
        len_mask,
        index
    )    
    for sample_num in range(num_samples):
        pred_graph = utils_models.obtain_graph_3(
            graph_helper,
            graph_info,
            pred_edge[sample_num],
            pred_state[sample_num],
            pred_change[sample_num],
            mask_edges_orig,
            input_edges,
            len_mask,
            index
        )
        pred_graph_samples.append(pred_graph)

    other_info = {
        'prog_gt': program_gt,
        'index': index,
        'metrics': metrics_item,
        'metrics_tstep': metrics_item_tstep
    }

    results = {'gt_graph': gt_graph, 'pred_graph': pred_graph_samples, 'other_info': other_info}


    html_str = utils_models.get_html(results, graph_helper)
                # ipdb.set_trace()
    
    # ipdb.set_trace()
    dir_name = os.path.dirname(html_name)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    with open(html_name, 'w+') as f:
        f.write(html_str)

def inference(
    data_loader,
    model,
    args,
    logger,
    criterions
):
    epoch = 0
    model.eval()


    metric_dict = get_metrics(args)

    progress = ProgressMeter(
        len(data_loader),
        list(metric_dict.values()),
        prefix="Epoch: [{}]".format(epoch),
    )


    end = time.time()
    rows_total = []


    if args.plot_inference:
        threaded_plotter = utils_models.ThreadedPlotter(plot_func, use_threading=False)
    for it, data_item in enumerate(data_loader):
        # print("HEre")
        if it < args['test']['num_iters']:
            metric_dict['data_time'].update(time.time() - end)

            (
                graph_info,
                program,
                label,
                len_mask,
                goal,
                label_agent,
                real_label_agent,
                ind,
            ) = data_item
            

            gt , predictions, misc, losses_dict, inp, loss = compute_forward_pass(args, data_item, data_loader, model, criterions, evaluation=True, posterior=args.inference_posterior)

    
            update_metrics(metric_dict, args, losses_dict, gt, predictions, misc)
            label_action = program['action']
            index_label_obj1 = program['indobj1']
            index_label_obj2 = program['indobj2']

            prog_gt = {
                'action': label_action,
                'o1': index_label_obj1,
                'o2': index_label_obj2,
                'graph': graph_info,
                'mask_len': len_mask,
            }
            # ipdb.set_trace()
            num_samples = args.samples_per_graph

            # Sample here:
            predicted_edge, predicted_state, predicted_change = [], [], []

            if not 'VAE' in args.model.time_aggregate:
                pred_change = (nn.functional.softmax(predictions['pred_change'], dim=3)).cpu().numpy()
                pred_edge = (nn.functional.softmax(predictions['pred_edge'], dim=-1)).cpu().numpy()
                
                pred_state = torch.sigmoid(predictions['pred_state'][..., None]).cpu()
                pred_state = torch.cat([1-pred_state, pred_state], -1).cpu().numpy()

            for sample_num in range(num_samples):
                if 'VAE' in args.model.time_aggregate:
                    with torch.no_grad():
                        output2 = model(inp, inference=not args.inference_posterior)
                    b, t, n, _ = output2['states'].shape
                    cpred_change = output2['node_change'][:, :-1, ...].argmax(-1).cpu()
                    cpred_state = (output2['states'][:, :-1, ...] > 0).cpu()
                    cpred_edge = output2['edges'][:, :-1, ...].reshape(b, t-1, n, n).argmax(-1).cpu()
                else:
                    if sample_num == 0:
                        cpred_change = pred_change.argmax(-1)
                        cpred_edge = pred_edge.argmax(-1)
                        cpred_state = pred_state.argmax(-1)
                    else:
                        cpred_change = utils_models.vectorized(pred_change)
                        cpred_edge = utils_models.vectorized(pred_edge)
                        cpred_state = utils_models.vectorized(pred_state)
                predicted_edge.append(cpred_edge)
                predicted_state.append(cpred_state)
                predicted_change.append(cpred_change)


            pred_edge_c = np.concatenate([x[None, :] for x in predicted_edge], 0)
            pred_change_c = np.concatenate([x[None, :] for x in predicted_change], 0)
            metrics_item_tstep, metrics_item = update_metrics_recall_prec(metric_dict, args, gt, pred_edge_c, pred_change_c, misc)

            input_edges = misc['input_edges']
            gt_state = gt['gt_state']
            gt_change = gt['gt_change']
            gt_edge = gt['gt_edge']            
            mask_edges_orig = misc['mask_nodes']

            
            indices = list(range(ind.shape[0])) * num_samples # (#batch) #samples
            # edge_l = []
            # state_l = []
            # change_l = []

            # for ns in range(num_samples):
            #     edge_l +=  predicted_edge[ns] * bs
            #     state_l += predicted_state[ns] * bs
            #     change_l += predicted_change[ns] * bs


            # graphs = p_map(partial_func, edge_l, state_l, change_l, indices)

            for index in range(0, ind.shape[0], 4):

                # Get the name of html
                current_index = ind[index]
                fname = data_loader.dataset.pkl_files[current_index]
                sfname = fname.split('/')[-1] + "_result"
                pv = ''
                if args.inference_posterior:
                    sfname += '_posterior'
                    pv = '_posterior'
                expath = logger.results_path



                cpath = '/'.join(fname.split('/')[-3:-1])
                dir_name = f'{expath}/{cpath}/'
                result_name = f'{dir_name}/{sfname}.pkl'
                result_name_html = f'{dir_name}/{sfname}.html'
                result_name_html_total = f'{dir_name}/total{pv}.html'
                # ipdb.set_trace()
                dict_plot = {
                    'html_name': result_name_html,
                    'graph_helper': data_loader.dataset.graph_helper,
                    'graph_info': graph_info,
                    'len_mask': len_mask,
                    'index': index,
                    'input_edges': input_edges, 
                    'pred_edge': predicted_edge,
                    'pred_change': predicted_change,
                    'pred_state': predicted_state,
                    'gt_edge': gt_edge,
                    'gt_change': gt_change,
                    'gt_state': gt_state,
                    'prog_gt': prog_gt,
                    'metrics_item': metrics_item,
                    'metrics_item_tstep': metrics_item_tstep,
                    'mask_edges_orig': mask_edges_orig
                       
                }
                # ipdb.set_trace()
                if args.plot_inference:
                    goal_str = ''
                    # goal_str = '<br>'.join([f'{elem}: x{cont}' for elem, cont in ct.items()])
                    threaded_plotter.put_plot_dict(dict_plot)
                    score_total_str = '<br>'.join(['{}: {:03f}'.format(name, value[index]) for name, value in metrics_item.items()])
                    rows_total.append(['<a href="{}.html"> {} </a>'.format(sfname, sfname), score_total_str, goal_str])
                    html_str_total = utils_models.build_table(rows_total, ['Link', 'Scores', 'Goals'])
                    with open(result_name_html_total, 'w+') as f:
                        f.write(html_str_total)








                # ipdb.set_trace()
                if args.save_inference:
                    if not os.path.isdir(dir_name):
                        os.makedirs(dir_name)

                    with open(result_name, 'wb') as f:
                       pkl.dump(results, f)

                # ipdb.set_trace()
            # Update accuracy
            
            # ipdb.set_trace()
            metric_dict['batch_time'].update(time.time() - end)
            end = time.time()
            
            progress.display(it)
            # if it == 4:
            #     ipdb.set_trace()
            # ipdb.set_trace()
            # ipdb.set_trace()

        else:
            continue

        progress.display(it)

# def concat_predictions(pred_graph_samples):
#     all_edges, all_from, all_to = [], [], []
#     for elem in pred_graph_samples:
#         all_edges.append(elem['edge_pred'])
#         all_from.append(elem['from_id'])
#         all_to.append(elem['to_id'])
#     all_edges = np.concatenate(all_edges, 0)
#     all_from = np.concatenate(all_from, 0)
#     all_to = np.concatenate(all_to, 0)

        
#     info['edge_pred'] = all_edges
#     info['from_id'] = all_from
#     info['to_id'] = all_to
#     info['all_edges_input'] = pred_graph_samples[0]['all_edges_input']
#     info['from_id_input'] = pred_graph_samples[0]['from_id_input']
#     info['to_id_input'] = pred_graph_samples[0]['to_id_input']
#     info['states'] = pred_graph_samples[0]['states']
#     info['nodes'] = pred_graph_samples[0]['nodes']
    
#     pred_graph_dict = info
#     return pred_graph_dict

def get_metrics(args):
    metric_dict = {}    
    metric_dict['batch_time'] = AverageMeter('Time', ':6.3f')
    metric_dict['batch_time'] = AverageMeter('Time', ':6.3f')
    metric_dict['data_time'] = AverageMeter('Data', ':6.3f')
    metric_dict['losses'] = AverageMeter('Loss', ':.4e')
    metric_dict['losses_state'] = AverageMeter('LossState', ':.4e')
    metric_dict['losses_edge'] = AverageMeter('LossEdge', ':.4e')
    metric_dict['losses_change'] = AverageMeter('LossChange', ':.4e')
    metric_dict['prec_state'] = AverageMeter('Prec State', ':6.2f')
    metric_dict['recall_state'] = AverageMeter('Rec State', ':6.2f')
    metric_dict['prec_change'] = AverageMeter('Prec Change', ':6.2f')
    metric_dict['recall_change'] = AverageMeter('Rec Change', ':6.2f')
    metric_dict['accuracy_edge'] = AverageMeter('Accuracy Edge', ':6.2f')
    metric_dict['accuracy_edge_pos'] = AverageMeter('Accuracy Edge Pos', ':6.2f')
    metric_dict['accuracy_edge_interest'] = AverageMeter('Accuracy Edge Interest', ':6.2f')
    metric_dict['accuracy_edge_interest_pos'] = AverageMeter('Accuracy Edge Interest Pos', ':6.2f')
    metric_dict['precision_edge_sample'] = AverageMeter('Precision Edge Sample', ':6.2f')
    metric_dict['recall_edge_sample'] = AverageMeter('Recall Edge Sample', ':6.2f')
    metric_dict['f1_edge_sample'] = AverageMeter('F1 Edge Sample', ':6.2f')

    if 'VAE' in args.model.time_aggregate:
        metric_dict['kldiv'] = AverageMeter('KLDIV', ':6.3f')
    return metric_dict


def get_gt_edge(args, goal_graph, edge_dict, len_mask):
    gt_edges = edge_dict['gt_edges']
    if args.model.predict_last:
        nt = gt_edges.shape[1]
        numnode = gt_edges.shape[-1]
        tsteps = len_mask.sum(-1)[:, None, None].repeat(1, 1, numnode).long() - 1
        gt_edge = (
            torch.gather(gt_edges, 1, tsteps).repeat(1, nt - 1, 1).cuda()
        )


    else:
        gt_edge = gt_edges[:, 1:, ...].cuda()
    return gt_edge




def evaluate(
    data_loader,
    data_loader_train,
    model,
    epoch,
    args,
    logger,
    criterions
):
    model.eval()


    metric_dict = get_metrics(args)
    progress = ProgressMeter(
        len(data_loader),
        list(metric_dict.values()),
        prefix="Epoch: [{}]".format(epoch),
    )

    end = time.time()
    for it, data_item in enumerate(data_loader):
        if it < args['test']['num_iters']:
            metric_dict['data_time'].update(time.time() - end)

            (
                graph_info,
                program,
                label,
                len_mask,
                goal,
                label_agent,
                real_label_agent,
                ind,
            ) = data_item


            gt, predictions, misc, losses_dict, inp, loss = compute_forward_pass(args, data_item, data_loader, model, criterions, evaluation=True)
            # Update accuracy
            
            update_metrics(metric_dict, args, losses_dict, gt, predictions, misc)

            num_samples = args.samples_per_graph

            # Sample here:
            predicted_edge, predicted_state, predicted_change = [], [], []

            if not 'VAE' in args.model.time_aggregate:
                pred_change = (nn.functional.softmax(predictions['pred_change'], dim=3)).cpu().numpy()
                pred_edge = (nn.functional.softmax(predictions['pred_edge'], dim=-1)).cpu().numpy()
                
                pred_state = torch.sigmoid(predictions['pred_state'][..., None]).cpu()
                pred_state = torch.cat([1-pred_state, pred_state], -1).cpu().numpy()

            for sample_num in range(num_samples):
                if 'VAE' in args.model.time_aggregate:
                    with torch.no_grad():
                        output2 = model(inp)
                    b, t, n, _ = output2['states'].shape
                    cpred_change = output2['node_change'][:, :-1, ...].argmax(-1).cpu()
                    cpred_state = (output2['states'][:, :-1, ...] > 0).cpu()
                    cpred_edge = output2['edges'][:, :-1, ...].reshape(b, t-1, n, n).argmax(-1).cpu()
                else:
                    if sample_num == 0:
                        cpred_change = pred_change.argmax(-1)
                        cpred_edge = pred_edge.argmax(-1)
                        cpred_state = pred_state.argmax(-1)
                    else:
                        cpred_change = utils_models.vectorized(pred_change)
                        cpred_edge = utils_models.vectorized(pred_edge)
                        cpred_state = utils_models.vectorized(pred_state)
                predicted_edge.append(cpred_edge)
                predicted_state.append(cpred_state)
                predicted_change.append(cpred_change)

            pred_edge_c = np.concatenate([x[None, :] for x in predicted_edge], 0)
            pred_change_c = np.concatenate([x[None, :] for x in predicted_change], 0)
            update_metrics_recall_prec(metric_dict, args, gt, pred_edge_c, pred_change_c, misc)

            input_edges = misc['input_edges'].cpu().numpy()
            gt_state = gt['gt_state']
            gt_change = gt['gt_change']
            gt_edge = gt['gt_change']            
            mask_edges_orig = misc['mask_nodes']
            for index in range(1):
                current_index = ind[index]
                fname = data_loader.dataset.pkl_files[current_index]

                if True:
                    print("************************")
                    print(f"File: {current_index}:{fname}")

                    print("\nGroundTrurth")
                    # print(gt_edge.max())
                    utils_models.print_graph_3(
                        data_loader.dataset.graph_helper,
                        graph_info,
                        gt_edge.cpu().numpy(),
                        mask_edges_orig.cpu().numpy(),
                        gt_state.cpu().numpy(),
                        input_edges,
                        gt_change.cpu().numpy(),
                        index,
                        0,
                    )


                    print("\nPrediction at {}".format(0))
                    #ipdb.set_trace()
                    utils_models.print_graph_3(
                        data_loader.dataset.graph_helper,
                        graph_info,
                        predicted_edge[0],
                        mask_edges_orig.cpu().numpy(),
                        predicted_state[0] > 0,
                        input_edges,
                        predicted_change[0],
                        index,
                        0,
                    )

                    print("************************")
                    # ipdb.set_trace()

            
            

            # ipdb.set_trace()
            end = time.time()

        else:
            continue

        metric_dict['batch_time'].update(time.time() - end)
        progress.display(it)

        # # Print the prediction
        # prog_gt = {'action': label_action, 'o1': index_label_obj1, 'o2': index_label_obj2, 'graph': graph_info, 'mask_len': len_mask}
        # prog_pred = {'action': pred_action, 'o1': pred_o1, 'o2': pred_o2, 'graph': graph_info, 'mask_len': len_mask}

        # str_results = utils_models.get_pred_results_str(data_loader.dataset.graph_helper, prog_gt, prog_pred)

        # info_res = {
        #     'str': progress.display(it, do_print=False)+'\n'+str_results
        # }
        # logger.log_info(info_res)

    info_log = {
        'losses': {
            'total_val': metric_dict['losses'].avg,
            'state_val': metric_dict['losses_state'].avg,
            'edge_val': metric_dict['losses_edge'].avg,
        },
        'accuracy': {
            'state_prec_val': metric_dict['prec_state'].avg,
            'state_recall_val': metric_dict['recall_state'].avg,
            'edge_accuracy_val': metric_dict['accuracy_edge'].avg,
            'edge_accuracy_pos_val': metric_dict['accuracy_edge_pos'].avg,
            'edge_accuracy_interest_val': metric_dict['accuracy_edge_interest'].avg,
            'edge_accuracy_interest_pos_val': metric_dict['accuracy_edge_interest_pos'].avg,
            'recall_edge_sample_val': metric_dict['recall_edge_sample'].avg,
            'precision_edge_sample_val': metric_dict['precision_edge_sample'].avg
        },
        'misc': {'epoch': epoch},
    }
    if 'VAE' in args.model.time_aggregate:
        info_log['losses']['kldiv_val'] = metric_dict['kldiv'].avg
    if args.model.predict_edge_change:
        info_log['losses']['loss_change_val'] = metric_dict['losses_change'].avg
        info_log['accuracy']['edge_prec_change_val'] = metric_dict['prec_change'].avg
        info_log['accuracy']['edge_recall_change_val'] = metric_dict['recall_change'].avg

    if args.model.predict_node_change:
        info_log['losses']['loss_change_node_val'] = metric_dict['losses_change'].avg
        info_log['accuracy']['node_prec_change_val'] = metric_dict['prec_change'].avg
        info_log['accuracy']['node_recall_change_val'] = metric_dict['recall_change'].avg

    logger.log_data(len(data_loader_train) * epoch, info_log)

def update_metrics_recall_prec(metric_dict, args, gt, predicted_edge, changed_edge, misc):
    gt_edge = gt['gt_edge'].cpu().numpy()
    input_edge = misc['input_edges'].cpu().numpy()
    constructed_edge = changed_edge * predicted_edge + (1-changed_edge) * input_edge[None, :]

    samples, b, t, n = constructed_edge.shape
    pred_one_hot = np.zeros((samples, b, t, n, n)).astype(np.uint8)
    np.put_along_axis(pred_one_hot, constructed_edge[..., None], 1, 4)
    pred_max = pred_one_hot.max(0)
    
    # Remove edges connected to None from the prediction
    index_nothing = misc['id_nothing'][:, None, None, None].repeat(1, t, n, n).cpu().numpy()
    np.put_along_axis(pred_max, index_nothing, 0, 3)
    mask_nodes = misc['mask_nodes'].cpu().numpy()[..., 0]
    
    # Remove edges connected to nothing
    gt_max = np.zeros((b, t, n, n))
    np.put_along_axis(gt_max, gt_edge[..., None], 1, 3)
    np.put_along_axis(gt_max, index_nothing, 0, 3)
    mask_idnothing = gt_max.sum(-1)

    true_positives = np.take_along_axis(pred_max, gt_edge[..., None], 3)[..., 0]
    
    true_positives = (true_positives * mask_nodes).sum(-1)
    pred_positives = (pred_max.sum(-1) * mask_nodes).sum(-1)
    gt_positives = (mask_nodes * mask_idnothing).sum(-1)

    # ipdb.set_trace()
    recall = true_positives / (gt_positives + 1e-9)
    prec = true_positives / (pred_positives + 1e-9)
    f1 = recall * prec

    # ipdb.set_trace()
    mask_length = misc['mask_length'].cpu().numpy()
    mask_length = mask_length / mask_length.sum(-1)[:, None]

    # Average over time steps
    recall_item = (recall * mask_length).sum(-1)
    prec_item = (prec * mask_length).sum(-1)
    f1_item = (f1 * mask_length).sum(-1)
    
    metrics_item_tstep = {
        'recall': recall, 
        'prec': prec,
        'f1': f1

    }
    metrics_item = {
        'recall': recall_item,
        'prec': prec_item,
        'f1': f1_item
    }
    recall = recall_item.mean(0)
    prec = prec_item.mean(0)
    f1 = f1_item.mean(0)

    # ipdb.set_trace()
    metric_dict['recall_edge_sample'].update(recall)
    metric_dict['precision_edge_sample'].update(prec)
    metric_dict['f1_edge_sample'].update(f1)
    return metrics_item_tstep, metrics_item

def update_metrics(metric_dict, args, losses_dict, gt, predictions, misc):
    
    gt_state, gt_edge, gt_change = gt['gt_state'], gt['gt_edge'], gt['gt_change']
    pred_state, pred_edge, pred_changes = predictions['pred_state'], predictions['pred_edge'], predictions['pred_change']
    id_nothing, mask_length, mask_state, edge_interest = misc['id_nothing'], misc['mask_length'], misc['mask_nodes'], misc['edge_interest']
    mask_edges = mask_state[..., 0]
    # ipdb.set_trace()
    for loss_name, loss_val in losses_dict.items():
        metric_dict[loss_name].update(loss_val)

    if args.model.predict_edge_change or args.model.predict_node_change:
        # ipdb.set_trace()
        (
            state_prec,
            state_recall,
            change_prec,
            change_recall,
            edge_accuracy,
            edge_accuracy_pos,
            edge_accuracy_interest,
            edge_accuracy_interest_pos
        ) = compute_metrics_change(
            gt_state,
            gt_edge,
            pred_state,
            mask_state,
            pred_edge,
            mask_length,
            mask_edges,
            gt_change,
            pred_changes,
            edge_interest[:, 1:, :].cuda(),
            id_nothing,
            change_type="edge" if args.model.predict_edge_change else "node"
        )
        metric_dict['prec_change'].update(change_prec.item())
        metric_dict['recall_change'].update(change_recall.item())
    else:
        (
            state_prec,
            state_recall,
            edge_accuracy,
            edge_accuracy_pos,
            edge_accuracy_interest,
            edge_accuracy_interest_pos
        ) = compute_metrics(
            gt_state,
            gt_edge,
            pred_state,
            mask_state,
            pred_edge,
            mask_length,
            mask_edges,
            edge_interest[:, 1:, :].cuda()
        )

    metric_dict['prec_state'].update(state_prec.item())
    metric_dict['recall_state'].update(state_recall.item())
    metric_dict['accuracy_edge'].update(edge_accuracy.item())
    metric_dict['accuracy_edge_pos'].update(edge_accuracy_pos.item())
    metric_dict['accuracy_edge_interest'].update(edge_accuracy_interest.item())
    metric_dict['accuracy_edge_interest_pos'].update(edge_accuracy_interest_pos.item())


def train_epoch(
    data_loader,
    model,
    epoch,
    args,
    optimizer,
    logger,
    criterions
):
    print(colored(f"Training epoch {epoch}", "yellow"))

    metric_dict = get_metrics(args)

    progress = ProgressMeter(
        len(data_loader),
        list(metric_dict.values()),
        prefix="Epoch: [{}]".format(epoch),
    )
    

    model.train()

    end = time.time()

    for it, data_item in enumerate(data_loader):
        # ipdb.set_trace()
        metric_dict['data_time'].update(time.time() - end)

        (
            graph_info,
            program,
            label,
            len_mask,
            goal,
            label_agent,
            real_label_agent,
            ind,
        ) = data_item


        gt , predictions, misc, losses_dict, inp, loss = compute_forward_pass(
            args, data_item, data_loader, model, criterions, evaluation=False)

        label_action = program['action']
        index_label_obj1 = program['indobj1']
        index_label_obj2 = program['indobj2']

        prog_gt = {
            'action': label_action,
            'o1': index_label_obj1,
            'o2': index_label_obj2,
            'graph': graph_info,
            'mask_len': len_mask,
        }
        # if int(len_mask[0,:].sum()) == 30:
        #     ipdb.set_trace()
        # ipdb.set_trace()
        program_gt = utils_models.decode_program(
            data_loader.dataset.graph_helper, prog_gt
        )



        update_metrics(metric_dict, args, losses_dict, gt, predictions, misc)
        pred_edge_c = predictions['pred_edge'].argmax(-1)[None, :].cpu().numpy()
        pred_change_c = predictions['pred_change'].argmax(-1)[None, :].cpu().numpy()
        update_metrics_recall_prec(metric_dict, args, gt, pred_edge_c, pred_change_c, misc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_dict['batch_time'].update(time.time() - end)
        end = time.time()

        if it % args['log']['print_every'] == 0:
            progress.display(it)
        if it % args['log']['print_long_every'] == 0:
            # ipdb.set_trace()
            info_log = {
                'losses': {
                    'total': metric_dict['losses'].val,
                    'state': metric_dict['losses_state'].val,
                    'edge': metric_dict['losses_edge'].val,
                },
                'accuracy': {
                    'state_prec': metric_dict['prec_state'].val,
                    'state_recall': metric_dict['recall_state'].val,
                    'edge_accuracy': metric_dict['accuracy_edge'].val,
                    'edge_accuracy_pos': metric_dict['accuracy_edge_pos'].val,
                    'edge_accuracy_interest': metric_dict['accuracy_edge_interest'].val,
                    'edge_accuracy_interest_pos': metric_dict['accuracy_edge_interest_pos'].val

                },
                'misc': {'epoch': epoch},
            }


            if 'VAE' in args.model.time_aggregate:
                info_log['losses']['kldiv'] = metric_dict['kldiv'].val

     
            

            if args.model.predict_edge_change:
                info_log['losses']['loss_change'] = metric_dict['losses_change'].val
                info_log['accuracy']['edge_prec_change'] = metric_dict['prec_change'].val
                info_log['accuracy']['edge_recall_change'] = metric_dict['recall_change'].val

            if args.model.predict_node_change:
                info_log['losses']['loss_change_node'] = metric_dict['losses_change'].val
                info_log['accuracy']['node_prec_change'] = metric_dict['prec_change'].val
                info_log['accuracy']['node_recall_change'] = metric_dict['recall_change'].val

            logger.log_data(it + len(data_loader) * epoch, info_log)

            # Print the prediction
            # logger.log_info(info_res)

    # logger.log_embeds(len(data_loader) * epoch, model.module.agent_embedding)
    print("Failed Elements...", data_loader.dataset.get_failures())

def compute_kl_loss(net_outputs, mask_len):

    # https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians
    mu_prior, logvar_prior, mu_posterior, logvar_posterior = net_outputs['vae_params']
    var_prior = logvar_prior.exp()
    var_posterior = logvar_posterior.exp()
    kl_per_dim = logvar_prior - logvar_posterior - 1 + var_posterior/var_prior + (mu_prior - mu_posterior)**2 / var_prior
    mask_norm = mask_len / mask_len.sum(-1)[:, None]
    res = torch.sum(kl_per_dim, -1) * mask_norm.cuda()
    kldiv = 0.5 * res.sum(1).mean(0)
    # ipdb.set_trace()
    return kldiv


def compute_metrics_change(
    gt_state,
    gt_edges,
    pred_state,
    mask_state,
    pred_edges,
    mask_length,
    mask_edges,
    gt_changes,
    pred_changes,
    edge_int,
    id_nothing,
    change_type="edge"
):
    
    assert change_type in ["node", "edge"]
    # if change_type == "node":
    #     B, T, N = gt_state.shape[:3]
    #     gt_changes = gt_changes.reshape(B, T, N, N)
    #     gt_changes = (gt_changes.sum(-1) > 0).long() 


    # ipdb.set_trace()
    # How many GT positives
    pos_state = gt_state.sum(-1).sum(-1) + 1e-9

    pos_change = gt_changes.sum(-1) + 1e-9

    state_avg = gt_state / (pos_state[:, :, None, None])
    edge_avg = gt_changes / (pos_change[:, :, None])
    # How many predicted positives
    # ipdb.set_trace()

    pred_changes_label = pred_changes.argmax(-1)
    if change_type == "edge":
        edge_avg_pos = (pred_changes_label * mask_edges).sum(-1) + 1e-9
    else:
        edge_avg_pos = (pred_changes_label * mask_state[..., 0]).sum(-1) + 1e-9

    state_avg_pos = ((pred_state > 0) * mask_state).sum(-1).sum(-1) + 1e-9
    # ipdb.set_trace()
    tp_edge_change = (gt_changes * pred_changes_label).sum(-1)
    tp_state = (gt_state * (pred_state > 0)).sum(-1).sum(-1)

    # Recall
    state_recall = tp_state / pos_state
    change_recall = tp_edge_change / pos_change
    # Precision
    state_prec = tp_state / state_avg_pos
    change_prec = tp_edge_change / edge_avg_pos

    # Accuracy
    edge_pred = pred_edges.argmax(-1)
    edge_accuracy = edge_pred == gt_edges

    # Average over timesteps and batch
    mask_timesteps = mask_length / mask_length.sum(-1)[:, None]
    state_recall = (state_recall * mask_timesteps).sum(-1).mean()
    state_prec = (state_prec * mask_timesteps).sum(-1).mean()

    change_recall = (change_recall * mask_timesteps).sum(-1).mean()
    change_prec = (change_prec * mask_timesteps).sum(-1).mean()

    mask_edge_norm = mask_edges / (mask_edges.sum(-1)[..., None] + 1e-9)
    edge_accuracy = (edge_accuracy * mask_edge_norm).sum(-1)
    edge_accuracy = (edge_accuracy * mask_timesteps).sum(-1).mean()

    b, t, n = mask_edges.shape 
    id_nothing = id_nothing[:, None, None].repeat(1, t, n)
    mask_edges_pos = mask_edges.clone()
    mask_edges_pos[gt_edges == id_nothing] = 0

    edge_acc = edge_pred == gt_edges
    mask_edge_norm_pos = mask_edges_pos / (mask_edges_pos.sum(-1)[..., None] + 1e-9)
    edge_accuracy_pos = (edge_acc * mask_edge_norm_pos).sum(-1)
    edge_accuracy_pos = (edge_accuracy_pos * mask_timesteps).sum(-1).mean()
    
    mask_edges_interest = mask_edges.clone()
    mask_edges_interest = mask_edges_interest * edge_int
    mask_edge_norm_interest = mask_edges_interest / (mask_edges_interest.sum(-1)[..., None] + 1e-9)
    edge_accuracy_interest = (edge_acc * mask_edge_norm_interest).sum(-1)
    edge_accuracy_interest = (edge_accuracy_interest * mask_timesteps).sum(-1).mean()
    
    # The ones where the edge is somewhere
    mask_edges_interest_pos = mask_edges * edge_int * mask_edges_pos
    mask_edge_norm_interest_pos = mask_edges_interest_pos / (mask_edges_interest_pos.sum(-1)[..., None] + 1e-9)
    edge_accuracy_interest_pos = (edge_acc * mask_edge_norm_interest_pos).sum(-1)    
    edge_accuracy_interest_pos = (edge_accuracy_interest_pos * mask_timesteps).sum(-1).mean()

    # ipdb.set_trace()

    return (
        state_prec,
        state_recall,
        change_prec,
        change_recall,
        edge_accuracy,
        edge_accuracy_pos,
        edge_accuracy_interest,
        edge_accuracy_interest_pos
    )


def compute_metrics(
    gt_state, gt_edges, pred_state, mask_state, pred_edges, mask_length, mask_edges, edge_int
):

    # How many GT positives
    pos_state = gt_state.sum(-1).sum(-1) + 1e-9

    state_avg = gt_state / (pos_state[:, :, None, None])
    # How many predicted positives
    # ipdb.set_trace()

    state_avg_pos = ((pred_state > 0) * mask_state).sum(-1).sum(-1) + 1e-9

    tp_state = (gt_state * (pred_state > 0)).sum(-1).sum(-1)

    # Recall
    state_recall = tp_state / pos_state
    # Precision
    state_prec = tp_state / state_avg_pos

    # Accuracy
    edge_pred = pred_edges.argmax(-1)
    edge_accuracy = edge_pred == gt_edges

    # Average over timesteps and batch
    mask_timesteps = mask_length / mask_length.sum(-1)[:, None]
    state_recall = (state_recall * mask_timesteps).sum(-1).mean()
    state_prec = (state_prec * mask_timesteps).sum(-1).mean()

    mask_edge_norm = mask_edges / (mask_edges.sum(-1)[..., None] + 1e-9)
    edge_accuracy = (edge_accuracy * mask_edge_norm).sum(-1)
    edge_accuracy = (edge_accuracy * mask_timesteps).sum(-1).mean()

    mask_edges_pos = mask_edges.clone()
    mask_edges_pos[gt_edges == 0] = 0

    edge_acc = edge_pred == gt_edges
    mask_edge_norm_pos = mask_edges_pos / (mask_edges_pos.sum(-1)[..., None] + 1e-9)
    edge_accuracy_pos = (edge_acc * mask_edge_norm_pos).sum(-1)
    edge_accuracy_pos = (edge_accuracy_pos * mask_timesteps).sum(-1).mean()


    mask_edges_interest = mask_edges.clone()
    mask_edges_interest = mask_edges_interest * edge_int
    mask_edge_norm_interest = mask_edges_interest / (mask_edges_interest.sum(-1)[..., None] + 1e-9)
    edge_accuracy_interest = (edge_acc * mask_edge_norm_interest).sum(-1)
    edge_accuracy_interest = (edge_accuracy_interest * mask_timesteps).sum(-1).mean()

    mask_edges_interest_pos = mask_edges * edge_int * (gt_edges ==  2)
    mask_edge_norm_interest_pos = mask_edges_interest_pos / (mask_edges_interest_pos.sum(-1)[..., None] + 1e-9)
    edge_accuracy_interest_pos = (edge_acc * mask_edge_norm_interest_pos).sum(-1)    
    edge_accuracy_interest_pos = (edge_accuracy_interest_pos * mask_timesteps).sum(-1).mean()
    # ipdb.set_trace()
    return state_prec, state_recall, edge_accuracy, edge_accuracy_pos, edge_accuracy_interest, edge_accuracy_interest_pos


def get_loaders(args):
    print("Loading dataset...")
    print("Train: {}".format(args['data']['train_data']))
    print("Test: {}".format(args['data']['test_data']))
    curr_file = os.path.dirname(get_original_cwd())
    dataset = AgentTypeDataset(
        path_init='{}/agent_preferences/dataset/{}'.format(
            curr_file, args['data']['train_data']
        ),
        args_config=args,
    )
    if not args['train']['overfit']:
        dataset_test = AgentTypeDataset(
            path_init='{}/agent_preferences/dataset/{}'.format(
                curr_file, args['data']['test_data']
            ),
            args_config=args,
        )
    else:
        dataset_test = AgentTypeDataset(
            path_init='{}/agent_preferences/dataset/{}'.format(
                curr_file, args['data']['train_data']
            ),
            args_config=args,
        )

    if args['model']['state_encoder'] == 'GNN':
        collate_fn = dataloader_v2.collate_fn
    else:
        collate_fn = None
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args['train']['batch_size'],
        shuffle=True,
        num_workers=args['train']['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args['train']['batch_size'],
        shuffle=not args.inference,
        num_workers=args['train']['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    return train_loader, test_loader


@hydra.main(config_path="../config/agent_pred_graph", config_name="config_default_large_excl")
def main(cfg: DictConfig):
    config = cfg
    print("Config")
    print(OmegaConf.to_yaml(cfg))
    # ipdb.set_trace()

    assert not (cfg.model.predict_edge_change)
    assert cfg['model']['exclusive_edge']
    cfg.model.input_goal = False

    # cfg.num_gpus = torch.cuda.device_count()
    
    train_loader, test_loader = get_loaders(config)
    if config.model.input_goal:

        model = agent_pref_policy.GoalConditionedGraphPredNetwork(config)
    else:
        if 'VAE' not in config.model.time_aggregate:
            model = agent_pref_policy.GraphPredNetwork(config)
        else:
            model = agent_pref_policy.GraphPredNetworkVAE(config)
    print("CUDA: {}".format(cfg.cuda))
    if cfg.cuda:
        model = model.cuda()
        model = nn.DataParallel(model)

    if len(cfg.ckpt_load) > 0:
        model.load_state_dict(torch.load(cfg.ckpt_load)['model'])

    # loss states
    weight = None
    if config['train']['loss_weighted_edge']:
        weight = torch.Tensor([0.05, 1, 1, 1, 1])
        if cfg.cuda:
            weight = weight.cuda()
    criterion_state = torch.nn.BCEWithLogitsLoss(reduction='none')
    criterion_edge = torch.nn.CrossEntropyLoss(weight=weight, reduction='none')
    criterion_change = torch.nn.CrossEntropyLoss(reduction='none')

    criterions = {
        'change': criterion_change,
        'state': criterion_state,
        'edge': criterion_edge
    }
    if config.inference:
        logger = LoggerSteps(config, log_steps=False)
        print("Saving results at")
        print(logger.results_path)
        # ipdb.set_trace()
        inference(
            test_loader,
            model,
            config,
            logger,
            criterions
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=config['train']['lr'])
        print("Failures: ", train_loader.dataset.get_failures())

        logger = LoggerSteps(config, log_steps=config.logging)

        if config.logging:
            logger.save_model(0, model, optimizer)

        # evaluate(test_loader, train_loader, model, 0, config, logger, criterions)
        # ipdb.set_trace()

        evaluate(
            test_loader,
            train_loader,
            model,
            0,
            config,
            logger,
            criterions
        )
        # ipdb.set_trace()
        for epoch in range(config['train']['epochs']):
            # ipdb.set_trace()
            train_epoch(
                train_loader,
                model,
                epoch,
                config,
                optimizer,
                logger,
                criterions
            )
            evaluate(
                test_loader,
                train_loader,
                model,
                epoch,
                config,
                logger,
                criterions
            )
            if epoch % config.log.save_every == 0:
                logger.save_model(epoch, model, optimizer)


if __name__ == '__main__':
    main()
