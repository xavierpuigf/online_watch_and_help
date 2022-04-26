import torch
import sys
sys.path.append('.')
import time
import os
import glob
import yaml
import pickle as pkl
from tqdm import tqdm
import ipdb
import numpy as np
from dataloader.dataloader_v2_task_reduced import AgentTypeDataset
from dataloader import dataloader_v2_task_reduced as dataloader_v2
from arguments import *
from torch import nn
import torch.optim as optim
from models import agent_pref_policy_task as agent_pref_policy
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


def set_kl_coeff(args, epoch):
    kl_coeff_max = args.model.kl_coeff
    if epoch >= args.model.kl_anneal_epoch:
        return kl_coeff_max
    else:
        # do cosine annealing
        val = np.pi * epoch * 1./args.model.kl_anneal_epoch
        curr_kl_coeff = (-np.cos(val)*0.5 + 0.5) * kl_coeff_max
        return curr_kl_coeff



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


def compute_forward_pass(args, data_item, data_loader, model, criterions, misc={}, evaluation=False, posterior=False):

    (
        program,
        len_mask,
        goal,
        task_graph,
        ind,
        task_id,
    ) = data_item

    inputs = {
        'program': program,
        'task_graph': task_graph,
        'mask_len': len_mask,
        'goal': goal,
        'task_id': task_id
    }
    # ipdb.set_trace()
    # T = graph_info['mask_object'].shape[1]
    # object_coords_dim = graph_info['object_coords'].shape[3]
    # states_objects_dim = graph_info['states_objects'].shape[3]
    # mask_obs_node = graph_info['mask_obs_node']

    # goal_graph = build_goal_graph(graph_info, len_mask)
    # inputs['goal_graph'] = goal_graph


    ##################
    # Obtain GT
    ##################
    
    # edge_dict = utils_models.build_gt_edge(
    #     graph_info, data_loader.dataset.graph_helper, 
    #     exclusive_edge=args.model.exclusive_edge)
    # gt_state = goal_graph['states_objects'][:, 1:, ...]
    # gt_edge = get_gt_edge(args, goal_graph, edge_dict, len_mask) 

    # input_edges = edge_dict['gt_edges'][:, :-1, :]
    # edge_interest = edge_dict['edge_interest']

    # Obtain GT Change
    # gt_change = (gt_edge != input_edges.cuda()).long().cuda()
    # gt_change *= mask_obs_node[:, 1:, ...].long().cuda()

    # inputs['input_edges'] = edge_dict['gt_edges']
    # try:
    # print(len_mask.shape)
    if not evaluation:
        output = model(inputs)
        
    else:
        with torch.no_grad():
            output = model(inputs, inference=not posterior)
    # ipdb.set_trace()
    # except:
    #     print("ERROR!!!")
    #     ipdb.set_trace()
    ##################
    # Get Predictions
    ##################
    # B x T x P x #count 

    mask_length = len_mask[:, 1:].cuda()
    pred_mask = output['pred_mask'][:, :-1, ...]
    pred_task = output['pred_graph'][:, :-1, ...]
    pred_task_total = output['pred_graph_total'][:, :-1, ...]

    tsteps = pred_mask.shape[1]


    if args.model.predict_diff:
        gt_mask = inputs['task_graph']['mask_task_graph'][:, :-1, ...].float().cuda()
    else:
        if args.model.predict_diff_preds:
            T = inputs['task_graph']['mask_task_graph'].shape[1]
            gt_mask = (inputs['task_graph']['gt_task_graph'] != 0)[:, None, ...].repeat(1, T-1, 1).float().cuda()
        else:
            gt_mask = torch.ones_like(inputs['task_graph']['mask_task_graph'][:, :-1, ...].float()).cuda()

    # ipdb.set_trace()
    gt_task = inputs['task_graph']['gt_task_graph'][:, None, ...].repeat(1, tsteps, 1).long().cuda()
    input_task = inputs['task_graph']['task_graph'][:, :-1, ...].long().cuda()
    ##################
    # Compute losses
    ##################
    loss = 0
    losses_dict = {}

    # Loss state
    if args.model.use_only_input or not args.model.predict_diff:
        loss_mask = 0
    else:   
        loss_mask = criterions['mask'](
            pred_mask,
            gt_mask,
        )
        loss_mask = loss_mask.mean(-1)
        loss_mask = (loss_mask * mask_length).mean(-1).mean(-1)



    loss_task = criterions['task'](
        pred_task.permute(0,3,1,2),
        gt_task
    )
    # ipdb.set_trace()
    loss_task = loss_task.mean(-1)
    loss_task = (loss_task * mask_length).mean(-1).mean(-1)


    losses_dict['losses_task'] = loss_task.item()

    if args.model.use_only_input or not args.model.predict_diff:
        losses_dict['losses_mask'] = 0
    else:
        
        losses_dict['losses_mask'] = loss_mask.item()

    
    loss += loss_mask + loss_task

    if args.model.predict_category:
        predict_category = output['predict_category']
        # ipdb.set_trace()
        loss_category = criterions['category'](predict_category, inputs['task_id'].cuda()).mean(0)
        losses_dict['losses_category'] = loss_category.item()
        loss += loss_category

    losses_dict.update({
        'losses': loss.item()
    })

    if 'VAE' in args.model.time_aggregate and args.model.input_vae != 'none':
        loss_kl = compute_kl_loss(output, len_mask)
        losses_dict['kldiv'] = loss_kl.item()
        if args.model.kl_annealing and 'kl_coeff' in misc:
            loss += misc['kl_coeff']*loss_kl
        else:
            loss += args.model.kl_coeff*loss_kl

    gt = {
        'gt_task': gt_task,
        'gt_mask': gt_mask
    }

    predictions = {
        'pred_task': pred_task,
        'pred_mask': pred_mask
        
    }
    if args.model.predict_category:
        # B x num_cat
        predictions['predict_category'] = output['predict_category']
        gt['gt_category'] = task_id
    misc = {
        # 'input_edges': input_edges,
        # 'id_nothing': edge_dict['id_nothing'].cuda(),
        # 'mask_nodes': mask_state,
        'input_task': input_task,
        'mask_length': mask_length,
        'vae_params': output['vae_params'],
        # 'edge_interest': edge_dict['edge_interest']
    }
    # ipdb.set_trace()
    return gt, predictions, misc, losses_dict, inputs, loss


def plot_func(html_name, graph_helper, len_mask, index, 
              gt_task, gt_mask, input_task, pred_task, pred_mask, prog_gt, metrics_item, metrics_item_tstep):
    # print(prog_gt)
    # program_gt = utils_models.decode_program(graph_helper, prog_gt, index=index)
    pred_graph_samples = []
    num_samples = len(pred_task)

    # Obtain GT Graph
    gt_graph = utils_models.obtain_task_graph(
            graph_helper,
            None,
            gt_task,
            gt_mask,
            input_task,
            index,
            len_mask
    )  
    # ipdb.set_trace()
    
    program_gt = ['' for _ in gt_graph]
    for sample_num in range(num_samples):
        pred_graph = utils_models.obtain_task_graph(
            graph_helper,
            None,
            pred_task[sample_num],
            pred_mask[sample_num],
            input_task,
            index,
            len_mask
        )
        pred_graph_samples.append(pred_graph)

    other_info = {
        'prog_gt': program_gt,
        'index': index,
        'metrics': metrics_item,
        'metrics_tstep': metrics_item_tstep
    }

    results = {'gt_task': [gt_graph, gt_task[index]], 
               'pred_task': [pred_graph_samples, [pt[index] for pt in pred_task]], 
               'other_info': other_info}


    html_str = utils_models.get_html_task_update(results, graph_helper)
                # ipdb.set_trace()
    
    dir_name = os.path.dirname(html_name)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    with open(html_name, 'w+') as f:
        f.write(html_str)
    # ipdb.set_trace()

def inference(
    data_loader,
    model,
    args,
    logger,
    criterions,
    posterior
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
                program,
                len_mask,
                goal,
                task_graph,
                ind,
                task_index
            ) = data_item

            

            gt , predictions, misc, losses_dict, inp, loss = compute_forward_pass(args, data_item, data_loader, model, criterions, evaluation=True, posterior=posterior, misc={})

    
            update_metrics(metric_dict, args, losses_dict, gt, predictions, misc)
            label_action = program['action']
            index_label_obj1 = program['indobj1']
            index_label_obj2 = program['indobj2']

            prog_gt = None 
            # {
            #     'action': label_action,
            #     'o1': index_label_obj1,
            #     'o2': index_label_obj2,
            #     'graph': graph_info,
            #     'mask_len': len_mask,
            # }
            # ipdb.set_trace()
            num_samples = args.samples_per_graph

            # Sample here:

            predicted_mask, predicted_graph = [], []

            if not 'VAE' in args.model.time_aggregate:
                #pass
                pred_mask = (nn.functional.sigmoid(predictions['pred_mask']))[..., None].cpu().numpy()
                pred_mask = np.concatenate([pred_mask, 1 - pred_mask], -1)
                pred_graph = (nn.functional.softmax(predictions['pred_task'], dim=-1)).cpu().numpy()

            for sample_num in range(num_samples):
                if 'VAE' in args.model.time_aggregate:
                    with torch.no_grad():
                        output2 = model(inp, inference=not posterior)
                    cpred_mask = gt['gt_mask'].cpu().numpy() # (output2['pred_mask'][:, :-1, ...] > 0).cpu()
                    cpred_graph = (output2['pred_graph'][:, :-1, ...].argmax(-1)).cpu()
                else:
                    if sample_num == 0:
                        cpred_mask = gt['gt_mask'].cpu().numpy() #pred_mask.argmax(-1)
                        cpred_graph = pred_graph.argmax(-1)
                    else:
                        cpred_mask = gt['gt_mask'].cpu().numpy()
                        cpred_graph = utils_models.vectorized(pred_graph)
                predicted_mask.append(cpred_mask)
                predicted_graph.append(cpred_graph)
            # ipdb.set_trace()

            # pred_edge_c = np.concatenate([x[None, :] for x in predicted_edge], 0)
            # pred_change_c = np.concatenate([x[None, :] for x in predicted_change], 0)
            predicted_maskc = np.concatenate([x[None, :] for x in predicted_mask], 0)
            predicted_graphc = np.concatenate([x[None, :] for x in predicted_graph], 0)


            metrics_item_tstep, metrics_item = update_metrics_recall_prec(metric_dict, args, predicted_maskc, predicted_graphc, gt['gt_mask'].cpu().numpy(), gt['gt_task'].cpu().numpy(), misc)

            
            # ipdb.set_trace()
            if args.model.predict_category:
                # B x T
                category_accuracy = (gt['gt_category'].cuda() == predictions['predict_category'].argmax(-1)).float().cpu().numpy()

                metrics_item['category_accuracy'] = category_accuracy
            # input_edges = misc['input_edges']
            # gt_state = gt['gt_state']
            # gt_change = gt['gt_change']
            # gt_edge = gt['gt_edge']            
            # mask_edges_orig = misc['mask_nodes']

            
            # indices = list(range(ind.shape[0])) * num_samples # (#batch) #samples
            # edge_l = []
            # state_l = []
            # change_l = []

            # for ns in range(num_samples):
            #     edge_l +=  predicted_edge[ns] * bs
            #     state_l += predicted_state[ns] * bs
            #     change_l += predicted_change[ns] * bs


            # graphs = p_map(partial_func, edge_l, state_l, change_l, indices)

            gt_task = gt['gt_task'].cpu().numpy()
            gt_mask = gt['gt_mask'].cpu().numpy()

            # gt_edge = gt['gt_change']         
            input_task = misc['input_task'] 

            # ipdb.set_trace()
            # metrics_item = {metric_name: met.cpu().numpy() for metric_name, met in metrics_item.items()}
            # metrics_item_tstep = {metric_name: met.cpu().numpy() for metric_name, met in metrics_item_tstep.items()}

            res_total_new = [{metric_name: metric_value[:, index] for metric_name, metric_value in metrics_item.items() if '_seed' in metric_name} for index in range(0, ind.shape[0]) ]
            res_total_new_tstep = [{metric_name: metric_value[:, index] for metric_name, metric_value in metrics_item_tstep.items() if '_seed' in metric_name} for index in range(0, ind.shape[0]) ]

            inp_task = input_task.cpu().numpy()

            for index in range(0, ind.shape[0]):
                # ipdb.set_trace()
                # pred_task = predicted_graph[index]
                # pred_mask = predicted_mask[index]


                # Get the name of html
                current_index = ind[index]
                fname = data_loader.dataset.pkl_files[current_index]
                sfname = fname.split('/')[-1] + "_result"
                pv = ''
                if posterior:
                    sfname += '_posterior'
                    pv = '_posterior'
                expath = logger.results_path

                # if 'logs_episode.121_iter.0.pik_result' in sfname:
                #     ipdb.set_trace()
                # else:
                #     pass
                #     # continue


                dir_name = f'{expath}'
                result_name = f'{dir_name}/{sfname}.pkl'
                result_name_html = f'{dir_name}/{sfname}.html'
                result_name_html_total = f'{dir_name}/total{pv}.html'
                dict_result_name = f'{dir_name}/pred_dict.pkl'
                # ipdb.set_trace()
                dict_plot = {
                    'html_name': result_name_html,
                    'graph_helper': data_loader.dataset.graph_helper,
                    'len_mask': len_mask,
                    'index': index,
                    'gt_task': gt_task,
                    'gt_mask': gt_mask,
                    'input_task': input_task,
                    'pred_task': predicted_graph,
                    'pred_mask': predicted_mask,
                    'prog_gt': prog_gt,
                    'metrics_item_tstep': metrics_item_tstep,
                    'metrics_item': metrics_item
                       
                }
                # ipdb.set_trace()
                if args.plot_inference:
                    goal_str = ''
                    # goal_str = '<br>'.join([f'{elem}: x{cont}' for elem, cont in ct.items()])
                    threaded_plotter.put_plot_dict(dict_plot)
                    # score_total_str = '<br>'.join(['{}: {:03f}'.format(name, value[index]) for name, value in metrics_item.items()])
                    score_total_str = '<br>'.join(['{}: {:.03f}'.format(name, value[index]) for name, value in metrics_item.items() if 'seed' not in name])
                    # ipdb.set_trace()
                    rows_total.append(['<a href="{}.html"> {} </a>'.format(sfname, sfname), score_total_str, goal_str])
                    metric_names = list(metrics_item.keys())
                    html_str_total = utils_models.build_table(rows_total, ['Link']+metric_names+['Goals'])
                    if index == 0:
                        print(colored(result_name_html_total, 'cyan'))
                    if not os.path.isdir(dir_name):
                        os.makedirs(dir_name)
                    with open(result_name_html_total, 'w+') as f:
                        f.write(html_str_total)



                # ipdb.set_trace()




                if args.save_inference:
                    if not os.path.isfile(dict_result_name):
                        list_names = []
                        # save dict of preds
                        for index_pred in range(gt_task.shape[-1]):
                            curr_tuple = data_loader.dataset.graph_helper.task_graph_list[index_pred]
                            curr_names = data_loader.dataset.graph_helper.task_graph_dict[curr_tuple][0]
                            list_names.append(curr_names)
                        with open(dict_result_name, 'wb+') as f:
                            pkl.dump(list_names, f)
                    #ipdb.set_trace()
                    results = {
                        'results_total': res_total_new[index],
                        'results_total_tstep': res_total_new_tstep[index],
                        'gt_task': gt_task[index],
                        'inp_task': inp_task[index],
                        'pred_task': predicted_graphc[:, index],
                        'length': len_mask[index].sum().cpu().numpy()
                    }

                    # ipdb.set_trace()
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
    metric_dict['data_time'] = AverageMeter('DataTime', ':6.3f')
    metric_dict['model_time'] = AverageMeter('ModelTime', ':6.3f')
    metric_dict['losses'] = AverageMeter('Loss', ':.4e')
    metric_dict['losses_task'] = AverageMeter('LossTask', ':.4e')
    metric_dict['losses_mask'] = AverageMeter('LossMask', ':.4e')
    
    metric_dict['prec_change'] = AverageMeter('Prec Change', ':6.2f')
    metric_dict['recall_change'] = AverageMeter('Rec Change', ':6.2f')
    
    metric_dict['task_accuracy'] = AverageMeter('Acc. Task', ':6.2f')
    metric_dict['task_accuracy_pos'] = AverageMeter('Acc. Task Pos', ':6.2f')

    if args.model.predict_category:
        metric_dict['category_accuracy'] = AverageMeter('Acc. Category', ':6.2f')
        metric_dict['losses_category'] = AverageMeter('LossCategory', ':6.2f')

    
    # metric_dict['precision_edge_sample'] = AverageMeter('Precision Edge Sample', ':6.2f')
    # metric_dict['recall_edge_sample'] = AverageMeter('Recall Edge Sample', ':6.2f')
    # metric_dict['f1_edge_sample'] = AverageMeter('F1 Edge Sample', ':6.2f')

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
    criterions,
    use_posterior
):
    model.eval()


    metric_dict = get_metrics(args)
    progress = ProgressMeter(
        len(data_loader),
        list(metric_dict.values()),
        prefix="Epoch: [{}]".format(epoch),
    )

    end = time.time()
    print("Evaluation")
    # ipdb.set_trace()

    if args.model.kl_annealing:
        misc_dict = {}
        misc_dict['kl_coeff'] = set_kl_coeff(args, epoch) 

    for it, data_item in enumerate(data_loader):
        if it < args['test']['num_iters']:
            metric_dict['data_time'].update(time.time() - end)

            (
                program,
                len_mask,
                goal,
                task_graph,
                ind,
                category_index
            ) = data_item


            t1 = time.time()
            gt, predictions, misc, losses_dict, inp, loss = compute_forward_pass(args, data_item, data_loader, model, criterions, evaluation=True, posterior=use_posterior, misc=misc_dict)

            metric_dict['model_time'].update(time.time() - t1)
            # print(loss)
            # gt1, predictions1, misc1, losses_dict1, inp1, loss1 = compute_forward_pass(args, data_item, data_loader, model, criterions, evaluation=True, posterior=False)
            # gt2, predictions2, misc2, losses_dict2, inp2, loss2 = compute_forward_pass(args, data_item, data_loader, model, criterions, evaluation=True, posterior=True)
            # print(loss1, loss2)
            # ipdb.set_trace()
            # Update accuracy
            
            update_metrics(metric_dict, args, losses_dict, gt, predictions, misc)

            num_samples = args.samples_per_graph

            # Sample here:
            predicted_mask, predicted_graph = [], []

            if not 'VAE' in args.model.time_aggregate:
                pass
                #pred_mask = (nn.functional.sigmoid(predictions['pred_mask'], dim=3)).cpu().numpy()
                #pred_mask = np.concatenate([pred_mask, 1 - pred_mask], -1)
                #pred_graph = (nn.functional.softmax(predictions['pred_graph'], dim=-1)).cpu().numpy()
                
                # pred_state = torch.sigmoid(predictions['pred_state'][..., None]).cpu()
                # pred_state = torch.cat([1-pred_state, pred_state], -1).cpu().numpy()

        #     for sample_num in range(num_samples):
        #         if 'VAE' in args.model.time_aggregate:
        #             with torch.no_grad():
        #                 output2 = model(inp)

        #             cpred_mask = (output2['pred_mask'][:, :-1, ...] > 0).cpu()
        #             cpred_graph = (output2['pred_graph'][:, :-1, ...].argmax(-1)).cpu()
        #             # cpred_edge = output2['edges'][:, :-1, ...].reshape(b, t-1, n, n).argmax(-1).cpu()
        #         else:
        #             if sample_num == 0:
        #                 cpred_mask = pred_mask.argmax(-1)
        #                 cpred_graph = pred_graph.argmax(-1)
        #                 # cpred_state = pred_state.argmax(-1)
        #             else:
        #                 cpred_mask = utils_models.vectorized(pred_mask)
        #                 cpred_graph = utils_models.vectorized(pred_graph)
        #         predicted_mask.append(cpred_mask)
        #         predicted_graph.append(cpred_graph)

        #     # ipdb.set_trace()
        #     # pred_edge_c = np.concatenate([x[None, :] for x in predicted_edge], 0)
        #     # pred_change_c = np.concatenate([x[None, :] for x in predicted_change], 0)
        #     # update_metrics_recall_prec(metric_dict, args, gt, pred_edge_c, pred_change_c, misc)

        #     # input_edges = misc['input_edges'].cpu().numpy()
            
        #     gt_task = gt['gt_task'].cpu().numpy()
        #     gt_mask = gt['gt_mask'].cpu().numpy()

        #     # gt_edge = gt['gt_change']         
        #     input_task = misc['input_task']   
        #     # mask_edges_orig = misc['mask_nodes']
        #     for index in range(1):
        #         # ipdb.set_trace()
        #         current_index = ind[index]
        #         fname = data_loader.dataset.pkl_files[current_index]
        #         pred_graph = predicted_graph[index]
        #         pred_mask = predicted_mask[index]
        #         if True:
        #             print("************************")
        #             print(f"File: {current_index}:{fname}")

        #             print("\nGroundTrurth")
        #             # print(gt_edge.max())
        #             utils_models.print_task_graph(
        #                 data_loader.dataset.graph_helper,
        #                 graph_info,
        #                 gt_task,
        #                 gt_mask,
        #                 input_task,
        #                 index,
        #                 0,
        #             )
        #             print('---')
        #             # utils_models.print_task_graph(
        #             #     data_loader.dataset.graph_helper,
        #             #     graph_info,
        #             #     input_task,
        #             #     gt_mask,
        #             #     input_task,
        #             #     index,
        #             #     0,
        #             # )

        #             # ipdb.set_trace()

        #             print("\nPrediction at {}".format(0))
        #             #ipdb.set_trace()
        #             utils_models.print_task_graph(
        #                 data_loader.dataset.graph_helper,
        #                 graph_info,
        #                 pred_graph,
        #                 pred_mask,
        #                 input_task,
        #                 index,
        #                 0,
        #             )

        #             print("************************")
        #             # ipdb.set_trace()

            
            

        #     # ipdb.set_trace()
        #     end = time.time()

        # else:
        #     continue

        metric_dict['batch_time'].update(time.time() - end)
        
        # # Print the prediction
        # prog_gt = {'action': label_action, 'o1': index_label_obj1, 'o2': index_label_obj2, 'graph': graph_info, 'mask_len': len_mask}
        # prog_pred = {'action': pred_action, 'o1': pred_o1, 'o2': pred_o2, 'graph': graph_info, 'mask_len': len_mask}

        # str_results = utils_models.get_pred_results_str(data_loader.dataset.graph_helper, prog_gt, prog_pred)

        # info_res = {
        #     'str': progress.display(it, do_print=False)+'\n'+str_results
        # }
        # logger.log_info(info_res)

    # Take vae params of last step    
    progress.display(it)

    mu_prior, logvar_prior, mu_posterior, logvar_posterior = misc['vae_params']

    # ipdb.set_trace()
    suffix = 'val'
    if use_posterior:
        suffix = 'val_posterior'
    info_log = {
        'losses': {
            'total_'+suffix: metric_dict['losses'].avg,
            'task_'+suffix: metric_dict['losses_task'].avg,
            'mask_'+suffix: metric_dict['losses_mask'].avg,
        },
        'accuracy': {
            'change_prec_'+suffix: metric_dict['prec_change'].avg,
            'change_recall_'+suffix: metric_dict['recall_change'].avg,
            'task_accuracy_'+suffix: metric_dict['task_accuracy'].avg,
            'task_accuracy_pos_'+suffix: metric_dict['task_accuracy_pos'].avg
            # 'edge_accuracy_interest': metric_dict['accuracy_edge_interest'].val,
            # 'edge_accuracy_interest_pos': metric_dict['accuracy_edge_interest_pos'].val

        },
        'misc': {
            'epoch': epoch, 

        },
        'misc_hist': {
            'muprior': mu_prior,
            'muposterior': mu_posterior,
            'logvar_prior': logvar_prior,
            'logvar_posterior': logvar_posterior
        }
    }


    if args.model.kl_annealing:
        info_log['misc']['kl_coeff'] = set_kl_coeff(args, epoch) 
    if 'VAE' in args.model.time_aggregate:
        info_log['losses']['kldiv_'+suffix] = metric_dict['kldiv'].avg


    if args.model.predict_category:
        info_log['accuracy']['category_accuracy_'+suffix] = metric_dict['category_accuracy'].avg

    logger.log_data(len(data_loader_train) * epoch, info_log)

def update_metrics_recall_prec(metric_dict, args, pred_mask_task, pred_task, mask_task, gt_task, misc):
    eps = 1e-9
    # return None, None
    # gt_edge = gt['gt_edge'].cpu().numpy()
    # input_edge = misc['input_edges'].cpu().numpy()
    # constructed_edge = changed_edge * predicted_edge + (1-changed_edge) * input_edge[None, :]

    # samples, b, t, n = constructed_edge.shape
    # pred_one_hot = np.zeros((samples, b, t, n, n)).astype(np.uint8)
    # np.put_along_axis(pred_one_hot, constructed_edge[..., None], 1, 4)
    # pred_max = pred_one_hot.max(0)
    
    # # Remove edges connected to None from the prediction
    # index_nothing = misc['id_nothing'][:, None, None, None].repeat(1, t, n, n).cpu().numpy()
    # np.put_along_axis(pred_max, index_nothing, 0, 3)
    # mask_nodes = misc['mask_nodes'].cpu().numpy()[..., 0]
    
    # # Remove edges connected to nothing
    # gt_max = np.zeros((b, t, n, n))
    # np.put_along_axis(gt_max, gt_edge[..., None], 1, 3)
    # np.put_along_axis(gt_max, index_nothing, 0, 3)
    # mask_idnothing = gt_max.sum(-1)

    # true_positives = np.take_along_axis(pred_max, gt_edge[..., None], 3)[..., 0]
    
    # true_positives = (true_positives * mask_nodes).sum(-1)
    # pred_positives = (pred_max.sum(-1) * mask_nodes).sum(-1)
    # gt_positives = (mask_nodes * mask_idnothing).sum(-1)

    # # ipdb.set_trace()
    # recall = true_positives / (gt_positives + 1e-9)
    # prec = true_positives / (pred_positives + 1e-9)
    # f1 = recall * prec

    # # ipdb.set_trace()
    mask_length = misc['mask_length'] 
    mask_length = (mask_length / mask_length.sum(-1)[:, None]).cpu().numpy()

    # ipdb.set_trace()
    pos_change = mask_task.sum(-1) + 1e-9

    # How many predicted positives
    # ipdb.set_trace()

    change_avg_pos = pred_mask_task.sum(-1)
    
    # ipdb.set_trace()
    tp_edge_change = (mask_task[None, ...] * pred_mask_task).sum(-1)
    # recall = tp_edge_change / (pos_change+1e-9)

    recall =  np.minimum(pred_task, gt_task[None, :]).sum(-1) / (1e-9 + gt_task[None, ...].sum(-1))
    prec = (np.minimum(pred_task, gt_task).sum(-1) / (eps+pred_task.sum(-1))).mean(0)[None, ...]
    
    mask_task_norm = mask_task / (1e-9 + mask_task.sum(-1)[..., None])
    # Accuracy
    # ipdb.set_trace()
    # ipdb.set_trace()
    accuracy = (pred_task == gt_task).astype(np.float32).mean(-1)
    accuracy_pos = ((pred_task == gt_task).astype(np.float32) * mask_task_norm).sum(-1)
    

    # ipdb.set_trace()


    # # Average over time steps
    recall_item = (recall * mask_length).sum(-1)
    prec_item = (prec * mask_length).sum(-1)
    accuracy_pos_item = (accuracy_pos * mask_length).sum(-1)
    accuracy_item = (accuracy * mask_length).sum(-1)
    # f1_item = (f1 * mask_length).sum(-1)
    
    metrics_item_tstep = {
        'recall': recall.max(0), 
        'prec': prec.max(0),
        'accuracy': accuracy.max(0),
        'accuracy_pos': accuracy_pos.max(0),
        'accuracy_pos_seed': accuracy_pos,
        'accuracy_seed': accuracy

    }
    metrics_item = {
        'recall': recall_item.max(0), 
        'prec': prec_item.max(0),
        'accuracy': accuracy_item.max(0),
        'accuracy_pos': accuracy_pos_item.max(0),
        'accuracy_pos_seed': accuracy_pos_item,
        'accuracy_seed': accuracy_item

    }


    # recall = recall_item.mean(0)
    # prec = prec_item.mean(0)
    # f1 = f1_item.mean(0)

    # ipdb.set_trace()
    return metrics_item_tstep, metrics_item

def update_metrics(metric_dict, args, losses_dict, gt, predictions, misc):
    
    gt_task, mask_task = gt['gt_task'], gt['gt_mask']
    pred_mask_task, pred_task = predictions['pred_mask'], predictions['pred_task']
    mask_length = misc['mask_length'] 
    for loss_name, loss_val in losses_dict.items():
        metric_dict[loss_name].update(loss_val)

        # ipdb.set_trace()
    (
        change_prec, change_recall, task_accuracy, task_accuracy_change
    ) = compute_metrics_change(
        gt_task, pred_task, mask_task, (pred_mask_task > 0.), mask_length
        )
    metric_dict['prec_change'].update(change_prec.item())
    metric_dict['recall_change'].update(change_recall.item())
    metric_dict['task_accuracy'].update(task_accuracy.item())
    metric_dict['task_accuracy_pos'].update(task_accuracy_change.item())
    if args.model.predict_category:
        # B x T
        category_accuracy = (gt['gt_category'].cuda() == predictions['predict_category'].argmax(-1)).float().mean()

        metric_dict['category_accuracy'].update(category_accuracy.item())



def compute_kl_loss(net_outputs, mask_len):

    # https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians
    mu_prior, logvar_prior, mu_posterior, logvar_posterior = net_outputs['vae_params']
    var_prior = logvar_prior.exp()
    var_posterior = logvar_posterior.exp()
    kl_per_dim = logvar_prior - logvar_posterior - 1 + var_posterior/var_prior + (mu_prior - mu_posterior)**2 / var_prior
    # mask_norm = mask_len / mask_len.sum(-1)[:, None]
    # res = torch.sum(kl_per_dim, -1) * mask_norm.cuda()
    res = kl_per_dim
    kldiv = 0.5 * res.sum(1).mean(0)
    # ipdb.set_trace()
    return kldiv


def compute_metrics_change(
    gt_task, pred_task, mask_task, pred_mask_task, mask_length
):
    
    # assert change_type in ["node", "edge"]
    # if change_type == "node":
    #     B, T, N = gt_state.shape[:3]
    #     gt_changes = gt_changes.reshape(B, T, N, N)
    #     gt_changes = (gt_changes.sum(-1) > 0).long() 


    # ipdb.set_trace()
    # How many GT positives

    pos_change = mask_task.sum(-1) + 1e-9

    # How many predicted positives
    # ipdb.set_trace()

    change_avg_pos = pred_mask_task.sum(-1)
    
    # ipdb.set_trace()
    tp_edge_change = (mask_task * pred_mask_task).sum(-1)
    
    change_recall = tp_edge_change / pos_change
    change_prec = tp_edge_change / (change_avg_pos + 1e-9)

    mask_task_norm = mask_task / (1e-9 + mask_task.sum(-1)[..., None])
    # Accuracy
    # ipdb.set_trace()
    task_pred = pred_task.argmax(-1).long()
    task_accuracy = (task_pred == gt_task).float().mean(-1)
    task_accuracy_change = ((task_pred == gt_task).float() * mask_task_norm).sum(-1)
    

    # Average over timesteps and batch
    mask_timesteps = mask_length / mask_length.sum(-1)[:, None]

    change_recall = (change_recall * mask_timesteps).sum(-1).mean()
    change_prec = (change_prec * mask_timesteps).sum(-1).mean()

    # ipdb.set_trace()

    task_accuracy = (task_accuracy * mask_timesteps).sum(-1).mean()
    task_accuracy_change = (task_accuracy_change * mask_timesteps).sum(-1).mean()  

    return change_prec, change_recall, task_accuracy, task_accuracy_change




def get_loaders(args):
    print("Loading dataset...")
    print("Train: {}".format(args['data']['train_data']))
    print("Test: {}".format(args['data']['test_data']))
    curr_file = os.path.dirname(get_original_cwd())
    first_last = False
    dataset = AgentTypeDataset(
        path_init='{}/agent_preferences/dataset/{}'.format(
            curr_file, args['data']['train_data'] 
        ),
        first_last=first_last,
        args_config=args,
    )
    if not args['train']['overfit']:
        dataset_test = AgentTypeDataset(
            path_init='{}/agent_preferences/dataset/{}'.format(
                curr_file, args['data']['test_data']
            ),
            first_last=first_last,
            args_config=args,
        )
    else:
        dataset_test = AgentTypeDataset(
            path_init='{}/agent_preferences/dataset/{}'.format(
                curr_file, args['data']['train_data']
            ),
            first_last=first_last,
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
        shuffle=False,
        num_workers=args['train']['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False
    )
    return train_loader, test_loader


@hydra.main(config_path="..", config_name="inference")
def main(cfg: DictConfig):
    config = cfg
    print("Config")
    print(OmegaConf.to_yaml(cfg))


    # assert not (cfg.model.predict_edge_change)
    assert cfg['model']['exclusive_edge']

    cfg.model.input_goal = False

    
    train_loader, test_loader = get_loaders(config)
    config.model.num_task_preds = len(train_loader.dataset.graph_helper.task_graph_list)
    # ipdb.set_trace()
    # if config.model.input_goal:

    #     model = agent_pref_policy.GoalConditionedGraphPredNetwork(config)
    # else:
    #     if 'VAE' not in config.model.time_aggregate:
    #         model = agent_pref_policy.GraphPredNetwork(config)
    #     else:
    
    model = agent_pref_policy.GraphPredNetworkVAETask3(config)
    print("CUDA: {}".format(cfg.cuda))
    if cfg.cuda:
        model = model.cuda()
        model = nn.DataParallel(model)

    if len(cfg.ckpt_load) > 0:
        model.load_state_dict(torch.load(cfg.ckpt_load)['model'])

    # loss states
    
    criterion_task = torch.nn.CrossEntropyLoss(reduction='none')
    criterion_mask = torch.nn.BCEWithLogitsLoss(reduction='none')
    
    criterions = {
        'task': criterion_task,
        'mask': criterion_mask
    }

    if config.model.predict_category:
        criterions['category'] = criterion_task
    
    logger = LoggerSteps(config, log_steps=False)
    logger.results_path = 'results_inference/{}'.format(config.name_log)
        
    inference(
        test_loader,
        model,
        config,
        logger,
        criterions,
        False
    )

    inference(
        test_loader,
        model,
        config,
        logger,
        criterions,
        True
    )

    # ipdb.set_trace()





if __name__ == '__main__':
    main()
