import torch
import time
import os
import glob
import yaml
import pickle as pkl
from tqdm import tqdm
import ipdb
from dataloader.dataloader_v2 import AgentTypeDataset
from dataloader import dataloader_v2 as dataloader_v2
from arguments import *
from torch import nn
import torch.optim as optim
from models import agent_pref_policy
from hydra.utils import get_original_cwd, to_absolute_path
from termcolor import colored
import utils.utils_models_wb as utils_models
from utils.utils_models_wb import AverageMeter, ProgressMeter, LoggerSteps
import math
import hydra
from omegaconf import DictConfig, OmegaConf
import pathlib


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


def inference(
    data_loader,
    model,
    args,
    logger,
    criterion_state,
    criterion_edge,
    criterion_change=None,
):
    epoch = 0
    model.eval()


    metric_dict = get_metrics()

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
            # ipdb.set_trace()
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
            
            goal_graph = build_goal_graph(graph_info, len_mask)
            

            inputs['goal_graph'] = goal_graph

            edge_dict = utils_models.build_gt_edge(graph_info, data_loader.dataset.graph_helper,  exclusive_edge=args.model.exclusive_edge)
            gt_edges = edge_dict['gt_edges']
            edge_interest = edge_dict['edge_interest']

            gt_states = graph_info['states_objects']
            if args.model.predict_last:

                nt = gt_edges.shape[1]
                numnode = gt_edges.shape[-1]
                tsteps = (
                    len_mask.sum(-1)[:, None, None].repeat(1, 1, numnode).long() - 1
                )
                gt_edge = (
                    torch.gather(gt_edges.cuda(), 1, tsteps.cuda()).repeat(1, nt - 1, 1).cuda()
                )
                gt_state = goal_graph['states_objects'][:, 1:, :, :]
            else:
                gt_edge = gt_edges[:, 1:, ...].cuda()
                gt_state = gt_states[:, 1:, ...].cuda()

            with torch.no_grad():
                output = model(inputs)

            pred_edge = output['edges'][:, :-1, ...]
            if args.model.exclusive_edge:
                b, t, n, _ = output['states'].shape
                pred_edge = pred_edge.reshape([b, t-1, n, n])

            pred_state = output['states'][:, :-1, ...]
            mask_state = graph_info['mask_object'][:, 1:, :, None].cuda()
            mask_length = len_mask[:, 1:].cuda()

            loss_state = criterion_state(
                output['states'][:, :-1, ...],
                graph_info['states_objects'][:, 1:, ...].cuda(),
            )
            loss_state = loss_state * graph_info['mask_object'][:, 1:, :, None].cuda()
            loss_state = loss_state.mean()

            # loss edges edges in prediction are stored as a B x Time x N x N x num_edge_class tensor
            # GT is stored as B x Time x Num_edges, we need to convert
            num_nodes = output['states'].shape[-2]


            mask_obs_node = graph_info['mask_obs_node']

            medges1 = mask_obs_node.repeat([1, 1, num_nodes]).cuda()
            medges2 = mask_obs_node.repeat_interleave(num_nodes, dim=2).cuda()
            

            if args.model.exclusive_edge:
                mask_edges = mask_obs_node.cuda()
            else:
                mask_edges = medges1 * medges2

            mask_edges = mask_edges[:, 1:, ...]

            predicted_edge = nn.functional.softmax(pred_edge, dim=-1).cpu().numpy()
            predicted_state = pred_state.cpu()
            try:
                gt_edge_onehot = torch.nn.functional.one_hot(
                    gt_edge, predicted_edge.shape[-1]
                )
            except:
                ipdb.set_trace()

            loss = 0
            pred_changes_list, edge_changes_list = [], []
            mask_edges_orig = mask_edges

            
            pred_changes = None
            changed_edges = None
            changed_nodes = None
            if args.model.predict_node_change and args.model.exclusive_edge:
                changed_edges_pre = (gt_edge != gt_edges[:, :-1, :].cuda()).long().cuda()
                changed_edges_pre *= mask_obs_node[:, 1:, ...].long().cuda()
                pred_changes = output['node_change'][:, :-1, ...]

                # Only care about the from edges
                changed_nodes = changed_edges_pre


                loss_change = criterion_change(
                    pred_changes.permute(0, 3, 1, 2), changed_nodes
                )
                loss_change = loss_change * mask_obs_node[:, 1:, :].cuda()
                loss_change = loss_change.mean()
                
                metric_dict['losses_change'].update(loss_change.item())

                # Only loss for changed edges
                loss += loss_change


                gt_edges_onehot = torch.nn.functional.one_hot(
                    gt_edges, predicted_edge.shape[-2]
                )



                changed_nodes_onehot = torch.nn.functional.one_hot(
                    changed_nodes, 2
                )

                # ipdb.set_trace()
                pred_changes_list = [
                    (nn.functional.softmax(pred_changes, dim=3)).cpu().numpy(),
                    gt_edges_onehot[:, :-1, :].cpu().numpy(),
                ]

                edge_changes_list = [
                    changed_nodes_onehot.cpu().numpy(),
                    gt_edges_onehot[:, :-1, :].cpu().numpy(),
                ]


                loss_edges = criterion_edge(pred_edge.permute(0, 3, 1, 2), gt_edge)
                loss_edges = loss_edges * changed_nodes


            loss_edges = loss_edges.mean()

            loss += loss_edges + loss_state
            metric_dict['losses'].update(loss.item())
            metric_dict['losses_state'].update(loss_state.item())
            metric_dict['losses_edge'].update(loss_edges.item())


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
            for index in range(1): #ind.shape[0]):
                try:
                    program_gt = utils_models.decode_program(
                        data_loader.dataset.graph_helper, prog_gt, index=index
                    )
                except:
                    continue
                current_index = ind[index]
                fname = data_loader.dataset.pkl_files[current_index]
                pred_graph = utils_models.obtain_graph_3_2(
                    data_loader.dataset.graph_helper,
                    graph_info,
                    predicted_edge,
                    predicted_state,
                    mask_edges_orig.cpu(),
                    pred_changes_list,
                    len_mask,
                    index,
                    samples=args.samples_per_graph if args.inference_sample else None,
                )

                gt_graph = utils_models.obtain_graph_3_2(
                    data_loader.dataset.graph_helper,
                    graph_info,
                    gt_edge_onehot.cpu().numpy(),
                    gt_state.cpu(),
                    mask_edges_orig.cpu(),
                    edge_changes_list,
                    len_mask,
                    index
                )

                print("************************")
                print(f"File: {current_index}:{fname}")
                print("\nGroundTrurth")
                utils_models.print_graph_3_2(
                    data_loader.dataset.graph_helper,
                    graph_info,
                    gt_edge.cpu().numpy(),
                    mask_edges_orig.cpu().numpy(),
                    gt_state.cpu().numpy(),
                    [edge_changes_list[0].argmax(-1), edge_changes_list[1].argmax(-1)] if len(edge_changes_list) > 0 else [],
                    index,
                    0,
                )
                tsteps =  int(len_mask[index].sum()) - 1

                for t in [0, tsteps//2, tsteps-1]:
                    print("\nInput at {}".format(t))
                    utils_models.print_graph_3_2(
                        data_loader.dataset.graph_helper,
                        graph_info,
                        gt_edges.cpu().numpy(),
                        mask_edges_orig.cpu().numpy(),
                        gt_states.cpu().numpy(),
                        [],
                        index,
                        t,
                    )


                    print("\nPrediction at {}".format(t))
                    # ipdb.set_trace()c
                    utils_models.print_graph_3_2(
                        data_loader.dataset.graph_helper,
                        graph_info,
                        pred_edge.argmax(-1).cpu().numpy(),
                        mask_edges_orig.cpu().numpy(),
                        (pred_state > 0).cpu().numpy(),
                        [pred_changes_list[0].argmax(-1), 
                            edge_changes_list[1].argmax(-1) if len(pred_changes_list) > 0 else []],
                        index,
                        t,
                    )
                print("************************")
                
                results = {'gt_graph': gt_graph, 'pred_graph': pred_graph}
                sfname = fname.split('/')[-1] + "_result"
                expath = logger.results_path
                cpath = '/'.join(fname.split('/')[-3:-1])
                dir_name = f'{expath}/{cpath}/'
                result_name = f'{dir_name}/{sfname}.pkl'
                if args.save_inference:
                    if not os.path.isdir(dir_name):
                        os.makedirs(dir_name)

                    with open(result_name, 'wb') as f:
                       pkl.dump(results, f)

                # ipdb.set_trace()
            # Update accuracy
            update_metrics(metric_dict, args, gt_state, gt_edge, pred_state, pred_edge, mask_state, mask_length, 
                           mask_edges, changed_edges, changed_nodes, pred_changes, edge_interest, edge_dict['id_nothing'].cuda())

            progress.display(it)
            ipdb.set_trace()
            # ipdb.set_trace()
            metric_dict['batch_time'].update(time.time() - end)
            end = time.time()

        else:
            continue

        progress.display(it)

def get_metrics():
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
    return metric_dict

def evaluate(
    data_loader,
    data_loader_train,
    model,
    epoch,
    args,
    logger,
    criterion_state,
    criterion_edge,
    criterion_change=None,
):
    model.eval()


    metric_dict = get_metrics()
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
            # ipdb.set_trace()
            inputs = {
                'program': program,
                'graph': graph_info,
                'mask_len': len_mask,
                'goal': goal,
                'label_agent': label_agent,
            }
            goal_graph = build_goal_graph(graph_info, len_mask)
            inputs['goal_graph'] = goal_graph

            edge_dict = utils_models.build_gt_edge(graph_info, data_loader.dataset.graph_helper,  exclusive_edge=args.model.exclusive_edge)
            gt_edges = edge_dict['gt_edges']
            edge_interest = edge_dict['edge_interest']

            gt_states = graph_info['states_objects']
            if args.model.predict_last:

                nt = gt_edges.shape[1]
                numnode = gt_edges.shape[-1]
                tsteps = (
                    len_mask.sum(-1)[:, None, None].repeat(1, 1, numnode).long() - 1
                )
                gt_edge = (
                    torch.gather(gt_edges, 1, tsteps).repeat(1, nt - 1, 1).cuda()
                )

                gt_state = goal_graph['states_objects'][:, 1:, :, :]

            else:
                gt_edge = gt_edges[:, 1:, ...].cuda()
                gt_state = gt_states[:, 1:, ...].cuda()

            with torch.no_grad():
                output = model(inputs)


            pred_edge = output['edges'][:, :-1, ...]

            if args.model.exclusive_edge:
                b, t, n, _ = output['states'].shape
                pred_edge = pred_edge.reshape([b, t-1, n, n])



            pred_state = output['states'][:, :-1, ...]
            mask_state = graph_info['mask_object'][:, 1:, :, None].cuda()
            mask_length = len_mask[:, 1:].cuda()

            loss_state = criterion_state(
                output['states'][:, :-1, ...],
                graph_info['states_objects'][:, 1:, ...].cuda(),
            )
            loss_state = loss_state * graph_info['mask_object'][:, 1:, :, None].cuda()
            loss_state = loss_state.mean()

            # loss edges edges in prediction are stored as a B x Time x N x N x num_edge_class tensor
            # GT is stored as B x Time x Num_edges, we need to convert
            num_nodes = output['states'].shape[-2]

            mask_obs_node = graph_info['mask_obs_node']

            medges1 = mask_obs_node.repeat([1, 1, num_nodes]).cuda()
            medges2 = mask_obs_node.repeat_interleave(num_nodes, dim=2).cuda()

            if args.model.exclusive_edge:
                mask_edges = mask_obs_node.cuda()
            else:
                mask_edges = medges1 * medges2

            mask_edges = mask_edges[:, 1:, ...]

            
            loss = 0

            pred_changes_list, edge_changes_list = [], []
            mask_edges_orig = mask_edges

            pred_changes = None
            changed_edges = None
            changed_nodes = None

            if args.model.predict_edge_change:
                changed_edges = (gt_edge != gt_edges[:, :-1, :].cuda()).long().cuda()
                pred_changes = output['edge_change'][
                    :, :-1, ...
                ]  # TODO: is this correct? Xavi: correct
                loss_change = criterion_change(
                    pred_changes.permute(0, 3, 1, 2), changed_edges
                )
                loss_change = loss_change * mask_edges
                loss_change = loss_change.mean()
                
                metric_dict['losses_change'].update(loss_change.item())

                # Only loss for changed edges
                loss += loss_change

                pred_changes_list = [
                    (pred_changes[...].argmax(-1)).cpu().long(),
                    gt_edges[:, :-1, :].cpu(),
                ]
                edge_changes_list = [changed_edges.cpu(), gt_edges[:, :-1, :].cpu()]

                loss_edges = criterion_edge(pred_edge.permute(0, 3, 1, 2), gt_edge)
                loss_edges = loss_edges * changed_edges

            elif args.model.predict_node_change:
                
                # A node is if the from_edge is changed
                # An edge is changed if any of the nodes is changed
                
                if args['model']['exclusive_edge']:
                    # ipdb.set_trace()
                    changed_edges_pre = (gt_edge != gt_edges[:, :-1, :].cuda()).long().cuda()
                    changed_edges_pre *= mask_obs_node[:, 1:, ...].long().cuda()
                    pred_changes = output['node_change'][:, :-1, ...]
                    
                    # Only care about the from edges
                    changed_nodes = changed_edges_pre

                    loss_change = criterion_change(
                        pred_changes.permute(0, 3, 1, 2), changed_nodes
                    )
                    loss_change = loss_change * mask_obs_node[:, 1:, :].cuda()
                    loss_change = loss_change.mean()
                    metric_dict['losses_change'].update(loss_change.item())
                    loss += loss_change

                    pred_changes_list = [
                        (pred_changes[...].argmax(-1)).cpu().long(),
                        gt_edges[:, :-1, :].cpu(),
                    ]
                    edge_changes_list = [changed_nodes.cpu(), gt_edges[:, :-1, :].cpu()]

                    loss_edges = criterion_edge(pred_edge.permute(0, 3, 1, 2), gt_edge)
                    loss_edges = loss_edges * changed_nodes

                else:

                    changed_edges_pre = (gt_edge != gt_edges[:, :-1, :].cuda()).long().cuda()
                    changed_edges_pre *= mask_edges.long()

                    B, T, N = output['states'].shape[:3]
                    changed_edges_from_to = changed_edges_pre.reshape([B, T-1, N, N])
                    changed_nodes = (changed_edges_from_to.sum(-1)  > 0).long()
                    changed_edges = changed_nodes.repeat_interleave(N, dim=2)



                    pred_changes = output['node_change'][:, :-1, ...]  
                    loss_change = criterion_change(
                        pred_changes.permute(0, 3, 1, 2), changed_nodes
                    )
                    loss_change = loss_change * mask_obs_node[:, 1:, :].cuda()
                    loss_change = loss_change.mean()
                    
                    metric_dict['losses_change'].update(loss_change.item())

                    # Only loss for changed edges
                    loss += loss_change

                    pred_changes_list = [
                        (pred_changes[...].argmax(-1)).cpu().long(),
                        gt_edges[:, :-1, :].cpu(),
                    ]
                    edge_changes_list = [changed_edges.cpu(), gt_edges[:, :-1, :].cpu()]
                    
                    loss_edges = criterion_edge(pred_edge.permute(0, 3, 1, 2), gt_edge)
                    loss_edges = loss_edges * changed_edges
            else:
                loss_edges = criterion_edge(pred_edge.permute(0, 3, 1, 2), gt_edge)
                loss_edges = loss_edges * mask_edges   
            
            loss_edges = loss_edges.mean()
            loss += loss_edges + loss_state
            metric_dict['losses'].update(loss.item())
            metric_dict['losses_state'].update(loss_state.item())
            metric_dict['losses_edge'].update(loss_edges.item())

            for index in range(1):
                current_index = ind[index]
                fname = data_loader.dataset.pkl_files[current_index]

                if True:
                    print("************************")
                    print(f"File: {current_index}:{fname}")
                    print("\nGroundTrurth")
                    utils_models.print_graph_3_2(
                        data_loader.dataset.graph_helper,
                        graph_info,
                        gt_edge.cpu(),
                        mask_edges_orig.cpu(),
                        gt_state.cpu(),
                        edge_changes_list,
                        index,
                        0,
                    )

                    print("\nPrediction")
                    utils_models.print_graph_3_2(
                        data_loader.dataset.graph_helper,
                        graph_info,
                        pred_edge.argmax(-1).cpu(),
                        mask_edges_orig.cpu(),
                        (pred_state > 0).cpu(),
                        pred_changes_list,
                        index,
                        0,
                    )

                    print("************************")
                    # ipdb.set_trace()

            # Update accuracy
            update_metrics(metric_dict, args, gt_state, gt_edge, pred_state, pred_edge, mask_state, mask_length, 
                           mask_edges, changed_edges, changed_nodes, pred_changes, edge_interest, edge_dict['id_nothing'].cuda())
            

            # ipdb.set_trace()
            metric_dict['batch_time'].update(time.time() - end)
            end = time.time()

        else:
            continue

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
            'edge_accuracy_interest_pos_val': metric_dict['accuracy_edge_interest_pos'].avg
        },
        'misc': {'epoch': epoch},
    }
    if args.model.predict_edge_change:
        info_log['losses']['loss_change_val'] = metric_dict['losses_change'].avg
        info_log['accuracy']['edge_prec_change_val'] = metric_dict['prec_change'].avg
        info_log['accuracy']['edge_recall_change_val'] = metric_dict['recall_change'].avg

    if args.model.predict_node_change:
        info_log['losses']['loss_change_node_val'] = metric_dict['losses_change'].avg
        info_log['accuracy']['node_prec_change_val'] = metric_dict['prec_change'].avg
        info_log['accuracy']['node_recall_change_val'] = metric_dict['recall_change'].avg

    logger.log_data(len(data_loader_train) * epoch, info_log)


def update_metrics(metric_dict, args, gt_state, gt_edge, pred_state, pred_edge, mask_state, mask_length, 
                mask_edges, changed_edges, changed_nodes, pred_changes, edge_interest, id_nothing):

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
            changed_edges if args.model.predict_edge_change else changed_nodes,
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
    criterion_state,
    criterion_edge,
    criterion_change=None,
):
    print(colored(f"Training epoch {epoch}", "yellow"))

    metric_dict = get_metrics()

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
        # print(colored("graph", "yellow"))

        # utils_models.print_graph(data_loader.dataset.graph_helper, graph_info, 0, 0)

        # utils_models.print_graph(data_loader.dataset.graph_helper, graph_info, 0, 1)

        # utils_models.print_graph(data_loader.dataset.graph_helper, graph_info, 0, 2)
        # ipdb.set_trace()

        inputs = {
            'program': program,
            'graph': graph_info,
            'mask_len': len_mask,
            'goal': goal,
            'label_agent': label_agent,
        }

        # print(graph_info['mask_object'].shape)
        # print(graph_info['class_objects'].shape)
        # print(graph_info['object_coords'].shape)
        # print(graph_info['states_objects'].shape)
        # print(len_mask.shape)

        # print(inputs['graph']['graph'], inputs['graph']['mask_object'].sum())

        goal_graph = build_goal_graph(graph_info, len_mask)
        inputs['goal_graph'] = goal_graph

        edge_dict = utils_models.build_gt_edge(graph_info, data_loader.dataset.graph_helper, exclusive_edge=args.model.exclusive_edge)
        gt_edges = edge_dict['gt_edges']
        edge_interest = edge_dict['edge_interest']

        if args.model.predict_last:

            nt = gt_edges.shape[1]
            numnode = gt_edges.shape[-1]
            tsteps = len_mask.sum(-1)[:, None, None].repeat(1, 1, numnode).long() - 1
            gt_edge = (
                torch.gather(gt_edges, 1, tsteps).repeat(1, nt - 1, 1).cuda()
            )
            
            gt_state = goal_graph['states_objects'][:, 1:, ...].cuda()

        else:
            gt_edge = gt_edges[:, 1:, ...].cuda()
            gt_state = graph_info['states_objects'][:, 1:, ...].cuda()

        output = model(inputs)
        
        pred_edge = output['edges'][:, :-1, ...]

        if args.model.exclusive_edge:
            b, t, n, _ = output['states'].shape
            pred_edge = pred_edge.reshape([b, t-1, n, n])

        pred_state = output['states'][:, :-1, ...]
        mask_state = graph_info['mask_object'][:, 1:, :, None].cuda()
        mask_length = len_mask[:, 1:].cuda()

        loss_state = criterion_state(pred_state, gt_state)
        loss_state = loss_state * mask_state
        loss_state = loss_state.mean()

        # loss edges edges in prediction are stored as a B x Time x N x N x num_edge_class tensor
        # GT is stored as B x Time x Num_edges, we need to convert
        num_nodes = output['states'].shape[-2]

        mask_obs_node = graph_info['mask_obs_node']

        medges1 = mask_obs_node.repeat([1, 1, num_nodes]).cuda()
        medges2 = mask_obs_node.repeat_interleave(num_nodes, dim=2).cuda()

        if args.model.exclusive_edge:
            mask_edges = mask_obs_node.cuda()
        else:
            mask_edges = medges1 * medges2
            
        mask_edges = mask_edges[:, 1:, ...]

        loss = 0

        pred_changes = None
        changed_edges = None
        changed_nodes = None

        if args.model.predict_edge_change:
            if args.model.exclusive_edge:
                raise Exception

            changed_edges = (gt_edge != gt_edges[:, :-1, :].cuda()).long()
            changed_edges *= mask_edges.long()

            pred_changes = output['edge_change'][:, :-1, ...]  # TODO: is this correct?
            loss_change = criterion_change(
                pred_changes.permute(0, 3, 1, 2), changed_edges
            )
            if torch.any(torch.isnan(loss_change)):
                print(loss_change)
                ipdb.set_trace()

            if torch.any(torch.isnan(mask_edges)):
                print(mask_edges)
                ipdb.set_trace()

            loss_change = loss_change * mask_edges
            loss_change = loss_change.mean()

            metric_dict['losses_change'].update(loss_change.item())

            # Only loss for changed edges
            loss += loss_change
            loss_edges = criterion_edge(pred_edge.permute(0, 3, 1, 2), gt_edge)
            loss_edges = loss_edges * changed_edges

        elif args.model.predict_node_change:
                
            # A node is if the from_edge is changed
            # An edge is changed if any of the nodes is changed

            if args['model']['exclusive_edge']:
                # ipdb.set_trace()
                changed_edges_pre = (gt_edge != gt_edges[:, :-1, :].cuda()).long().cuda()
                changed_edges_pre *= mask_obs_node[:, 1:, ...].long().cuda()
                pred_changes = output['node_change'][:, :-1, ...]
                
                # Only care about the from edges
                changed_nodes = changed_edges_pre

                loss_change = criterion_change(
                    pred_changes.permute(0, 3, 1, 2), changed_nodes
                )
                loss_change = loss_change * mask_obs_node[:, 1:, :].cuda()
                loss_change = loss_change.mean()
                metric_dict['losses_change'].update(loss_change.item())
                loss += loss_change

                pred_changes_list = [
                    (pred_changes[...].argmax(-1)).cpu().long(),
                    gt_edges[:, :-1, :].cpu(),
                ]
                edge_changes_list = [changed_nodes.cpu(), gt_edges[:, :-1, :].cpu()]

                # ipdb.set_trace()
                loss_edges = criterion_edge(pred_edge.permute(0, 3, 1, 2), gt_edge)
                loss_edges = loss_edges * changed_nodes

            else:
                changed_edges_pre = (gt_edge != gt_edges[:, :-1, :].cuda()).long().cuda()
                changed_edges_pre *= mask_edges.long()

                B, T, N = output['states'].shape[:3]
                changed_edges_from_to = changed_edges_pre.reshape([B, T-1, N, N])
                changed_nodes = (changed_edges_from_to.sum(-1)  > 0).long()
                # changed_edges = changed_nodes.repeat(1, 1, N)
                changed_edges = changed_nodes.repeat_interleave(N, dim=2)

                # ipdb.set_trace()

                pred_changes = output['node_change'][:, :-1, ...]
                # ipdb.set_trace()
                loss_change = criterion_change(
                    pred_changes.permute(0, 3, 1, 2), changed_nodes
                )
                loss_change = loss_change * mask_obs_node[:, 1:, :].cuda()
                loss_change = loss_change.mean()
                
                metric_dict['losses_change'].update(loss_change.item())

                # Only loss for changed edges
                # ipdb.set_trace()
                loss += loss_change

                pred_changes_list = [
                    (pred_changes[...].argmax(-1)).cpu().long(),
                    gt_edges[:, :-1, :].cpu(),
                ]
                edge_changes_list = [changed_edges.cpu(), gt_edges[:, :-1, :].cpu()]
                # ipdb.set_trace()

                loss_edges = criterion_edge(pred_edge.permute(0, 3, 1, 2), gt_edge)
                loss_edges = loss_edges * changed_edges
        else:        
            loss_edges = criterion_edge(pred_edge.permute(0, 3, 1, 2), gt_edge)
            loss_edges = loss_edges * mask_edges
        

        loss_edges = loss_edges.mean()
        # ipdb.set_trace()
        loss += loss_edges + loss_state

        metric_dict['losses'].update(loss.item())
        metric_dict['losses_state'].update(loss_state.item())
        metric_dict['losses_edge'].update(loss_edges.item())

        
        update_metrics(metric_dict, args, gt_state, gt_edge, pred_state, pred_edge, mask_state, mask_length, 
                           mask_edges, changed_edges, changed_nodes, pred_changes, edge_interest, edge_dict['id_nothing'].cuda())
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

            if args.model.predict_edge_change:
                info_log['losses']['loss_change'] = metric_dict['losses_change'].val

            if args.model.predict_node_change:
                info_log['losses']['loss_change_node'] = metric_dict['losses_change'].val
            

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
    )

    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args['train']['batch_size'],
        shuffle=not args.inference,
        num_workers=args['train']['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn,
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

    train_loader, test_loader = get_loaders(config)
    if config.model.input_goal:
        model = agent_pref_policy.GoalConditionedGraphPredNetwork(config)
    else:
        model = agent_pref_policy.GraphPredNetwork(config)

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
            criterion_state,
            criterion_edge,
            criterion_change
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=config['train']['lr'])
        print("Failures: ", train_loader.dataset.get_failures())

        logger = LoggerSteps(config, log_steps=config.logging)

        if config.logging:
            logger.save_model(0, model, optimizer)

        # evaluate(test_loader, train_loader, model, 0, config, logger, criterion_state, criterion_edge)
        # ipdb.set_trace()

        # evaluate(
        #     test_loader,
        #     train_loader,
        #     model,
        #     0,
        #     config,
        #     logger,
        #     criterion_state,
        #     criterion_edge,
        #     criterion_change,
        # )
        for epoch in range(config['train']['epochs']):
            # ipdb.set_trace()
            train_epoch(
                train_loader,
                model,
                epoch,
                config,
                optimizer,
                logger,
                criterion_state,
                criterion_edge,
                criterion_change,
            )
            evaluate(
                test_loader,
                train_loader,
                model,
                epoch,
                config,
                logger,
                criterion_state,
                criterion_edge,
                criterion_change,
            )
            if epoch % config.log.save_every == 0:
                logger.save_model(epoch, model, optimizer)


if __name__ == '__main__':
    main()