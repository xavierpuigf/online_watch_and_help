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


def merge2d(tensor):
    dim = list(tensor.shape)
    return tensor.reshape([-1] + dim[2:])


def unmerge(tensor, firstdim):
    dim = list(tensor.shape)
    return tensor.reshape([firstdim, -1] + dim[1:])


# Convert adjacency list to adjacency matrix
def build_gt_edge(graph_info):
    batch, time, num_nodes = graph_info['mask_object'].shape
    gt_edges = torch.zeros([batch, time, num_nodes ** 2])
    # num_nodes = graph_info['mask_object'].shape[-1]

    # num_edges = gt_edges.shape[-1]
    edge_tuples = graph_info['edge_tuples']
    index_edges = edge_tuples[..., 0] * num_nodes + edge_tuples[..., 1]
    edge_types = graph_info['edge_classes']  # - 1
    # ipdb.set_trace()
    # gt_edges[..., index_edges.long()] = edge_types
    gt_edges = gt_edges.scatter(2, index_edges.long(), edge_types)
    gt_edges = gt_edges.long()
    # for it_edge in range(num_edges):
    #     index_edge = edge_types == it_edge
    #     index_edge_curr = index_edges[index_edge]
    #     gt_edges[..., index_edge_curr.long(), it_edge] = 1

    return gt_edges.cuda()


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
    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    losses_state = AverageMeter('LossState', ':.4e')
    losses_edge = AverageMeter('LossEdge', ':.4e')
    losses_change = AverageMeter('LossChange', ':.4e')
    prec_state = AverageMeter('Prec State', ':6.2f')
    recall_state = AverageMeter('Rec State', ':6.2f')
    prec_change = AverageMeter('Prec Change', ':6.2f')
    recall_change = AverageMeter('Rec Change', ':6.2f')
    accuracy_edge = AverageMeter('Accuracy Edge', ':6.2f')
    accuracy_edge_pos = AverageMeter('Accuracy Edge Pos', ':6.2f')

    if args.model.predict_edge_change:
        progress = ProgressMeter(
            len(data_loader),
            [
                batch_time,
                data_time,
                losses,
                losses_state,
                losses_edge,
                losses_change,
                prec_state,
                recall_state,
                prec_change,
                recall_change,
                accuracy_edge,
                accuracy_edge_pos,
            ],
            prefix="Epoch: [{}]".format(epoch),
        )
    else:
        progress = ProgressMeter(
            len(data_loader),
            [
                batch_time,
                data_time,
                losses,
                losses_state,
                losses_edge,
                prec_state,
                recall_state,
                accuracy_edge,
                accuracy_edge_pos,
            ],
            prefix="Epoch: [{}]".format(epoch),
        )

    end = time.time()
    for it, data_item in enumerate(data_loader):
        if it < args['test']['num_iters']:
            data_time.update(time.time() - end)

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
            with torch.no_grad():
                output = model(inputs)

            pred_edge = output['edges'][:, :-1, ...]
            pred_state = output['states'][:, :-1, ...]
            gt_state = graph_info['states_objects'][:, 1:, ...].cuda()
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

            pred_edge = output['edges'][:, :-1, ...]
            gt_edges = build_gt_edge(graph_info)

            if args.model.predict_last:
                nt = gt_edges.shape[1]
                numnode = gt_edges.shape[-1]
                tsteps = (
                    len_mask.sum(-1)[:, None, None].repeat(1, 1, numnode).long() - 1
                )
                gt_edge = (
                    torch.gather(gt_edges, 1, tsteps.cuda()).repeat(1, nt - 1, 1).cuda()
                )

            else:
                gt_edge = gt_edges[:, 1:, ...].cuda()
            inputs['goal_graph'] = gt_edge

            mask_obs_node = graph_info['mask_obs_node']

            medges1 = mask_obs_node.repeat([1, 1, num_nodes]).cuda()
            medges2 = mask_obs_node.repeat_interleave(num_nodes, dim=2).cuda()
            mask_edges = medges1 * medges2
            mask_edges = mask_edges[:, 1:, ...]

            predicted_edge = pred_edge.cpu()
            predicted_state = pred_state.cpu()
            try:
                gt_edge_onehot = torch.nn.functional.one_hot(
                    gt_edge, predicted_edge.shape[-1]
                )
            except:
                ipdb.set_trace()

            for index in range(ind.shape[0]):
                current_index = ind[index]
                fname = data_loader.dataset.pkl_files[current_index]
                pred_graph = utils_models.obtain_graph(
                    data_loader.dataset.graph_helper,
                    graph_info,
                    predicted_edge,
                    predicted_state,
                    mask_edges.cpu(),
                    index,
                    len_mask,
                )
                gt_graph = utils_models.obtain_graph(
                    data_loader.dataset.graph_helper,
                    graph_info,
                    gt_edge_onehot.cpu(),
                    gt_state.cpu(),
                    mask_edges.cpu(),
                    index,
                    len_mask,
                )
                results = {'gt_graph': gt_graph, 'pred_graph': pred_graph}
                sfname = fname.split('/')[-1] + "_result"
                expath = logger.results_path
                cpath = '/'.join(fname.split('/')[-3:-1])
                dir_name = f'{expath}/{cpath}/'
                result_name = f'{dir_name}/{sfname}.pkl'
                if not os.path.isdir(dir_name):
                    os.makedirs(dir_name)

                with open(result_name, 'wb') as f:
                    pkl.dump(results, f)

            loss = 0

            if args.model.predict_edge_change:
                changed_edges = (gt_edge != gt_edges[:, :-1, :]).long()
                pred_changes = output['edge_change'][
                    :, :-1, ...
                ]  # TODO: is this correct?
                loss_change = criterion_change(
                    pred_changes.permute(0, 3, 1, 2), changed_edges
                )
                loss_change = loss_change * mask_edges
                loss_change = loss_change.mean()
                losses_change.update(loss_change.item())

                # Only loss for changed edges
                mask_edges = mask_edges * changed_edges
                loss += loss_change

            loss_edges = criterion_edge(pred_edge.permute(0, 3, 1, 2), gt_edge)
            loss_edges = loss_edges * mask_edges
            loss_edges = loss_edges.mean()

            loss += loss_edges + loss_state
            losses.update(loss.item())
            losses_state.update(loss_state.item())
            losses_edge.update(loss_edges.item())

            # Update accuracy
            if args.model.predict_edge_change:
                (
                    state_prec,
                    state_recall,
                    change_prec,
                    change_recall,
                    edge_accuracy,
                    edge_accuracy_pos,
                ) = compute_metrics_change(
                    gt_state,
                    gt_edge,
                    pred_state,
                    mask_state,
                    pred_edge,
                    mask_length,
                    mask_edges,
                    changed_edges,
                    pred_changes,
                )
                prec_change.update(change_prec.item())
                recall_change.update(change_recall.item())
            else:
                (
                    state_prec,
                    state_recall,
                    edge_accuracy,
                    edge_accuracy_pos,
                ) = compute_metrics(
                    gt_state,
                    gt_edge,
                    pred_state,
                    mask_state,
                    pred_edge,
                    mask_length,
                    mask_edges,
                )

            prec_state.update(state_prec.item())
            recall_state.update(state_recall.item())
            accuracy_edge.update(edge_accuracy.item())
            accuracy_edge_pos.update(edge_accuracy_pos.item())

            # ipdb.set_trace()
            batch_time.update(time.time() - end)
            end = time.time()

        else:
            continue

        progress.display(it)


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
    batch_time = AverageMeter('Time', ':6.3f')
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    losses_state = AverageMeter('LossState', ':.4e')
    losses_edge = AverageMeter('LossEdge', ':.4e')
    losses_change = AverageMeter('LossChange', ':.4e')
    prec_state = AverageMeter('Prec State', ':6.2f')
    recall_state = AverageMeter('Rec State', ':6.2f')
    prec_change = AverageMeter('Prec Change', ':6.2f')
    recall_change = AverageMeter('Rec Change', ':6.2f')
    accuracy_edge = AverageMeter('Accuracy Edge', ':6.2f')
    accuracy_edge_pos = AverageMeter('Accuracy Edge Pos', ':6.2f')

    if args.model.predict_edge_change:
        progress = ProgressMeter(
            len(data_loader),
            [
                batch_time,
                data_time,
                losses,
                losses_state,
                losses_edge,
                losses_change,
                prec_state,
                recall_state,
                prec_change,
                recall_change,
                accuracy_edge,
                accuracy_edge_pos,
            ],
            prefix="Epoch: [{}]".format(epoch),
        )
    else:
        progress = ProgressMeter(
            len(data_loader),
            [
                batch_time,
                data_time,
                losses,
                losses_state,
                losses_edge,
                prec_state,
                recall_state,
                accuracy_edge,
                accuracy_edge_pos,
            ],
            prefix="Epoch: [{}]".format(epoch),
        )

    end = time.time()
    for it, data_item in enumerate(data_loader):
        if it < args['test']['num_iters']:
            data_time.update(time.time() - end)

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
            with torch.no_grad():
                output = model(inputs)

            pred_edge = output['edges'][:, :-1, ...]
            pred_state = output['states'][:, :-1, ...]
            gt_state = graph_info['states_objects'][:, 1:, ...].cuda()
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

            pred_edge = output['edges'][:, :-1, ...]
            gt_edges = build_gt_edge(graph_info)
            if args.model.predict_last:
                nt = gt_edges.shape[1]
                numnode = gt_edges.shape[-1]
                tsteps = (
                    len_mask.sum(-1)[:, None, None].repeat(1, 1, numnode).long() - 1
                )
                gt_edge = (
                    torch.gather(gt_edges, 1, tsteps.cuda()).repeat(1, nt - 1, 1).cuda()
                )

            else:
                gt_edge = gt_edges[:, 1:, ...].cuda()

            inputs['goal_graph'] = gt_edge

            mask_obs_node = graph_info['mask_obs_node']

            medges1 = mask_obs_node.repeat([1, 1, num_nodes]).cuda()
            medges2 = mask_obs_node.repeat_interleave(num_nodes, dim=2).cuda()
            mask_edges = medges1 * medges2
            mask_edges = mask_edges[:, 1:, ...]

            for index in range(1):
                current_index = ind[index]
                fname = data_loader.dataset.pkl_files[current_index]

                if True:
                    print("************************")
                    print(f"File: {current_index}:{fname}")
                    print("\nGroundTrurth")
                    utils_models.print_graph_2(
                        data_loader.dataset.graph_helper,
                        graph_info,
                        gt_edge.cpu(),
                        mask_edges.cpu(),
                        gt_state.cpu(),
                        index,
                        0,
                    )
                    print("\nPrediction")
                    utils_models.print_graph_2(
                        data_loader.dataset.graph_helper,
                        graph_info,
                        pred_edge.argmax(-1).cpu(),
                        mask_edges.cpu(),
                        (pred_state > 0).cpu(),
                        index,
                        0,
                    )

                    print("************************")
                    # ipdb.set_trace()
            loss_edges = criterion_edge(pred_edge.permute(0, 3, 1, 2), gt_edge)
            loss_edges = loss_edges * mask_edges
            loss_edges = loss_edges.mean()

            loss = 0

            if args.model.predict_edge_change:
                changed_edges = (gt_edge != gt_edges[:, :-1, :]).long()
                pred_changes = output['edge_change'][
                    :, :-1, ...
                ]  # TODO: is this correct?
                loss_change = criterion_change(
                    pred_changes.permute(0, 3, 1, 2), changed_edges
                )
                loss_change = loss_change * mask_edges
                loss_change = loss_change.mean()
                losses_change.update(loss_change.item())

                # Only loss for changed edges
                mask_edges = mask_edges * changed_edges
                loss += loss_change

            loss += loss_edges + loss_state
            losses.update(loss.item())
            losses_state.update(loss_state.item())
            losses_edge.update(loss_edges.item())

            # Update accuracy
            if args.model.predict_edge_change:
                (
                    state_prec,
                    state_recall,
                    change_prec,
                    change_recall,
                    edge_accuracy,
                    edge_accuracy_pos,
                ) = compute_metrics_change(
                    gt_state,
                    gt_edge,
                    pred_state,
                    mask_state,
                    pred_edge,
                    mask_length,
                    mask_edges,
                    changed_edges,
                    pred_changes,
                )
                prec_change.update(change_prec.item())
                recall_change.update(change_recall.item())
            else:
                (
                    state_prec,
                    state_recall,
                    edge_accuracy,
                    edge_accuracy_pos,
                ) = compute_metrics(
                    gt_state,
                    gt_edge,
                    pred_state,
                    mask_state,
                    pred_edge,
                    mask_length,
                    mask_edges,
                )

            prec_state.update(state_prec.item())
            recall_state.update(state_recall.item())
            accuracy_edge.update(edge_accuracy.item())
            accuracy_edge_pos.update(edge_accuracy_pos.item())

            # ipdb.set_trace()
            batch_time.update(time.time() - end)
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
            'total_val': losses.avg,
            'state_val': losses_state.avg,
            'edge_val': losses_edge.avg,
        },
        'accuracy': {
            'state_prec_val': prec_state.val,
            'state_recall_val': recall_state.avg,
            'edge_accuracy_val': accuracy_edge.avg,
            'edge_accuracy_pos_val': accuracy_edge_pos.avg,
        },
        'misc': {'epoch': epoch},
    }
    logger.log_data(len(data_loader_train) * epoch, info_log)


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

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    losses_state = AverageMeter('LossState', ':.4e')
    losses_edge = AverageMeter('LossEdge', ':.4e')
    losses_change = AverageMeter('LossChange', ':.4e')
    prec_state = AverageMeter('Prec State', ':6.2f')
    recall_state = AverageMeter('Rec State', ':6.2f')
    prec_change = AverageMeter('Prec Change', ':6.2f')
    recall_change = AverageMeter('Rec Change', ':6.2f')
    accuracy_edge = AverageMeter('Accuracy Edge', ':6.2f')
    accuracy_edge_pos = AverageMeter('Accuracy Edge Pos', ':6.2f')

    if args.model.predict_edge_change:
        progress = ProgressMeter(
            len(data_loader),
            [
                batch_time,
                data_time,
                losses,
                losses_state,
                losses_edge,
                losses_change,
                prec_state,
                recall_state,
                prec_change,
                recall_change,
                accuracy_edge,
                accuracy_edge_pos,
            ],
            prefix="Epoch: [{}]".format(epoch),
        )
    else:
        progress = ProgressMeter(
            len(data_loader),
            [
                batch_time,
                data_time,
                losses,
                losses_state,
                losses_edge,
                prec_state,
                recall_state,
                accuracy_edge,
                accuracy_edge_pos,
            ],
            prefix="Epoch: [{}]".format(epoch),
        )

    model.train()

    end = time.time()

    for it, data_item in enumerate(data_loader):
        data_time.update(time.time() - end)

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

        # print(len_mask.shape)

        label_action = program['action'][:, 1:]
        index_label_obj1 = program['indobj1'][:, 1:]
        index_label_obj2 = program['indobj2'][:, 1:]

        prog_gt = {
            'action': label_action,
            'o1': index_label_obj1,
            'o2': index_label_obj2,
            'graph': graph_info,
            'mask_len': len_mask,
        }
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
        # print(inputs['graph']['graph'], inputs['graph']['mask_object'].sum())
        output = model(inputs)

        pred_edge = output['edges'][:, :-1, ...]
        pred_state = output['states'][:, :-1, ...]
        gt_state = graph_info['states_objects'][:, 1:, ...].cuda()
        mask_state = graph_info['mask_object'][:, 1:, :, None].cuda()
        mask_length = len_mask[:, 1:].cuda()

        loss_state = criterion_state(pred_state, gt_state)
        loss_state = loss_state * mask_state
        loss_state = loss_state.mean()

        # loss edges edges in prediction are stored as a B x Time x N x N x num_edge_class tensor
        # GT is stored as B x Time x Num_edges, we need to convert
        num_nodes = output['states'].shape[-2]

        gt_edges = build_gt_edge(graph_info)

        if args.model.predict_last:
            nt = gt_edges.shape[1]
            numnode = gt_edges.shape[-1]
            tsteps = len_mask.sum(-1)[:, None, None].repeat(1, 1, numnode).long() - 1
            gt_edge = (
                torch.gather(gt_edges, 1, tsteps.cuda()).repeat(1, nt - 1, 1).cuda()
            )
        else:
            gt_edge = gt_edges[:, 1:, ...].cuda()

        inputs['goal_graph'] = gt_edge

        mask_obs_node = graph_info['mask_obs_node']

        medges1 = mask_obs_node.repeat([1, 1, num_nodes]).cuda()
        medges2 = mask_obs_node.repeat_interleave(num_nodes, dim=2).cuda()
        mask_edges = medges1 * medges2
        mask_edges = mask_edges[:, 1:, ...]

        loss = 0

        if args.model.predict_edge_change:
            changed_edges = (gt_edge != gt_edges[:, :-1, :]).long().cuda()
            pred_changes = output['edge_change'][:, :-1, ...]  # TODO: is this correct?
            loss_change = criterion_change(
                pred_changes.permute(0, 3, 1, 2), changed_edges
            )
            loss_change = loss_change * mask_edges
            loss_change = loss_change.mean()
            losses_change.update(loss_change.item())

            # Only loss for changed edges
            mask_edges = mask_edges * changed_edges
            loss += loss_change

        loss_edges = criterion_edge(pred_edge.permute(0, 3, 1, 2), gt_edge)
        loss_edges = loss_edges * mask_edges
        loss_edges = loss_edges.mean()

        loss += loss_edges + loss_state

        losses.update(loss.item())
        losses_state.update(loss_state.item())
        losses_edge.update(loss_edges.item())

        # Update accuracy
        if args.model.predict_edge_change:
            (
                state_prec,
                state_recall,
                change_prec,
                change_recall,
                edge_accuracy,
                edge_accuracy_pos,
            ) = compute_metrics_change(
                gt_state,
                gt_edge,
                pred_state,
                mask_state,
                pred_edge,
                mask_length,
                mask_edges,
                changed_edges,
                pred_changes,
            )
            prec_change.update(change_prec.item())
            recall_change.update(change_recall.item())
        else:
            (
                state_prec,
                state_recall,
                edge_accuracy,
                edge_accuracy_pos,
            ) = compute_metrics(
                gt_state,
                gt_edge,
                pred_state,
                mask_state,
                pred_edge,
                mask_length,
                mask_edges,
            )

        prec_state.update(state_prec.item())
        recall_state.update(state_recall.item())
        accuracy_edge.update(edge_accuracy.item())
        accuracy_edge_pos.update(edge_accuracy_pos.item())

        # ipdb.set_trace()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if it % args['log']['print_every'] == 0:
            progress.display(it)
        if it % args['log']['print_long_every'] == 0:
            # ipdb.set_trace()
            info_log = {
                'losses': {
                    'total': losses.val,
                    'state': losses_state.val,
                    'edge': losses_edge.val,
                },
                'accuracy': {
                    'state_prec': prec_state.val,
                    'state_recall': recall_state.val,
                    'edge_accuracy': accuracy_edge.val,
                    'edge_accuracy_pos': accuracy_edge_pos.val,
                },
                'misc': {'epoch': epoch},
            }
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
):

    # How many GT positives
    pos_state = gt_state.sum(-1).sum(-1) + 1e-9
    pos_edge_change = gt_changes.sum(-1) + 1e-9

    state_avg = gt_state / (pos_state[:, :, None, None])
    edge_avg = gt_changes / (pos_edge_change[:, :, None])
    # How many predicted positives
    # ipdb.set_trace()

    pred_changes_label = pred_changes.argmax(-1)
    edge_avg_pos = (pred_changes_label * mask_edges).sum(-1) + 1e-9
    state_avg_pos = ((pred_state > 0) * mask_state).sum(-1).sum(-1) + 1e-9

    tp_edge_change = (gt_changes * pred_changes_label).sum(-1)
    tp_state = (gt_state * (pred_state > 0)).sum(-1).sum(-1)

    # Recall
    state_recall = tp_state / pos_state
    change_recall = tp_edge_change / pos_edge_change
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

    mask_edges_pos = mask_edges.clone()
    mask_edges_pos[gt_edges == 0] = 0

    edge_acc = edge_pred == gt_edges
    mask_edge_norm_pos = mask_edges_pos / (mask_edges_pos.sum(-1)[..., None] + 1e-9)
    edge_accuracy_pos = (edge_acc * mask_edge_norm_pos).sum(-1)
    edge_accuracy_pos = (edge_accuracy_pos * mask_timesteps).sum(-1).mean()
    return (
        state_prec,
        state_recall,
        change_prec,
        change_recall,
        edge_accuracy,
        edge_accuracy_pos,
    )


def compute_metrics(
    gt_state, gt_edges, pred_state, mask_state, pred_edges, mask_length, mask_edges
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
    return state_prec, state_recall, edge_accuracy, edge_accuracy_pos


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
    dataset_test = AgentTypeDataset(
        path_init='{}/agent_preferences/dataset/{}'.format(
            curr_file, args['data']['test_data']
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
        shuffle=True,
        num_workers=args['train']['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return train_loader, test_loader


@hydra.main(config_path="../config/agent_pred_graph", config_name="config_default_toy")
def main(cfg: DictConfig):
    config = cfg
    print("Config")
    print(OmegaConf.to_yaml(cfg))
    # ipdb.set_trace()

    train_loader, test_loader = get_loaders(config)
    if config.model.gated:
        model = agent_pref_policy.GoalConditionedGraphPredNetwork(config)
    else:
        model = agent_pref_policy.GoalConditionedGraphPredNetwork(config)

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
        logger = LoggerSteps(config)
        inference(
            test_loader,
            model,
            config,
            logger,
            criterion_state,
            criterion_edge,
            criterion_change,
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=config['train']['lr'])
        print("Failures: ", train_loader.dataset.get_failures())

        logger = LoggerSteps(config)

        logger.save_model(0, model, optimizer)

        # evaluate(test_loader, train_loader, model, 0, config, logger, criterion_state, criterion_edge)
        # ipdb.set_trace()

        for epoch in range(config['train']['epochs']):
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
            if epoch % 10 == 0:
                logger.save_model(epoch, model, optimizer)


if __name__ == '__main__':
    main()
