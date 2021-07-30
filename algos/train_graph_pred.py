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

import utils.utils_models_wb as utils_models
from utils.utils_models_wb import AverageMeter, ProgressMeter, LoggerSteps

import hydra
from omegaconf import DictConfig, OmegaConf
import pathlib


def merge2d(tensor):
    dim = list(tensor.shape)
    return tensor.reshape([-1]+dim[2:])

def unmerge(tensor, firstdim):
    dim = list(tensor.shape)
    return tensor.reshape([firstdim, -1] + dim[1:])

def evaluate(data_loader, data_loader_train, model, epoch, args, criterion, logger):
    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    losses_state = AverageMeter('LossState', ':.4e')
    losses_edge = AverageMeter('LossState', ':.4e')

    

    prec_state = AverageMeter('Prec State', ':6.2f')
    prec_edge = AverageMeter('Prec Edge', ':6.2f')
    
    recall_state = AverageMeter('Rec State', ':6.2f')
    recall_edge = AverageMeter('Rec Edge', ':6.2f')

    progress = ProgressMeter(
            len(data_loader),
            [batch_time, data_time, losses, losses_state, losses_edge, prec_state, recall_state, prec_edge, recall_edge],
        prefix="Epoch: [{}]".format(epoch))




    end = time.time()
    for it, data_item in enumerate(data_loader):
        if it < args['test']['num_iters']:
            data_time.update(time.time() - end)


            graph_info, program, label, len_mask, goal, label_agent, real_label_agent = data_item
            # ipdb.set_trace()
            inputs = {
                'program': program,
                'graph': graph_info,
                'mask_len': len_mask,
                'goal': goal,
                'label_agent': label_agent

            }
            with torch.no_grad():
                output = model(inputs)


            pred_edge = output['edges'][:, :-1, ...]
            pred_state = output['states'][:, :-1, ...]
            gt_state = graph_info['states_objects'][:, 1:, ...].cuda()
            mask_state = graph_info['mask_object'][:, 1:, :, None].cuda()
            mask_length = len_mask[:, 1:].cuda()


            # loss states
            criterion_state = torch.nn.BCEWithLogitsLoss(reduction='none')
            loss_state = criterion_state(output['states'][:, :-1, ...], graph_info['states_objects'][:, 1:, ...].cuda()) 
            loss_state = loss_state *  graph_info['mask_object'][:, 1:, :, None].cuda()
            loss_state = loss_state.mean()

            # loss edges edges in prediction are stored as a B x Time x N x N x num_edge_class tensor
            # GT is stored as B x Time x Num_edges, we need to convert
            num_nodes = output['states'].shape[-2]
            gt_edges = torch.zeros_like(output['edges'])
            num_edges = gt_edges.shape[-1]
            edge_tuples = graph_info['edge_tuples']
            index_edges = edge_tuples[..., 0] * num_nodes + edge_tuples[..., 1]
            edge_types = graph_info['edge_classes'] - 1
            for it_edge in range(num_edges):
                index_edge = edge_types == it_edge
                index_edge_curr = index_edges[index_edge]
                gt_edges[..., index_edge_curr.long(), it_edge] = 1


            gt_edge = gt_edges[:, 1:, ...].cuda()

            mask_obs_node = graph_info['mask_obs_node']
            
            medges1 = mask_obs_node.repeat([1, 1, num_nodes]).cuda()
            medges2 = mask_obs_node.repeat_interleave(num_nodes, dim=2).cuda()
            mask_edges = medges1 * medges2
            mask_edges = mask_edges[:, 1:, ..., None]
            loss_edges = criterion_state(output['edges'][:, :-1, ...], gt_edges[:, 1:, ...].cuda()) 
            loss_edges = loss_edges * mask_edges
            loss_edges = loss_edges.mean()


            loss = loss_edges + loss_state
            losses.update(loss.item())
            losses_state.update(loss_state.item())
            losses_edge.update(loss_edges.item())


            # How many GT positives
            pos_state = gt_state.sum(-1).sum(-1) + 1e-9
            pos_edge = gt_state.sum(-1).sum(-1) + 1e-9

            state_avg = gt_state / (pos_state[:, :, None, None])
            edge_avg = gt_edge / (pos_edge[:, :, None, None])
            # How many predicted positives
            # ipdb.set_trace()
            edge_avg_pos = ((pred_edge > 0.5) * mask_edges).sum(-1).sum(-1) + 1e-9
            state_avg_pos = ((pred_state > 0.5) * mask_state).sum(-1).sum(-1) + 1e-9

            tp_edge = (gt_edge * (pred_edge > 0.5)).sum(-1).sum(-1)
            tp_state = (gt_state * (pred_state > 0.5)).sum(-1).sum(-1)

            # Recall
            edge_recall = tp_edge / pos_edge
            state_recall = tp_state / pos_state
            
            # Precision
            edge_prec = (tp_edge / edge_avg_pos)
            state_prec = (tp_state / state_avg_pos)


            # Average over timesteps and batch
            mask_timesteps = mask_length / mask_length.sum(-1)[:, None]
            edge_recall = (edge_recall * mask_timesteps).sum()
            state_recall = (state_recall * mask_timesteps).sum()
            edge_prec = (edge_prec * mask_timesteps).sum()
            state_prec = (state_prec * mask_timesteps).sum()
            prec_state.update(state_prec.item())
            recall_state.update(state_recall.item())
            prec_edge.update(edge_prec.item())
            recall_edge.update(edge_recall.item())
                
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
                'losses': {'total_val': losses.avg, 'state_val': losses_state.avg, 'edge_val': losses_edge.avg},
                'accuracy': {'state_prec_val': prec_state.val, 'edge_prec_val': prec_edge.avg, 'state_recall_val': recall_state.avg, 'edge_recall_val': recall_edge.avg},
                'misc': {'epoch': epoch}
            }
    logger.log_data(len(data_loader_train) * epoch, info_log)



def train_epoch(data_loader, model, epoch, args, criterion, optimizer, logger):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    losses_state = AverageMeter('LossState', ':.4e')
    losses_edge = AverageMeter('LossState', ':.4e')

    

    prec_state = AverageMeter('Prec State', ':6.2f')
    prec_edge = AverageMeter('Prec Edge', ':6.2f')
    
    recall_state = AverageMeter('RecallState', ':6.2f')
    recall_edge = AverageMeter('RecallEdge', ':6.2f')

    progress = ProgressMeter(
            len(data_loader),
            [batch_time, data_time, losses, losses_state, losses_edge, prec_state, recall_state, prec_edge, recall_edge],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    
    end = time.time()


    for it, data_item in enumerate(data_loader):
        data_time.update(time.time() - end)

        graph_info, program, label, len_mask, goal, label_agent, real_label_agent = data_item
        

        # utils_models.print_graph(data_loader.dataset.graph_helper, graph_info, 0, 0)
        # ipdb.set_trace()

        inputs = {
            'program': program,
            'graph': graph_info,
            'mask_len': len_mask,
            'goal': goal,
            'label_agent': label_agent
        }
        # print(inputs['graph']['graph'], inputs['graph']['mask_object'].sum())
        output = model(inputs)

        pred_edge = output['edges'][:, :-1, ...]
        pred_state = output['states'][:, :-1, ...]
        gt_state = graph_info['states_objects'][:, 1:, ...].cuda()
        mask_state = graph_info['mask_object'][:, 1:, :, None].cuda()
        mask_length = len_mask[:, 1:].cuda()

        # loss states
        criterion_state = torch.nn.BCEWithLogitsLoss(reduction='none')
        loss_state = criterion_state(pred_state, gt_state) 
        loss_state = loss_state *  mask_state
        loss_state = loss_state.mean()

        # loss edges edges in prediction are stored as a B x Time x N x N x num_edge_class tensor
        # GT is stored as B x Time x Num_edges, we need to convert
        num_nodes = output['states'].shape[-2]
        gt_edges = torch.zeros_like(output['edges'])
        num_edges = gt_edges.shape[-1]
        edge_tuples = graph_info['edge_tuples']
        index_edges = edge_tuples[..., 0] * num_nodes + edge_tuples[..., 1]
        edge_types = graph_info['edge_classes'] - 1
        for it_edge in range(num_edges):
            index_edge = edge_types == it_edge
            index_edge_curr = index_edges[index_edge]
            gt_edges[..., index_edge_curr.long(), it_edge] = 1


        gt_edge = gt_edges[:, 1:, ...].cuda()

        mask_obs_node = graph_info['mask_obs_node']
        
        medges1 = mask_obs_node.repeat([1, 1, num_nodes]).cuda()
        medges2 = mask_obs_node.repeat_interleave(num_nodes, dim=2).cuda()
        mask_edges = medges1 * medges2
        mask_edges = mask_edges[:, 1:, ..., None]
        loss_edges = criterion_state(pred_edge, gt_edge) 
        loss_edges = loss_edges * mask_edges
        loss_edges = loss_edges.mean()


        loss = loss_edges + loss_state
        losses.update(loss.item())
        losses_state.update(loss_state.item())
        losses_edge.update(loss_edges.item())


        # Update accuracy
        # TODO: add accuracy metrics


        # How many GT positives
        pos_state = gt_state.sum(-1).sum(-1) + 1e-9
        pos_edge = gt_state.sum(-1).sum(-1) + 1e-9

        state_avg = gt_state / (pos_state[:, :, None, None])
        edge_avg = gt_edge / (pos_edge[:, :, None, None])
        # How many predicted positives
        # ipdb.set_trace()
        edge_avg_pos = ((pred_edge > 0.5) * mask_edges).sum(-1).sum(-1) + 1e-9
        state_avg_pos = ((pred_state > 0.5) * mask_state).sum(-1).sum(-1) + 1e-9

        tp_edge = (gt_edge * (pred_edge > 0.5)).sum(-1).sum(-1)
        tp_state = (gt_state * (pred_state > 0.5)).sum(-1).sum(-1)

        # Recall
        edge_recall = tp_edge / pos_edge
        state_recall = tp_state / pos_state
        
        # Precision
        edge_prec = (tp_edge / edge_avg_pos)
        state_prec = (tp_state / state_avg_pos)
        
        # Average over timesteps and batch
        mask_timesteps = mask_length / mask_length.sum(-1)[:, None]
        edge_recall = (edge_recall * mask_timesteps).sum()
        state_recall = (state_recall * mask_timesteps).sum()
        edge_prec = (edge_prec * mask_timesteps).sum()
        state_prec = (state_prec * mask_timesteps).sum()

        # ipdb.set_trace()
        prec_state.update(state_prec.item())
        recall_state.update(state_recall.item())
        prec_edge.update(edge_prec.item())
        recall_edge.update(edge_recall.item())
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
                'losses': {'total': losses.val, 'state': losses_state.val, 'edge': losses_edge.val},
                'accuracy': {'state_prec': prec_state.val, 'edge_rec': prec_edge.val, 'state_recall': recall_state.val, 'edge_recall': recall_edge.val},
                'misc': {'epoch': epoch}
            }
            logger.log_data(it + len(data_loader) * epoch, info_log)
            
            # Print the prediction
            #logger.log_info(info_res)

    #logger.log_embeds(len(data_loader) * epoch, model.module.agent_embedding)
    print("Failed Elements...", data_loader.dataset.get_failures())


def get_loaders(args):
    print("Loading dataset...")
    print("Train: {}".format(args['data']['train_data']))
    print("Test: {}".format(args['data']['test_data']))
    curr_file = os.path.dirname(get_original_cwd())
    dataset = AgentTypeDataset(path_init='{}/agent_preferences/dataset/{}'.format(curr_file, args['data']['train_data']), args_config=args)
    dataset_test = AgentTypeDataset(path_init='{}/agent_preferences/dataset/{}'.format(curr_file, args['data']['test_data']), args_config=args)
    if args['model']['state_encoder'] == 'GNN':
        collate_fn = dataloader_v2.collate_fn
    else:
        collate_fn = None
    train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args['train']['batch_size'], 
            shuffle=True, num_workers=args['train']['num_workers'], pin_memory=True, collate_fn=collate_fn)

    test_loader = torch.utils.data.DataLoader(
            dataset_test, batch_size=args['train']['batch_size'], 
            shuffle=True, num_workers=args['train']['num_workers'], pin_memory=True, collate_fn=collate_fn)
    return train_loader, test_loader



@hydra.main(config_path="../config/agent_pred_graph", config_name="config_default")
def main(cfg: DictConfig):
    config = cfg
    print("Config")
    print(OmegaConf.to_yaml(cfg))
    # ipdb.set_trace()

    train_loader, test_loader = get_loaders(config)
    if config.model.gated:
        model = agent_pref_policy.GraphPredNetwork(config)
    else:
        model = agent_pref_policy.GraphPredNetwork(config)

    print("CUDA: {}".format(cfg.cuda))
    if cfg.cuda:
        model = model.cuda()
        model = nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=config['train']['lr'])
    print("Failures: ", train_loader.dataset.get_failures())

    logger = LoggerSteps(config)

    logger.save_model(0, model, optimizer)

    for epoch in range(config['train']['epochs']):
        train_epoch(train_loader, model, epoch, config, criterion, optimizer, logger)
        evaluate(test_loader, train_loader, model, epoch, config, criterion, logger)
        if epoch % 10 == 0:
            logger.save_model(epoch, model, optimizer)


if __name__ == '__main__':
      main()
