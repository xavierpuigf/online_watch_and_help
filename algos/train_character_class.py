import torch
import time

import glob
import yaml
import pickle as pkl
from tqdm import tqdm
import ipdb
from dataloader.dataloader import AgentTypeDataset
from arguments import *
from torch import nn
import torch.optim as optim
from models import agent_pref_policy
import utils.utils_models as utils_models
from utils.utils_models import AverageMeter, ProgressMeter, LoggerSteps


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
    losses_action = AverageMeter('LossAction', ':.4e')
    losses_o1 = AverageMeter('LossO1', ':.4e')
    losses_o2 = AverageMeter('LossO2', ':.4e')
    acc_action = AverageMeter('AccAction', ':6.2f')
    acc_o1 = AverageMeter('Acc O1', ':6.2f')
    acc_o2 = AverageMeter('Acc O2', ':6.2f')
    progress = ProgressMeter(
            len(data_loader),
            [batch_time, data_time, losses, losses_action, losses_o1, losses_o2, acc_action, acc_o1, acc_o2],
        prefix="Epoch: [{}]".format(epoch))
    
    end = time.time()
    for it, data_item in enumerate(data_loader):
        if it < args['test']['num_iters']:
            data_time.update(time.time() - end)


            graph_info, program, label, len_mask, goal = data_item
            inputs = {
                'program': program,
                'graph': graph_info,
                'mask_len': len_mask,
                'goal': goal
            }
            # ipdb.set_trace()
            with torch.no_grad():
                output = model(inputs)
            action_l, o1_l, o2_l = output['action_logits'], output['o1_logits'], output['o2_logits']
            bs = action_l.shape[0]

            index_label_obj1 = program['indobj1'][:, 1:]
            index_label_obj2 = program['indobj2'][:, 1:]

            if index_label_obj2.max() > args['model']['max_nodes'] or index_label_obj1.max() > args['model']['max_nodes']:
                print("Error with indices", index_label_obj1.max().item(), index_label_obj2.max().item())
            label_action = program['action'][:, 1:]

            if args['cuda']:
                label_action = label_action.cuda()
                index_label_obj2 = index_label_obj2.cuda()
                index_label_obj1 = index_label_obj1.cuda()
                len_mask = len_mask.cuda()

            # ipdb.set_trace()
            # ipdb.set_trace()
            loss_action = unmerge(criterion(merge2d(action_l), merge2d(label_action)), bs)
            loss_object1 = unmerge(criterion(merge2d(o1_l), merge2d(index_label_obj1)), bs)
            loss_object2 = unmerge(criterion(merge2d(o2_l), merge2d(index_label_obj2)), bs)

            len_mask_avg = len_mask/len_mask.sum(1)[:, None]
            loss_action = (loss_action * len_mask_avg).sum(1).mean(0)
            loss_object1 = (loss_object1 * len_mask_avg).sum(1).mean(0)
            loss_object2 = (loss_object2 * len_mask_avg).sum(1).mean(0)


            loss = loss_action + loss_object1 + loss_object2

            # Update losses
            losses.update(loss.item())
            losses_action.update(loss_action.item())
            losses_o1.update(loss_object1.item())
            losses_o2.update(loss_object2.item())

            # Update accuracy
            pred_action = action_l.argmax(-1)
            pred_o1 = o1_l.argmax(-1)
            pred_o2 = o2_l.argmax(-1)

            action_accuracy = ((pred_action == label_action) * len_mask_avg).sum(1).mean(0)
            o1_accuracy = ((pred_o1 == index_label_obj1) * len_mask_avg).sum(1).mean(0)
            o2_accuracy = ((pred_o2 == index_label_obj2) * len_mask_avg).sum(1).mean(0)
            
            acc_action.update(action_accuracy.item())
            acc_o1.update(o1_accuracy.item())
            acc_o2.update(o2_accuracy.item())
            
            # ipdb.set_trace()
            batch_time.update(time.time() - end)
            end = time.time()


        else:
            continue

        progress.display(it)
        
        info_log = {
            'losses': {'total_val': losses.val, 'action_val': losses_action.val, 'object1_val': losses_o1.val, 'object2_val': losses_o2.val},
            'accuracy': {'action_val': acc_action.val, 'object1_val': acc_o1.val, 'object2_val': acc_o2.val }
        }
        logger.log_data(len(data_loader_train) * epoch, info_log)
        
        # Print the prediction
        prog_gt = {'action': label_action, 'o1': index_label_obj1, 'o2': index_label_obj2, 'graph': graph_info, 'mask_len': len_mask}
        prog_pred = {'action': pred_action, 'o1': pred_o1, 'o2': pred_o2, 'graph': graph_info, 'mask_len': len_mask}

        str_results = utils_models.get_pred_results_str(data_loader.dataset.graph_helper, prog_gt, prog_pred)
        
        info_res = {
            'str': progress.display(it, do_print=False)+'\n'+str_results
        }
        logger.log_info(info_res)

def train_epoch(data_loader, model, epoch, args, criterion, optimizer, logger):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    losses_action = AverageMeter('LossAction', ':.4e')
    losses_o1 = AverageMeter('LossO1', ':.4e')
    losses_o2 = AverageMeter('LossO2', ':.4e')
    acc_action = AverageMeter('AccAction', ':6.2f')
    acc_o1 = AverageMeter('Acc O1', ':6.2f')
    acc_o2 = AverageMeter('Acc O2', ':6.2f')
    progress = ProgressMeter(
            len(data_loader),
            [batch_time, data_time, losses, losses_action, losses_o1, losses_o2, acc_action, acc_o1, acc_o2],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    
    end = time.time()
    for it, data_item in enumerate(data_loader):
        data_time.update(time.time() - end)

        graph_info, program, label, len_mask, goal = data_item
        inputs = {
            'program': program,
            'graph': graph_info,
            'mask_len': len_mask,
            'goal': goal
        }
        # ipdb.set_trace()
        output = model(inputs)
        action_l, o1_l, o2_l = output['action_logits'], output['o1_logits'], output['o2_logits']
        bs = action_l.shape[0]

        index_label_obj1 = program['indobj1'][:, 1:]
        index_label_obj2 = program['indobj2'][:, 1:]

        if index_label_obj2.max() > args['model']['max_nodes'] or index_label_obj1.max() > args['model']['max_nodes']:
            print("Error with indices", index_label_obj1.max().item(), index_label_obj2.max().item())
        label_action = program['action'][:, 1:]

        if args['cuda']:
            label_action = label_action.cuda()
            index_label_obj2 = index_label_obj2.cuda()
            index_label_obj1 = index_label_obj1.cuda()
            len_mask = len_mask.cuda()

        # ipdb.set_trace()
        # ipdb.set_trace()
        loss_action = unmerge(criterion(merge2d(action_l), merge2d(label_action)), bs)
        loss_object1 = unmerge(criterion(merge2d(o1_l), merge2d(index_label_obj1)), bs)
        loss_object2 = unmerge(criterion(merge2d(o2_l), merge2d(index_label_obj2)), bs)

        len_mask_avg = len_mask/len_mask.sum(1)[:, None]
        loss_action = (loss_action * len_mask_avg).sum(1).mean(0)
        loss_object1 = (loss_object1 * len_mask_avg).sum(1).mean(0)
        loss_object2 = (loss_object2 * len_mask_avg).sum(1).mean(0)


        loss = loss_action + loss_object1 + loss_object2

        # Update losses
        losses.update(loss.item())
        losses_action.update(loss_action.item())
        losses_o1.update(loss_object1.item())
        losses_o2.update(loss_object2.item())

        # Update accuracy
        pred_action = action_l.argmax(-1)
        pred_o1 = o1_l.argmax(-1)
        pred_o2 = o2_l.argmax(-1)

        action_accuracy = ((pred_action == label_action) * len_mask_avg).sum(1).mean(0)
        o1_accuracy = ((pred_o1 == index_label_obj1) * len_mask_avg).sum(1).mean(0)
        o2_accuracy = ((pred_o2 == index_label_obj2) * len_mask_avg).sum(1).mean(0)
        
        acc_action.update(action_accuracy.item())
        acc_o1.update(o1_accuracy.item())
        acc_o2.update(o2_accuracy.item())
        
        # ipdb.set_trace()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if it % args['log']['print_every'] == 0:
            progress.display(it)
        if it % args['log']['print_long_every'] == 0:
        
            info_log = {
                'losses': {'total': losses.val, 'action': losses_action.val, 'object1': losses_o1.val, 'object2': losses_o2.val},
                'accuracy': {'action': acc_action.val, 'object1': acc_o1.val, 'object2': acc_o2.val },
                'misc': {'epoch': epoch}
            }
            logger.log_data(it + len(data_loader) * epoch, info_log)
            
            # Print the prediction
            prog_gt = {'action': label_action, 'o1': index_label_obj1, 'o2': index_label_obj2, 'graph': graph_info, 'mask_len': len_mask}
            prog_pred = {'action': pred_action, 'o1': pred_o1, 'o2': pred_o2, 'graph': graph_info, 'mask_len': len_mask}

            str_results = utils_models.get_pred_results_str(data_loader.dataset.graph_helper, prog_gt, prog_pred)
            
            info_res = {
                'str': progress.display(it, do_print=False)+'\n'+str_results
            }
            logger.log_info(info_res)
    print("Failed Elements...", data_loader.dataset.get_failures())


def get_loaders(args):
    dataset = AgentTypeDataset(path_init='../data_scratch/large_data/train_env_task_set_20_full_reduced_tasks/', args_config=args)
    dataset_test = AgentTypeDataset(path_init='../data_scratch/large_data/test_env_task_set_10_full_reduced_tasks/', args_config=args)
    train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args['train']['batch_size'], 
            shuffle=True, num_workers=args['train']['num_workers'], pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
            dataset_test, batch_size=args['train']['batch_size'], 
            shuffle=True, num_workers=args['train']['num_workers'], pin_memory=True)
    return train_loader, test_loader



def main():
    args = get_args_pref_agent()

    with open(args.config, 'r') as f:
        config = yaml.load(f)
    config['cuda'] = args.cuda

    train_loader, test_loader = get_loaders(config)
    model = agent_pref_policy.ActionPredNetwork(config)
    print("CUDA: {}".format(args.cuda))
    if args.cuda:
        model = model.cuda()
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=config['train']['lr'])
    print("Failures: ", train_loader.dataset.get_failures())

    logger = LoggerSteps(config)
    for epoch in range(config['train']['epochs']):
        train_epoch(train_loader, model, epoch, config, criterion, optimizer, logger)
        evaluate(test_loader, train_loader, model, epoch, config, criterion, logger)
        if epoch % 10 == 0:
            logger.save_model(epoch, model, optimizer)


if __name__ == '__main__':
      main()
