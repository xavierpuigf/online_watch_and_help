import torch

import glob
import pickle as pkl
from tqdm import tqdm
import ipdb
from dataloader.dataloader import AgentTypeDataset
from arguments import *
from torch import nn
from models import agent_pref_policy

class AuxModel(nn.Module):
    def __init__(self, args):
        super(AuxModel, self).__init__()
        self.num_attributes = args.num_attributes
        self.linear_layer = nn.Linear(5, args.num_attributes)
    def __forward__(self, graph_input):
        ipdb.set_trace()
        output_class = None
        return output_class

def train_epoch(data_loader, model, epoch, args):
    for data_item in data_loader:
        graph_info, program, label = data_item
        inputs = {
            'program': program,
            'graph': graph_info
        }
        output_class = model(inputs)
        ipdb.set_trace()

def get_loaders(args):
    dataset = AgentTypeDataset(path_init='../data_scratch/train_env_task_set_20_full_reduced_tasks/', args=args)
    train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, 
            shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = train_loader
    return train_loader, test_loader



def main():
    args = get_args_pref_agent()
    train_loader, test_loader = get_loaders(args)
    model = agent_pref_policy.ActionPredNetwork(args)
    print("CUDA: {}".format(args.cuda))
    if args.cuda:
        model = model.cuda()
        model = nn.DataParallel(model)
    for epoch in range(args.epochs):
        train_epoch(train_loader, model, epoch, args)


if __name__ == '__main__':
      main()
