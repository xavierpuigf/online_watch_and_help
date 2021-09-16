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
import numpy as np
from utils import utils_models_wb

@hydra.main(config_path="../config/agent_pred_graph", config_name="config_default_toy_excl")
def main(cfg: DictConfig):
    config = cfg
    args = config

    model = agent_pref_policy.GraphPredNetwork(config)
    state_dict = torch.load(cfg.ckpt_load)['model']
    state_dict_new = {}
    
    for param_name, param_value in state_dict.items():
        state_dict_new[param_name.replace('module.', '')] = param_value

    model.load_state_dict(state_dict_new)
    model.eval()




    curr_file = os.path.dirname(get_original_cwd())
    dataset_test = AgentTypeDataset(
        path_init='{}/agent_preferences/dataset/{}'.format(
            curr_file, args['data']['test_data']
        ),
        args_config=args,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=None,
    )
    

    # Output by the loader
    (
        graph_info,
        program,
        label,
        len_mask,
        goal,
        label_agent,
        real_label_agent,
        ind
    ) = next(iter(test_loader))
    

    inputs_loader = {
        'program': program,
        'graph': graph_info,
        'mask_len': len_mask,
        'goal': goal
    }



    # Output by the standalone function
    file_name = dataset_test.pkl_files[0]
    with open(file_name, 'rb') as f:
        content = pkl.load(f)

    graphs = content['graph']
    observations = content['obs']
    program_hist = content['action'][0]

    # list_miss =  list((set(node['id'] for node in content['graph'][0]['nodes']) - set(content['obs'][0])))
    # print([node['class_name'] for node in content['graph'][0]['nodes'] if node['id'] in list_miss])


    # How to load the data from a graph
    inputs_func = utils_models_wb.prepare_graph_for_model(graphs, observations, program_hist, args, dataset_test)
    with torch.no_grad():
        output_func = model(inputs_func)




    with torch.no_grad():
        output_loader = model(inputs_loader)

    # Compare inputs
    assert inputs_func['mask_len'].sum() == inputs_loader['mask_len'].sum(), "Different mask lengths"
    mask_len = int(inputs_loader['mask_len'].sum().item())
    # print(mask_len)

    for content in ['program', 'graph']:
        inp_loader = inputs_loader[content]
        inp_func = inputs_func[content]
        for key_elem in inp_func.keys():

            val_loader = inp_loader[key_elem][0,:mask_len]
            val_func = inp_func[key_elem][0,:mask_len]

            # print("\n", key_elem,  inp_loader[key_elem].shape, inp_func[key_elem].shape)
            assert np.all(np.array(val_loader.shape) == np.array(val_func.shape)), f"Shapes from {key_elem} differ"
            assert np.all(val_loader.numpy() == val_func.numpy()), f"Values from {key_elem} differ"
            print(key_elem, np.all(val_loader.numpy() == val_func.numpy()), val_loader.shape, val_func.shape)
            # print('\n')

    print('\n#############')
    print('Compare Results')
    # Compare outputs
    for key in output_func.keys():
        print(key, output_func[key].shape, output_loader[key].shape)
        # print(key, output_loader[key].shape)
        val_loader = output_loader[key][0,:mask_len]
        val_func = output_func[key][0,:mask_len]
        try:
            assert np.sum(val_loader.numpy() - val_func.numpy()) < 0.01, f"values from {key} differ"
        except:
            ipdb.set_trace()

    ipdb.set_trace()


if __name__ == '__main__':
    main()