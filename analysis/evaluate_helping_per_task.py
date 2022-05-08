import sys

sys.path.append(".")
import shutil
import os
import logging
import traceback
import pickle as pkl
import random
import pickle
import copy
from pathlib import Path
import numpy as np
import ipdb

# import ipdb
import hydra
import time
import multiprocessing as mp
from omegaconf import DictConfig, OmegaConf
from dataloader.dataloader_v2 import AgentTypeDataset
from dataloader import dataloader_v2 as dataloader_v2
from models import agent_pref_policy
from hydra.utils import get_original_cwd, to_absolute_path
from utils import utils_models_wb, utils_rl_agent

from envs.unity_environment import UnityEnvironment
from agents import MCTS_agent, MCTS_agent_particle_v2, MCTS_agent_particle

# from arguments import get_args
from algos.arena_mp2 import ArenaMP
from utils import utils_goals
from utils import utils_exception
import torch


def get_class_mode(agent_args):
    mode_str = "{}_opencost{}_closecost{}_walkcost{}_forgetrate{}".format(
        agent_args["obs_type"],
        agent_args["open_cost"],
        agent_args["should_close"],
        agent_args["walk_cost"],
        agent_args["belief"]["forget_rate"],
    )
    return mode_str


def get_edge_class0(pred, t, source="pred"):
    # pred_edge_prob = pred['edge_prob']
    edge_pred = pred["edge_pred"][t] if source == "pred" else pred["edge_input"][t]
    pred_edge_names = pred["edge_names"]
    pred_nodes = pred["nodes"]
    pred_from_ids = pred["from_id"]  # if source == 'pred' else pred['from_id_input']
    pred_to_ids = pred["to_id"]  # if source == 'pred' else pred['to_id_input']

    # edge_prob = pred_edge_prob[t]
    # edge_pred = np.argmax(edge_prob, 1)

    edge_pred_class = {}

    num_edges = len(edge_pred)
    for edge_id in range(num_edges):
        from_id = pred_from_ids[t][edge_id]
        to_id = pred_to_ids[t][edge_id]
        from_node_name = pred_nodes[from_id]
        to_node_name = pred_nodes[to_id]
        # if object_name in from_node_name or object_name in to_node_name:
        edge_name = pred_edge_names[edge_pred[edge_id]]
        if edge_name in ["inside", "on"]:  # disregard room locations + plate
            if to_node_name.split(".")[0] in [
                "kitchen",
                "livingroom",
                "bedroom",
                "bathroom",
                "plate",
            ]:
                continue
            # if from_node_name.split('.')[0]
            edge_class = "{}_{}_{}".format(
                edge_name, from_node_name.split(".")[0], to_node_name.split(".")[1]
            )
            # print(from_node_name, to_node_name, edge_name)
            if edge_class not in edge_pred_class:
                edge_pred_class[edge_class] = 1
            else:
                edge_pred_class[edge_class] += 1
    return edge_pred_class


def get_edge_class(pred, t, source="pred"):
    # pred_edge_prob = pred['edge_prob']
    # print(len(pred['edge_input'][t]), len(pred['edge_pred'][t]))
    edge_pred = pred["edge_pred"][t] if source == "pred" else pred["edge_input"][t]
    pred_edge_names = pred["edge_names"]
    pred_nodes = pred["nodes"]
    pred_from_ids = pred["from_id"] if source == "pred" else pred["from_id_input"]
    pred_to_ids = pred["to_id"] if source == "pred" else pred["to_id_input"]

    # edge_prob = pred_edge_prob[t]
    # edge_pred = np.argmax(edge_prob, 1)

    edge_pred_class = {}

    num_edges = len(edge_pred)
    # print(pred_from_ids[t], num_edges)
    for edge_id in range(num_edges):
        from_id = pred_from_ids[t][edge_id]
        to_id = pred_to_ids[t][edge_id]
        from_node_name = pred_nodes[from_id]
        to_node_name = pred_nodes[to_id]
        # if object_name in from_node_name or object_name in to_node_name:
        edge_name = pred_edge_names[edge_pred[edge_id]]
        if to_node_name.split(".")[1] == "-1":
            continue
        if edge_name in ["inside", "on"]:  # disregard room locations + plate
            if to_node_name.split(".")[0] in [
                "kitchen",
                "livingroom",
                "bedroom",
                "bathroom",
                "plate",
            ]:
                continue
        else:
            continue
        if from_node_name.split(".")[0] not in [
            "apple",
            "cupcake",
            "plate",
            "waterglass",
        ]:
            continue

        # if from_node_name.split('.')[0]

        # # TODO: need to infer the correct edge class
        # if 'table' in to_node_name.split('.')[0]:
        #     ipdb.set_trace()
        #     edge_name = 'on'

        edge_class = "{}_{}_{}".format(
            edge_name, from_node_name.split(".")[0], to_node_name.split(".")[1]
        )
        # print(from_node_name, to_node_name, edge_name)
        if edge_class not in edge_pred_class:
            edge_pred_class[edge_class] = 1
        else:
            edge_pred_class[edge_class] += 1
    return edge_pred_class


def aggregate_multiple_pred(preds, t, change=False):
    edge_classes = []
    edge_pred_class_all = {}
    N_preds = len(preds)
    for pred in preds:
        edge_pred_class = get_edge_class(pred, t)
        edge_classes += list(edge_pred_class.keys())
        for edge_class, count in edge_pred_class.items():
            if edge_class not in edge_pred_class_all:
                edge_pred_class_all[edge_class] = [count]
            else:
                edge_pred_class_all[edge_class] += [count]
    if change:
        edge_input_class = get_edge_class(preds[0], t, "input")
        edge_classes += list(edge_input_class.keys())

    edge_classes = sorted(list(set(edge_classes)))
    edge_pred_class_estimated = {}
    for edge_class in edge_classes:
        if edge_class not in edge_pred_class_all:
            edge_pred_class_estimated[edge_class] = (-edge_input_class[edge_class], 0)
            continue
        curr_len = len(edge_pred_class_all[edge_class])
        if curr_len < N_preds:
            edge_pred_class_all[edge_class] += [0] * (N_preds - curr_len)
        if change:
            c = (
                np.mean(edge_pred_class_all[edge_class]) - edge_input_class[edge_class]
                if edge_class in edge_input_class
                else np.mean(edge_pred_class_all[edge_class])
            )
        else:
            c = np.mean(edge_pred_class_all[edge_class])
        edge_pred_class_estimated[edge_class] = (
            c,
            np.std(edge_pred_class_all[edge_class]),
        )
        # print(edge_class, edge_pred_class_estimated[edge_class])
    return edge_pred_class_estimated


def get_metrics_reward(
    alice_results, test_results, episode_ids, num_tries, time_limit=30
):
    mS = []
    mL = []
    mSP = []
    mSwS = []
    # pdb.set_trace()
    for seed in range(num_tries):
        alice_S = []
        alice_L = []
    normalized_by_suc = True
    for episode_id in episode_ids:
        Ls = []
        Ss = []
        SWSs = []
        L_A_seeds = []
        for seed_alice in range(num_tries):
            if episode_id not in alice_results:
                S_A, L_A = 0, time_limit
                pdb.set_trace()
                # continue
            else:
                if alice_results[episode_id]["S"][seed_alice] == "":
                    # print(episode_id, seed)
                    continue

                S_A = alice_results[episode_id]["S"][seed_alice]
                L_A = alice_results[episode_id]["L"][seed_alice]
                L_A_seeds.append(L_A)
        if episode_id not in test_results:
            # print(episode_id, seed)
            continue
        L_A_seeds = [t for t in L_A_seeds if t is not None]
        if normalized_by_suc:
            L_A_seeds = [t for t in L_A_seeds if t < time_limit]

        Ls = []
        Ss = []
        for seed_bob in range(num_tries):
            if seed_bob >= len(test_results[episode_id]["S"]):
                continue
            try:
                if test_results[episode_id]["S"][seed_bob] == "":
                    # print(episode_id, seed)
                    continue
                if test_results[episode_id]["S"][seed_bob] is None:
                    # print(episode_id, seed)
                    continue
                S_B = test_results[episode_id]["S"][seed_bob]
                L_B = test_results[episode_id]["L"][seed_bob]
                if L_B == time_limit:
                    S_B = 0.0
            except:
                pdb.set_trace()

            Ls.append(L_B)
            Ss.append(S_B)

        Ls = [t for t in Ls if t is not None]
        if normalized_by_suc:
            Ls = [t for t in Ls if t < time_limit]
        if len(Ls) > 0:
            # if len([t for t in Ss if t == 0.]) > 0:
            #     pdb.set_trace()
            SWSs.append(
                np.mean([-ls * 1.0 / time_limit + sb for ls, sb in zip(Ls, Ss)])
            )
            # mSwS.append(SWSs)
            # if SWSs > 0:
            #     cont_better += 1
            #

            # pdb.set_trace()
            # print(episode_id)
            if len(L_A_seeds) > 0:
                mSP.append(np.mean(L_A_seeds) / np.mean(Ls) - 1.0)
        else:
            Ls = [time_limit] * len(Ss)
            SWSs.append(0)

        # print(episode_id, Ss, Ls)

        mS.append(np.mean(Ss))
        mL.append(np.mean(Ls))
        mSwS.append(np.mean(SWSs))

    # print('Alice:', np.mean(alice_S), np.mean(alice_L))
    # print('Alice:', np.mean(alice_S), '({})'.format(np.std(alice_S)), np.mean(alice_L), '({})'.format(np.std(alice_L)))
    # print('Bob:', np.mean(Ss), '({})'.format(np.std(Ss)), np.mean(Ls), '({})'.format(np.std(Ls)), np.mean(SWSs), '({})'.format(np.std(SWSs)))

    ns = np.sqrt(len(mS))
    nsp = np.sqrt(len(mSP))
    nw = np.sqrt(len(mSwS))
    # Success, Length, SpeedUp, Reward 
    return (
        np.mean(mS),
        np.mean(mL),
        np.mean(mSP),
        np.mean(mSwS),
        np.std(mS) / ns,
        np.std(mL) / ns,
        np.std(mSP) / nsp,
        np.std(mSwS) / nw,
    )


@hydra.main(config_path="../config/", config_name="config_default_toy_excl_plan")
def main(cfg: DictConfig):
    config = cfg
    print("Config")
    print(OmegaConf.to_yaml(cfg))
    args = config
    args_pred = args.agent_pred_graph
    num_proc = 0


    with open(f"{get_original_cwd()}/analysis/metadata/task_name_dict.pik", 'rb') as f:
        task_dict = pkl.load(f)
    task_dict = task_dict['test']


    num_tries = 5
    args.executable_file = "/data/vision/torralba/frames/data_acquisition/SyntheticStories/website/release/simulator/v2.0/v2.2.5_beta/linux_exec.v2.2.5_beta.x86_64"
    args.max_episode_length = 250
    args.num_per_apartment = 20
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    # home_path = '../'
    rootdir = ""

    # args.dataset_path = f'{rootdir}/dataset/train_env_task_set_100_full.pik'
    args.dataset_path = f"/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/agent_preferences/dataset/test_env_task_set_60_full_task.all.pik"
    # args.dataset_path = './dataset/train_env_task_set_20_full_reduced_tasks_single.pik'



    #cachedir = '/data/vision/torralba/frames/data_acquisition/SyntheticStories/agent_preferences/tshu/agent_preferences/outputs/helping_states_1_3_ip1_detfull_encoder_task_graph_20_1.0_1.0_5.0'
    # cachedir = f"{get_original_cwd()}/outputs/helping_states_nohold_20_1.0_1.0"
    # cachedir = f"{get_original_cwd()}/outputs/helping_states_20_1.0_1.0"
    # cachedir = f"{get_original_cwd()}/outputs/helping_states_newvaefull_encoder_task_graph_10_1.0_1.0_5.0"
    # cachedir = f"{get_original_cwd()}/outputs/helping_states_newvaefull_encoder_task_graph.kl0.001_10_1.0_1.0_5.0"
    # cachedir = f"{get_original_cwd()}/outputs/helping_states_ip1_newvaefull_encoder_task_graph.kl0.001_20_1.0_1.0_5.0"
    # cachedir = f"{get_original_cwd()}/outputs/helping_states_1_3_ip1_newvaefull_encoder_task_graph.kl0.001_20_1.0_1.0_5.0"
    # cachedir = f"{get_original_cwd()}/outputs/helping_states_ip1_detfull_encoder_task_graph_20_1.0_1.0_5.0"
    #cachedir = f"{get_original_cwd()}/outputs/helping_states_1_3_ip1_detfull_encoder_task_graph_20_1.0_1.0_5.0"
    # cachedir = f"{get_original_cwd()}/outputs/helping_states_detfull_encoder_task_graph_20_1.0_1.0_5.0"

    # cachedir = f'{get_original_cwd()}/outputs/helping_action_freq_v2_20'
    # cachedir = f'{get_original_cwd()}/outputs/helping_action_freq_1'
    tshu_dir = '/data/vision/torralba/frames/data_acquisition/SyntheticStories/agent_preferences/tshu/agent_preferences'

    cachedir_main = f"{get_original_cwd()}/outputs/main_agent_only_large"
    cachedir_main = "/data/vision/torralba/frames/data_acquisition/SyntheticStories/agent_preferences/tshu/agent_preferences/outputs/main_agent_only_large"
    cachedir_det = "/data/vision/torralba/frames/data_acquisition/SyntheticStories/agent_preferences/tshu/agent_preferences/outputs/helping_states_fastwalk_1_3_ip1_detfull_encoder_task_graph_20_c10_1.0_5.0"
    #cachedir = cachedir_main
    results = {}
    results_per_task = {}
    cachedirs = [
                 ("Main", cachedir_main),
                 ("GT", f"{get_original_cwd()}/outputs/helping_gt_goal_fast"),
                 ("Uniform", f"{tshu_dir}/outputs/helping_states_fastwalk_r_1_3_ip1_uniform_20_1.0_1.0_5.0"),
                 ("w/o InvPlanning", f"{tshu_dir}/outputs/helping_states_fastwalk_r_1_3_ip0_detfull_encoder_task_graph_20_1.0_1.0_5.0"),
                 ("1 Goal", f"{tshu_dir}/outputs/helping_states_fastwalk_r_1_3_ip0_detfull_encoder_task_graph_1_1.0_1.0_5.0"),
                 ("Ours", f"{tshu_dir}/outputs/helping_states_fastwalk_r_1_3_ip1_detfull_encoder_task_graph_20_1.0_1.0_5.0")
                 ]
    for name_exp, cachedir in cachedirs:
        random_start = random.Random()

        datafile = args.dataset_path.split("/")[-1].replace(".pik", "")
        
        # TODO: add num_samples to the argument
        num_samples = args.num_samples
        num_processes = args.num_processes
        # args.mode = "{}_".format(agent_id + 1) + "action_freq_{}".format(num_samples)
        # args.mode += 'v9_particles_v2'

        # env_task_set = pickle.load(open(args.dataset_path, 'rb'))
        # # print(env_task_set)
        # print(len(env_task_set))

        args.record_dir = "{}/{}".format(cachedir, datafile)
        record_dir_main = "{}/{}".format(cachedir_main, datafile)
        error_dir = "{}/logging/{}".format(cachedir, datafile)
        # if not os.path.exists(args.record_dir):
        #     os.makedirs(args.record_dir)

        # if not os.path.exists(error_dir):
        #     os.makedirs(error_dir)

        executable_args = {
            "file_name": args.executable_file,
            "x_display": 0,
            "no_graphics": True,
        }

        id_run = 0
        # random.seed(id_run)
        # episode_ids = list(range(len(env_task_set)))
        # episode_ids = sorted(episode_ids)
        # random_start.shuffle(episode_ids)
        # episode_ids = episode_ids[10:]

        valid_set_path = "/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/agent_preferences/analysis/test_set_reduced.txt"
        f = open(valid_set_path, "r")
        episode_ids = []
        for filename in f:
            episode_ids.append(int(filename.split("episode.")[-1].split("_")[0]))
        episode_ids = sorted(episode_ids)
        print(len(episode_ids))
        f.close()

        # episode_ids = [3]

        # # episode_ids = [20] #episode_ids
        # # num_tries = 1
        # episode_ids = [1]
        # ndict = {'on_book_329': 1}
        # env_task_set[91]['init_rooms'] = ['bedroom', 'bedroom']
        # env_task_set[91]['task_goal'] = {0: ndict, 1: ndict}

        # test_results_

        main_results, help_results = {}, {}
        num_tries = 3

        for iter_id in range(0, num_tries):
            # if iter_id > 0:
            # iter_id = 1

            steps_list, failed_tasks = [], []
            current_tried = iter_id

            # test_results = {}
            print(args.record_dir)

            if not os.path.isfile(args.record_dir + "/results_{}.pik".format(iter_id)):
                test_results = {}
            else:
                test_results = pickle.load(
                    open(args.record_dir + "/results_{}.pik".format(iter_id), "rb")
                )
                help_results = dict(test_results)

            # print(iter_id, len(test_results))

            if not os.path.isfile(record_dir_main + "/results_{}.pik".format(iter_id)):
                test_results = {}
            else:
                test_results = pickle.load(
                    open(record_dir_main + "/results_{}.pik".format(iter_id), "rb")
                )
                main_results = dict(test_results)

            # print(test_results)

        print(len(help_results))

        for episode_id in help_results:
            try:
                print(episode_id, main_results[episode_id], help_results[episode_id])
            except:
                ipdb.set_trace()

            if (
                np.mean(help_results[episode_id]["S"])
                > 0
                # and 250 in help_results[episode_id]["L"]
            ):
                tmp_list = list(help_results[episode_id]["S"])
                help_results[episode_id]["S"] = [x for x in tmp_list if x > 0]
                tmp_list = list(help_results[episode_id]["L"])
                help_results[episode_id]["L"] = [x for x in tmp_list if x < 250]

        # if np.mean(help_results[episode_id]["S"]) < np.mean(
        #     main_results[episode_id]["S"]
        # ):
        #     log_file_name = args.record_dir + "/logs_episode.{}_iter.{}.pik".format(
        #         episode_id, 0
        #     )
        #     log_res = pickle.load(open(log_file_name, "rb"))
        #     ipdb.set_trace()
        # print(main_results[152])
        # print(help_results)

        SR, AL, SP, SWS, stdSR, stdL, stdSP, stdR = get_metrics_reward(
            main_results,
            help_results,
            episode_ids,
            num_tries,
            time_limit=args.max_episode_length,
        )

        # Success, Length, SpeedUp, Reward 
        print("SR", "AL", "SP", "Reward")
        print(SR, AL, SP, SWS)

        print("stdSR", "stdL", "stdSP", "stdR")
        results[name_exp] = {}
        print(stdSR, stdL, stdSP, stdR)
        results[name_exp]['all'] = {
            'SR': [SR, stdSR],
            'AL': [AL, stdL],
            'SP': [SP, stdSP],
            'Reward': [SWS, stdR]
        }
        task_types = task_dict.keys()
        for task_type in task_types:
            episode_ids_task = task_dict[task_type]
            episode_ids_task = list(set(episode_ids).intersection(episode_ids_task))
            SR, AL, SP, SWS, stdSR, stdL, stdSP, stdR = get_metrics_reward(
                main_results,
                help_results,
                episode_ids_task,
                num_tries,
                time_limit=args.max_episode_length,
            )
            results[name_exp][task_type] = {
                'SR': [SR, stdSR],
                'AL': [AL, stdL],
                'SP': [SP, stdSP],
                'Reward': [SWS, stdR]
            }   

    with open(f"{get_original_cwd()}/results_eval_helping.pik", "wb+") as f:
        pkl.dump(results, f)

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("running time: %s sec" % (time.time() - start_time))
