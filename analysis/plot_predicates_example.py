import sys

sys.path.append(".")
import shutil
import os
import logging
import traceback
import random
import pickle
import copy
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
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


info_objects = {
    "objects_inside": [
        "bathroomcabinet",
        "kitchencabinet",
        "cabinet",
        "fridge",
        "stove",
        "dishwasher",
        "microwave",
    ],
    "objects_surface": [
        "bench",
        "cabinet",
        "chair",
        "coffeetable",
        "desk",
        "kitchencounter",
        "kitchentable",
        "nightstand",
        "sofa",
    ],
    "objects_grab": [
        "apple",
        "book",
        "coffeepot",
        "cupcake",
        "cutleryfork",
        "juice",
        "pancake",
        "plate",
        "poundcake",
        "pudding",
        "remotecontrol",
        "waterglass",
        "whippedcream",
        "wine",
        "wineglass",
    ],
    "others": ["character"],
}


def get_pred_name(container_name):
    pred_name = "on"
    room_list = ["kitchen", "livingroom", "bedroom", "bathroom"]
    if container_name in info_objects["objects_inside"] or container_name in room_list:
        pred_name = "inside"
    return pred_name


def get_edge_class(pred, t, source="pred"):
    # index 0 has pred, index 1 has mask, index 2 has input

    # pred_edge_prob = pred['edge_prob']
    # print(len(pred['edge_input'][t]), len(pred['edge_pred'][t]))
    t = min(t, len(pred) - 1)
    if source == "pred":
        curr_task_dict = pred[t][0]
    else:
        curr_task_dict = pred[t][2]

    # which edge to use?
    return {
        "{}_{}_{}".format(get_pred_name(obj2), obj1, obj2): num
        for (obj1, obj2), num in curr_task_dict.items()
    }


@hydra.main(config_path="../config/", config_name="config_default_toy_excl_plan")
def main(cfg: DictConfig):
    config = cfg
    print("Config")
    print(OmegaConf.to_yaml(cfg))
    args = config
    args_pred = args.agent_pred_graph
    num_proc = 0

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

    # cachedir = "/data/vision/torralba/frames/data_acquisition/SyntheticStories/agent_preferences/tshu/agent_preferences/outputs/helping_states_1_3_ip1_detfull_encoder_task_graph_20_1.0_1.0_5.0"
    # cachedir = f"{get_original_cwd()}/outputs/helping_states_nohold_20_1.0_1.0"
    # cachedir = f"{get_original_cwd()}/outputs/helping_states_20_1.0_1.0"
    # cachedir = f"{get_original_cwd()}/outputs/helping_states_newvaefull_encoder_task_graph_10_1.0_1.0_5.0"
    # cachedir = f"{get_original_cwd()}/outputs/helping_states_newvaefull_encoder_task_graph.kl0.001_10_1.0_1.0_5.0"
    # cachedir = f"{get_original_cwd()}/outputs/helping_states_ip1_newvaefull_encoder_task_graph.kl0.001_20_1.0_1.0_5.0"
    # cachedir = f"{get_original_cwd()}/outputs/helping_states_fastwalk_1_3_ip1_newvaefull_encoder_task_graph.kl0.001_20_1.0_1.0_5.0"
    # cachedir = f"{get_original_cwd()}/outputs/helping_states_ip1_detfull_encoder_task_graph_20_1.0_1.0_5.0"
    # cachedir = f"{get_original_cwd()}/outputs/helping_states_fastwalk_1_3_ip1_detfull_encoder_task_graph_20_1.0_1.0_5.0"
    # cachedir = f"{get_original_cwd()}/outputs/helping_states_detfull_encoder_task_graph_20_1.0_1.0_5.0"

    # cachedir = f'{get_original_cwd()}/outputs/helping_action_freq_v2_20'
    # cachedir = f'{get_original_cwd()}/outputs/helping_action_freq_1'

    cachedir_main = f"{get_original_cwd()}/outputs/main_agent_only_large"
    cachedir_main = "/data/vision/torralba/frames/data_acquisition/SyntheticStories/agent_preferences/tshu/agent_preferences/outputs/main_agent_only_large"

    # # =======================
    # # oracle
    # # =======================
    # cachedir = f"{get_original_cwd()}/outputs/helping_gt_goal"

    # =======================
    # ours
    # =======================
    cachedir = f"/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/agent_preferences//results/results_smallset_help/helping_states_fastwalk_r15_0_5_ip1_detfull_alldata_20_1.0_1.0_5.0"
    # cachedir = f"{get_original_cwd()}/outputs/helping_states_fastwalk_r2_1_3_ip1_detfull_encoder_task_graph_20_1.0_1.0_5.0"

    # # =======================
    # # single particle
    # # =======================
    # # cachedir = f"/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/agent_preferences//results/results_smallset_help/helping_states_fastwalk_r15_0_5_ip0_detfull_alldata_1_1.0_1.0_5.0"
    #  cachedir = f"{get_original_cwd()}/outputs/helping_action_freq_fastwalk_r15_0_3_ip1_detfull_encoder_task_graph_1_1.0_1.0_5.0"

    # # =======================
    # # w/o inv plan
    # # =======================
    # cachedir = f"{get_original_cwd()}/outputs/helping_states_fastwalk_r15_0_3_ip0_detfull_encoder_task_graph_20_1.0_1.0_5.0"

    # # # =======================
    # # ours w/ uniform proposals
    # # # =======================
    # cachedir = f"{get_original_cwd()}/outputs/helping_states_fastwalk_r_1_3_ip1_uniform_20_1.0_1.0_5.0"

    # # # =======================
    # # empowerment
    # # # =======================
    # cachedir = f"{get_original_cwd()}/outputs/helping_empowerment_fastwalk_r15_1_3_ip0_uniform_20_1.0_1.0_5.0"

    # # =======================
    # # action frequency
    # # =======================
    # # cachedir = f"{get_original_cwd()}/outputs/helping_action_freq_fastwalk_r15_1_3_ip1_detfull_encoder_task_graph_20_1.0_1.0_5.0"
    # cachedir = f"{get_original_cwd()}/outputs/helping_action_freq_fastwalk_r15_0_3_ip1_detfull_encoder_task_graph_20_1.0_1.0_5.0"
    # # cachedir = f"{get_original_cwd()}/outputs/helping_action_freq_fastwalk_r15_0_3_ip1_detfull_r0.05_20_1.0_1.0_5.0"

    # # # =======================
    # # # ours w/o returning
    # # # =======================
    # cachedir = f"{get_original_cwd()}/outputs/helping_states_fastwalk_r15_0_3_ip1_detfull_encoder_task_graph_20_1.0_1.0_0.0"

    agent_types = [
        ["full", 0, 0.05, False, 0, "uniform"],  # 0
        ["full", 0.5, 0.01, False, 0, "uniform"],  # 1
        ["full", -5, 0.05, False, 0, "uniform"],  # 2
        ["partial", 0, 0.05, False, 0, "uniform"],  # 3
        ["partial", 0, 0.05, False, 0, "spiked"],  # 4
        ["partial", 0, 0.05, False, 0.2, "uniform"],  # 5
        ["partial", 0, 0.01, False, 0.01, "spiked"],  # 6
        ["partial", -5, 0.05, False, 0.2, "uniform"],  # 7
        ["partial", 0.5, 0.05, False, 0.2, "uniform"],  # 8
    ]
    random_start = random.Random()
    agent_id = 0
    (
        args.obs_type,
        open_cost,
        walk_cost,
        should_close,
        forget_rate,
        belief_type,
    ) = agent_types[0]
    datafile = args.dataset_path.split("/")[-1].replace(".pik", "")
    agent_args = {
        "obs_type": args.obs_type,
        "open_cost": open_cost,
        "should_close": should_close,
        "walk_cost": walk_cost,
        "belief": {"forget_rate": forget_rate, "belief_type": belief_type},
    }
    # TODO: add num_samples to the argument
    num_samples = args.num_samples
    num_processes = args.num_processes
    args.mode = "{}_".format(agent_id + 1) + "action_freq_{}".format(num_samples)
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

    fig_dir = "{}/analysis/plots/examples_{}".format(
        get_original_cwd(), cachedir.split("/")[-1]
    )
    Path(fig_dir).mkdir(parents=True, exist_ok=True)

    # episode_ids = [3]

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

        print(iter_id, len(test_results))

        if not os.path.isfile(record_dir_main + "/results_{}.pik".format(iter_id)):
            test_results = {}
        else:
            test_results = pickle.load(
                open(record_dir_main + "/results_{}.pik".format(iter_id), "rb")
            )
            main_results = dict(test_results)

        # print(test_results)

    print(len(help_results))
    results = {}

    num_tries = 3

    main_dist = {}
    helper_dist = {}
    all_dist = []

    for episode_id in help_results:
        for iter_id in range(0, 1):
            if episode_id not in [523]:
                continue
            log_file_name = args.record_dir + "/logs_episode.{}_iter.{}.pik".format(
                episode_id, iter_id
            )
            if os.path.isfile(log_file_name):
                for shown_len in range(1, 47):
                    helper_log = pickle.load(open(log_file_name, "rb"))
                    T = len(helper_log["proposals"])
                    pred_goals = [None] * T
                    all_predicates = []
                    all_objects = []
                    all_locations = []
                    all_cnts = []
                    for t in range(T):
                        proposals = helper_log["proposals"][t]
                        pred_goals[t] = []
                        for pred_id, proposal in proposals.items():
                            pred_goal = get_edge_class(
                                proposal["pred"], len(proposal["pred"]) - 1
                            )
                            pred_goals[t].append(pred_goal)
                            all_predicates += [
                                f"{pred}: {cnt}" for pred, cnt in pred_goal.items()
                            ]
                            all_objects += [
                                pred.split("_")[1] for pred, cnt in pred_goal.items()
                            ]
                            all_locations += [
                                pred.split("_")[2] for pred, cnt in pred_goal.items()
                            ]
                            all_cnts += [f"{cnt}" for pred, cnt in pred_goal.items()]

                    all_predicates = sorted(list(set(all_predicates)))
                    predicates_mean_seq = {}
                    predicates_std_seq = {}

                    all_objects = sorted(list(set(all_objects)))
                    objects_mean_seq = {}
                    objects_std_seq = {}

                    all_locations = sorted(list(set(all_locations)))
                    locations_mean_seq = {}
                    locations_std_seq = {}

                    all_cnts = sorted(list(set(all_cnts)))
                    cnts_mean_seq = {}
                    cnts_std_seq = {}

                    different_preds = []

                    max_cnt = {}

                    filtered_predicastes = []

                    for predicate in all_predicates:
                        if "character" in predicate:
                            continue
                        filtered_predicastes.append(predicate)
                        predicates_mean_seq[predicate] = []
                        predicates_std_seq[predicate] = []
                        pred, cnt = predicate.split(": ")
                        cnt = int(cnt)
                        for t in range(T):
                            cnt_list = []
                            for pred_goal in pred_goals[t]:
                                if pred in pred_goal and pred_goal[pred] == cnt:
                                    cnt_list.append(1)
                                else:
                                    cnt_list.append(0)
                            predicates_mean_seq[predicate].append(np.mean(cnt_list))
                            predicates_std_seq[predicate].append(np.mean(cnt_list))

                        mean_seq = np.array(predicates_mean_seq[predicate])
                        std_seq = np.array((predicates_std_seq[predicate]))
                        # if max(mean_seq) > 0.4 or predicates_mean_seq[predicate][0] > 0:
                        if pred not in different_preds:
                            different_preds.append(pred)
                            max_cnt[pred] = cnt
                        else:
                            max_cnt[pred] = max(max_cnt[pred], cnt)

                    colormap = plt.cm.gist_ncar  # nipy_spectral, Set1,Paired
                    colors = [
                        colormap(i) for i in np.linspace(0, 1, len(different_preds))
                    ]
                    alpha_levels = [0, 0.2, 0.4, 1.0]

                    colors_rgb = {
                        # 1: {1: [1.0, 0.4, 1.0], 2: [1.0, 0, 1.0], 3: [0.6, 0.0, 0.6]},
                        0: {
                            1: [0.75, 0.75, 0.75],
                            2: [0.5, 0.5, 0.5],
                            3: [0.25, 0.25, 0.25],
                        },
                        1: {1: [1.0, 0.6, 0.6], 2: [1.0, 0.2, 0.2], 3: [0.8, 0.0, 0.0]},
                        2: {1: [0.8, 1.0, 0.6], 2: [0.6, 1.0, 0.2], 3: [0.2, 0.8, 0.0]},
                        3: {1: [1.0, 0.7, 0.4]},
                        4: {1: [0.6, 0.8, 1.0], 2: [0.2, 0.6, 1.0], 3: [0, 0.4, 0.8]},
                    }

                    legend_lists = []
                    fig, ax = plt.subplots(figsize=(12, 5.5))
                    fig_name = "{}/{}_{}_{}.pdf".format(
                        fig_dir, episode_id, iter_id, shown_len
                    )
                    selected_predicates = [
                        # "on_apple_coffeetable: 1",
                        "on_salmon_coffeetable: 1",
                        "on_salmon_coffeetable: 2",
                        "on_cutleryfork_kitchentable: 1",
                        "on_cutleryfork_kitchentable: 2",
                        "on_cutleryfork_kitchentable: 3",
                        "on_plate_kitchentable: 1",
                        "on_plate_kitchentable: 2",
                        "on_plate_kitchentable: 3",
                        "on_wineglass_coffeetable: 1",
                        "on_wineglass_kitchentable: 1",
                        "on_wineglass_kitchentable: 2",
                        "on_wineglass_kitchentable: 3",
                    ]
                    shown_predicates = []
                    curves = []
                    for predicate in selected_predicates:
                        if "character" in predicate:
                            continue
                        if (
                            predicate != "on_wineglass_coffeetable: 1"
                            # and not predicate.startswith("on_apple_coffeetable")
                            and not predicate.startswith("on_salmon_coffeetable")
                        ):
                            goal_objects = ["plate", "cutleryfork", "wineglass"]
                            relevant = False
                            for goal_object in goal_objects:
                                if goal_object in predicate:
                                    relevant = True
                                    break
                            if not relevant:
                                continue
                            relevant = False
                            goal_locations = ["kitchentable"]
                            for goal_location in goal_locations:
                                if goal_location in predicate:
                                    relevant = True
                                    break

                            if not relevant:
                                continue

                        print(predicate)

                        mean_seq = list(predicates_mean_seq[predicate])
                        # if max(mean_seq) > 0.4:
                        pred, cnt = predicate.split(": ")
                        if pred not in shown_predicates:
                            shown_predicates.append(pred)
                        cnt = int(cnt)
                        color_index = shown_predicates.index(pred)

                        ax.plot(
                            range(0, shown_len + 1),
                            [mean_seq[0]] + mean_seq[:shown_len],
                            color=colors_rgb[color_index][cnt]
                            # color=f"C{color_index}",  # colors[color_index],
                            # alpha=alpha_levels[cnt],
                        )
                        elements = pred.replace("cutlery", "").split("_")
                        label = "{}({},{}): {}".format(
                            elements[0].upper(), elements[1], elements[2], cnt
                        )

                        legend_lists.append(label)
                        # ax.fill_between(
                        #     range(T), mean_seq - std_seq, mean_seq + std_seq, alpha=0.5
                        # )
                    # plt.legend(
                    #     legend_lists,
                    #     bbox_to_anchor=(1.04, 1),
                    #     loc="upper right",
                    #     fontsize=12,
                    # )
                    legend = plt.legend(
                        legend_lists,
                        bbox_to_anchor=(0.5, 1.35),
                        loc="upper center",
                        fontsize=12,
                        ncol=3,
                        facecolor="white",
                        framealpha=1,
                    )
                    ax.set_xlim([0, len(mean_seq)])
                    ax.set_ylim([-0.05, 1.05])
                    plt.tight_layout()
                    plt.savefig(fig_name)
                    print(fig_name)

                # # objects
                # filtered_objects = []
                # for obj in all_objects:
                #     if "character" in obj:
                #         continue
                #     filtered_objects.append(obj)
                #     objects_mean_seq[obj] = []
                #     objects_std_seq[obj] = []
                #     for t in range(T):
                #         cnt_list = []
                #         for pred_goal in pred_goals[t]:
                #             exists = False
                #             for predicate in pred_goal:
                #                 if obj in predicate:
                #                     exists = True
                #                     break
                #             cnt_list.append(int(exists))
                #         objects_mean_seq[obj].append(np.mean(cnt_list))
                #         objects_std_seq[obj].append(np.mean(cnt_list))

                #     mean_seq = np.array(objects_mean_seq[obj])
                #     std_seq = np.array((objects_std_seq[obj]))
                #     # if max(mean_seq) > 0.4 or predicates_mean_seq[predicate][0] > 0:

                # legend_lists = []
                # fig, ax = plt.subplots(figsize=(12, 3))
                # fig_name = "{}/{}_{}_obj.pdf".format(fig_dir, episode_id, iter_id)
                # shown_objects = []
                # for obj in filtered_objects:
                #     if "character" in obj:
                #         continue
                #     mean_seq = np.array(objects_mean_seq[obj])
                #     if obj not in shown_objects:
                #         shown_objects.append(obj)
                #     color_index = shown_objects.index(obj)

                #     ax.plot(
                #         mean_seq,
                #         # color=colors_rgb[color_index][cnt]
                #         color=f"C{color_index}",  # colors[color_index],
                #         # alpha=alpha_levels[cnt],
                #     )
                #     legend_lists.append(obj.replace("cutlery", ""))
                # # plt.legend(
                # #     shown_objects,
                # #     bbox_to_anchor=(1.04, 1),
                # #     loc="upper left",
                # #     fontsize=12,
                # # )
                # plt.tight_layout()
                # plt.savefig(fig_name)

                # # locations
                # filtered_locations = []
                # for loc in all_locations:
                #     if loc in [
                #         "kitchen",
                #         "livingroom",
                #         "bedroom",
                #         "bathroom",
                #         "plate",
                #     ]:
                #         continue
                #     filtered_locations.append(loc)
                #     locations_mean_seq[loc] = []
                #     locations_std_seq[loc] = []
                #     for t in range(T):
                #         cnt_list = []
                #         for pred_goal in pred_goals[t]:
                #             exists = False
                #             for predicate in pred_goal:
                #                 if loc in predicate:
                #                     exists = True
                #                     break
                #             cnt_list.append(int(exists))
                #         locations_mean_seq[loc].append(
                #             np.mean(cnt_list) if t < 10 or loc != "coffeetable" else 0
                #         )
                #         locations_std_seq[loc].append(np.mean(cnt_list))

                #     mean_seq = np.array(locations_mean_seq[loc])
                #     std_seq = np.array((locations_std_seq[loc]))
                #     # if max(mean_seq) > 0.4 or predicates_mean_seq[predicate][0] > 0:

                # legend_lists = []
                # fig, ax = plt.subplots(figsize=(12, 3))
                # fig_name = "{}/{}_{}_loc.pdf".format(fig_dir, episode_id, iter_id)
                # shown_locations = []
                # for loc in filtered_locations:
                #     mean_seq = np.array(locations_mean_seq[loc])
                #     if loc not in shown_locations:
                #         shown_locations.append(loc)
                #     color_index = shown_locations.index(loc)

                #     ax.plot(
                #         mean_seq,
                #         # color=colors_rgb[color_index][cnt]
                #         color=f"C{color_index}",  # colors[color_index],
                #         # alpha=alpha_levels[cnt],
                #     )
                #     legend_lists.append(loc)
                # # plt.legend(
                # #     shown_locations,
                # #     bbox_to_anchor=(1.04, 1),
                # #     loc="upper left",
                # #     fontsize=12,
                # # )
                # plt.tight_layout()
                # plt.savefig(fig_name)

                # # cnts
                # filtered_cnts = []
                # for cnt in all_cnts:
                #     filtered_cnts.append(cnt)
                #     cnts_mean_seq[cnt] = []
                #     cnts_std_seq[cnt] = []
                #     for t in range(T):
                #         cnt_list = []
                #         for pred_goal in pred_goals[t]:
                #             exists = False
                #             for predicate, count in pred_goal.items():
                #                 if "character" in predicate:
                #                     continue
                #                 # print(predicate, count)
                #                 # ipdb.set_trace()
                #                 goal_objects = [
                #                     "plate",
                #                     "cutleryfork",
                #                     "wineglass",
                #                     "apple",
                #                     "salmon",
                #                 ]
                #                 relevant = False
                #                 for goal_object in goal_objects:
                #                     if goal_object in predicate:
                #                         relevant = True
                #                         break
                #                 if not relevant:
                #                     continue
                #                 relevant = False
                #                 goal_locations = ["kitchentable"]
                #                 for goal_location in goal_locations:
                #                     if goal_location in predicate:
                #                         relevant = True
                #                         break
                #                 if not relevant:
                #                     continue
                #                 if int(cnt) == count:
                #                     exists = True
                #                     break
                #             cnt_list.append(int(exists))
                #         cnts_mean_seq[cnt].append(np.mean(cnt_list))
                #         cnts_std_seq[cnt].append(np.mean(cnt_list))

                #     mean_seq = np.array(cnts_mean_seq[cnt])
                #     std_seq = np.array((cnts_std_seq[cnt]))
                #     # if max(mean_seq) > 0.4 or predicates_mean_seq[predicate][0] > 0:

                # legend_lists = []
                # fig, ax = plt.subplots(figsize=(12, 3))
                # fig_name = "{}/{}_{}_cnt.pdf".format(fig_dir, episode_id, iter_id)
                # shown_cnts = []
                # for cnt in filtered_cnts:
                #     mean_seq = np.array(cnts_mean_seq[cnt])
                #     if np.mean(mean_seq) < 0.1:
                #         continue
                #     if cnt not in shown_cnts:
                #         shown_cnts.append(cnt)
                #     color_index = shown_cnts.index(cnt)

                #     ax.plot(
                #         mean_seq,
                #         # color=colors_rgb[color_index][cnt]
                #         color=f"C{color_index}",  # colors[color_index],
                #         # alpha=alpha_levels[cnt],
                #     )
                #     legend_lists.append(cnt)
                # # plt.legend(
                # #     shown_cnts,
                # #     bbox_to_anchor=(1.04, 1),
                # #     loc="upper left",
                # #     fontsize=12,
                # # )
                # plt.tight_layout()
                # plt.savefig(fig_name)

                # # print("main")
                # # for t, action in enumerate(helper_log["action"][0][1:]):
                # #     print(t, action)
                # # print("helper")
                # # for t, action in enumerate(helper_log["action"][1]):
                # #     print(t, action)
                # # # ipdb.set_trace()


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("running time: %s sec" % (time.time() - start_time))
