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
import pdb
import ipdb

from torch import nn
import hydra
import time
import multiprocessing as mp
from omegaconf import DictConfig, OmegaConf
from dataloader.dataloader_v2 import AgentTypeDataset
from dataloader import dataloader_v2 as dataloader_v2
from models import agent_pref_policy_task as agent_pref_policy
from hydra.utils import get_original_cwd, to_absolute_path
from utils import utils_models_wb, utils_rl_agent

from envs.unity_environment import UnityEnvironment
from agents import MCTS_agent, MCTS_agent_particle_v2_instance, MCTS_agent_particle

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

all_object_types = [
    "chips",
    "remotecontrol",
    "condimentbottle",
    "condimentshaker",
    "salmon",
    "apple",
    "cupcake",
    "pudding",
    "wineglass",
    "waterglass",
    "plate",
    "cutleryfork",
]


def get_class_mode(agent_args):
    mode_str = "{}_opencost{}_closecost{}_walkcost{}_forgetrate{}".format(
        agent_args["obs_type"],
        agent_args["open_cost"],
        agent_args["should_close"],
        agent_args["walk_cost"],
        agent_args["belief"]["forget_rate"],
    )
    return mode_str


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


def get_class_from_state(state):
    id2node = {node["id"]: node["class_name"] for node in state["nodes"]}
    edges = state["edges"]
    nodes = state["nodes"]

    edge_class_count = {}

    num_edges = len(edges)
    # print(pred_from_ids[t], num_edges)
    for edge_id in range(num_edges):
        from_id = edges[edge_id]["from_id"]
        to_id = edges[edge_id]["to_id"]
        from_node_name = id2node[from_id]
        to_node_name = id2node[to_id]
        # if object_name in from_node_name or object_name in to_node_name:
        edge_name = edges[edge_id]["relation_type"].lower()
        if edge_name in ["inside", "on"]:  # disregard room locations + plate
            if to_node_name in [
                "kitchen",
                "livingroom",
                "bedroom",
                "bathroom",
                "plate",
            ]:
                continue
            if from_node_name in [
                "kitchen",
                "livingroom",
                "bedroom",
                "bathroom",
                "character",
            ]:
                continue
        else:
            continue

        edge_class = "{}_{}_{}".format(edge_name, from_node_name, to_node_name)
        if edge_class not in edge_class_count:
            edge_class_count[edge_class] = 1
        else:
            edge_class_count[edge_class] += 1
    return edge_class_count


def compute_dist_instance(init_state, curr_state, subgoal_instance):
    init_edge_class = get_class_from_state(init_state)
    curr_edge_class = get_class_from_state(curr_state)
    base_dist = 0
    for edge in init_edge_class:
        if edge not in curr_edge_class:
            curr_edge_class[edge] = 0
    for edge in curr_edge_class:
        if edge not in init_edge_class:
            init_edge_class[edge] = 0
    for edge in init_edge_class:
        # if init_edge_class[edge] != curr_edge_class[edge]:
        #     print(edge, init_edge_class[edge], curr_edge_class[edge])
        base_dist += abs(init_edge_class[edge] - curr_edge_class[edge])

    id2node = {node["id"]: node["class_name"] for node in curr_state["nodes"]}

    elements = subgoal_instance.split("_")
    edge_type, from_id, to_id = (
        elements[0],
        int(elements[1].split(".")[1]),
        int(elements[2].split(".")[1]),
    )
    object_id = from_id if edge_type in ["on", "inside"] else to_id
    found = False
    for edge in curr_state["edges"]:
        if edge["from_id"] == object_id:
            edge_class = "{}_{}_{}".format(
                edge["relation_type"].lower(),
                id2node[edge["from_id"]],
                id2node[edge["to_id"]],
            )
            # print(edge_class)
            if edge["relation_type"] in ["ON", "INSIDE"]:
                if id2node[edge["to_id"]] in [
                    "kitchen",
                    "livingroom",
                    "bedroom",
                    "bathroom",
                    "plate",
                ]:
                    continue

                curr_edge_class[edge_class] -= 1
                # print(edge_class, "-")
                found = True
    if edge_type in ["on", "inside"]:
        edge_class = "{}_{}_{}".format(
            edge_type, elements[1].split(".")[0], elements[2].split(".")[0]
        )
        # print(edge_class, "+")
        if edge_class not in curr_edge_class:
            curr_edge_class[edge_class] = 1
        else:
            curr_edge_class[edge_class] += 1
    dist = -base_dist
    for edge in init_edge_class:
        if edge not in curr_edge_class:
            curr_edge_class[edge] = 0
    for edge in curr_edge_class:
        if edge not in init_edge_class:
            init_edge_class[edge] = 0
    for edge in init_edge_class:
        # if init_edge_class[edge] != curr_edge_class[edge]:
        #     print(edge, init_edge_class[edge], curr_edge_class[edge])
        dist += abs(init_edge_class[edge] - curr_edge_class[edge])

    if dist == 1 and edge_type in ["on", "inside"]:
        if not found:
            dist += 1
    return dist


def get_pred_name(container_name):
    pred_name = "on"
    room_list = ["kitchen", "livingroom", "bedroom", "bathroom"]
    if container_name in info_objects["objects_inside"] or container_name in room_list:
        pred_name = "inside"
    return pred_name


def get_edge_instance(pred, class2id, gt_container_id, t, source="pred"):
    # pred_edge_prob = pred['edge_prob']
    # print(len(pred['edge_input'][t]), len(pred['edge_pred'][t]))

    t = min(t, len(pred) - 1)

    # Note: we will assume that all predicates should be achieved, some of the predicates
    # will be already correct, but this should make sure we don't undo tasks that
    # are already correct

    preds = pred[t][0]
    input_graph = pred[t][2]
    edge_pred_ins = {}
    # ipdb.set_trace()
    for predicate_name, count in preds.items():
        # print(cpred)
        # pred_name, count = cpred.split(':')
        obj_name, container_name = predicate_name
        obj_name, container_name = obj_name.strip(), container_name.strip()

        pred_name = get_pred_name(container_name)
        if obj_name == "character" or container_name in [
            "kitchen",
            "livingroom",
            "bedroom",
            "bathroom",
            "plate",
        ]:
            continue

        if obj_name in class2id:
            if gt_container_id in class2id[container_name]:
                container_id = gt_container_id
            else:
                container_id = class2id[container_name][0]
            pred_name = f"{pred_name}_{obj_name}_{container_id}"
            edge_pred_ins[pred_name] = {
                "count": count,
                "grab_obj_ids": class2id[obj_name],
                "container_ids": [container_id],
            }
        else:
            # Wrong prediction, this prediction did not exist
            pass
    # if len(edge_pred_ins) == 0:
    #     ipdb.set_trace()
    return edge_pred_ins


def get_subgoals_from_init_state(state):
    id2node = {node["id"]: node["class_name"] for node in state["nodes"]}
    subgoals = []
    for edge in state["edges"]:
        edge_name = edge["relation_type"].lower()
        from_node_id = edge["from_id"]
        from_node_name = id2node[from_node_id]
        to_node_id = edge["to_id"]
        to_node_name = id2node[edge["to_id"]]
        if edge_name in ["inside", "on"]:  # disregard room locations + plate
            if to_node_name in [
                "kitchen",
                "livingroom",
                "bedroom",
                "bathroom",
                "plate",
            ]:
                continue

            if from_node_name not in [
                "plate",
                "cutleryfork",
                "waterglass",
                "cupcake",
                "salmon",
                "apple",
                "remotecontrol",
                "chips",
                "condimentbottle",
                "condimentshaker",
                "wineglass",
                "pudding",
            ]:
                continue
            edge_instance = "{}_{}.{}_{}.{}_init".format(
                edge_name, from_node_name, from_node_id, to_node_name, to_node_id
            )
            subgoals.append(edge_instance)
    return subgoals


def get_edge_instance_from_pred(pred, class2id, gt_container_id):
    # pred_edge_prob = pred['edge_prob']
    # print(len(pred['edge_input'][t]), len(pred['edge_pred'][t]))

    preds = pred[-1][0]
    input_graph = pred[-1][2]

    edge_pred_ins = {}
    edge_list = []

    for predicate_name, count in preds.items():
        # print(cpred)
        # pred_name, count = cpred.split(':')
        from_node_name, to_node_name = predicate_name
        from_node_name, to_node_name = from_node_name.strip(), to_node_name.strip()

        edge_name = get_pred_name(to_node_name)

        # if "hold" in edge_name:  # ignore left or right hand
        #     edge_name = "offer"
        #     # ipdb.set_trace()
        #     # continue  # TODO: add handing over plan

        if edge_name in ["inside", "on"]:  # disregard room locations + plate
            if to_node_name in [
                "kitchen",
                "livingroom",
                "bedroom",
                "bathroom",
                "plate",
            ]:
                continue

        if from_node_name not in [
            "plate",
            "cutleryfork",
            "waterglass",
            "cupcake",
            "salmon",
            "apple",
            "remotecontrol",
            "chips",
            "condimentbottle",
            "condimentshaker",
            "wineglass",
            "pudding",
        ]:
            continue
        elif edge_name in ["close"]:
            continue

        if from_node_name in class2id:
            if gt_container_id in class2id[to_node_name]:
                to_node_id = gt_container_id
            else:
                to_node_id = class2id[to_node_name][0]
            edge_class = f"{edge_name}_{from_node_name}_{to_node_id}"
            edge_pred_ins[edge_name] = {
                "count": count,
                "grab_obj_ids": class2id[from_node_name],
                "container_ids": [to_node_id],
            }
            for from_node_id in class2id[from_node_name]:
                edge_list.append(
                    "{}_{}.{}_{}.{}".format(
                        edge_name,
                        from_node_name,
                        from_node_id,
                        to_node_name,
                        to_node_id,
                    )
                )
        else:
            # Wrong prediction, this prediction did not exist
            pass

    # print(edge_list)
    # ipdb.set_trace()
    return edge_pred_ins, edge_list


def get_edge_instance_from_state(state):
    id2node = {node["id"]: node["class_name"] for node in state["nodes"]}
    # print(id2node)
    edge_pred_ins = {}
    edge_list = []
    for edge in state["edges"]:
        edge_name, from_id, to_id = (
            edge["relation_type"].lower(),
            edge["from_id"],
            edge["to_id"],
        )
        if "hold" in edge_name:  # ignore left or right hand
            edge_name = "offer"
            # ipdb.set_trace()
            # continue  # TODO: add handing over plan
        from_node_name = id2node[from_id]
        to_node_name = id2node[to_id]
        if edge_name in ["inside", "on"]:  # disregard room locations + plate
            if to_node_name in [
                "kitchen",
                "livingroom",
                "bedroom",
                "bathroom",
                "plate",
            ]:
                continue
            if from_node_name.split(".")[0] in [
                "kitchen",
                "livingroom",
                "bedroom",
                "bathroom",
                "character",
            ]:
                continue
        elif edge_name in ["close"]:
            continue
        # if from_node_name not in ['apple', 'cupcake', 'plate', 'waterglass']:
        #     continue

        edge_class = "{}_{}_{}".format(edge_name, from_node_name, to_id)
        if edge_name == "offer":
            if edge_class not in edge_pred_ins:
                edge_pred_ins[edge_class] = {
                    "count": 0,
                    "grab_obj_ids": [],
                    "container_ids": [from_id],
                }
            edge_pred_ins[edge_class]["count"] += 1
            edge_pred_ins[edge_class]["grab_obj_ids"].append(to_id)
        else:
            if edge_class not in edge_pred_ins:
                edge_pred_ins[edge_class] = {
                    "count": 0,
                    "grab_obj_ids": [],
                    "container_ids": [to_id],
                }
            edge_pred_ins[edge_class]["count"] += 1
            edge_pred_ins[edge_class]["grab_obj_ids"].append(from_id)
        edge_list.append(
            "{}_{}.{}_{}.{}".format(
                edge_name, from_node_name, from_id, to_node_name, to_id
            )
        )
    return edge_pred_ins, edge_list


def aggregate_multiple_pred(preds, t, change=False):
    print("Aggregating preds")
    # ipdb.set_trace()
    t = min(t, len(preds[0]) - 1)
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


def edge2goal(edge):
    # edge format: edgeName_fromNodeName.id_toNodeName.id
    elements = edge.split("_")
    edge_name = elements[0]
    from_node_id = int(elements[1].split(".")[-1])
    from_node_name = elements[1].split(".")[0]
    to_node_id = int(elements[2].split(".")[-1])

    goal_name = "{}_{}_{}".format(edge_name, from_node_name, to_node_id)
    if edge_name == "offer":
        goal = {
            goal_name: {
                "count": 1,
                "grab_obj_ids": [to_node_id],
                "container_ids": [from_node_id],
            }
        }
    else:
        goal = {
            goal_name: {
                "count": 1,
                "grab_obj_ids": [from_node_id],
                "container_ids": [to_node_id],
            }
        }
    return goal


def edge2name(edge):
    elements = edge.split("_")
    edge_name = elements[0]
    from_node_id = int(elements[1].split(".")[-1])
    from_node_name = elements[1].split(".")[0]
    to_node_id = int(elements[2].split(".")[-1])
    to_node_name = elements[2].split(".")[0]
    if edge_name == "offer":
        goal_name = "{}_{}_{}".format(edge_name, to_node_name, from_node_id)
    else:
        goal_name = "{}_{}_{}".format(edge_name, from_node_name, to_node_id)
    return goal_name


def same_action(action1, action2):

    if action1 == action2:
        return True

    elements1 = action1.split(" ")
    elements2 = action2.split(" ")

    if elements1[0] != elements2[0]:
        return False

    if "grab" in elements1[0]:
        return elements1[1] == elements2[1]

    if "put" in elements1[0]:
        return (
            elements1[1] == elements2[1]
            and elements1[3] == elements2[3]
            and elements1[4] == elements2[4]
        )

    if "walk" in elements1[0] and elements1[1] in all_object_types:
        return elements1[1] == elements2[1]

    return False


def is_in_plan(action, plan):
    for tmp_action in plan:
        if same_action(action, tmp_action):
            return True
    return False


def is_in_goal(grabbed_obj, goals):
    """check if grabbed objects are part of the goal"""
    objs_goals = []
    for predicate in goals:
        goal_obj = predicate.split("_")[1]
        if goal_obj != "character":
            objs_goals.append(goal_obj)
    for obj, grabbed in grabbed_obj.items():
        if grabbed and obj not in objs_goals:
            # ipdb.set_trace()
            return False
    return True


def pred_main_agent_plan(
    process_id,
    pred_task,
    class2id,
    gt_container_id,
    t,
    pred_actions_fn,
    obs,
    length_plan,
    must_replan,
    agent_id,
    res,
    verbose=False
):
    inferred_goal = get_edge_instance(pred_task, class2id, gt_container_id, t)
    if verbose:
        print("pred {}:".format(process_id), inferred_goal)
    plan_states, opponent_subgoal = None, None
    if len(inferred_goal) > 0:  # if no edge prediction then None action
        opponent_actions, opponent_info = pred_actions_fn(
            obs,
            length_plan=length_plan,
            must_replan=must_replan,
            agent_id=1 - agent_id,
            inferred_goal=inferred_goal,
        )
        plan_states = opponent_info[1 - agent_id]["plan_states"]
        plan_cost = opponent_info[1 - agent_id]["plan_cost"]
        plan = opponent_info[1 - agent_id]["plan"]
        if (
            opponent_info[1 - agent_id]["subgoals"] is not None
            and len(opponent_info[1 - agent_id]["subgoals"]) > 0
        ):
            opponent_subgoal = opponent_info[1 - agent_id]["subgoals"][0][0]
        else:
            opponent_subgoal = None
        res[process_id] = (opponent_subgoal, plan, plan_states, plan_cost)
    else:
        # This particle has not plan
        res[process_id] = (None, [], [], 0.0)
    # print('main pred {}:'.format(process_id), inferred_goal)
    # print('main plan {}:'.format(process_id), plan)


def convert_walktowards(action):
    if action is not None and "walktowards" not in action:
        return action.replace("walk", "walktowards")
    else:
        return action


def get_helping_plan(
    process_id,
    edge,
    t,
    opponent_subgoal,
    get_actions_fn,
    obs,
    length_plan,
    must_replan,
    agent_id,
    res,
    verbose=False
):
    inferred_goal = edge2goal(edge)
    if verbose:
        print("pred {}:".format(process_id), inferred_goal)
    subgoal, action, plan, plan_states = None, None, None, None
    if len(inferred_goal) > 0:  # if no edge prediction then None action

        actions, info = get_actions_fn(
            obs,
            length_plan=length_plan,
            must_replan=must_replan,
            agent_id=agent_id,
            inferred_goal=inferred_goal,
            opponent_subgoal=opponent_subgoal,
        )
        # print('actions:', actions)
        if verbose:
            print("pred {}:".format(process_id), inferred_goal)
            print("plan {}:".format(process_id), opponent_subgoal, info[1]["subgoals"])
        if info[1]["subgoals"] is None or len(info[1]["subgoals"]) == 0:
            res[process_id] = (None, None, None, None)
            return

        # Here you can get the intermediate states
        plan_states = info[agent_id]["plan_states"]
        plan_cost = info[agent_id]["plan_cost"]
        plan = info[agent_id]["plan"]
        action = actions[agent_id]
        subgoal = (
            info[agent_id]["subgoals"][0][0]
            if info[agent_id]["subgoals"] is not None
            and len(info[agent_id]["subgoals"]) > 0
            else None
        )
    res[process_id] = (subgoal, plan, plan_states, plan_cost)


@hydra.main(config_path="../config/configs_for_help", config_name="config_diff_state")
def main(cfg: DictConfig):
    config = cfg
    print("Config")
    print(OmegaConf.to_yaml(cfg))
    args = config
    args_pred = args.agent_pred_graph
    num_proc = 0
    
    verbose = False
    num_tries = 3
    # args.executable_file = '/data/vision/torralba/frames/data_acquisition/SyntheticStories/website/release/simulator/v2.0/v2.2.5_beta4/linux_exec.v2.2.5_beta4.x86_64'
    # args.max_episode_length = 250
    # args.num_per_apartment = 20
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    # home_path = '../'
    rootdir = curr_dir + "/../"

    # args.dataset_path = f'{rootdir}/dataset/train_env_task_set_100_full.pik'
    # args.dataset_path = f'/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/agent_preferences/dataset/test_env_task_set_10_full.pik'
    args.dataset_path = f"/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/agent_preferences/dataset/structured_agent/test_env_task_set_60_full_task.all.pik"
    # args.dataset_path = f"/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/agent_preferences/dataset/structured_agent/dataset/dataset_graph_full_150step_larger_test.pkl"

    # args.dataset_path = './dataset/train_env_task_set_20_full_reduced_tasks_single.pik'

    valid_set_path = "/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/agent_preferences/analysis/test_set_reduced.txt"
    f = open(valid_set_path, "r")
    episode_ids = []
    for filename in f:
        episode_ids.append(int(filename.split("episode.")[-1].split("_")[0]))
    episode_ids0 = sorted(episode_ids)
    if args.small_set:
        pass
    else:
        episode_ids = list(episode_ids0)
        #episode_ids = [episode_id for episode_id in episode_ids if episode_id not in [180, 323, 523, 556, 573, 573, 591, 621]]
        #episode_ids_red = list(episode_ids0[::5])
        #episode_ids = [episode for episode in list(episode_ids0) if episode not in episode_ids_red]
        #episode_ids = sorted(set(episode_ids+[180, 323, 523, 556, 573, 573, 591, 621]))
    if args.debug:
        episode_ids = [420]
    print(len(episode_ids))
    f.close()

    network_name = args_pred.name_log

    if network_name == "newvaefull_encoder_task_graph":
        network_name += ".kl{}".format(args_pred.model.kl_coeff)
    if not args.debug:
        cachedir = f"{get_original_cwd()}/results/results_smallset_help/helping_states_fastwalk_r{int(args.reset_steps)}_{int(args.small_set)}_{args.num_tries}_ip{int(args.inv_plan)}_{network_name}_{args.num_samples}_{args.alpha}_{args.beta}_{args.lam}"
    else:
        cachedir = f"{get_original_cwd()}/results/debug_results_smallset_help/helping_states_fastwalk_r{int(args.reset_steps)}_{int(args.small_set)}_{args.num_tries}_ip{int(args.inv_plan)}_{network_name}_{args.num_samples}_{args.alpha}_{args.beta}_{args.lam}"

    # cachedir = f'{get_original_cwd()}/outputs/helping_toy_states_{args.num_samples}_{args.alpha}_{args.beta}'
    # cachedir = f'{rootdir}/dataset_episodes/helping_toy'

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

    env_task_set = pickle.load(open(args.dataset_path, "rb"))
    # print(env_task_set)
    print(len(env_task_set))

    for env in env_task_set:
        init_gr = env["init_graph"]
        gbg_can = [
            node["id"]
            for node in init_gr["nodes"]
            if node["class_name"] in ["garbagecan", "clothespile"]
        ]
        init_gr["nodes"] = [
            node for node in init_gr["nodes"] if node["id"] not in gbg_can
        ]
        init_gr["edges"] = [
            edge
            for edge in init_gr["edges"]
            if edge["from_id"] not in gbg_can and edge["to_id"] not in gbg_can
        ]
        for node in init_gr["nodes"]:
            if node["class_name"] == "cutleryfork":
                node["obj_transform"]["position"][1] += 0.1

    args.record_dir = "{}/{}".format(cachedir, datafile)
    error_dir = "{}/logging/{}".format(cachedir, datafile)
    if not os.path.exists(args.record_dir):
        os.makedirs(args.record_dir)

    if not os.path.exists(error_dir):
        os.makedirs(error_dir)

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
    # # episode_ids = episode_ids[10:]

    # episode_ids = [episode_ids[0]]

    S = {episode_id: [] for episode_id in episode_ids}
    L = {episode_id: [] for episode_id in episode_ids}

    test_results = {}

    def env_fn(env_id):
        return UnityEnvironment(
            num_agents=2,
            max_episode_length=args.max_episode_length,
            port_id=env_id,
            env_task_set=env_task_set,
            observation_types=[args.obs_type, args.obs_type],
            use_editor=args.use_editor,
            executable_args=executable_args,
            base_port=args.base_port,
            convert_goal=True,
        )

    args_common = dict(
        recursive=False,
        max_episode_length=20,
        num_simulation=100,
        max_rollout_steps=5,
        c_init=0.1,
        c_base=100,
        num_samples=1,
        num_processes=num_proc,
        num_particles=20,
        logging=True,
        logging_graphs=True,
        get_plan_states=True,
        get_plan_cost=True,
    )
    if args.obs_type == "full":
        args_common["num_particles"] = 1
    else:
        args_common["num_particles"] = 20

    args_agent1 = {"agent_id": 1, "char_index": 0}
    args_agent1.update(args_common)
    args_agent1["agent_params"] = agent_args

    args_agent2 = {"agent_id": 2, "char_index": 1}
    args_agent2.update(args_common)
    args_agent2["agent_params"] = agent_args
    args_agent2["num_simulation"] = 50

    agents = [
        lambda x, y: MCTS_agent_particle_v2_instance(**args_agent1),
        lambda x, y: MCTS_agent_particle_v2_instance(**args_agent2),
    ]
    arena = ArenaMP(args.max_episode_length, id_run, env_fn, agents, use_sim_agent=True)

    # # episode_ids = [20] #episode_ids
    # # num_tries = 1
    # episode_ids = [0]
    # ndict = {'on_book_329': 1}
    # env_task_set[91]['init_rooms'] = ['bedroom', 'bedroom']
    # env_task_set[91]['task_goal'] = {0: ndict, 1: ndict}

    # episode_ids = [0, 1, 2, 3, 4, 20, 21, 22, 23, 24]
    # episode_ids = [21, 22, 23, 24]

    for iter_id in range(num_tries):
        # if iter_id > 0:
        # iter_id = 1

        steps_list, failed_tasks = [], []
        current_tried = iter_id

        # test_results = {}
        if not os.path.isfile(args.record_dir + "/results_{}.pik".format(iter_id - 1)):
            test_results = {}
        else:
            test_results = pickle.load(
                open(args.record_dir + "/results_{}.pik".format(iter_id - 1), "rb")
            )

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # gt_dir = "/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/agent_preferences/dataset_episodes/large_data_toy/test_env_task_set_10_full/1_full_opencost0_closecostFalse_walkcost0.05_forgetrate0"
        # # pred_dir = "/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/results/predict_graph/train_data.dataset_graph_pred_train.pkl-agentsall/time_model.LSTM-stateenc.TF-edgepred.concat-lr0.0001-bs.16-goalenc.False_extended._costclose.1.0_costgoal.1.0_agentembed.False/test_env_task_set_10_full/1_full_opencost0_closecostFalse_walkcost0.05_forgetrate0"
        # # pred_dir = "/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/results/predict_graph/train_data.dataset_graph_pred_30step_train.pkl-agentsall/time_model.LSTM-stateenc.TF-globalrepr.pool-edgepred.concat-lr0.0001-bs.8-goalenc.False_extended._costclose.1.0_costgoal.1.0_agentembed.False_predchangeedge.True_inputgoal.True/test_env_task_set_10_full/1_full_opencost0_closecostFalse_walkcost0.05_forgetrate0"

        # root = "/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/results/predict_graph/train_data.dataset_graph_pred_30step_train.pkl-agentsall/"
        # pred_dir = (
        #     root
        #     + "time_model.LSTM-stateenc.TF-globalrepr.pool-edgepred.concat-lr0.0001-bs.8-goalenc.False_extended._costclose.1.0_costgoal.1.0_agentembed.False_predchange.edge_inputgoal.False_excledge.False/test_env_task_set_10_full/1_full_opencost0_closecostFalse_walkcost0.05_forgetrate0"
        #     # + "time_model.LSTM-stateenc.TF-globalrepr.pool-edgepred.concat-lr0.0001-bs.32-goalenc.False_extended._costclose.1.0_costgoal.1.0_agentembed.False_predchange.node_inputgoal.False_excledge.True/test_env_task_set_10_full/1_full_opencost0_closecostFalse_walkcost0.05_forgetrate0"
        # )
        # gt_p = Path(gt_dir).glob("*.pik")
        # ipdb.set_trace()
        if args_pred.name_log == "uniform":
            model = agent_pref_policy.UniformModel()
        else:
            model = agent_pref_policy.GraphPredNetworkVAETask3(args_pred)
            state_dict = torch.load(args_pred.ckpt_load)["model"]
            state_dict_new = {}

            for param_name, param_value in state_dict.items():
                state_dict_new[param_name.replace("module.", "")] = param_value

            model.load_state_dict(state_dict_new)
            model.eval()

        curr_file = (
            "/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah"
        )
        # dataset_test = AgentTypeDataset(
        #     path_init='{}/agent_preferences/dataset/{}'.format(
        #         curr_file, args_pred['data']['test_data']
        #     ),
        #     args_config=args_pred,
        # )
        graph_helper = utils_rl_agent.GraphHelper(
            max_num_objects=args_pred["model"]["max_nodes"],
            toy_dataset=args_pred["model"]["reduced_graph"],
        )

        num_episodes = 0
        # gt_p = [gp for gp in gt_p if 'logs_episode.26_iter.2.pik_result.pkl' in gp]
        # ipdb.set_trace()

        max_steps = args.max_episode_length

        # for env_task in env_task_set:

        for episode_id in episode_ids:
            steps_list, failed_tasks = [], []
            current_tried = iter_id

            # print('gt goal:', gt_goal)

            # episode_id = env_task['task_id']

            # if episode_id not in episode_ids:
            #     continue

            log_file_name = args.record_dir + "/logs_episode.{}_iter.{}.pik".format(
                episode_id, iter_id
            )
            failure_file = "{}/{}_{}.txt".format(error_dir, episode_id, iter_id)

            # if os.path.isfile(log_file_name):  # or os.path.isfile(failure_file):
            #     print(log_file_name)
            #     continue

            if os.path.isfile(failure_file):
                os.remove(failure_file)
            fileh = logging.FileHandler(failure_file, "a")
            fileh.setLevel(logging.DEBUG)
            logger.addHandler(fileh)

            print("episode:", episode_id)

            for it_agent, agent in enumerate(arena.agents):
                agent.seed = (it_agent + current_tried * 2) * 5

            try:
                # if True:
                obs = arena.reset(episode_id)
                init_state = obs[1]
                arena.task_goal = None
                gt_goal = arena.env.task_goal[0]

                gt_container_id = list(gt_goal.values())[0]["container_ids"][0]

                tv = False
                food = False
                dish = False

                print(arena.env.task_goal, arena.env.agent_goals)
                # print(
                #     [edge for edge in obs[0]['edges'] if edge['from_id'] == 351]
                #     or edge['to_id'] == 351
                # )
                # for t, action in enumerate(actions):
                steps = 2
                history_obs = []
                history_graph = []
                history_action = []

                grabbed_obj = {obj_type: False for obj_type in all_object_types}

                saved_info = {
                    "task_id": arena.env.task_id,
                    "env_id": arena.env.env_id,
                    "task_name": arena.env.task_name,
                    "gt_goals": arena.env.task_goal[0],
                    "goals": arena.task_goal,
                    "action": {0: [], 1: []},
                    "executed_action": {0: [], 1: []},
                    "plan": {0: [], 1: []},
                    "subgoal": {0: [], 1: []},
                    "finished": None,
                    "init_unity_graph": arena.env.init_graph,
                    "goals_finished": [],
                    "belief": {0: [], 1: []},
                    "belief_room": {0: [], 1: []},
                    "belief_graph": {0: [], 1: []},
                    "graph": [arena.env.init_unity_graph],
                    "obs": [],
                    "graph_results": [],
                    "helping_subgoal": [],
                    "opponent_subgoal": [],
                    "proposals": [],
                }

                actions, curr_info = arena.get_actions(
                    obs, length_plan=10, must_replan={0: False, 1: True}, agent_id=0
                )
                (prev_obs, reward, done, infos) = arena.step_given_action(
                    {0: actions[0]}
                )
                prev_graph = infos["graph"]

                if "satisfied_goals" in infos:
                    saved_info["goals_finished"].append(infos["satisfied_goals"])

                for agent_id in range(2):
                    if agent_id in actions:
                        saved_info["action"][agent_id].append(actions[agent_id])
                    else:
                        saved_info["action"][agent_id].append(None)
                    if agent_id == 2:
                        if (
                            agent_id in action
                            and len(saved_info["action"][agent_id]) > 1
                            and saved_info["action"][agent_id][-1]
                            == saved_info["action"][agent_id][-2]
                        ):
                            cnt_same_help_action_steps += 1
                            if cnt_same_help_action_steps >= 10:
                                early_stopping = True
                        else:
                            cnt_same_help_action_steps = 1

                if "executed_script" in infos:
                    for agent_id in range(2):
                        if agent_id in infos["executed_script"]:
                            saved_info["executed_action"][agent_id].append(
                                infos["executed_script"][agent_id]
                            )
                        else:
                            saved_info["executed_action"][agent_id].append(None)
                if "graph" in infos:
                    saved_info["graph"].append(infos["graph"])
                for agent_id, info in curr_info.items():
                    if "belief_room" in info:
                        saved_info["belief_room"][agent_id].append(info["belief_room"])
                    if "belief" in info:
                        saved_info["belief"][agent_id].append(info["belief"])
                    if "plan" in info:
                        saved_info["plan"][agent_id].append(info["plan"][:3])
                    if "obs" in info:
                        saved_info["obs"].append([node["id"] for node in info["obs"]])

                actions, curr_info = arena.get_actions(
                    prev_obs,
                    length_plan=10,
                    must_replan={0: False, 1: True},
                    agent_id=0,
                )
                prev_action = actions[0]
                history_action.append(prev_action)
                new_action = True
                ipdb.set_trace()
                (curr_obs, reward, done, infos) = arena.step_given_action(
                    {0: actions[0]}
                )
                if verbose:
                    print("agents' positions")
                    print(
                        [
                            (node["id"], node["bounding_box"]["center"])
                            for node in curr_obs[0]["nodes"]
                            if node["id"] < 3
                        ]
                    )

                curr_graph = infos["graph"]

                if "satisfied_goals" in infos:
                    saved_info["goals_finished"].append(infos["satisfied_goals"])

                for agent_id in range(2):
                    if agent_id in actions:
                        saved_info["action"][agent_id].append(actions[agent_id])
                    else:
                        saved_info["action"][agent_id].append(None)

                if "executed_script" in infos:
                    for agent_id in range(2):
                        if agent_id in infos["executed_script"]:
                            saved_info["executed_action"][agent_id].append(
                                infos["executed_script"][agent_id]
                            )
                        else:
                            saved_info["executed_action"][agent_id].append(None)

                if "graph" in infos:
                    saved_info["graph"].append(infos["graph"])
                for agent_id, info in curr_info.items():
                    if "belief_room" in info:
                        saved_info["belief_room"][agent_id].append(info["belief_room"])
                    if "belief" in info:
                        saved_info["belief"][agent_id].append(info["belief"])
                    if "plan" in info:
                        saved_info["plan"][agent_id].append(info["plan"][:3])
                    if "obs" in info:
                        saved_info["obs"].append([node["id"] for node in info["obs"]])

                # history_obs.append([node['id'] for node in curr_info[0]['obs']])
                # history_graph.append(prev_graph)

                proposals = {}
                success = False
                last_goal_edge = None
                max_plan_length = 10
                pred_main_plan_length = 15
                steps_since_last_prediction = 0

                # Build mapping from class 2 id
                class2id = {}
                for node in curr_graph["nodes"]:
                    if node["class_name"] not in class2id:
                        class2id[node["class_name"]] = []
                    class2id[node["class_name"]].append(node["id"])

                early_stopping = False
                cnt_same_action_steps = 0

                while steps < max_steps and not early_stopping:
                    steps += 1

                    # ======================================================
                    # reject inconsistent proposals
                    all_reject = False
                    if (
                        args.inv_plan
                        and len(proposals) > 0
                        and steps_since_last_prediction < args.reset_steps
                    ):
                        last_observed_main_action = history_action[-1]
                        if last_observed_main_action is not None:
                            last_observed_main_action = (
                                last_observed_main_action.replace("walktowards", "walk")
                            )
                        remained_proposals = {}
                        for pred_id, proposal in proposals.items():
                            if verbose:
                                print(pred_id)
                            goal_pred = get_edge_class(
                                proposal["pred"],
                                len(proposal["pred"]) - 1,
                            )
                            if verbose:
                                print(goal_pred)
                            if not is_in_goal(grabbed_obj, goal_pred):
                                if verbose:
                                    print("reject")
                            else:
                                print(last_observed_main_action, proposal["plan"])
                                if last_observed_main_action is None:
                                    remained_proposals[pred_id] = proposal
                                    if verbose:
                                        print("accept")
                                else:
                                    if is_in_plan(
                                        last_observed_main_action, proposal["plan"]
                                    ):
                                        remained_proposals[pred_id] = proposal
                                        if verbose:
                                            print("accept")
                                    else:
                                        if verbose:
                                            print("reject")
                        proposals = dict(remained_proposals)
                        if len(proposals) == 0:
                            all_reject = True
                        # ipdb.set_trace()
                    else:
                        proposals = {}

                    # proposals = {}

                    # new proposals
                    if new_action:
                        history_obs.append([node["id"] for node in curr_info[0]["obs"]])
                        history_graph.append(prev_graph)

                    replan_for_helper = True
                    if last_goal_edge is not None and "offer" not in last_goal_edge:
                        in_helper_hands = [
                            edge
                            for edge in curr_obs[1]["edges"]
                            if edge["from_id"] == 2 and "HOLD" in edge["relation_type"]
                        ]
                        if len(in_helper_hands) > 0:
                            inferred_goal = edge2goal(last_goal_edge)
                            if verbose:
                                print("last helper goal:", inferred_goal)
                            if (
                                len(inferred_goal) > 0
                            ):  # if no edge prediction then None action
                                actions, info = arena.get_actions(
                                    obs,
                                    length_plan=10,
                                    must_replan=[False, True],
                                    agent_id=2,
                                    inferred_goal=inferred_goal,
                                    opponent_subgoal=None,
                                )
                                helper_action = (
                                    actions[0]
                                    if actions is not None and len(actions) > 0
                                    else None
                                )
                            else:
                                helper_action = None
                            if helper_action is not None:
                                replan_for_helper = False
                            else:
                                last_goal_edge = None

                    if len(proposals) < 1:  # args.num_samples / 3:
                        steps_since_last_prediction = 0
                        if verbose:
                            print(len(history_graph))
                            print(len(history_action))
                        assert len(history_graph) == len(history_obs)
                        assert len(history_graph) == len(history_action)
                        # if len(proposals) == 0:
                        #     history_graph = [history_graph[-1:]]
                        #     history_obs = [history_obs[-1:]]
                        #     history_action = [history_action[-1:]]
                        if history_action[-1] is not None:
                            inputs_func = (
                                utils_models_wb.prepare_graph_for_task_model_diff(
                                    history_graph,
                                    history_obs,
                                    history_action,
                                    args_pred,
                                    graph_helper,
                                    batch_repeat=num_samples,
                                )
                            )
                            with torch.no_grad():
                                output_func = model(inputs_func, inference=True)
                            # ipdb.set_trace()
                            # First particle, first timestep, since all the particles have the same time graph
                            task_graph_input = graph_helper.get_task_graph(
                                inputs_func["input_task_graph"][0, 0], use_dict=True
                            )
                            task_result = []
                            num_tsteps = output_func["pred_graph"].shape[1]
                            if not model.use_vae and args.num_samples > 1:
                                # ipdb.set_trace()
                                sample = True

                                pred_graph_prob = output_func["pred_graph"]
                                if not sample:
                                    pred_graph = (
                                        pred_graph_prob.argmax(-1).cpu().numpy()
                                    )
                                else:
                                    pred_graph_prob = (
                                        nn.functional.softmax(pred_graph_prob, dim=-1)
                                        .cpu()
                                        .numpy()
                                    )
                                    pred_graph = utils_models_wb.vectorized(
                                        pred_graph_prob
                                    )

                            else:
                                if args_pred.name_log == "uniform":
                                    pred_graph = output_func["pred_graph"]
                                else:
                                    # VAE, take max
                                    pred_graph = output_func["pred_graph"].argmax(-1)
                            for ind in range(num_samples):
                                task_graphs = []
                                for tstep in range(num_tsteps):
                                    try:
                                        curr_task_graph = graph_helper.get_task_graph(
                                            pred_graph[ind, tstep], use_dict=True
                                        )
                                    except:
                                        ipdb.set_trace()
                                    # ipdb.set_trace()
                                    # curr_mask_task = mask_task_graph[ind, tstep]
                                    curr_mask_task = None
                                    task_graphs.append(
                                        (
                                            curr_task_graph,
                                            curr_mask_task,
                                            task_graph_input,
                                        )
                                    )
                                if args.debug:

                                    print(ind, task_graphs)
                                    # ipdb.set_trace()

                                goal_pred = get_edge_class(
                                    task_graphs,
                                    len(task_graphs) - 1,
                                )
                                if (
                                    is_in_goal(grabbed_obj, goal_pred)
                                    or not args.inv_plan
                                ):
                                    task_result.append(task_graphs)

                        print("planning for the helper agent")
                        action_freq = {}
                        opponent_subgoal_freq = {}
                        manager = mp.Manager()

                        if args.num_processes == 0:
                            res = {}
                            for index in range(len(task_result)):

                                pred_main_agent_plan(
                                    index,
                                    task_result[index],
                                    class2id,
                                    gt_container_id,
                                    steps - 3,
                                    arena.pred_actions,
                                    curr_obs,
                                    pred_main_plan_length,
                                    {0: True, 1: True},
                                    1,
                                    res,
                                )
                        else:
                            res = manager.dict()
                            for start_root_id in range(
                                0, len(task_result), self.num_processes
                            ):
                                end_root_id = min(
                                    start_root_id + self.num_processes, len(task_result)
                                )
                                jobs = []
                                for process_id in range(start_root_id, end_root_id):
                                    # print(process_id)
                                    p = mp.Process(
                                        target=pred_main_agent_plan,
                                        args=(
                                            process_id,
                                            task_result[process_id],
                                            class2id,
                                            gt_container_id,
                                            steps - 3,
                                            arena.pred_actions,
                                            curr_obs,
                                            15,
                                            {0: True, 1: True},
                                            1,
                                            res,
                                        ),
                                    )
                                    jobs.append(p)
                                    p.start()
                                for p in jobs:
                                    p.join()
                        # all_plan_states = []
                        edge_freq = {}
                        edge_steps = {}
                        proposals = {}
                        combined_edge_freq = {}
                        for pred_id, (
                            subgoal,
                            plan,
                            plan_states,
                            plan_cost,
                        ) in res.items():
                            proposals[pred_id] = {
                                "pred": task_result[pred_id],
                                "subgoal": subgoal,
                                "plan": plan,
                                "plan_states": plan_states,
                                "plan_cost": plan_cost,
                                "edge_steps": {},
                            }
                            if subgoal is not None:
                                if subgoal not in opponent_subgoal_freq:
                                    opponent_subgoal_freq[subgoal] = 1
                                else:
                                    opponent_subgoal_freq[subgoal] += 1
                                # print(len(plan_states))
                                if plan_states is not None and len(plan_states) > 0:
                                    # all_plan_states.append(plan_states)
                                    all_edges = []
                                    estimated_steps = 0

                                    for t, (action, state, cost) in enumerate(
                                        zip(plan, plan_states, plan_cost)
                                    ):
                                        (
                                            edge_pred_ins,
                                            edge_list,
                                        ) = get_edge_instance_from_state(state)

                                        all_edges += edge_list
                                        estimated_steps += max(int(cost + 0.5), 1)

                                        for edge in edge_list:
                                            if edge not in edge_freq:
                                                edge_freq[edge] = 0
                                                if edge not in edge_steps:
                                                    edge_steps[edge] = []
                                                edge_steps[edge].append(estimated_steps)
                                                proposals[pred_id]["edge_steps"][
                                                    edge
                                                ] = estimated_steps
                                # ipdb.set_trace()
                                edge_pred_ins, edge_list = get_edge_instance_from_pred(
                                    task_result[pred_id], class2id, gt_container_id
                                )
                                all_edges += edge_list
                                estimated_steps = 100  # TODO: tune this
                                for edge in edge_list:
                                    if edge not in edge_freq:
                                        edge_freq[edge] = 0
                                        if edge not in edge_steps:
                                            edge_steps[edge] = []
                                        edge_steps[edge].append(estimated_steps)
                                        proposals[pred_id]["edge_steps"][
                                            edge
                                        ] = estimated_steps

                                all_edges = list(set(all_edges))
                                for edge in all_edges:
                                    edge_freq[edge] += 1.0 / len(res)
                        for edge, freq in edge_freq.items():
                            edge_goal_name = edge2name(edge)
                            if edge_goal_name not in combined_edge_freq:
                                combined_edge_freq[edge_goal_name] = 0
                            combined_edge_freq[edge_goal_name] += freq
                        # print(edge_freq)
                        # ipdb.set_trace()
                    else:
                        steps_since_last_prediction += 1
                        edge_freq = {}
                        edge_steps = {}
                        combined_edge_freq = {}
                        opponent_subgoal_freq = {}
                        for pred_id, proposal in proposals.items():
                            subgoal, plan, plan_states, plan_cost = (
                                proposal["subgoal"],
                                proposal["plan"],
                                proposal["plan_states"],
                                proposal["plan_cost"],
                            )

                            if subgoal is not None:
                                if subgoal not in opponent_subgoal_freq:
                                    opponent_subgoal_freq[subgoal] = 1
                                else:
                                    opponent_subgoal_freq[subgoal] += 1
                                # print(len(plan_states))
                                all_edges = []
                                if plan_states is not None and len(plan_states) > 0:
                                    # all_plan_states.append(plan_states)
                                    estimated_steps = 0
                                    for t, (action, state, cost) in enumerate(
                                        zip(plan, plan_states, plan_cost)
                                    ):
                                        (
                                            edge_pred_ins,
                                            edge_list,
                                        ) = get_edge_instance_from_state(state)

                                        all_edges += edge_list
                                        estimated_steps += max(int(cost + 0.5), 1)

                                        for edge in edge_list:
                                            if edge not in edge_freq:
                                                edge_freq[edge] = 0
                                                if edge not in edge_steps:
                                                    edge_steps[edge] = []
                                                edge_steps[edge].append(estimated_steps)
                                edge_pred_ins, edge_list = get_edge_instance_from_pred(
                                    proposal["pred"], class2id, gt_container_id
                                )
                                all_edges += edge_list
                                estimated_steps = 100
                                for edge in edge_list:
                                    if edge not in edge_freq:
                                        edge_freq[edge] = 0
                                        if edge not in edge_steps:
                                            edge_steps[edge] = []
                                        edge_steps[edge].append(estimated_steps)
                                        proposals[pred_id]["edge_steps"][
                                            edge
                                        ] = estimated_steps
                                all_edges = list(set(all_edges))
                                for edge in all_edges:
                                    edge_freq[edge] += 1.0 / len(proposals)
                                if args.debug:
                                    print(pred_id)
                                    print(edge_pred_ins)
                                    for edge in all_edges:
                                        print(edge, edge_freq[edge])
                                    # ipdb.set_trace()

                        for edge, freq in edge_freq.items():
                            edge_goal_name = edge2name(edge)
                            if edge_goal_name not in combined_edge_freq:
                                combined_edge_freq[edge_goal_name] = 0
                            combined_edge_freq[edge_goal_name] += freq

                    # ======================================================
                    # get main agent's action
                    # arena.task_goal = None
                    print("planning for the main agent")
                    selected_actions, curr_info = arena.get_actions(
                        curr_obs,
                        length_plan=10,
                        must_replan={0: False, 1: True},
                        agent_id=0,
                    )

                    #if steps > 35:
                    #    # arena.agents[0].verbose = True
                    #    # arena.agents[0].mcts.verbose = True
                    #    Path(args.record_dir).mkdir(parents=True, exist_ok=True)
                    #    # print(saved_info)
                    #    if len(saved_info["obs"]) > 0:
                    #        pickle.dump(saved_info, open(log_file_name, "wb"))
                    #    ipdb.set_trace()

                    if selected_actions[0] is not None:
                        for obj_name in all_object_types:
                            if obj_name in selected_actions[0]:
                                grabbed_obj[obj_name] = True

                    if selected_actions[0] is not None:
                        if (
                            "chips" in selected_actions[0]
                            or "remotecontrol" in selected_actions[0]
                            or "condimentbottle" in selected_actions[0]
                            or "condimentshaker" in selected_actions[0]
                        ):
                            tv = True
                        if (
                            "salmon" in selected_actions[0]
                            or "apple" in selected_actions[0]
                            or "cupcake" in selected_actions[0]
                            or "pudding" in selected_actions[0]
                        ):
                            food = True
                        if (
                            "plate" in selected_actions[0]
                            or "glass" in selected_actions[0]
                            or "fork" in selected_actions[0]
                        ):
                            dish = True
                    if verbose:
                        print("main agent subgoal:", curr_info[0]["subgoals"])

                    saved_info["graph_results"].append(task_result)
                    saved_info["proposals"].append(proposals)

                    if len(task_result) == 0:
                        last_goal_edge = None
                        selected_actions[1] = None
                    else:
                        if replan_for_helper:

                            # ======================================================
                            # get helper agent's actio
                            if verbose:
                                print("gt goal:", gt_goal)
                                print("pred goal")
                            edge_pred_class_estimated = aggregate_multiple_pred(
                                task_result, steps - 3, change=True
                            )
                            if verbose:
                                for edge_class, count in edge_pred_class_estimated.items():
                                    if (
                                        edge_pred_class_estimated[edge_class][0] < 1e-6
                                        and edge_pred_class_estimated[edge_class][1] < 1e-6
                                    ):
                                        continue
                                    print(edge_class, edge_pred_class_estimated[edge_class])
                                print("edge freq:")
                            _, curr_edge_list = get_edge_instance_from_state(
                                curr_obs[1]
                            )
                            goal_edges = []
                            for edge in edge_freq:
                                edge_steps[edge] = np.mean(edge_steps[edge])
                                if verbose:
                                    print(
                                        edge,
                                        edge_freq[edge],
                                        edge_steps[edge],
                                        edge_steps[edge] > 1 + 1e-6,
                                        edge not in curr_edge_list,
                                    )
                                if (
                                    edge_steps[edge] > 1 + 1e-6
                                    and edge not in curr_edge_list
                                ):
                                    # if tv and (
                                    #     not (
                                    #         "chips" in edge
                                    #         or "remotecontrol" in edge
                                    #         or "condimentbottle" in edge
                                    #         or "condimentshaker" in edge
                                    #     )
                                    # ):
                                    #     continue
                                    # if food and (
                                    #     not (
                                    #         "salmon" in edge
                                    #         or "apple" in edge
                                    #         or "cupcake" in edge
                                    #         or "pudding" in edge
                                    #     )
                                    # ):
                                    #     continue
                                    # if dish and (
                                    #     not (
                                    #         "plate" in edge
                                    #         or "fork" in edge
                                    #         or "glass" in edge
                                    #     )
                                    # ):
                                    #     continue
                                    goal_edges.append(edge)
                            # if args.debug and steps == 4:
                            #     ipdb.set_trace()
                            max_freq = 0
                            opponent_subgoal = None
                            for subgoal, count in opponent_subgoal_freq.items():
                                if count > max_freq:
                                    max_freq = count
                                    opponent_subgoal = subgoal
                                if verbose:
                                    print(subgoal, count / len(proposals))
                            if verbose:
                                print("predicted main's subgoal:", opponent_subgoal)
                            # ipdb.set_trace()
                            del res

                            return_subgoals = get_subgoals_from_init_state(init_state)
                            # print(goal_edges)
                            # print(return_subgoals)
                            # ipdb.set_trace()

                            num_pred_goal_edges = len(goal_edges)

                            goal_edges += list(
                                return_subgoals
                            )  # add returning objects subgoals

                            res = manager.dict()
                            num_goals = len(goal_edges)
                            if num_processes == 0:
                                for process_id in range(num_goals):
                                    get_helping_plan(
                                        process_id,
                                        goal_edges[process_id],
                                        steps - 3,
                                        None,
                                        arena.get_actions,
                                        curr_obs,
                                        10,
                                        {0: True, 1: True},
                                        1,
                                        res,
                                    )
                            else:
                                for start_root_id in range(0, num_goals, num_processes):
                                    end_root_id = min(
                                        start_root_id + num_processes, num_goals
                                    )
                                    jobs = []
                                    for process_id in range(start_root_id, end_root_id):
                                        # print(process_id)
                                        p = mp.Process(
                                            target=get_helping_plan,
                                            args=(
                                                process_id,
                                                goal_edges[process_id],
                                                steps - 3,
                                                None,
                                                arena.get_actions,
                                                curr_obs,
                                                10,
                                                {0: True, 1: True},
                                                1,
                                                res,
                                            ),
                                        )
                                        jobs.append(p)
                                        p.start()
                                    for p in jobs:
                                        p.join()

                            # select best subgoal and action based on value
                            best_value = args.min_acceptable_value
                            best_estimated_steps = 1e6
                            last_goal_edge_curr = None
                            if True:  # not all_reject:
                                for pred_id, (
                                    subgoal,
                                    plan,
                                    plan_states,
                                    plan_cost,
                                ) in res.items():
                                    estimated_steps = 0
                                    if plan is None:
                                        if pred_id >= num_pred_goal_edges:
                                            estimated_steps = 0
                                        else:
                                            estimated_steps = None
                                    else:
                                        first_walk_steps = 0
                                        for action, cost in zip(plan, plan_cost):
                                            if "walk" in action:
                                                if (
                                                    "kitchen" in action
                                                    or "livingroom" in action
                                                    or "bedroom" in action
                                                    or "bathroom" in action
                                                ):
                                                    if estimated_steps == 0:
                                                        first_walk_steps += int(
                                                            cost + 0.5
                                                        )
                                            #        estimated_steps += 5
                                            #     else:
                                            #         estimated_steps += 2
                                            # else:
                                            #     estimated_steps += 1
                                            estimated_steps += max(int(cost + 0.5), 1)
                                    dist = compute_dist_instance(
                                        init_state, curr_obs[1], goal_edges[pred_id]
                                    )
                                    if goal_edges[pred_id].startswith("offer"):
                                        dist = 2
                                    if goal_edges[pred_id].endswith("init"):
                                        if (
                                            dist == 0
                                            or estimated_steps
                                            == 0  # TODO: check when dist !=0 but estimated_steps = 0
                                        ):  # still in the initial location
                                            continue
                                        value = (
                                            -args.beta * estimated_steps
                                            - args.lam * dist
                                        )
                                        estimated_steps_back = 0
                                    elif estimated_steps is None:
                                        value = -1e6
                                        estimated_steps_back = None
                                    else:
                                        if (
                                            dist < 0
                                        ):  # remove accidental retuning subgoals?
                                            continue
                                        estimated_steps_back = (
                                            estimated_steps - first_walk_steps
                                        )
                                        edge_goal_name = edge2name(goal_edges[pred_id])
                                        value = (
                                            args.alpha
                                            * (
                                                min(
                                                    edge_steps[goal_edges[pred_id]]
                                                    - estimated_steps,
                                                    args.max_benefit,
                                                )
                                            )
                                            * min(1, edge_freq[goal_edges[pred_id]])
                                            - args.beta
                                            * estimated_steps_back
                                            * max(0, 1 - edge_freq[goal_edges[pred_id]])
                                            - args.lam * dist
                                        )
                                    if goal_edges[pred_id].endswith("init") and verbose:
                                        print(
                                            goal_edges[pred_id],
                                            1,
                                            0,
                                            0,
                                            estimated_steps,
                                            dist,
                                            value,
                                        )
                                    else:
                                        if verbose:
                                            print(
                                                goal_edges[pred_id],
                                                edge_freq[goal_edges[pred_id]],
                                                edge_steps[goal_edges[pred_id]],
                                                estimated_steps,
                                                estimated_steps_back,
                                                dist,
                                                value,
                                            )
                                    # if (
                                    #     value > best_value
                                    #     or abs(value - best_value) < 1e-6
                                    #     and last_goal_edge is not None
                                    #     and (
                                    #         'chips' in last_goal_edge
                                    #         or 'remotecontrol' in last_goal_edge
                                    #     )
                                    #     and estimated_steps < best_estimated_steps
                                    # ):

                                    # if (
                                    #     value > best_value
                                    #     or abs(value - best_value) < 1e-6
                                    #     and (
                                    #         last_goal_edge_curr != last_goal_edge
                                    #         and (
                                    #             estimated_steps < best_estimated_steps
                                    #             and (
                                    #                 'chips' not in goal_edges[pred_id]
                                    #                 and 'remotecontrol'
                                    #                 not in goal_edges[pred_id]
                                    #             )
                                    #             or last_goal_edge is not None
                                    #             and (
                                    #                 'chips' in last_goal_edge
                                    #                 or 'remotecontrol' in last_goal_edge
                                    #             )
                                    #         )
                                    #     )
                                    # ):
                                    if (
                                        value > best_value
                                        or abs(value - best_value) < 1e-6
                                        and (
                                            last_goal_edge_curr != last_goal_edge
                                            and (
                                                estimated_steps
                                                < best_estimated_steps
                                                # or last_goal_edge is not None
                                            )
                                        )
                                    ):
                                        best_value = value
                                        selected_actions[1] = convert_walktowards(
                                            plan[0]
                                        )
                                        last_goal_edge_curr = goal_edges[pred_id]
                                        best_estimated_steps = estimated_steps
                                        # print('accept', last_goal_edge)
                                        if value > 0:
                                            print(
                                                "accept",
                                                value,
                                                best_value,
                                                estimated_steps,
                                                best_estimated_steps,
                                                goal_edges[pred_id],
                                                last_goal_edge_curr,
                                                last_goal_edge,
                                            )
                                last_goal_edge = last_goal_edge_curr
                                # ipdb.set_trace()

                                edge_pred_class_estimated = aggregate_multiple_pred(
                                    task_result, steps - 3, change=True
                                )

                                # for goal_object in goal_objects:
                                if verbose:
                                    print("-------------------------------------")
                                    print("gt goal")
                                    # print(gt_goal)
                                    for pred, count in gt_goal.items():
                                        print(pred, count)
                                    print("pred goal")
                                    for (
                                        edge_class,
                                        count,
                                    ) in edge_pred_class_estimated.items():
                                        if (
                                            edge_pred_class_estimated[edge_class][0] < 1e-6
                                            and edge_pred_class_estimated[edge_class][1]
                                            < 1e-6
                                        ):
                                            continue
                                        print(
                                            edge_class,
                                            edge_pred_class_estimated[edge_class],
                                        )
                        else:
                            selected_actions[1] = convert_walktowards(helper_action)
                    if verbose:
                        print("selected_actions:", selected_actions, best_value)
                        print("opponent_subgoal:", opponent_subgoal)
                        print("last_goal_edge:", last_goal_edge)
                        print("step:", steps)

                    prev_obs = copy.deepcopy(curr_obs)
                    prev_graph = copy.deepcopy(curr_graph)

                    try:
                        from termcolor import colored

                        print(colored(("taking step", selected_actions), "green"))
                        (curr_obs, reward, done, infos) = arena.step_given_action(
                            selected_actions
                        )
                    except:
                        ipdb.set_trace()
                    curr_graph = infos["graph"]
                    # history_obs.append(curr_obs[0])
                    # history_graph.append(curr_graph)
                    if prev_action != selected_actions[0]:
                        history_action.append(selected_actions[0])
                        prev_action = selected_actions[0]
                        new_action = True
                        cnt_same_action_steps = 1
                    else:
                        new_action = False
                        cnt_same_action_steps += 1
                        if cnt_same_action_steps >= 10 and (
                            "put" in prev_action or "grab" in prev_action
                        ):
                            early_stopping = True

                    if "satisfied_goals" in infos:
                        saved_info["goals_finished"].append(infos["satisfied_goals"])
                    for agent_id in range(2):
                        if agent_id in selected_actions:
                            saved_info["action"][agent_id].append(
                                selected_actions[agent_id]
                            )
                        else:
                            saved_info["action"][agent_id].append(None)

                    if "executed_script" in infos:
                        for agent_id in range(2):
                            if agent_id in infos["executed_script"]:
                                saved_info["executed_action"][agent_id].append(
                                    infos["executed_script"][agent_id]
                                )
                            else:
                                saved_info["executed_action"][agent_id].append(None)

                    saved_info["opponent_subgoal"].append(opponent_subgoal)
                    saved_info["helping_subgoal"].append(last_goal_edge)

                    if "graph" in infos:
                        saved_info["graph"].append(infos["graph"])
                    for agent_id, info in curr_info.items():
                        if "belief_room" in info:
                            saved_info["belief_room"][agent_id].append(
                                info["belief_room"]
                            )
                        if "belief" in info:
                            saved_info["belief"][agent_id].append(info["belief"])
                        if "plan" in info:
                            saved_info["plan"][agent_id].append(info["plan"][:3])
                        if "obs" in info:
                            saved_info["obs"].append(
                                [node["id"] for node in info["obs"]]
                            )

                    print("success:", infos["finished"])
                    # if steps > 30:
                    #     pickle.dump(saved_info, open(log_file_name, "wb"))
                    #     ipdb.set_trace()
                    # if args.debug:
                    #     ipdb.set_trace()
                    if infos["finished"]:
                        success = True
                        break

                print("-------------------------------------")
                print("success" if success else "failure")
                print("steps:", steps)
                print("-------------------------------------")
                # ipdb.set_trace()

                if not success:
                    failed_tasks.append(episode_id)
                else:
                    steps_list.append(steps)
                is_finished = 1 if success else 0

                saved_info["obs"].append([node["id"] for node in curr_obs[0]["nodes"]])
                saved_info["finished"] = success

                # if not args.debug:
                Path(args.record_dir).mkdir(parents=True, exist_ok=True)
                if len(saved_info["obs"]) > 0:
                    pickle.dump(saved_info, open(log_file_name, "wb"))
                else:
                    with open(log_file_name, "w+") as f:
                        f.write(json.dumps(saved_info, indent=4))

                logger.removeHandler(logger.handlers[0])
                os.remove(failure_file)

            except utils_exception.UnityException as e:
                traceback.print_exc()

                print("Unity exception")
                arena.reset_env()
                # ipdb.set_trace()
                continue

            except utils_exception.ManyFailureException as e:
                traceback.print_exc()

                print("ERRO HERE")
                logging.exception("Many failure Error")
                # print("OTHER ERROR")
                logger.removeHandler(logger.handlers[0])
                # exit()
                # arena.reset_env()
                print("Dione")
                # ipdb.set_trace()
                arena.reset_env()
                continue

            except Exception as e:
                with open(failure_file, "w+") as f:
                    error_str = "Failure"
                    error_str += "\n"
                    stack_form = "".join(traceback.format_stack())
                    error_str += stack_form

                    f.write(error_str)
                traceback.print_exc()

                logging.exception("Error")
                print("OTHER ERROR")
                logger.removeHandler(logger.handlers[0])
                # exit()
                arena.reset_env()
                # ipdb.set_trace()
                continue
            S[episode_id].append(is_finished)
            L[episode_id].append(steps)
            test_results[episode_id] = {"S": S[episode_id], "L": L[episode_id]}
            # pdb.set_trace()

            print(test_results)
            if not args.debug:
                pickle.dump(
                    test_results,
                    open(args.record_dir + "/results_{}.pik".format(iter_id), "wb"),
                )

        print(
            "average steps (finishing the tasks):",
            np.array(steps_list).mean() if len(steps_list) > 0 else None,
        )
        print("failed_tasks:", failed_tasks)
        if not args.debug:
            pickle.dump(
                test_results,
                open(args.record_dir + "/results_{}.pik".format(iter_id), "wb"),
            )


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("running time: %s sec" % (time.time() - start_time))
