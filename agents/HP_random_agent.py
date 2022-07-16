import multiprocessing as mp
import numpy as np
import random
import time
import math
import copy
import importlib
import json
import multiprocessing
import ipdb
import pickle
from pathlib import Path
import os
from omegaconf import OmegaConf
import torch
from . import belief
from envs.graph_env import VhGraphEnv
from utils import utils_rl_agent

from torch import nn
import sys

sys.path.append("..")
from utils import utils_environment as utils_env
from utils import utils_models_wb
from . import MCTS_agent_particle_v2_instance

from models import agent_pref_policy_task as agent_pref_policy


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


def convert_walktowards(action):
    if action is not None and "walktowards" not in action:
        return action.replace("walk", "walktowards")
    else:
        return action


def get_pred_name(container_name):
    pred_name = "on"
    room_list = ["kitchen", "livingroom", "bedroom", "bathroom"]
    if container_name in info_objects["objects_inside"] or container_name in room_list:
        pred_name = "inside"
    return pred_name


def get_edge_instance(pred, class2id, gt_container_id, t, source="pred"):
    # pred_edge_prob = pred['edge_prob']
    # print(len(pred['edge_input'][t]), len(pred['edge_pred'][t]))

    t = max(0, min(t, len(pred) - 1))

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
):
    inferred_goal = get_edge_instance(pred_task, class2id, gt_container_id, t)
    # print("pred {}:".format(process_id), inferred_goal)
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


def get_helping_plan(
    process_id,
    pred_task,
    class2id,
    gt_container_id,
    t,
    opponent_subgoal,
    get_actions_fn,
    obs,
    length_plan,
    must_replan,
    agent_id,
    res,
):
    inferred_goal = get_edge_instance(pred_task, class2id, gt_container_id, t)
    # print("pred {}:".format(process_id), inferred_goal)
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
        # print("pred {}:".format(process_id), inferred_goal)
        # print("plan {}:".format(process_id), opponent_subgoal, info[1]["subgoals"])
        if info[1]["subgoals"] is None or len(info[1]["subgoals"]) == 0:
            res[process_id] = None
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
    res[process_id] = action  # (subgoal, action)


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


class HP_random_agent:
    """
    Hierarchical Planner with Random Goals agent
    """

    def __init__(
        self,
        agent_id,
        char_index,
        max_episode_length,
        num_simulation,
        max_rollout_steps,
        c_init,
        c_base,
        recursive=False,
        num_samples=1,
        num_processes=1,
        comm=None,
        logging=False,
        logging_graphs=False,
        inv_plan=True,
        seed=None,
    ):
        self.agent_type = "HP_Random"
        self.verbose = False
        self.recursive = recursive

        # self.env = unity_env.env
        if seed is None:
            seed = random.randint(0, 100)
        self.seed = seed
        self.logging = logging
        self.logging_graphs = logging_graphs

        self.agent_id = agent_id
        self.char_index = char_index
        self.sim_env = VhGraphEnv()
        self.sim_env.pomdp = True
        self.belief = None
        self.max_episode_length = max_episode_length
        self.num_simulation = num_simulation
        self.max_rollout_steps = max_rollout_steps
        self.c_init = c_init
        self.c_base = c_base
        self.num_samples = 1  # TODO: adjustable
        self.num_processes = num_processes

        self.previous_belief_graph = None
        self.verbose = False

        # Indicates whether there is a unity simulation
        self.comm = comm
        self.history_action = []
        self.history_obs = []
        self.history_graph = []

        self.proposals = {}
        self.max_plan_length = 10
        self.pred_main_plan_length = 15
        self.steps_since_last_prediction = 0
        self.all_reject = False
        self.inv_plan = False  # TODO: adjustable
        self.reset_steps = None
        self.steps_since_last_prediction = 0
        self.new_action = True

        args_common = dict(
            recursive=False,
            max_episode_length=20,
            num_simulation=100,
            max_rollout_steps=5,
            c_init=0.1,
            c_base=100,
            num_samples=1,
            num_processes=0,
            num_particles=1,
            logging=True,
            logging_graphs=True,
            get_plan_states=True,
            get_plan_cost=True,
        )

        agent_args = {
            "obs_type": "full",
            "open_cost": 0,
            "should_close": False,
            "walk_cost": 0.05,
            "belief": {"forget_rate": 0, "belief_type": "uniform"},
        }

        args_agent1 = {"agent_id": 1, "char_index": 0}
        args_agent1.update(args_common)
        args_agent1["agent_params"] = agent_args

        args_agent2 = {"agent_id": 2, "char_index": 1}
        args_agent2.update(args_common)
        args_agent2["agent_params"] = agent_args
        args_agent2["num_simulation"] = 50
        self.agents = {
            0: MCTS_agent_particle_v2_instance(**args_agent1),
            1: MCTS_agent_particle_v2_instance(**args_agent2),
        }

        self.args = OmegaConf.load(
            "/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/agent_preferences/config/configs_for_help/config_diff_state.yaml"
        )
        self.num_processes = self.args.num_processes
        self.reset_steps = self.args.reset_steps
        self.args_pred = OmegaConf.load(
            "/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/agent_preferences/config/configs_for_help/agent_pred_graph/config_det_0.05.yaml"
        )
        self.args_pred.ckpt_load = "/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/ckpts/predict_graph/train_data.dataset_graph_full_150step_larger_train.pkl/lr0.0009-bs.256-klcoff.1-goalenc.False_predchange.none_inputgoal.False_excledge.True_preddiff.Falsereduced_walk_logname.detfull_encoder_task_graph/290.pt"
        self.graph_helper = None
        self.args_pred.name_log = "uniform"
        if self.args_pred.name_log == "uniform":
            model = agent_pref_policy.UniformModel()
        else:
            model = agent_pref_policy.GraphPredNetworkVAETask3(self.args_pred)
            state_dict = torch.load(self.args_pred.ckpt_load)["model"]
            state_dict_new = {}

            for param_name, param_value in state_dict.items():
                state_dict_new[param_name.replace("module.", "")] = param_value

            model.load_state_dict(state_dict_new)
            model.eval()
        self.model = model
        self.class2id = {}

    def filtering_graph(self, graph):
        new_edges = []
        edge_dict = {}
        for edge in graph["edges"]:
            key = (edge["from_id"], edge["to_id"])
            if key not in edge_dict:
                edge_dict[key] = [edge["relation_type"]]
                new_edges.append(edge)
            else:
                if edge["relation_type"] not in edge_dict[key]:
                    edge_dict[key] += [edge["relation_type"]]
                    new_edges.append(edge)

        graph["edges"] = new_edges
        return graph

    def get_relations_char(self, graph):
        # TODO: move this in vh_mdp
        char_id = [
            node["id"] for node in graph["nodes"] if node["class_name"] == "character"
        ][0]
        edges = [edge for edge in graph["edges"] if edge["from_id"] == char_id]
        print("Character:")
        print(edges)
        print("---")

    def is_in_plan(self, action, plan):
        for tmp_action in plan:
            if same_action(action, tmp_action):
                return True
        return False

    def is_in_goal(self, grabbed_obj, goals):
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

    def get_edge_class(self, pred, t, source="pred"):
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

    def get_action(
        self, obs, goal_spec, previous_main_action, steps, opponent_subgoal=None
    ):
        if previous_main_action is not None:
            for obj_name in all_object_types:
                if obj_name in previous_main_action:
                    self.grabbed_obj[obj_name] = True

        curr_obs = {0: obs, 1: obs}
        curr_graph = obs
        num_samples = self.num_samples
        selected_actions = {1: None}
        if (
            len(self.history_action) == 0
            or self.history_action[-1] != previous_main_action
        ):
            self.history_action.append(previous_main_action)
            self.new_action = True
            # cnt_same_action_steps = 1
        else:
            self.new_action = False
            # cnt_same_action_steps += 1
            # if cnt_same_action_steps >= 10 and (
            #     "put" in prev_action or "grab" in prev_action
            # ):
            #     early_stopping = True

        # NEW PROPOSALS
        if self.new_action:
            self.history_obs.append([node["id"] for node in obs["nodes"]])
            self.history_graph.append(copy.deepcopy(obs))

        if len(self.task_result) < 1:  # args.num_samples / 3:
            self.steps_since_last_prediction = 0
            print(len(self.history_graph))
            print(len(self.history_action))
            print(len(self.history_obs))
            assert len(self.history_graph) == len(self.history_obs)
            assert len(self.history_graph) == len(self.history_action)
            # if len(proposals) == 0:
            #     history_graph = [history_graph[-1:]]
            #     history_obs = [history_obs[-1:]]
            #     history_action = [history_action[-1:]]
            if self.history_action[-1] is not None:
                # ipdb.set_trace()
                inputs_func = utils_models_wb.prepare_graph_for_task_model_diff(
                    self.history_graph,
                    self.history_obs,
                    self.history_action,
                    self.args_pred,
                    self.graph_helper,
                    batch_repeat=num_samples,
                )
                with torch.no_grad():
                    output_func = self.model(inputs_func, inference=True)
                # ipdb.set_trace()
                # First particle, first timestep, since all the particles have the same time graph
                task_graph_input = self.graph_helper.get_task_graph(
                    inputs_func["input_task_graph"][0, 0], use_dict=True
                )
                num_tsteps = output_func["pred_graph"].shape[1]
                if not self.model.use_vae and self.num_samples > 1:
                    # ipdb.set_trace()
                    sample = True

                    pred_graph_prob = output_func["pred_graph"]
                    if not sample:
                        pred_graph = pred_graph_prob.argmax(-1).cpu().numpy()
                    else:
                        pred_graph_prob = (
                            nn.functional.softmax(pred_graph_prob, dim=-1).cpu().numpy()
                        )
                        pred_graph = utils_models_wb.vectorized(pred_graph_prob)

                else:
                    if self.args_pred.name_log == "uniform":
                        pred_graph = output_func["pred_graph"]
                    else:
                        # VAE, take max
                        pred_graph = output_func["pred_graph"].argmax(-1)
                for ind in range(num_samples):
                    task_graphs = []
                    for tstep in range(num_tsteps):
                        try:
                            curr_task_graph = self.graph_helper.get_task_graph(
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
                    # if args.debug:
                    #     print(ind, task_graphs)
                    #     # ipdb.set_trace()

                    # goal_pred = self.get_edge_class(
                    #     task_graphs,
                    #     len(task_graphs) - 1,
                    # )
                    self.task_result.append(task_graphs)

        action_freq = {}
        # opponent_subgoal_freq = {}
        manager = mp.Manager()
        task_result = self.task_result

        # if self.num_processes == 0:
        #     res = {}
        #     for index in range(len(task_result)):

        #         pred_main_agent_plan(
        #             index,
        #             task_result[index],
        #             self.class2id,
        #             self.gt_container_id,
        #             steps - 3,
        #             self.get_actions,
        #             curr_obs,
        #             self.pred_main_plan_length,
        #             {0: True, 1: True},
        #             1,
        #             res,
        #         )
        # else:
        #     res = manager.dict()
        #     for start_root_id in range(0, len(task_result), self.num_processes):
        #         end_root_id = min(start_root_id + self.num_processes, len(task_result))
        #         jobs = []
        #         for process_id in range(start_root_id, end_root_id):
        #             # print(process_id)
        #             p = mp.Process(
        #                 target=pred_main_agent_plan,
        #                 args=(
        #                     process_id,
        #                     task_result[process_id],
        #                     self.class2id,
        #                     self.gt_container_id,
        #                     steps - 3,
        #                     self.get_actions,
        #                     curr_obs,
        #                     15,
        #                     {0: True, 1: True},
        #                     1,
        #                     res,
        #                 ),
        #             )
        #             jobs.append(p)
        #             p.start()
        #         for p in jobs:
        #             p.join()
        # # all_plan_states = []
        # edge_freq = {}
        # edge_steps = {}
        # self.proposals = {}
        # combined_edge_freq = {}
        # for pred_id, (
        #     subgoal,
        #     plan,
        #     plan_states,
        #     plan_cost,
        # ) in res.items():
        #     self.proposals[pred_id] = {
        #         "pred": task_result[pred_id],
        #         "subgoal": subgoal,
        #         "plan": plan,
        #         "plan_states": plan_states,
        #         "plan_cost": plan_cost,
        #         "edge_steps": {},
        #     }

        #     if subgoal is not None:
        #         if subgoal not in opponent_subgoal_freq:
        #             opponent_subgoal_freq[subgoal] = 1
        #         else:
        #             opponent_subgoal_freq[subgoal] += 1
        #         # print(len(plan_states))
        #         if plan_states is not None and len(plan_states) > 0:
        #             # all_plan_states.append(plan_states)
        #             all_edges = []
        #             estimated_steps = 0

        #             for t, (action, state, cost) in enumerate(
        #                 zip(plan, plan_states, plan_cost)
        #             ):
        #                 (
        #                     edge_pred_ins,
        #                     edge_list,
        #                 ) = get_edge_instance_from_state(state)

        #                 all_edges += edge_list
        #                 estimated_steps += max(int(cost + 0.5), 1)

        #                 for edge in edge_list:
        #                     if edge not in edge_freq:
        #                         edge_freq[edge] = 0
        #                         if edge not in edge_steps:
        #                             edge_steps[edge] = []
        #                         edge_steps[edge].append(estimated_steps)
        #                         self.proposals[pred_id]["edge_steps"][
        #                             edge
        #                         ] = estimated_steps
        #         # ipdb.set_trace()
        #         edge_pred_ins, edge_list = get_edge_instance_from_pred(
        #             task_result[pred_id], self.class2id, self.gt_container_id
        #         )
        #         all_edges += edge_list
        #         estimated_steps = 100  # TODO: tune this
        #         for edge in edge_list:
        #             if edge not in edge_freq:
        #                 edge_freq[edge] = 0
        #                 if edge not in edge_steps:
        #                     edge_steps[edge] = []
        #                 edge_steps[edge].append(estimated_steps)
        #                 self.proposals[pred_id]["edge_steps"][edge] = estimated_steps

        #         all_edges = list(set(all_edges))
        #         for edge in all_edges:
        #             edge_freq[edge] += 1.0 / len(res)
        # for edge, freq in edge_freq.items():
        #     edge_goal_name = edge2name(edge)
        #     if edge_goal_name not in combined_edge_freq:
        #         combined_edge_freq[edge_goal_name] = 0
        #     combined_edge_freq[edge_goal_name] += freq
        # # print(edge_freq)
        # # ipdb.set_trace()

        if len(task_result) == 0:
            selected_actions[1] = None
        else:
            # ======================================================
            # get helper agent's action
            action_freq = {}
            # print("gt goal:", gt_goal)
            # print("pred goal")
            # edge_pred_class_estimated = aggregate_multiple_pred(
            #     task_result, steps - 3, change=True
            # )
            # for edge_class, count in edge_pred_class_estimated.items():
            #     if (
            #         edge_pred_class_estimated[edge_class][0] < 1e-6
            #         and edge_pred_class_estimated[edge_class][1] < 1e-6
            #     ):
            #         continue
            #     print(edge_class, edge_pred_class_estimated[edge_class])
            # print("edge freq:")
            # _, curr_edge_list = get_edge_instance_from_state(curr_obs[1])

            # max_freq = 0
            # opponent_subgoal = None
            # for subgoal, count in opponent_subgoal_freq.items():
            #     if count > max_freq:
            #         max_freq = count
            #         opponent_subgoal = subgoal
            #     print(subgoal, count / len(self.proposals))
            # print("predicted main's subgoal:", opponent_subgoal)
            # # ipdb.set_trace()
            # del res

            res = manager.dict()
            num_goals = len(self.task_result)
            curr_pred_goals = [pred for pred in self.task_result]
            if self.num_processes == 0:
                for process_id in range(num_goals):
                    get_helping_plan(
                        process_id,
                        curr_pred_goals[process_id],
                        self.class2id,
                        self.gt_container_id,
                        steps - 3,
                        None,
                        self.get_actions,
                        curr_obs,
                        10,
                        {0: True, 1: True},
                        1,
                        res,
                    )
            else:
                for start_root_id in range(0, num_goals, self.num_processes):
                    end_root_id = min(start_root_id + self.num_processes, num_goals)
                    jobs = []
                    for process_id in range(start_root_id, end_root_id):
                        # print(process_id)
                        p = mp.Process(
                            target=get_helping_plan,
                            args=(
                                process_id,
                                curr_pred_goals[process_id],
                                self.class2id,
                                self.gt_container_id,
                                steps - 3,
                                None,
                                self.get_actions,
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

            # ipdb.set_trace()

            for pred_id, action in res.items():
                if action is not None:
                    if action not in action_freq:
                        action_freq[action] = 1
                    else:
                        action_freq[action] += 1

                    # edge_pred_class_estimated = aggregate_multiple_pred(
                    #     task_result, steps - 3, change=True
                    # )

            N_preds = num_goals
            max_freq = 0
            for action, count in action_freq.items():
                curr_freq = count / N_preds
                if curr_freq > max_freq:
                    max_freq = curr_freq
                    selected_actions[1] = action
                print(action, curr_freq)

        prev_obs = copy.deepcopy(curr_obs)
        prev_graph = copy.deepcopy(curr_graph)

        return selected_actions[1], {}

    def get_goal2(self, task_spec):
        # pred = [x for x, y in task_spec.items() if y['count'] > 0 and x.split('_')[0] in ['on', 'inside']]
        # object_grab = [pr.split('_')[1] for pr in pred]
        # predicates_grab = {'holds_{}_1'.format(obj_gr): [1, False, 2] for obj_gr in object_grab}
        res_dict = {
            goal_k: copy.deepcopy(goal_c)
            for goal_k, goal_c in task_spec.items()
            if goal_c["count"] > 0
        }
        for goal_k, goal_dict in res_dict.items():
            goal_dict.update({"final": True, "reward": 2})
        # res_dict.update(predicates_grab)
        return res_dict

    def get_actions(
        self,
        obs,
        length_plan,
        must_replan,
        agent_id,
        inferred_goal=None,
        opponent_subgoal=None,
    ):
        goal_spec = self.get_goal2(inferred_goal)
        dict_actions = {}
        dict_info = {}
        dict_actions[agent_id], dict_info[agent_id] = self.agents[agent_id].get_action(
            obs[agent_id],
            goal_spec,
            opponent_subgoal,
            length_plan=length_plan,
            must_replan=must_replan,
        )
        return dict_actions, dict_info

    def reset(
        self,
        observed_graph,
        gt_graph,
        container_id,
        task_goal,
        seed=0,
        simulator_type="python",
        is_alice=False,
    ):

        self.class2id = {}

        self.gt_container_id = container_id

        class2id = {}
        for node in gt_graph["nodes"]:
            if node["class_name"] not in class2id:
                class2id[node["class_name"]] = []
            class2id[node["class_name"]].append(node["id"])
        self.class2id = class2id

        self.last_action = None
        self.last_subgoal = None
        for ind_agent in range(len(self.agents)):
            self.agents[ind_agent].reset(
                observed_graph, gt_graph, task_goal, seed, simulator_type, is_alice
            )
        self.history_action = []
        self.history_obs = []
        self.history_graph = []
        self.graph_helper = utils_rl_agent.GraphHelper(
            max_num_objects=self.args_pred["model"]["max_nodes"],
            toy_dataset=self.args_pred["model"]["reduced_graph"],
        )
        self.proposals = {}
        self.task_result = []
        self.grabbed_obj = {obj_type: False for obj_type in all_object_types}

        self.steps_since_last_prediction = 0
        self.all_reject = False
        # self.reset_steps = None
        self.new_action = True
        self.steps_since_last_prediction = 0
