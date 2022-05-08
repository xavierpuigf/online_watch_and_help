import sys

sys.path.append(".")
import shutil
import os
from torch import nn
import logging
import traceback
import pickle as pkl
import random
from tqdm import tqdm
import pickle
import copy
from pathlib import Path
import numpy as np
import pdb
import ipdb
import hydra
import time
import multiprocessing as mp
from omegaconf import DictConfig, OmegaConf
from dataloader.dataloader_v2 import AgentTypeDataset
from dataloader import dataloader_v2 as dataloader_v2
from models import agent_pref_policy_task as agent_pref_policy
from hydra.utils import get_original_cwd, to_absolute_path
from utils import utils_models_wb, utils_rl_agent, utils_environment

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


def compute_metrics(pred_graphs, task_graph_gt):

    if len(pred_graphs) == 0:
        return {
            "recall": 0,
            "recallmax": 0,
            "accuracy": 0,
            "accuracymax": 0,
            "precision": 1.0,
            "precisionmax": 1.0,
        }

    eps = 1e-9
    pos_preds = np.array(
        [
            0,
            0,
            0,
            2,
            0,
            5,
            0,
            3,
            0,
            0,
            2,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            2,
            0,
            7,
            0,
            2,
            0,
            0,
            2,
            0,
            0,
            0,
            3,
            7,
            0,
            0,
            3,
            0,
            0,
            0,
            0,
            0,
            0,
            3,
            7,
            0,
            0,
            3,
            0,
            0,
            0,
            0,
            0,
            0,
            2,
            0,
            6,
            0,
            3,
            0,
            0,
            2,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            2,
            0,
            7,
            0,
            3,
            0,
            0,
            2,
            0,
            0,
            0,
            2,
            6,
            0,
            0,
            3,
            0,
            0,
            0,
            0,
            0,
            0,
            3,
            6,
            0,
            0,
            3,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
        ]
    )
    pred_task = np.concatenate([pred_graph[None, ...] for pred_graph in pred_graphs])
    gt_task = task_graph_gt[None, ...]

    pos_gt_p = (gt_task > 0) / ((gt_task > 0).sum(-1)[..., None] + eps)
    pred_p = (pred_task > 0) / ((pred_task > 0).sum(-1)[..., None] + eps)

    accuracy = (((gt_task == pred_task) * pos_gt_p).sum(-1)).mean(0)[None, ...]
    recall = (np.minimum(pred_task, gt_task).sum(-1) / (eps + gt_task.sum(-1))).mean(0)[
        None, ...
    ]
    prec = (np.minimum(pred_task, gt_task).sum(-1) / (eps + pred_task.sum(-1))).mean(0)[
        None, ...
    ]

    accuracymax = (((gt_task == pred_task) * pos_gt_p).sum(-1)).max(0)[None, ...]
    recallmax = (np.minimum(pred_task, gt_task).sum(-1) / (eps + gt_task.sum(-1))).max(
        0
    )[None, ...]
    precmax = (np.minimum(pred_task, gt_task).sum(-1) / (eps + pred_task.sum(-1))).max(
        0
    )[None, ...]
    return {
        "recall": recall,
        "recallmax": recallmax,
        "accuracy": accuracy,
        "accuracymax": accuracymax,
        "precision": prec,
        "precisionmax": precmax,
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


class GoalInferenceParticle:
    def __init__(
        self,
        planner,
        prediction_net,
        args_pred,
        graph_helper,
        class2id,
        arena,
        num_particles=10,
        num_proc=0,
        z_vec=None,
    ):
        self.planner = planner
        self.prediction_net = prediction_net
        self.num_particles = num_particles
        self.arena = arena
        self.z_vec = z_vec

        self.particles = []
        self.num_proc = num_proc
        self.args_pred = args_pred
        self.graph_helper = graph_helper
        self.class2id = class2id
        self.num_steps_plan = 15

        self.grabbed_obj = {x: False for x in all_object_types}

    def get_rejected_particles(self, current_action):
        index_reject = []
        if current_action is not None:
            current_action.replace("walktowards", "walk")
        for index in range(len(self.particles)):
            if current_action is not None and "plan" in self.particles[index]:
                try:
                    plan_particle = self.particles[index]["plan"][1]
                except:
                    ipdb.set_trace()
                is_present = is_in_plan(current_action, plan_particle)
            else:
                is_present = True
            goal_pred = get_edge_class(
                self.particles[index]["pred_graph"],
                len(
                    self.particles[index]["pred_graph"],
                )
                - 1,
            )
            if not is_present or not is_in_goal(self.grabbed_obj, goal_pred):
                index_reject.append(index)
            # print(
            #     index, is_present, is_in_goal(self.grabbed_obj, goal_pred), index_reject
            # )
            # ipdb.set_trace()
        return index_reject

    def get_pred_name(self, container_name):
        pred_name = "on"
        room_list = ["kitchen", "livingroom", "bedroom", "bathroom"]
        if (
            container_name in info_objects["objects_inside"]
            or container_name in room_list
        ):
            pred_name = "inside"
        return pred_name

    def get_edge_instance(self, pred, class2id, t, source="pred"):
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

            pred_name = self.get_pred_name(container_name)
            if obj_name == "character" or container_name in [
                "kitchen",
                "livingroom",
                "bedroom",
                "bathroom",
                "plate",
            ]:
                continue

            if obj_name in class2id:
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

    def get_agent_plan(
        self, t, index_plan, res, obs, length_plan, must_replan, agent_id, current_goal
    ):
        "Returns the plan of the agent"
        inferred_goal = self.get_edge_instance(current_goal, self.class2id, t)
        print("pred {}:".format(index_plan), inferred_goal)
        # ipdb.set_trace()
        plan_states, opponent_subgoal = None, None
        if len(inferred_goal) > 0:  # if no edge prediction then None action
            opponent_actions, opponent_info = self.planner(
                {0: obs},
                length_plan=length_plan,
                must_replan=must_replan,
                agent_id=agent_id,
                inferred_goal=inferred_goal,
            )
            plan_states = opponent_info[agent_id]["plan_states"]
            plan_cost = opponent_info[agent_id]["plan_cost"]
            plan = opponent_info[agent_id]["plan"]

            # ipdb.set_trace()
            if (
                opponent_info[agent_id]["subgoals"] is not None
                and len(opponent_info[agent_id]["subgoals"]) > 0
            ):
                opponent_subgoal = opponent_info[agent_id]["subgoals"][0][0]
            else:
                opponent_subgoal = None
            res[index_plan] = (opponent_subgoal, plan, plan_states, plan_cost)
        else:
            # This particle has not plan
            res[index_plan] = (None, [], [], 0.0)

    def plan_for_particles(self, curr_obs, particle_ids):
        num_goals = len(particle_ids)
        processes_used = min(len(particle_ids), self.num_proc)
        if processes_used == 0:
            res = {}
            t = 0
            for particle_id in particle_ids:
                self.get_agent_plan(
                    t,
                    particle_id,
                    res,
                    curr_obs,
                    self.num_steps_plan,  # len_plan
                    {0: True, 1: True},  # must_replan
                    0,  # agent_id
                    self.particles[particle_id]["pred_graph"],
                )

        else:
            t = 0
            # ipdb.set_trace()
            manager = mp.Manager()
            res = manager.dict()
            for start_root_id in range(0, num_goals, processes_used):
                end_root_id = min(start_root_id + processes_used, num_goals)
                jobs = []
                for process_id in range(start_root_id, end_root_id):
                    # print(process_id)
                    particle_id = particle_ids[process_id]
                    p = mp.Process(
                        target=self.get_agent_plan,
                        args=(
                            t,
                            process_id,
                            res,
                            curr_obs,
                            self.num_steps_plan,
                            {0: True, 1: True},
                            0,
                            self.particles[particle_id]["pred_graph"],
                        ),
                    )
                    jobs.append(p)
                    p.start()
                for p in jobs:
                    p.join()

        index = 0
        for particle_id in particle_ids:
            self.particles[particle_id]["plan"] = res[index]
            index += 1
        # ipdb.set_trace()

    def regen_particles(self, graphs, observations, actions, particle_ids, t=None):
        if particle_ids is None:
            particle_ids = list(range(self.num_particles))
            self.particles = [{} for _ in range(self.num_particles)]

        history_graph = graphs
        history_obs = observations
        history_action = actions

        # if len(particle_ids) > 0:
        #     particle_ids = list(range(self.num_particles))
        inputs_func = utils_models_wb.prepare_graph_for_task_model_diff(
            history_graph,
            history_obs,
            history_action,
            self.args_pred,
            self.graph_helper,
            batch_repeat=len(particle_ids),
        )

        with torch.no_grad():
            if self.z_vec is not None:
                z_vec = self.z_vec.copy()
                z_vec[: len(particle_ids), :] = z_vec[particle_ids, :]
                output_func = self.prediction_net(
                    inputs_func, inference=True, z_vec=torch.tensor(self.z_vec)
                )
            else:
                output_func = self.prediction_net(inputs_func, inference=True)

        num_tsteps = output_func["pred_graph"].shape[1]

        if not self.prediction_net.use_vae:
            pred_graph_prob = output_func["pred_graph"]
            pred_graph_prob = (
                nn.functional.softmax(pred_graph_prob, dim=-1).cpu().numpy()
            )
            pred_graph = utils_models_wb.vectorized(pred_graph_prob)
        else:
            pred_graph = output_func["pred_graph"].argmax(-1)

        # todo : get current plan?
        task_graph_input = self.graph_helper.get_task_graph(
            inputs_func["input_task_graph"][0, 0], use_dict=True
        )
        task_result = []
        # print(num_tsteps)
        for ind, index_particle in enumerate(particle_ids):
            task_graphs = []
            for tstep in [num_tsteps - 1]:
                try:
                    curr_task_graph = self.graph_helper.get_task_graph(
                        pred_graph[ind, tstep], use_dict=True
                    )
                except:
                    print("Error processing graph")
                    ipdb.set_trace()
                # ipdb.set_trace()
                # curr_mask_task = mask_task_graph[ind, tstep]
                curr_mask_task = None
                task_graphs.append(
                    (
                        curr_task_graph,
                        curr_mask_task,
                        task_graph_input,
                        pred_graph[ind, tstep],
                    )
                )
            task_result.append(task_graphs)

            self.particles[index_particle]["pred_graph"] = task_graphs
        elem = []
        for i in range(self.num_particles):
            tg = self.particles[i]["pred_graph"][-1][-1]
            new_str = ""
            if i in particle_ids:
                new_str = "*"
            elem.append(new_str + str(int(tg.sum(-1))))
        print("time {}".format(t))
        print(" ".join(elem))
        # ipdb.set_trace()

    def initialize(self, graphs, observations, actions):
        self.arena.sim_agents[0].reset(observations[0], graphs[0], None, seed=0)
        history_graph = graphs
        history_obs = observations
        history_action = actions
        inputs_func = utils_models_wb.prepare_graph_for_task_model_diff(
            history_graph,
            history_obs,
            history_action,
            self.args_pred,
            self.graph_helper,
            batch_repeat=self.num_particles,
        )

        with torch.no_grad():
            if self.z_vec is not None:
                output_func = self.prediction_net(
                    inputs_func, inference=True, z_vec=torch.tensor(self.z_vec)
                )
            else:
                output_func = self.prediction_net(inputs_func, inference=True)

        num_tsteps = output_func["pred_graph"].shape[1]
        pred_graph = output_func["pred_graph"].argmax(-1)

        # todo : get current plan?
        task_graph_input = self.graph_helper.get_task_graph(
            inputs_func["input_task_graph"][0, 0], use_dict=True
        )
        task_result = []
        for ind in range(self.num_particles):
            task_graphs = []
            for tstep in range(num_tsteps):
                try:
                    curr_task_graph = self.graph_helper.get_task_graph(
                        pred_graph[ind, tstep], use_dict=True
                    )
                except:
                    print("Error processing graph")
                    ipdb.set_trace()
                # ipdb.set_trace()
                # curr_mask_task = mask_task_graph[ind, tstep]
                curr_mask_task = None
                task_graphs.append(
                    (
                        curr_task_graph,
                        curr_mask_task,
                        task_graph_input,
                        pred_graph[ind, tstep],
                    )
                )
            task_result.append(task_graphs)
        self.particles = [{"pred_graph": task_res} for task_res in task_result]
        # ipdb.set_trace()
        all_particle_ids = [i for i in range(self.num_particles)]

        elem = []
        for i in range(self.num_particles):
            tg = self.particles[i]["pred_graph"][0][-1]
            new_str = ""

            elem.append(new_str + str(int(tg.sum(-1))))
        print(" ".join(elem))
        # ipdb.set_trace()

        # TODO: tianmin, this does not seem correct, but im not sure the logic of the func, can you check?
        obs = observations[-1]
        graph_obs = {
            "nodes": [node for node in graphs[-1]["nodes"] if node["id"] in obs],
            "edges": [
                edge
                for edge in graphs[-1]["edges"]
                if edge["from_id"] in obs and edge["to_id"] in obs
            ],
        }
        self.plan_for_particles(graph_obs, all_particle_ids)

    def filter():
        pass


@hydra.main(config_path="../config/configs_for_help", config_name="config_diff_state")
def main(cfg: DictConfig):
    config = cfg
    print("Config")
    print(OmegaConf.to_yaml(cfg))
    args = config
    args_pred = args.agent_pred_graph
    num_proc = 20
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    # home_path = '../'
    rootdir = curr_dir + "/../"

    # args.dataset_path = f'{rootdir}/dataset/train_env_task_set_100_full.pik'
    # args.dataset_path = f'/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/agent_preferences/dataset/test_env_task_set_10_full.pik'
    args.dataset_path = f"/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/agent_preferences/dataset/structured_agent/test_env_task_set_60_full_task.all.pik"
    # args.dataset_path = './dataset/train_env_task_set_20_full_reduced_tasks_single.pik'

    valid_set_path = "/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/agent_preferences/analysis/test_set_reduced.txt"
    pred_result_path = "/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/agent_preferences/results_inference"
    f = open(valid_set_path, "r")
    episode_ids = []
    filenames = []
    for filename in f:
        episode_id = int(filename.split("episode.")[-1].split("_")[0])

        episodes_keep = [
            3,
            139,
            162,
            180,
            193,
            225,
            290,
            304,
            323,
            366,
            401,
            419,
            428,
            466,
            523,
            556,
            573,
            591,
            606,
            621,
        ]
        if episode_id in episodes_keep:
            episode_ids.append(episode_id)
            filenames.append(filename)
    # episode_ids = sorted(episode_ids)
    # print(len(episode_ids))
    f.close()
    # episode_ids_prepare_food = [162, 180, 193, 225]

    all_content = {"smart_reset": []}
    for ep_id, filename in zip(episode_ids, filenames):
        # if ep_id not in episode_ids_prepare_food:
        #     continue

        graph_helper = utils_rl_agent.GraphHelper(
            max_num_objects=args_pred["model"]["max_nodes"],
            toy_dataset=args_pred["model"]["reduced_graph"],
        )

        print(filename)
        # ipdb.set_trace()
        with open(filename.strip(), "rb") as f:
            content = pkl.load(f)
        with open(filename.strip().replace(".pik", "_reduced.pik"), "rb") as f:
            content_reduced = pkl.load(f)

        # with open(
        #     "{}/VAE.KL.0.001/{}_result.pkl".format(
        #         pred_result_path, filename.split("/")[-1].strip()
        #     ),
        #     "rb",
        # ) as f:
        #     prev_infer = pkl.load(f)
        #     z_vec = prev_infer["z_vec"]

        # ipdb.set_trace()
        steps_keep = utils_rl_agent.condense_walking(content["action"][0])
        # ipdb.set_trace()
        curr_graph = content["graph"][0]
        class2id = {}
        for node in curr_graph["nodes"]:
            if node["class_name"] not in class2id:
                class2id[node["class_name"]] = []
            class2id[node["class_name"]].append(node["id"])

        model = agent_pref_policy.GraphPredNetworkVAETask3(args_pred)
        state_dict = torch.load(args_pred.ckpt_load)["model"]

        state_dict_new = {}

        for param_name, param_value in state_dict.items():
            state_dict_new[param_name.replace("module.", "")] = param_value

        model.load_state_dict(state_dict_new)
        model.eval()
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

        (
            args.obs_type,
            open_cost,
            walk_cost,
            should_close,
            forget_rate,
            belief_type,
        ) = agent_types[0]

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
        agent_args = {
            "obs_type": args.obs_type,
            "open_cost": open_cost,
            "should_close": should_close,
            "walk_cost": walk_cost,
            "belief": {"forget_rate": forget_rate, "belief_type": belief_type},
        }

        args_agent1 = {"agent_id": 1, "char_index": 0}
        args_agent1.update(args_common)
        args_agent1["agent_params"] = agent_args
        agents = [
            lambda x, y: MCTS_agent_particle_v2_instance(**args_agent1),
        ]

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

        executable_args = {
            "file_name": args.executable_file,
            "x_display": 0,
            "no_graphics": True,
        }

        def env_fn(env_id):
            return UnityEnvironment(
                num_agents=1,
                max_episode_length=args.max_episode_length,
                port_id=env_id,
                env_task_set=env_task_set,
                observation_types=[args.obs_type, args.obs_type],
                use_editor=True,
                executable_args=executable_args,
                base_port=args.base_port,
                convert_goal=True,
            )

        arena = ArenaMP(args.max_episode_length, 0, env_fn, agents, use_sim_agent=True)
        # ipdb.set_trace()
        particle_pred = GoalInferenceParticle(
            planner=arena.pred_actions,
            prediction_net=model,
            args_pred=args_pred,
            graph_helper=graph_helper,
            class2id=class2id,
            arena=arena,
            num_particles=args.num_samples,
            num_proc=num_proc,
            z_vec=None,
        )

        graph_info, _ = graph_helper.build_graph_for_task(
            content["graph"][0],
            character_id=1,
            include_edges=True,
            obs_ids=content["obs"][0],
            unique_from=True,
        )

        graph_info_end, _ = graph_helper.build_graph_for_task(
            content["graph"][-1],
            character_id=1,
            include_edges=True,
            obs_ids=content["obs"][-1],
            unique_from=True,
        )

        task_graph_init = graph_helper.build_task_graph(graph_info)
        task_graph_end = graph_helper.build_task_graph(graph_info_end)
        task_graph_gt = np.maximum(
            task_graph_end - task_graph_init, np.zeros(task_graph_init.shape)
        )

        curr_values = []
        curr_graphs = content["graph"][0]
        graphs = [utils_environment.inside_not_trans(curr_graph)]
        obs = [content["obs"][0]]
        actions = [None]
        particle_pred.z_vec = None
        particle_pred.initialize(graphs, obs, actions)
        pred_graphs = [
            particle["pred_graph"][-1][-1] for particle in particle_pred.particles
        ]
        curr_values.append({"pred_task": pred_graphs, "gt_task": task_graph_gt})
        steps_since_last_prediction = 0
        t = 1
        cont_t_keep = 1
        for action in tqdm(content["action"][0]):
            if t in steps_keep:
                curr_graphs = [content["graph"][0], content["graph"][t]]
                graphs = [
                    utils_environment.inside_not_trans(
                        utils_environment.clean_house_obj(graph)
                    )
                    for graph in curr_graphs
                ]

                obs = [content["obs"][0], content["obs"][t]]
                actions = [None, None]

                if action is not None:
                    for obj_name in all_object_types:
                        if obj_name in action:
                            particle_pred.grabbed_obj[obj_name] = True

                if steps_since_last_prediction < particle_pred.num_steps_plan:
                    rejected_particles = particle_pred.get_rejected_particles(action)
                    particle_pred.particles = [
                        particle_pred.particles[pid]
                        for pid in range(len(particle_pred.particles))
                        if pid not in rejected_particles
                    ]
                else:
                    particle_pred.particles = []

                print(t)
                print(rejected_particles)

                # particle_pred.particles = []

                if len(particle_pred.particles) < 1:
                    steps_since_last_prediction = 0
                    particle_pred.regen_particles(
                        graphs,
                        obs,
                        actions,
                        None,
                        cont_t_keep,
                    )
                    filtered_graph_obs = {
                        "nodes": [
                            node
                            for node in graphs[-1]["nodes"]
                            if node["id"] in obs[-1]
                        ],
                        "edges": [
                            edge
                            for edge in graphs[-1]["edges"]
                            if edge["from_id"] in obs[-1] and edge["to_id"] in obs[-1]
                        ],
                    }
                    # ipdb.set_trace()

                    particle_pred.arena.sim_agents[0].reset(
                        filtered_graph_obs, graphs[-1], None, seed=0
                    )
                    particle_pred.plan_for_particles(
                        filtered_graph_obs, rejected_particles
                    )

                    rejected_particles = particle_pred.get_rejected_particles(
                        current_action=None
                    )
                    particle_pred.particles = [
                        particle_pred.particles[pid]
                        for pid in range(len(particle_pred.particles))
                        if pid not in rejected_particles
                    ]
                else:
                    steps_since_last_prediction += 1

                pred_graphs = [
                    particle["pred_graph"][-1][-1]
                    for particle in particle_pred.particles
                ]
                print(len(pred_graphs))
                # ipdb.set_trace()
                # curr_metrics.append(compute_metrics(pred_graphs, task_graph_gt))
                curr_values.append({"pred_task": pred_graphs, "gt_task": task_graph_gt})
                cont_t_keep += 1

            t += 1

        # curr_metrics2 = []
        # curr_graphs = content["graph"][0]
        # graphs = [utils_environment.inside_not_trans(curr_graph)]
        # obs = [content["obs"][0]]
        # actions = [None]

        # particle_pred.z_vec = z_vec
        # particle_pred.initialize(graphs, obs, actions)
        # pred_graphs = [
        #     particle["pred_graph"][-1][-1] for particle in particle_pred.particles
        # ]
        # curr_metrics2.append(compute_metrics(pred_graphs, task_graph_gt))
        # t = 1
        # cont_t_keep = 0
        # for action in tqdm(content["action"][0]):
        #     if t in steps_keep:

        #         curr_graphs = [content["graph"][0], content["graph"][t]]
        #         graphs = [
        #             utils_environment.inside_not_trans(
        #                 utils_environment.clean_house_obj(graph)
        #             )
        #             for graph in curr_graphs
        #         ]

        #         obs = [content["obs"][0], content["obs"][t]]
        #         actions = [None, None]

        #         rejected_particles = range(particle_pred.num_particles)

        #         if len(rejected_particles):
        #             particle_pred.regen_particles(graphs, obs, actions, rejected_particles)
        #             filtered_graph_obs = {
        #                 "nodes": [
        #                     node for node in graphs[0]["nodes"] if node["id"] in obs[0]
        #                 ],
        #                 "edges": [
        #                     edge
        #                     for edge in graphs[0]["edges"]
        #                     if edge["from_id"] in obs[0] and edge["to_id"] in obs[0]
        #                 ],
        #             }
        #             # ipdb.set_trace()
        #             graphc = filtered_graph_obs

        #             # particle_pred.arena.sim_agents[0].reset(filtered_graph_obs, graphs[0], None, seed=0)
        #             # particle_pred.plan_for_particles(filtered_graph_obs, rejected_particles)

        #         pred_graphs = [
        #             particle["pred_graph"][-1][-1] for particle in particle_pred.particles
        #         ]
        #         curr_metrics2.append(compute_metrics(pred_graphs, task_graph_gt))

        #     t += 1
        method_name = args.log_name
        filename_last = ".".join(filename.split("/")[-1].split(".")[:-1])
        if not os.path.isdir(
            f"results/results_smallset_inference_online/{method_name}"
        ):
            os.makedirs(f"results/results_smallset_inference_online/{method_name}")

        with open(
            f"results/results_smallset_inference_online/{method_name}/{filename_last}.pkl",
            "wb+",
        ) as f:
            pkl.dump(curr_values, f)
    # ipdb.set_trace()


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("running time: %s sec" % (time.time() - start_time))
