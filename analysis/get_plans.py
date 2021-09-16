import sys
import shutil
import os
import logging
import traceback
import random
import pickle
from pathlib import Path
import numpy as np
import pdb
import ipdb

sys.path.append('.')
from envs.unity_environment import UnityEnvironment
from agents import MCTS_agent, MCTS_agent_particle_v2, MCTS_agent_particle
from arguments import get_args
from algos.arena_mp2 import ArenaMP
from utils import utils_goals
from utils import utils_exception


def get_class_mode(agent_args):
    mode_str = '{}_opencost{}_closecost{}_walkcost{}_forgetrate{}'.format(
        agent_args['obs_type'],
        agent_args['open_cost'],
        agent_args['should_close'],
        agent_args['walk_cost'],
        agent_args['belief']['forget_rate'],
    )
    return mode_str


def get_edge_class(pred, t, source='pred'):
    # pred_edge_prob = pred['edge_prob']
    edge_pred = pred['edge_pred'][t] if source == 'pred' else pred['edge_input'][t]
    pred_edge_names = pred['edge_names']
    pred_nodes = pred['nodes']
    pred_from_ids = pred['from_id']  # if source == 'pred' else pred['from_id_input']
    pred_to_ids = pred['to_id']  # if source == 'pred' else pred['to_id_input']

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
        if edge_name in ['inside', 'on']:  # disregard room locations + plate
            if to_node_name.split('.')[0] in [
                'kitchen',
                'livingroom',
                'bedroom',
                'bathroom',
                'plate',
            ]:
                continue
            # if from_node_name.split('.')[0]
            edge_class = '{}_{}_{}'.format(
                edge_name, from_node_name.split('.')[0], to_node_name.split('.')[1]
            )
            # print(from_node_name, to_node_name, edge_name)
            if edge_class not in edge_pred_class:
                edge_pred_class[edge_class] = 1
            else:
                edge_pred_class[edge_class] += 1
    return edge_pred_class


# def get_edge_class(pred, t, source='pred'):
#     # pred_edge_prob = pred['edge_prob']
#     # print(len(pred['edge_input'][t]), len(pred['edge_pred'][t]))
#     edge_pred = pred['edge_pred'][t] if source == 'pred' else pred['edge_input'][t]
#     pred_edge_names = pred['edge_names']
#     pred_nodes = pred['nodes']
#     pred_from_ids = pred['from_id'] if source == 'pred' else pred['from_id_input']
#     pred_to_ids = pred['to_id'] if source == 'pred' else pred['to_id_input']

#     # edge_prob = pred_edge_prob[t]
#     # edge_pred = np.argmax(edge_prob, 1)

#     edge_pred_class = {}

#     num_edges = len(edge_pred)
#     # print(pred_from_ids[t], num_edges)
#     for edge_id in range(num_edges):
#         from_id = pred_from_ids[t][edge_id]
#         to_id = pred_to_ids[t][edge_id]
#         from_node_name = pred_nodes[from_id]
#         to_node_name = pred_nodes[to_id]
#         # if object_name in from_node_name or object_name in to_node_name:
#         edge_name = pred_edge_names[edge_pred[edge_id]]
#         # if edge_name in ['inside', 'on']:  # disregard room locations + plate
#         # if to_node_name.split('.')[0] in [
#         #     'kitchen',
#         #     'livingroom',
#         #     'bedroom',
#         #     'bathroom',
#         #     'plate',
#         # ]:
#         #     continue
#         # if from_node_name.split('.')[0]
#         edge_class = '()_{}_{}'.format(
#             edge_name, from_node_name.split('.')[0], to_node_name.split('.')[1]
#         )
#         # print(from_node_name, to_node_name, edge_name)
#         if edge_class not in edge_pred_class:
#             edge_pred_class[edge_class] = 1
#         else:
#             edge_pred_class[edge_class] += 1
#     return edge_pred_class


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
        edge_input_class = get_edge_class(preds[0], t, 'input')
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


if __name__ == "__main__":
    args = get_args()
    num_proc = 5

    num_tries = 5
    args.executable_file = '/data/vision/torralba/frames/data_acquisition/SyntheticStories/website/release/simulator/v2.0/v2.2.5_beta/linux_exec.v2.2.5_beta.x86_64'
    args.max_episode_length = 250
    args.num_per_apartment = 20
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    # home_path = '../'
    rootdir = curr_dir + '/../'

    # args.dataset_path = f'{rootdir}/dataset/train_env_task_set_100_full.pik'
    args.dataset_path = f'/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/agent_preferences/dataset/test_env_task_set_10_full.pik'
    # args.dataset_path = './dataset/train_env_task_set_20_full_reduced_tasks_single.pik'
    cachedir = f'{rootdir}/dataset_episodes/large_data_toy'

    agent_types = [
        ['full', 0, 0.05, False, 0, "uniform"],  # 0
        ['full', 0.5, 0.01, False, 0, "uniform"],  # 1
        ['full', -5, 0.05, False, 0, "uniform"],  # 2
        ['partial', 0, 0.05, False, 0, "uniform"],  # 3
        ['partial', 0, 0.05, False, 0, "spiked"],  # 4
        ['partial', 0, 0.05, False, 0.2, "uniform"],  # 5
        ['partial', 0, 0.01, False, 0.01, "spiked"],  # 6
        ['partial', -5, 0.05, False, 0.2, "uniform"],  # 7
        ['partial', 0.5, 0.05, False, 0.2, "uniform"],  # 8
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
    datafile = args.dataset_path.split('/')[-1].replace('.pik', '')
    agent_args = {
        'obs_type': args.obs_type,
        'open_cost': open_cost,
        'should_close': should_close,
        'walk_cost': walk_cost,
        'belief': {'forget_rate': forget_rate, 'belief_type': belief_type},
    }
    args.mode = '{}_'.format(agent_id + 1) + get_class_mode(agent_args)
    # args.mode += 'v9_particles_v2'

    env_task_set = pickle.load(open(args.dataset_path, 'rb'))
    print(len(env_task_set))

    for env in env_task_set:
        init_gr = env['init_graph']
        gbg_can = [
            node['id']
            for node in init_gr['nodes']
            if node['class_name'] in ['garbagecan', 'clothespile']
        ]
        init_gr['nodes'] = [
            node for node in init_gr['nodes'] if node['id'] not in gbg_can
        ]
        init_gr['edges'] = [
            edge
            for edge in init_gr['edges']
            if edge['from_id'] not in gbg_can and edge['to_id'] not in gbg_can
        ]
        for node in init_gr['nodes']:
            if node['class_name'] == 'cutleryfork':
                node['obj_transform']['position'][1] += 0.1

    args.record_dir = '{}/{}/{}'.format(cachedir, datafile, args.mode)
    error_dir = '{}/logging/{}_{}'.format(cachedir, datafile, args.mode)
    if not os.path.exists(args.record_dir):
        os.makedirs(args.record_dir)

    if not os.path.exists(error_dir):
        os.makedirs(error_dir)

    executable_args = {
        'file_name': args.executable_file,
        'x_display': 0,
        'no_graphics': True,
    }

    id_run = 0
    # random.seed(id_run)
    episode_ids = list(range(len(env_task_set)))
    episode_ids = sorted(episode_ids)
    random_start.shuffle(episode_ids)
    # episode_ids = episode_ids[10:]

    S = [[] for _ in range(len(episode_ids))]
    L = [[] for _ in range(len(episode_ids))]

    test_results = {}
    # episode_ids = [episode_ids[0]]

    def env_fn(env_id):
        return UnityEnvironment(
            num_agents=1,
            max_episode_length=args.max_episode_length,
            port_id=env_id,
            env_task_set=env_task_set,
            observation_types=[args.obs_type, args.obs_type],
            use_editor=args.use_editor,
            executable_args=executable_args,
            base_port=args.base_port,
        )

    args_common = dict(
        recursive=False,
        max_episode_length=20,
        num_simulation=200,
        max_rollout_steps=5,
        c_init=0.1,
        c_base=100,
        num_samples=1,
        num_processes=num_proc,
        num_particles=20,
        logging=True,
        logging_graphs=True,
        get_plan_states=True,
    )
    if args.obs_type == 'full':
        args_common['num_particles'] = 1
    else:
        args_common['num_particles'] = 20

    args_agent1 = {'agent_id': 1, 'char_index': 0}
    args_agent1.update(args_common)
    args_agent1['agent_params'] = agent_args

    args_agent2 = {'agent_id': 2, 'char_index': 1}
    args_agent2.update(args_common)
    args_agent2['agent_params'] = agent_args

    agents = [
        lambda x, y: MCTS_agent_particle_v2(**args_agent1),
        # lambda x, y: MCTS_agent_particle_v2(**args_agent2),
    ]
    arena = ArenaMP(args.max_episode_length, id_run, env_fn, agents)

    # # episode_ids = [20] #episode_ids
    # # num_tries = 1
    # episode_ids = [0]
    # ndict = {'on_book_329': 1}
    # env_task_set[91]['init_rooms'] = ['bedroom', 'bedroom']
    # env_task_set[91]['task_goal'] = {0: ndict, 1: ndict}

    # for iter_id in range(1, num_tries):
    # if iter_id > 0:
    iter_id = 1

    cnt = 0
    steps_list, failed_tasks = [], []
    current_tried = iter_id

    if not os.path.isfile(args.record_dir + '/results_{}.pik'.format(0)):
        test_results = {}
    else:
        test_results = pickle.load(
            open(args.record_dir + '/results_{}.pik'.format(0), 'rb')
        )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    gt_dir = "/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/agent_preferences/dataset_episodes/large_data_toy/test_env_task_set_10_full/1_full_opencost0_closecostFalse_walkcost0.05_forgetrate0"
    # pred_dir = "/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/results/predict_graph/train_data.dataset_graph_pred_train.pkl-agentsall/time_model.LSTM-stateenc.TF-edgepred.concat-lr0.0001-bs.16-goalenc.False_extended._costclose.1.0_costgoal.1.0_agentembed.False/test_env_task_set_10_full/1_full_opencost0_closecostFalse_walkcost0.05_forgetrate0"
    # pred_dir = "/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/results/predict_graph/train_data.dataset_graph_pred_30step_train.pkl-agentsall/time_model.LSTM-stateenc.TF-globalrepr.pool-edgepred.concat-lr0.0001-bs.8-goalenc.False_extended._costclose.1.0_costgoal.1.0_agentembed.False_predchangeedge.True_inputgoal.True/test_env_task_set_10_full/1_full_opencost0_closecostFalse_walkcost0.05_forgetrate0"

    root = "/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/results/predict_graph/train_data.dataset_graph_pred_30step_train.pkl-agentsall/"
    pred_dir = (
        root
        + "time_model.LSTM-stateenc.TF-globalrepr.pool-edgepred.concat-lr0.0001-bs.8-goalenc.False_extended._costclose.1.0_costgoal.1.0_agentembed.False_predchange.edge_inputgoal.False_excledge.False/test_env_task_set_10_full/1_full_opencost0_closecostFalse_walkcost0.05_forgetrate0"
        # + "time_model.LSTM-stateenc.TF-globalrepr.pool-edgepred.concat-lr0.0001-bs.32-goalenc.False_extended._costclose.1.0_costgoal.1.0_agentembed.False_predchange.node_inputgoal.False_excledge.True/test_env_task_set_10_full/1_full_opencost0_closecostFalse_walkcost0.05_forgetrate0"
    )

    gt_p = Path(gt_dir).glob("*.pik")

    num_episodes = 0
    # gt_p = [gp for gp in gt_p if 'logs_episode.26_iter.2.pik_result.pkl' in gp]
    # ipdb.set_trace()
    for gt_path in gt_p:
        print(str(gt_path))
        # if num_episodes > 0:
        #     break
        if 'result' in str(gt_path):
            continue
        # if 'episode.26' not in str(gt_path):
        #     continue
        gt = pickle.load(open(str(gt_path), 'rb'))

        if not Path(
            pred_dir + '/' + str(gt_path).split('/')[-1] + '_result.pkl'
        ).exists():
            continue

        num_episodes += 1

        print(pred_dir + '/' + str(gt_path).split('/')[-1] + '_result.pkl')

        pred = pickle.load(
            open(pred_dir + '/' + str(gt_path).split('/')[-1] + '_result.pkl', 'rb')
        )

        print(len(pred['pred_graph']))
        print(pred.keys())
        print(gt['env_id'], gt['task_id'], gt['gt_goals'], len(gt['action'][0]))

        gt_goal = gt['gt_goals']
        actions = gt['action'][0]

        # goal_objects = [predicate.split('_')[1] for predicate in gt_goal] #only check current goal objects
        goal_objects = [
            'cupcake',
            'apple',
            'plate',
            'waterglass',
        ]  # check all possible goal objects

        T = len(actions)
        print(T, len(pred['pred_graph'][0]['edge_pred']))
        # if T != len(pred['pred_graph'][0]['edge_pred']):
        #     pdb.set_trace()
        #     continue

        print('init state')
        ipdb.set_trace()
        edge_input_class = get_edge_class(pred['pred_graph'][0], 0, 'input')
        for goal_object in goal_objects:
            for edge_class, count in edge_input_class.items():
                if goal_object in edge_class:
                    print(edge_class, edge_input_class[edge_class])

        episode_id = gt['task_id']
        log_file_name = args.record_dir + '/logs_episode.{}_iter.{}.pik'.format(
            episode_id, iter_id
        )
        failure_file = '{}/{}_{}.txt'.format(error_dir, episode_id, iter_id)

        # if os.path.isfile(log_file_name):  # or os.path.isfile(failure_file):
        #     print(log_file_name)
        #     continue

        if os.path.isfile(failure_file):
            os.remove(failure_file)
        fileh = logging.FileHandler(failure_file, 'a')
        fileh.setLevel(logging.DEBUG)
        logger.addHandler(fileh)

        print('episode:', episode_id)

        for it_agent, agent in enumerate(arena.agents):
            agent.seed = (it_agent + current_tried * 2) * 5

        try:

            arena.reset(episode_id)
            print(arena.env.task_goal, arena.env.agent_goals)
            for t, action in enumerate(actions):
                action_freq = {}
                (obs, reward, done, infos) = arena.step_given_action({0: action})
                for pred_id, pred_graph in enumerate(pred['pred_graph']):
                    edge_pred_class = get_edge_class(pred_graph, t)
                    arena.task_goal = {0: edge_pred_class}

                    # if pred_id == 2:
                    #     arena.agents[0].mcts.verbose = True
                    #     arena.agents[0].mcts.any_verbose = True

                    actions, info = arena.get_actions(
                        obs, length_plan=10, must_replan=[True]
                    )
                    # print('actions:', actions)
                    print('pred {}:'.format(pred_id), edge_pred_class)
                    print('plan {}:'.format(pred_id), info[0]['plan'])

                    # Here you can get the intermediate states
                    plan_states = info[0]['plan_states']
                    # ipdb.set_trace()
                    # if pred_id == 2:
                    #     ipdb.set_trace()
                    for action in info[0]['plan']:
                        if action not in action_freq:
                            action_freq[action] = 1
                        else:
                            action_freq[action] += 1
                edge_pred_class_estimated = aggregate_multiple_pred(
                    pred['pred_graph'], t, change=True
                )
                # for goal_object in goal_objects:
                print('-------------------------------------')
                for edge_class, count in edge_pred_class_estimated.items():
                    if (
                        edge_pred_class_estimated[edge_class][0] < 1e-6
                        and edge_pred_class_estimated[edge_class][1] < 1e-6
                    ):
                        continue
                    print(edge_class, edge_pred_class_estimated[edge_class])
                print('action freq:')
                N_preds = len(pred['pred_graph'])
                for action, count in action_freq.items():
                    print(action, count / N_preds)
                pdb.set_trace()
            print('success:', infos['finished'])

        # print('-------------------------------------')
        # print('success' if success else 'failure')
        # print('steps:', steps)
        # print('-------------------------------------')
        # if not success:
        #     failed_tasks.append(episode_id)
        # else:
        #     steps_list.append(steps)
        # is_finished = 1 if success else 0

        # Path(args.record_dir).mkdir(parents=True, exist_ok=True)
        # # if len(saved_info['obs']) > 0:
        # #     pickle.dump(saved_info, open(log_file_name, 'wb'))
        # # else:
        # #     with open(log_file_name, 'w+') as f:
        # #         f.write(json.dumps(saved_info, indent=4))

        # logger.removeHandler(logger.handlers[0])
        # os.remove(failure_file)

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
            # with open(failure_file, 'w+') as f:
            #    error_str = 'Failure'
            #    error_str += '\n'
            #    stack_form = ''.join(traceback.format_stack())
            #    error_str += stack_form

            #    f.write(error_str)
            traceback.print_exc()

            logging.exception("Error")
            print("OTHER ERROR")
            logger.removeHandler(logger.handlers[0])
            # exit()
            arena.reset_env()
            # ipdb.set_trace()
            # ipdb.set_trace()
            # pdb.set_trace()
            continue
        # S[episode_id].append(is_finished)
        # L[episode_id].append(steps)
        # test_results[episode_id] = {'S': S[episode_id], 'L': L[episode_id]}
        pdb.set_trace()

    #     for t in range(T):
    #         if (
    #             t == 0
    #             or actions[t].startswith('[grab]')
    #             or actions[t].startswith('[put')
    #         ):
    #             print(t, actions[t])
    #             print(gt_goal)
    #             if t:
    #                 print('prev')
    #                 # edge_pred_class_gt = aggregate_multiple_pred(
    #                 #     pred['gt_graph'], t - 1
    #                 # )
    #                 edge_pred_class_estimated = aggregate_multiple_pred(
    #                     pred['pred_graph'], t - 1, change=True
    #                 )
    #                 for goal_object in goal_objects:
    #                     for edge_class, count in edge_pred_class_estimated.items():
    #                         if goal_object in edge_class:
    #                             print(edge_class, edge_pred_class_estimated[edge_class])
    #             print('curr')
    #             # edge_pred_class_gt = aggregate_multiple_pred(pred['gt_graph'], t)
    #             edge_pred_class_estimated = aggregate_multiple_pred(
    #                 pred['pred_graph'], t, change=True
    #             )
    #             for goal_object in goal_objects:
    #                 for edge_class, count in edge_pred_class_estimated.items():
    #                     if goal_object in edge_class:
    #                         print(edge_class, edge_pr
