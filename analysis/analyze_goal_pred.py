import pickle
from pathlib import Path
import numpy as np
import pdb


def get_edge_pred_class(pred, t):
    # pred_edge_prob = pred['edge_prob']
    edge_pred = pred['edge_pred'][t]
    pred_edge_names = pred['edge_names']
    pred_nodes = pred['nodes']
    pred_from_ids = pred['from_id']
    pred_to_ids = pred['to_id']

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
        if edge_name in ['inside', 'on']:
            edge_class = '{}_{}_{}'.format(
                from_node_name.split('.')[0], to_node_name.split('.')[0], edge_name
            )
            # print(from_node_name, to_node_name, edge_name)
            if edge_class not in edge_pred_class:
                edge_pred_class[edge_class] = 1
            else:
                edge_pred_class[edge_class] += 1
    return edge_pred_class


def aggregate_multiple_pred(preds, t):
    edge_classes = []
    edge_pred_class_all = {}
    N_preds = len(preds)
    for pred in preds:
        edge_pred_class = get_edge_pred_class(pred, t)
        edge_classes += list(edge_pred_class.keys())
        for edge_class, count in edge_pred_class.items():
            if edge_class not in edge_pred_class_all:
                edge_pred_class_all[edge_class] = [count]
            else:
                edge_pred_class_all[edge_class] += [count]
    edge_classes = sorted(list(set(edge_classes)))
    edge_pred_class_estimated = {}
    for edge_class in edge_classes:
        edge_pred_class_estimated[edge_class] = (
            sum(edge_pred_class_all[edge_class]) / N_preds
        )
        # print(edge_class, edge_pred_class_estimated[edge_class])
    return edge_pred_class_estimated


if __name__ == "__main__":
    gt_dir = "/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/agent_preferences/dataset_episodes/large_data_toy/test_env_task_set_10_full/1_full_opencost0_closecostFalse_walkcost0.05_forgetrate0"
    # pred_dir = "/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/results/predict_graph/train_data.dataset_graph_pred_train.pkl-agentsall/time_model.LSTM-stateenc.TF-edgepred.concat-lr0.0001-bs.16-goalenc.False_extended._costclose.1.0_costgoal.1.0_agentembed.False/test_env_task_set_10_full/1_full_opencost0_closecostFalse_walkcost0.05_forgetrate0"

    pred_dir = "/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/results/predict_graph/train_data.dataset_graph_pred_30step_train.pkl-agentsall/time_model.LSTM-stateenc.TF-globalrepr.pool-edgepred.concat-lr0.0001-bs.8-goalenc.False_extended._costclose.1.0_costgoal.1.0_agentembed.False_predchangeedge.True_inputgoal.True/test_env_task_set_10_full/1_full_opencost0_closecostFalse_walkcost0.05_forgetrate0"

    gt_p = Path(gt_dir).glob("*.pik")

    for gt_path in gt_p:
        if 'result' in str(gt_path):
            continue
        gt = pickle.load(open(str(gt_path), 'rb'))
        pred = pickle.load(
            open(pred_dir + '/' + str(gt_path).split('/')[-1] + '_result.pkl', 'rb')
        )

        print(len(pred['pred_graph']))

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
        for t in range(T):
            if (
                t == 0
                or actions[t].startswith('[grab]')
                or actions[t].startswith('[put')
            ):
                print(t, actions[t])
                print(gt_goal)
                if t:
                    print('prev')
                    edge_pred_class_estimated = aggregate_multiple_pred(
                        pred['pred_graph'], t - 1
                    )
                    for goal_object in goal_objects:
                        for edge_class, count in edge_pred_class_estimated.items():
                            if goal_object in edge_class:
                                print(edge_class, edge_pred_class_estimated[edge_class])
                print('curr')
                edge_pred_class_estimated = aggregate_multiple_pred(
                    pred['pred_graph'], t
                )
                for goal_object in goal_objects:
                    for edge_class, count in edge_pred_class_estimated.items():
                        if goal_object in edge_class:
                            print(edge_class, edge_pred_class_estimated[edge_class])
                pdb.set_trace()
