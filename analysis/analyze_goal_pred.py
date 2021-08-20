import pickle
from pathlib import Path
import numpy as np
import pdb


def human_readable_graph_pred(pred, object_name, t):
    pred_edge_prob = pred['pred_graph']['edge_prob']
    pred_edge_names = pred['pred_graph']['edge_names']
    pred_nodes = pred['pred_graph']['nodes']
    pred_from_ids = pred['pred_graph']['from_id']
    pred_to_ids = pred['pred_graph']['to_id']

    edge_prob = pred_edge_prob[t]
    edge_pred = np.argmax(edge_prob, 1)

    num_edges = len(edge_pred)
    for edge_id in range(num_edges):
        from_id = pred_from_ids[t][edge_id]
        to_id = pred_to_ids[t][edge_id]
        from_node_name = pred_nodes[from_id]
        to_node_name = pred_nodes[to_id]
        if object_name in from_node_name or object_name in to_node_name:
            edge_name = pred_edge_names[edge_pred[edge_id]]
            if edge_name in ['inside', 'on']:
                print(from_node_name, to_node_name, edge_name)


if __name__ == "__main__":
    gt_dir = "/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/agent_preferences/dataset_episodes/large_data_toy/test_env_task_set_10_full/1_full_opencost0_closecostFalse_walkcost0.05_forgetrate0"
    pred_dir = "/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/results/predict_graph/train_data.dataset_graph_pred_train.pkl-agentsall/time_model.LSTM-stateenc.TF-edgepred.concat-lr0.0001-bs.16-goalenc.False_extended._costclose.1.0_costgoal.1.0_agentembed.False/test_env_task_set_10_full/1_full_opencost0_closecostFalse_walkcost0.05_forgetrate0"

    gt_p = Path(gt_dir).glob("*.pik")

    for gt_path in gt_p:
        if 'result' in str(gt_path):
            continue
        gt = pickle.load(open(str(gt_path), 'rb'))
        pred = pickle.load(
            open(pred_dir + '/' + str(gt_path).split('/')[-1] + '_result.pkl', 'rb')
        )

        gt_goal = gt['gt_goals']
        actions = gt['action'][0]

        goal_objects = [predicate.split('_')[1] for predicate in gt_goal]

        T = len(actions)
        for t in range(T):
            if actions[t].startswith('[grab]') or actions[t].startswith('[put'):
                print(t, actions[t])
                print(gt_goal)
                if t:
                    print('prev')
                    for goal_object in goal_objects:
                        human_readable_graph_pred(pred, goal_object, t - 1)
                print('curr')
                for goal_object in goal_objects:
                    human_readable_graph_pred(pred, goal_object, t)
                pdb.set_trace()
