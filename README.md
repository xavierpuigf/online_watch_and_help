Copy the folder: `/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/agent_preferences/dataset/`


Symlink the folder: `/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/agent_preferences/dataset_episodes/`


Run training

```
CUDA_VISIBLE_DEVICES=0,1 python algos/train_graph_pred.py
```

Run inference

```
CUDA_VISIBLE_DEVICES=0,1 python algos/train_graph_pred.py inference=True ckpt_load=
```

# Runs

```bash
CUDA_VISIBLE_DEVICES=1,2 python algos/train_graph_pred.py name_log=pred_last_graph model.predict_edge_change=True
CUDA_VISIBLE_DEVICES=3,4 python algos/train_graph_pred.py name_log=pred_last_graph model.predict_node_change=True
CUDA_VISIBLE_DEVICES=5,6 python algos/train_graph_pred.py name_log=pred_last_graph

CUDA_VISIBLE_DEVICES=5,6 python algos/train_graph_pred_excl.py name_log=pred_last_graph_excl model.predict_node_change=True model.exclusive_edge=True train.lr=0.001
```


# Evaluation

```bash
export HOME_CKPT="/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/ckpts/predict_graph/train_data.dataset_graph_pred_30step_train.pkl-agentsall/"
export ckpt_none=$HOME_CKPT"time_model.LSTM-stateenc.TF-globalrepr.pool-edgepred.concat-lr0.0001-bs.8-goalenc.False_extended._costclose.1.0_costgoal.1.0_agentembed.False_predchange.none_inputgoal.False_excledge.False"
export ckpt_edge=$HOME_CKPT"time_model.LSTM-stateenc.TF-globalrepr.pool-edgepred.concat-lr0.0001-bs.8-goalenc.False_extended._costclose.1.0_costgoal.1.0_agentembed.False_predchange.edge_inputgoal.False_excledge.False"
export ckpt_node=$HOME_CKPT"time_model.LSTM-stateenc.TF-globalrepr.pool-edgepred.concat-lr0.0001-bs.8-goalenc.False_extended._costclose.1.0_costgoal.1.0_agentembed.False_predchange.node_inputgoal.False_excledge.False"


CUDA_VISIBLE_DEVICES=1,2 python algos/train_graph_pred.py inference=True inference_sample=True model.predict_edge_change=True ckpt_load=$ckpt_edge"/490.pt"
CUDA_VISIBLE_DEVICES=3,4 python algos/train_graph_pred.py inference=True inference_sample=True model.predict_node_change=True ckpt_load=$ckpt_node"/490.pt"
CUDA_VISIBLE_DEVICES=5,6 python algos/train_graph_pred.py inference=True inference_sample=True ckpt_load=$ckpt_none"/490.pt"

CUDA_VISIBLE_DEVICES=5,6 python algos/train_graph_pred_excl.py name_log=pred_last_graph_excl model.predict_node_change=True logging=False model.exclusive_edge=True train.num_workers=0



export ckpt_excl="/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/ckpts/predict_graph/train_data.dataset_graph_pred_30step_train.pkl-agentsall/time_model.LSTM-stateenc.TF-globalrepr.pool-edgepred.concat-lr0.001-bs.32-goalenc.False_extended._costclose.1.0_costgoal.1.0_agentembed.False_predchange.node_inputgoal.False_excledge.True/"


export ckpt_excl="/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/ckpts/predict_graph/train_data.dataset_graph_pred_30step_train.pkl-agentsall/time_model.LSTM-stateenc.TF-globalrepr.pool-edgepred.concat-lr0.0001-bs.32-goalenc.False_extended._costclose.1.0_costgoal.1.0_agentembed.False_predchange.node_inputgoal.False_excledge.True_goodaction/"

CUDA_VISIBLE_DEVICES=1,2 python algos/train_graph_pred.py inference=True inference_sample=False model.predict_node_change=True ckpt_load=$ckpt_node"/490.pt"



CUDA_VISIBLE_DEVICES=1,2 python algos/train_graph_pred_excl.py inference=True inference_sample=False model.predict_node_change=True  model.exclusive_edge=True ckpt_load=$ckpt_excl"/100.pt"



CUDA_VISIBLE_DEVICES=0 python tests/test_solo_loader.py model.predict_node_change=True  model.exclusive_edge=True ckpt_load=$ckpt_excl"/490.pt"



python analysis/helping_action_freq.py agent_pred_graph.ckpt_load=$ckpt_excl"/490.pt"   agent_pred_graph.model.predict_node_change=True  agent_pred_graph.model.exclusive_edge=True

```

