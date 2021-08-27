# Setup
The task is now always to bring either:
1. A few apples and cupcakes into the table
2. A few glasses and plates into the table

The agent has full observability in the environment

# Instructions to run
Copy the folder: `/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/agent_preferences/dataset/`


Symlink the folder: `/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/agent_preferences/dataset_episodes/`


Run training

```
CUDA_VISIBLE_DEVICES=0,1 python algos/train_graph_pred.py train.num_workers=0
```

Run inference

```
CUDA_VISIBLE_DEVICES=0,1 python algos/train_graph_pred.py inference=True ckpt_load=
```

