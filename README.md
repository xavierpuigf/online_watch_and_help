# Instructions to run
Copy the folder: `/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/agent_preferences/dataset/`
Symlink the folder: `/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/agent_preferences/dataset_episodes/`
```CUDA_VISIBLE_DEVICES=0,1 python algos/train_graph_pred.py train.num_workers=0```
