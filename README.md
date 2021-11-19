Copy the folder: `/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/agent_preferences/dataset/`


Symlink the folder: `/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/agent_preferences/dataset_episodes/`


Run training


# New experiments


CUDA_VISIBLE_DEVICES=4,5,6,7 python algos/train_graph_pred_task.py name_log=pred_last_graph_excl_large_VAE_uncondprior_new \
	model.predict_node_change=True model.exclusive_edge=True train.lr=0.001 \
	model.cond_prior=False model.time_aggregate='seqVAE' model.state_encoder="GNN" logging=False

export ckpt_edge="/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/ckpts/predict_graph/train_data.dataset_graph_full_150step_larger_train.pkl-agentsall/time_model.seqVAE-stateenc.GNN-globalrepr.pool-edgepred.concat-lr0.001-bs.16-goalenc.False_extended._costclose.1.0_costgoal.1.0_agentembed.False_predchange.node_inputgoal.False_excledge.True_goodactionreduced_walk_condprior.False_zvec.node"



CUDA_VISIBLE_DEVICES=4,5,6,7 python algos/train_graph_pred_task.py name_log=pred_last_graph_excl_large_VAE_uncondprior_new \
	model.predict_node_change=True model.exclusive_edge=True train.lr=0.001 \
	model.cond_prior=False model.time_aggregate='seqVAE' model.state_encoder="GNN" inference=True inference_sample=True  ckpt_load=$ckpt_edge"/290.pt"
	


## Experiment autoencoder, train only the first step, no kl loss
CUDA_VISIBLE_DEVICES=4,5,6,7 python algos/train_graph_pred_task.py name_log=pred_last_graph_excl_large_VAE_uncondprior_new \
model.predict_node_change=True model.exclusive_edge=True train.lr=0.001 \
model.cond_prior=False model.time_aggregate='seqVAE' model.state_encoder="GNN" logging=False model.max_tsteps=2 train.batch_size=64 


##
Train autoencoder

CUDA_VISIBLE_DEVICES=4,5,6,7 python algos/train_autoencoder_task.py name_log=autoencoder_VAE_new \
	model.predict_node_change=True model.exclusive_edge=True train.lr=0.001 \
	model.cond_prior=False model.time_aggregate='seqVAE' model.state_encoder="GNN" logging=False args.model.max_tsteps=2



CUDA_VISIBLE_DEVICES=4,5,6,7 python algos/train_graph_pred_task.py name_log=autoencoder_VAE_new_oldencoder \
model.predict_node_change=True model.exclusive_edge=True train.lr=0.001 \
model.cond_prior=False model.time_aggregate='seqVAE' model.state_encoder="GNN" logging=False model.max_tsteps=2 args.model.max_tsteps=2


##
All is task graph

CUDA_VISIBLE_DEVICES=4,5,6,7 python algos/train_autoencoder_task.py name_log=autoencoder_VAE_new_task \
	model.predict_node_change=True model.exclusive_edge=True train.lr=0.001 \
	model.cond_prior=False model.time_aggregate='seqVAE' model.state_encoder="GNN" logging=False args.model.max_tsteps=2



CUDA_VISIBLE_DEVICES=4,5,6,7 python algos/train_graph_pred_task.py name_log=autoencoder_VAE_new_oldencoder \
model.predict_node_change=True model.exclusive_edge=True train.lr=0.001 \
model.cond_prior=False model.time_aggregate='seqVAE' model.state_encoder="GNN" logging=False model.max_tsteps=2 args.model.max_tsteps=2


<!-- ```
CUDA_VISIBLE_DEVICES=0,1 python algos/train_graph_pred.py
```

Run inference

```
CUDA_VISIBLE_DEVICES=0,1 python algos/train_graph_pred.py inference=True ckpt_load=
 -->```

# Runs

```bash
CUDA_VISIBLE_DEVICES=1,2 python algos/train_graph_pred.py name_log=pred_last_graph model.predict_edge_change=True
CUDA_VISIBLE_DEVICES=3,4 python algos/train_graph_pred.py name_log=pred_last_graph model.predict_node_change=True
CUDA_VISIBLE_DEVICES=5,6 python algos/train_graph_pred.py name_log=pred_last_graph

CUDA_VISIBLE_DEVICES=3,4,5,6 python algos/train_graph_pred_excl.py name_log=pred_last_graph_excl_large model.predict_node_change=True model.exclusive_edge=True train.lr=0.001


CUDA_VISIBLE_DEVICES=3,4,5,6 python algos/train_graph_pred_excl.py name_log=pred_last_graph_excl_large_VAE model.predict_node_change=True model.exclusive_edge=True train.lr=0.001


### VAE

CUDA_VISIBLE_DEVICES=3,4,5,6 python algos/train_graph_pred_excl.py name_log=pred_last_graph_excl_large_VAE model.predict_node_change=True model.exclusive_edge=True train.lr=0.001


CUDA_VISIBLE_DEVICES=3,4,5,6 python algos/train_graph_pred_excl.py name_log=pred_last_graph_excl_large_VAE_uncondprior model.predict_node_change=True model.exclusive_edge=True train.lr=0.001 model.cond_prior=False model.time_aggregate='seqVAE'

### VAE + GNN
CUDA_VISIBLE_DEVICES=0,1,2,3 python algos/train_graph_pred_excl.py name_log=pred_last_graph_excl_large_VAE_uncondprior_new \
	model.predict_node_change=True model.exclusive_edge=True train.lr=0.001 \
	model.cond_prior=False model.time_aggregate='seqVAE' model.state_encoder="TF"


CUDA_VISIBLE_DEVICES=4,5,6,7 python algos/train_graph_pred_excl.py name_log=pred_last_graph_excl_large_VAE_uncondprior_new \
	model.predict_node_change=True model.exclusive_edge=True train.lr=0.001 \
	model.cond_prior=False model.time_aggregate='seqVAE' model.state_encoder="GNN"
	
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


## LSTM
CUDA_VISIBLE_DEVICES=5,6 python algos/train_graph_pred_excl.py inference=True inference_sample=True \
model.predict_node_change=True logging=False model.exclusive_edge=True train.num_workers=0 ckpt_load=$ckpt_excl"/290.pt"


CUDA_VISIBLE_DEVICES=5,6 python algos/train_graph_pred_excl_old.py inference=True inference_sample=True \
model.predict_node_change=True logging=False model.exclusive_edge=True train.num_workers=0 ckpt_load=$ckpt_excl"/290.pt"


## VAE - cond prior

export ckpt_excl_vae="/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/ckpts/predict_graph/train_data.dataset_graph_full_150step_larger_train.pkl-agentsall/time_model.seqVAE-stateenc.TF-globalrepr.pool-edgepred.concat-lr0.001-bs.16-goalenc.False_extended._costclose.1.0_costgoal.1.0_agentembed.False_predchange.node_inputgoal.False_excledge.True_goodactionreduced_walk_condprior.True"
CUDA_VISIBLE_DEVICES=5,6 python algos/train_graph_pred_excl.py inference=True inference_sample=True \
model.predict_node_change=True logging=False model.exclusive_edge=True ckpt_load=$ckpt_excl_vae"/290.pt" model.time_aggregate='seqVAE'

## VAE - uncond prior
export ckpt_excl_vae="/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/ckpts/predict_graph/train_data.dataset_graph_full_150step_larger_train.pkl-agentsall/time_model.seqVAE-stateenc.TF-globalrepr.pool-edgepred.concat-lr0.001-bs.16-goalenc.False_extended._costclose.1.0_costgoal.1.0_agentembed.False_predchange.node_inputgoal.False_excledge.True_goodactionreduced_walk_condprior.False_zvec.node"
CUDA_VISIBLE_DEVICES=5,6 python algos/train_graph_pred_excl.py inference=True inference_sample=True \
model.predict_node_change=True logging=False model.exclusive_edge=True ckpt_load=$ckpt_excl_vae"/290.pt" model.time_aggregate='seqVAE'




## Other tests

CUDA_VISIBLE_DEVICES=0 python tests/test_solo_loader.py model.predict_node_change=True  model.exclusive_edge=True ckpt_load=$ckpt_excl"/490.pt"



python analysis/helping_action_freq.py agent_pred_graph.ckpt_load=$ckpt_excl"/490.pt"   agent_pred_graph.model.predict_node_change=True  agent_pred_graph.model.exclusive_edge=True

## Evaluation model with full dataset


export ckpt_excl="/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/ckpts/predict_graph/train_data.dataset_graph_full_150step_larger_train.pkl-agentsall/time_model.LSTM-stateenc.TF-globalrepr.pool-edgepred.concat-lr0.001-bs.16-goalenc.False_extended._costclose.1.0_costgoal.1.0_agentembed.False_predchange.node_inputgoal.False_excledge.True_goodactionreduced_walk/"

python analysis/helping_action_freq.py agent_pred_graph.ckpt_load=$ckpt_excl"/490.pt"   agent_pred_graph.model.predict_node_change=True  agent_pred_graph.model.exclusive_edge=True 


python analysis/helping_gt_goal.py agent_pred_graph.ckpt_load=$ckpt_excl"/290.pt"   agent_pred_graph.model.predict_node_change=True  agent_pred_graph.model.exclusive_edge=True 

python analysis/helping_states.py agent_pred_graph.ckpt_load=$ckpt_excl"/290.pt"   agent_pred_graph.model.predict_node_change=True  agent_pred_graph.model.exclusive_edge=True  num_processes=0 num_samples=1
	

export ckpt_excl_vae="/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/ckpts/predict_graph/train_data.dataset_graph_full_150step_larger_train.pkl-agentsall/time_model.seqVAE-stateenc.TF-globalrepr.pool-edgepred.concat-lr0.001-bs.16-goalenc.False_extended._costclose.1.0_costgoal.1.0_agentembed.False_predchange.node_inputgoal.False_excledge.True_goodactionreduced_walk_condprior.False_zvec.node"

python analysis/helping_states.py agent_pred_graph.ckpt_load=$ckpt_excl"/290.pt"   agent_pred_graph.model.predict_node_change=True  agent_pred_graph.model.exclusive_edge=True  num_processes=0 num_samples=1 agent_pred_graph.model.time_aggregate='seqVAE' agent_pred_graph.model.cond_prior=False

```

