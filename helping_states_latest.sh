
# KL 0.001
export rootpath="/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/ckpts/predict_graph/train_data.dataset_graph_full_150step_larger_train.pkl/lr0.0009-bs.256-klcoff.0.001-goalenc.False_predchange.none_inputgoal.False_excledge.True_preddiff.Falsereduced_walk_logname.newvaefull_encoder_task_graph_condprior.False_zvec.node"
export ck=""
export full_ck=$rootpath$ck

CUDA_VISIBLE_DEVICES=1 python analysis/helping_states_diffpred.py \
    agent_pred_graph="config_vae0.001" num_processes=30 \
    agent_pred_graph.ckpt_load=$full_ck"/290.pt" base_port=8183 hydra.run.dir=. hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled


# # KL 0.0
# export rootpath="/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/ckpts/predict_graph/train_data.dataset_graph_full_150step_larger_train.pkl/lr0.0009-bs.256-klcoff.0.0-goalenc.False_predchange.none_inputgoal.False_excledge.True_preddiff.Falsereduced_walk_logname.newvaefull_encoder_task_graph_condprior.False_zvec.node"
# export ck=""
# export full_ck=$rootpath$ck

# CUDA_VISIBLE_DEVICES=1 python analysis/helping_states_diffpred.py \
#     agent_pred_graph="config_vae0.000" num_processes=30 \
#     agent_pred_graph.ckpt_load=$full_ck"/290.pt" base_port=8183 hydra.run.dir=. hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled


# # KL 1.0
# export rootpath="/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/ckpts/predict_graph/train_data.dataset_graph_full_150step_larger_train.pkl/lr0.0009-bs.256-klcoff.1-goalenc.False_predchange.none_inputgoal.False_excledge.True_preddiff.Falsereduced_walk_logname.newvaefull_encoder_task_graph_condprior.False_zvec.node"
# export ck=""
# export full_ck=$rootpath$ck

# CUDA_VISIBLE_DEVICES=1 python analysis/helping_states_diffpred.py \
#     agent_pred_graph="config_vae1.000" num_processes=30 \
#     agent_pred_graph.ckpt_load=$full_ck"/290.pt" base_port=8183 hydra.run.dir=. hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled


# Deterministic
export rootpath="/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/ckpts/predict_graph/train_data.dataset_graph_full_150step_larger_train.pkl/lr0.0009-bs.256-klcoff.1-goalenc.False_predchange.none_inputgoal.False_excledge.True_preddiff.Falsereduced_walk_logname.detfull_encoder_task_graph"
export ck=""
export full_ck=$rootpath$ck

CUDA_VISIBLE_DEVICES=1 python analysis/helping_states_diffpred.py \
    agent_pred_graph="config_det" num_processes=30 \
    agent_pred_graph.ckpt_load=$full_ck"/290.pt" base_port=8183 hydra.run.dir=. hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled


#==========
# SMALL SET
#==========

# KL 0.001
export rootpath="/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/ckpts/predict_graph/train_data.dataset_graph_full_150step_larger_train.pkl/lr0.0009-bs.256-klcoff.0.001-goalenc.False_predchange.none_inputgoal.False_excledge.True_preddiff.Falsereduced_walk_logname.newvaefull_encoder_task_graph_condprior.False_zvec.node"
export ck=""
export full_ck=$rootpath$ck

CUDA_VISIBLE_DEVICES=1 python analysis/helping_states_diffpred.py \
    agent_pred_graph="config_vae0.001" num_processes=30 \
    small_set=True num_tries=3 \
    agent_pred_graph.ckpt_load=$full_ck"/290.pt" base_port=8183 hydra.run.dir=. hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled


# Deterministic
export rootpath="/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/ckpts/predict_graph/train_data.dataset_graph_full_150step_larger_train.pkl/lr0.0009-bs.256-klcoff.1-goalenc.False_predchange.none_inputgoal.False_excledge.True_preddiff.Falsereduced_walk_logname.detfull_encoder_task_graph"
export ck=""
export full_ck=$rootpath$ck

CUDA_VISIBLE_DEVICES=1 python analysis/helping_states_diffpred.py \
    agent_pred_graph="config_det" num_processes=30 \
    small_set=True num_tries=3 reset_steps=15 \
    agent_pred_graph.ckpt_load=$full_ck"/290.pt" base_port=8183 hydra.run.dir=. hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled



#==========
# SMALL SET 1 particle
#==========
export rootpath="/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/ckpts/predict_graph/train_data.dataset_graph_full_150step_larger_train.pkl/lr0.0009-bs.256-klcoff.1-goalenc.False_predchange.none_inputgoal.False_excledge.True_preddiff.Falsereduced_walk_logname.detfull_encoder_task_graph"
export ck=""
export full_ck=$rootpath$ck

CUDA_VISIBLE_DEVICES=1 python analysis/helping_states_diffpred.py \
    agent_pred_graph="config_det" num_processes=30 num_samples=1 inv_plan=False\
    small_set=True num_tries=3 \
    agent_pred_graph.ckpt_load=$full_ck"/290.pt" base_port=8184 hydra.run.dir=. hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled


#==========
# SMALL SET no inv plan
#==========
export rootpath="/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/ckpts/predict_graph/train_data.dataset_graph_full_150step_larger_train.pkl/lr0.0009-bs.256-klcoff.1-goalenc.False_predchange.none_inputgoal.False_excledge.True_preddiff.Falsereduced_walk_logname.detfull_encoder_task_graph"
export ck=""
export full_ck=$rootpath$ck

CUDA_VISIBLE_DEVICES=1 python analysis/helping_states_diffpred.py \
    agent_pred_graph="config_det" num_processes=30 inv_plan=False\
    small_set=True num_tries=3 \
    agent_pred_graph.ckpt_load=$full_ck"/290.pt" base_port=8185 hydra.run.dir=. hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled



#=========
# SMALL SET w/ uniform proposal
#=========
export rootpath="/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/ckpts/predict_graph/train_data.dataset_graph_full_150step_larger_train.pkl/lr0.0009-bs.256-klcoff.1-goalenc.False_predchange.none_inputgoal.False_excledge.True_preddiff.Falsereduced_walk_logname.detfull_encoder_task_graph"
export ck=""
export full_ck=$rootpath$ck

CUDA_VISIBLE_DEVICES=1 python analysis/helping_states_diffpred.py \
    agent_pred_graph="config_det" num_processes=30 agent_pred_graph.name_log="uniform" \
    small_set=True num_tries=3 \
    agent_pred_graph.ckpt_load=$full_ck"/290.pt" base_port=8186 hydra.run.dir=. hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled


#==========
# SMALL SET empowermetn: no inv plan + w/ uniform proposal
#==========
export rootpath="/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/ckpts/predict_graph/train_data.dataset_graph_full_150step_larger_train.pkl/lr0.0009-bs.256-klcoff.1-goalenc.False_predchange.none_inputgoal.False_excledge.True_preddiff.Falsereduced_walk_logname.detfull_encoder_task_graph"
export ck=""
export full_ck=$rootpath$ck

CUDA_VISIBLE_DEVICES=1 python analysis/helping_empowerment.py \
    agent_pred_graph="config_det" num_processes=30 inv_plan=False agent_pred_graph.name_log="uniform" \
    small_set=True num_tries=3 \
    agent_pred_graph.ckpt_load=$full_ck"/290.pt" base_port=8185 hydra.run.dir=. hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled


## 100 particles
export rootpath="/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/ckpts/predict_graph/train_data.dataset_graph_full_150step_larger_train.pkl/lr0.0009-bs.256-klcoff.1-goalenc.False_predchange.none_inputgoal.False_excledge.True_preddiff.Falsereduced_walk_logname.detfull_encoder_task_graph"
export ck=""
export full_ck=$rootpath$ck

CUDA_VISIBLE_DEVICES=1 python analysis/helping_empowerment.py \
    agent_pred_graph="config_det" num_processes=30 inv_plan=False agent_pred_graph.name_log="uniform" \
    small_set=True num_tries=3 num_samples=100\
    agent_pred_graph.ckpt_load=$full_ck"/290.pt" base_port=8186 hydra.run.dir=. hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled


#==========
# SMALL SET action frequency
#==========
export rootpath="/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/ckpts/predict_graph/train_data.dataset_graph_full_150step_larger_train.pkl/lr0.0009-bs.256-klcoff.1-goalenc.False_predchange.none_inputgoal.False_excledge.True_preddiff.Falsereduced_walk_logname.detfull_encoder_task_graph"
export ck=""
export full_ck=$rootpath$ck

CUDA_VISIBLE_DEVICES=1 python analysis/helping_action_freq_new.py \
    agent_pred_graph="config_det" num_processes=20 \
    small_set=True num_tries=3 reset_steps=15 \
    agent_pred_graph.ckpt_load=$full_ck"/290.pt" base_port=8187 hydra.run.dir=. hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled


#==========
# FULL SET action frequency
#==========
export rootpath="/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/ckpts/predict_graph/train_data.dataset_graph_full_150step_larger_train.pkl/lr0.0009-bs.256-klcoff.1-goalenc.False_predchange.none_inputgoal.False_excledge.True_preddiff.Falsereduced_walk_logname.detfull_encoder_task_graph"
export ck=""
export full_ck=$rootpath$ck

CUDA_VISIBLE_DEVICES=1 python analysis/helping_action_freq_new.py \
    agent_pred_graph="config_det" num_processes=20 \
    small_set=False num_tries=3 reset_steps=15 \
    agent_pred_graph.ckpt_load=$full_ck"/290.pt" base_port=8188 hydra.run.dir=. hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled



#========
# DEBUG
#========

# KL 0.001
export rootpath="/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/ckpts/predict_graph/train_data.dataset_graph_full_150step_larger_train.pkl/lr0.0009-bs.256-klcoff.0.001-goalenc.False_predchange.none_inputgoal.False_excledge.True_preddiff.Falsereduced_walk_logname.newvaefull_encoder_task_graph_condprior.False_zvec.node"
export ck=""
export full_ck=$rootpath$ck

CUDA_VISIBLE_DEVICES=2 python analysis/helping_states_diffpred.py \
    agent_pred_graph="config_vae0.001" num_processes=30 debug=True \
    agent_pred_graph.ckpt_load=$full_ck"/290.pt" base_port=8184 hydra.run.dir=. hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled


# # KL 0.0
# export rootpath="/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/ckpts/predict_graph/train_data.dataset_graph_full_150step_larger_train.pkl/lr0.0009-bs.256-klcoff.0.0-goalenc.False_predchange.none_inputgoal.False_excledge.True_preddiff.Falsereduced_walk_logname.newvaefull_encoder_task_graph_condprior.False_zvec.node"
# export ck=""
# export full_ck=$rootpath$ck

# CUDA_VISIBLE_DEVICES=2 python analysis/helping_states_diffpred.py \
#     agent_pred_graph="config_vae0.000" num_processes=30 debug=True \
#     agent_pred_graph.ckpt_load=$full_ck"/290.pt" base_port=8184 hydra.run.dir=. hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled


# # KL 1.0
# export rootpath="/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/ckpts/predict_graph/train_data.dataset_graph_full_150step_larger_train.pkl/lr0.0009-bs.256-klcoff.1-goalenc.False_predchange.none_inputgoal.False_excledge.True_preddiff.Falsereduced_walk_logname.newvaefull_encoder_task_graph_condprior.False_zvec.node"
# export ck=""
# export full_ck=$rootpath$ck

# CUDA_VISIBLE_DEVICES=2 python analysis/helping_states_diffpred.py \
#     agent_pred_graph="config_vae1.000" num_processes=30 debug=True \
#     agent_pred_graph.ckpt_load=$full_ck"/290.pt" base_port=8184 hydra.run.dir=. hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled


# Deterministic
export rootpath="/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/ckpts/predict_graph/train_data.dataset_graph_full_150step_larger_train.pkl/lr0.0009-bs.256-klcoff.1-goalenc.False_predchange.none_inputgoal.False_excledge.True_preddiff.Falsereduced_walk_logname.detfull_encoder_task_graph"
export ck=""
export full_ck=$rootpath$ck

CUDA_VISIBLE_DEVICES=2 python analysis/helping_states_diffpred.py \
    agent_pred_graph="config_det" num_processes=30 debug=True \
    agent_pred_graph.ckpt_load=$full_ck"/290.pt" base_port=8184 hydra.run.dir=. hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled



#==========
# SMALL SET DEBUG
#==========

# KL 0.001
export rootpath="/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/ckpts/predict_graph/train_data.dataset_graph_full_150step_larger_train.pkl/lr0.0009-bs.256-klcoff.0.001-goalenc.False_predchange.none_inputgoal.False_excledge.True_preddiff.Falsereduced_walk_logname.newvaefull_encoder_task_graph_condprior.False_zvec.node"
export ck=""
export full_ck=$rootpath$ck

CUDA_VISIBLE_DEVICES=1 python analysis/helping_states_diffpred.py \
    agent_pred_graph="config_vae0.001" num_processes=30 debug=True \
    small_set=True num_tries=3 \
    agent_pred_graph.ckpt_load=$full_ck"/290.pt" base_port=8183 hydra.run.dir=. hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled


# Deterministic
export rootpath="/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/ckpts/predict_graph/train_data.dataset_graph_full_150step_larger_train.pkl/lr0.0009-bs.256-klcoff.1-goalenc.False_predchange.none_inputgoal.False_excledge.True_preddiff.Falsereduced_walk_logname.detfull_encoder_task_graph"
export ck=""
export full_ck=$rootpath$ck

CUDA_VISIBLE_DEVICES=1 python analysis/helping_states_diffpred.py \
    agent_pred_graph="config_det" num_processes=30 debug=True \
    small_set=True num_tries=3 \
    agent_pred_graph.ckpt_load=$full_ck"/290.pt" base_port=8183 hydra.run.dir=. hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled
