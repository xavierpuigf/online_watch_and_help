
# KL 0.001
export rootpath="/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/ckpts/predict_graph/train_data.dataset_graph_full_150step_larger_train.pkl/lr0.0009-bs.256-klcoff.0.001-goalenc.False_predchange.none_inputgoal.False_excledge.True_preddiff.Falsereduced_walk_logname.newvaefull_encoder_task_graph_condprior.False_zvec.node"
export ck=""

export full_ck=$rootpath$ck

CUDA_VISIBLE_DEVICES=1 python analysis/goal_inference_particlefilter.py \
    agent_pred_graph="config_vae0.001" num_processes=20 \
    agent_pred_graph.ckpt_load=$full_ck"/290.pt" base_port=8185 hydra.run.dir=. hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled
