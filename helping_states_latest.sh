export rootpath="/data/vision/torralba/frames/data_acquisition/SyntheticStories/online_wah/ckpts/predict_graph/train_data.dataset_graph_full_150step_larger_train.pkl/lr0.0009-bs.256-klcoff.0.001-goalenc.False_predchange.none_inputgoal.False_excledge.True_preddiff.Falsereduced_walk_logname.newvaefull_encoder_task_graph_condprior.False_zvec.node"
export ck=""

export full_ck=$rootpath$ck
#CUDA_VISIBLE_DEVICES=3,4  python algos/inference_autoencoder_task.py \
#   --config-path=$full_ck --config-name="config" save_inference=True samples_per_graph=11 ckpt_load=$full_ck/290.pt name_log='VAE.KL.0.001' \
#   hydra.run.dir=. hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled

CUDA_VISIBLE_DEVICES=1 python analysis/helping_states_diffpred.py \
    agent_pred_graph="config_vae0.001" num_processes=0 \
    agent_pred_graph.ckpt_load=$full_ck"/290.pt" base_port=8183 hydra.run.dir=. hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled

#
#
#
#
