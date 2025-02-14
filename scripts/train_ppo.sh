#!/bin/bash

PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 CUDA_LAUNCH_BLOCKING=1 python -m verl.trainer.main_ppo \
 data.train_files=train_data.parquet \
 data.val_files=test_data.parquet \
 data.train_batch_size=256 \
 data.val_batch_size=256 \
 data.max_prompt_length=2048 \
 data.max_response_length=1024 \
 actor_rollout_ref.model.path=Qwen2.5-7B-Instruct \
 actor_rollout_ref.actor.optim.lr=5e-7 \
 actor_rollout_ref.actor.ppo_mini_batch_size=256 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 critic.optim.lr=5e-6 \
 critic.model.path=Qwen2.5-7B-Instruct \
 critic.ppo_micro_batch_size_per_gpu=4 \
 algorithm.kl_ctrl.kl_coef=0.001 \
 algorithm.adv_estimator=gae \
 +trainer.val_before_train=False \
 trainer.default_hdfs_dir=null \
 trainer.n_gpus_per_node=1 \
 trainer.nnodes=8 \
 trainer.save_freq=16 \
 trainer.test_freq=1000000 \
 trainer.total_epochs=15 \
 trainer.project_name=test \
 trainer.experiment_name=test_task \
 trainer.tensorboard_dir=logs/tensorboard \
 trainer.rl_logging_board_dir=logs/rl_logging_board  2>&1 | tee verl_demo.log