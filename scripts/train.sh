#!/bin/bash
model_dir="/glusterfs/dfs-gfs-dist/wangxida/masterarbeit/output/models/Entity_Blinding_15_05_21"
cache_dir="/glusterfs/dfs-gfs-dist/wangxida/masterarbeit/output/cache"
scibert_path="/glusterfs/dfs-gfs-dist/wangxida/masterarbeit/scibert_scivocab_uncased"
roberta_path="/glusterfs/dfs-gfs-dist/wangxida/masterarbeit/roberta-base-squad2" 

# Distributed Data Parallel (only for training!)
# For usage, following changes need to be made:
# number_gpus=2
# CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=${number_gpus} run/run_qa.py \
# --evaluate_during_training=False \

# Runtime Profiler
# For usage, following changes need to be made:
# CUDA_VISIBLE_DEVICES=2,3 python -m cProfile -o runstats run/run_qa.py \

# 0 1 2 57 3 4 5 58 6 7 8 59 9 10 11 60 18 19 20 63 21 22 23 64 42 43 53 71 49 50 51 52 72
# Use option -h or --help to get all available options

CUDA_VISIBLE_DEVICES=1,2,3 python run/run_qa.py \
--train_data="all_08_05_21_train" \
--dev_data="all_08_05_21_dev" \
--model_name_or_path=${scibert_path} \
--question_types 0 1 2 57 3 4 5 58 6 7 8 59 9 10 11 60 18 19 20 63 21 22 23 64 42 43 53 71 49 50 51 52 72 \
--output_dir=${model_dir} \
--cache_dir=${cache_dir} \
--cache_predictions=0 \
--predictions_suffix="_0" \
--do_train=True \
--do_eval=False \
--visualize_preds=False \
--overwrite_output_dir=True \
--overwrite_cache=False \
--retrieval_size=100 \
--entity_blinding=True \
--multiturn=False \
--multi_instance_learning=True \
--multi_instance_learning_neg=True \
--crf=False \
--use_simple_normalizer=False \
--learning_rate=2e-5 \
--weight_decay=0.01 \
--dropout=0.1 \
--max_grad_norm=1.0 \
--warmup_proportion=0.1 \
--num_train_epochs=6 \
--evaluate_during_training=True \
--evaluate_train_set=False \
--save_steps=1 \
--wandb=True
