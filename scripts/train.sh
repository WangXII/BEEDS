#!/bin/bash
model_dir="/glusterfs/dfs-gfs-dist/wangxida/masterarbeit/output/models/BioNLP_04_03_22"
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
# 0 2 57 3 5 58 6 8 59 9 11 60 18 20 63 21 23 64 42 53 49 51 72 44 74
# Use option -h or --help to get all available options

CUDA_VISIBLE_DEVICES=1,2,3 python run/run_qa.py \
--train_data="EVEX_17_01_21_train" \
--dev_data="EVEX_17_01_21_dev" \
--direct_data="BioNLP_New" \
--model_name_or_path=${scibert_path} \
--question_types 0 2 57 3 5 58 6 8 59 9 11 60 18 20 63 21 23 64 42 53 49 51 72 44 74 \
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
--entity_blinding=False \
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
--save_steps=1 \
--use_distantly_supervised_data=True \
--use_directly_supervised_data=True \
--direct_weight=4 \
--seed=1 \
--wandb=True
