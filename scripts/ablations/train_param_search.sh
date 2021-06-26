#!/bin/bash
model_dir="/glusterfs/dfs-gfs-dist/wangxida/masterarbeit/output/models/SciBERT_27_04_21"
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

# Use option -h or --help to get all available options

for params in "0.1 2e-5 0.01 0.1 1 1" "0 2e-5 0.1 0 4 1000" "0.25 2e-5 0 0.5 2 1000" "0 5e-5 0.001 0 2 10"  "0 3e-5 0.1 0.5 1 1" "0 4e-5 0 0.25 4 1"  "0.25 2e-5 0.01 0 1 1000" "0 3e-5 0.1 0.1 4 100" "0.1 2e-5 0.001 0.25 2 1" "0.1 2e-5 0.001 0.5 4 100" "0 4e-5 0.001 0.25 2 1000" "0.25 4e-5 0 0 2 1"
do
    set $params
    CUDA_VISIBLE_DEVICES=3 python run/run_qa.py \
    --train_data="scibert_27_04_21_train" \
    --dev_data="scibert_27_04_21_dev" \
    --model_name_or_path=${scibert_path} \
    --question_types 0 1 2 57 3 4 5 58 6 7 8 59 9 10 11 60 18 19 20 63 21 22 23 64 42 43 53 71 49 50 51 52 72 \
    --output_dir=${model_dir} \
    --cache_dir=${cache_dir} \
    --cache_predictions=0 \
    --predictions_suffix="_1" \
    --do_train=True \
    --do_eval=False \
    --visualize_preds=False \
    --overwrite_output_dir=True \
    --overwrite_cache=False \
    --retrieval_size=45 \
    --multiturn=False \
    --multi_instance_learning=True \
    --multi_instance_learning_neg=True \
    --crf=False \
    --use_simple_normalizer=False \
    --warmup_proportion=$1 \
    --learning_rate=$2 \
    --weight_decay=$3 \
    --dropout=$4 \
    --gradient_accumulation_steps=$5 \
    --max_grad_norm=$6 \
    --num_train_epochs=4 \
    --evaluate_during_training=True \
    --evaluate_train_set=False \
    --save_steps=-1 \
    --wandb=True
done
