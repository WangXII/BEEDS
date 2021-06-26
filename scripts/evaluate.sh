#!/bin/bash
model_dir="/glusterfs/dfs-gfs-dist/wangxida/masterarbeit/output/models/Final_Model_Large_09_05_21"
cache_dir="/glusterfs/dfs-gfs-dist/wangxida/masterarbeit/output/cache"
scibert_path="/glusterfs/dfs-gfs-dist/wangxida/masterarbeit/scibert_scivocab_uncased"
roberta_path="/glusterfs/dfs-gfs-dist/wangxida/masterarbeit/roberta-base-squad2" 

# 0 1 2 57 3 4 5 58 6 7 8 59 9 10 11 60 18 19 20 63 21 22 23 64 42 43 53 71 49 50 51 52 72
# Use option -h or --help to get all available options
# Final_Model_Large_09_05_21, all_08_05_21_test
# EVEX_Model_10_05_21, evex_10_05_21_test
# Entity_Blinding_15_05_21
# predictions_suffix (All) _0 multiturn, _1 no multiturn, (EVEX) _2 multiturn, _3 no multiturn

CUDA_VISIBLE_DEVICES=3 python run/run_qa.py \
--train_data="all_08_05_21_train" \
--dev_data="all_08_05_21_test" \
--model_name_or_path=${scibert_path} \
--question_types 0 1 2 57 3 4 5 58 6 7 8 59 9 10 11 60 18 19 20 63 21 22 23 64 42 43 53 71 49 50 51 52 72 \
--output_dir=${model_dir} \
--cache_dir=${cache_dir} \
--cache_predictions=0 \
--predictions_suffix="_3" \
--visualize_preds=True \
--do_eval=True \
--eval_all_checkpoints=False \
--overwrite_cache=False \
--retrieval_size=100 \
--entity_blinding=False \
--multiturn=True \
--crf=False \
--use_simple_normalizer=False \
--wandb=True
