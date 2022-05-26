#!/bin/bash
# model_dir="/glusterfs/dfs-gfs-dist/wangxida/masterarbeit/output/models/Final_Model_Large_09_05_21"
# model_dir="/glusterfs/dfs-gfs-dist/wangxida/masterarbeit/output/models/ALL_27_12_21" # pos and negative examples for direct supervision
# model_dir="/glusterfs/dfs-gfs-dist/wangxida/masterarbeit/output/models/ALL_28_12_21" # pos examples direct supervision old
# model_dir="/glusterfs/dfs-gfs-dist/wangxida/masterarbeit/output/models/EVEX_17_01_22_Paragraphs"
model_dir="/glusterfs/dfs-gfs-dist/wangxida/masterarbeit/output/models/BioNLP_04_03_22"
# model_dir="/glusterfs/dfs-gfs-dist/wangxida/masterarbeit/output/models/EVEX_18_01_22_Sentences"
cache_dir="/glusterfs/dfs-gfs-dist/wangxida/masterarbeit/output/cache"
scibert_path="/glusterfs/dfs-gfs-dist/wangxida/masterarbeit/scibert_scivocab_uncased"
roberta_path="/glusterfs/dfs-gfs-dist/wangxida/masterarbeit/roberta-base-squad2" 

# 0 2 57 3 5 58 6 8 59 9 11 60 18 20 63 21 23 64 42 53 49 51 72 44 74
# Use option -h or --help to get all available options
# EVEX_17_01_21_train, ALL_26_12_21_train
# predictions_suffix (All) _0 multiturn, _1 no multiturn, (EVEX) _2 multiturn, _3 no multiturn

CUDA_VISIBLE_DEVICES=0 python run/run_qa.py \
--train_data="EVEX_18_01_21_train" \
--dev_data="EVEX_18_01_21_dev_1000" \
--direct_data="BioNLP_New" \
--model_name_or_path=${scibert_path} \
--question_types 0 2 57 3 5 58 6 8 59 9 11 60 18 20 63 21 23 64 42 53 49 51 72 44 74 \
--output_dir=${model_dir} \
--cache_dir=${cache_dir} \
--cache_predictions=0 \
--overwrite_cache=False \
--predictions_suffix="_1" \
--visualize_preds=False \
--do_eval=True \
--eval_all_checkpoints=False \
--overwrite_cache=False \
--retrieval_size=1000 \
--entity_blinding=False \
--multiturn=True \
--crf=False \
--use_simple_normalizer=True \
--wandb=True
