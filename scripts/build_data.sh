#!/bin/bash
cache_dir="/glusterfs/dfs-gfs-dist/wangxida/masterarbeit/output/cache"
scibert_path="/glusterfs/dfs-gfs-dist/wangxida/masterarbeit/scibert_scivocab_uncased"
roberta_path="/glusterfs/dfs-gfs-dist/wangxida/masterarbeit/roberta-base-squad2" 

# question types split
# STATECHANGE_CAUSE: 72
# PHOSPHORYLATION ENZYMES: 0 1
# PTMS: 0 1 2 57 3 4 5 58 6 7 8 59 9 10 11 60 18 19 20 63 21 22 23 64
# OTHER: 42 43 53 71 49 50 51 52
# ALL: 0 1 2 57 3 4 5 58 6 7 8 59 9 10 11 60 18 19 20 63 21 22 23 64 42 43 53 71 49 50 51 52 72

# Change NEGATIVE_POSITIVE_EXAMPLES_RATIO in /configs/__init__.py
# Use option -h or --help to get all available options

python run/run_qa.py \
--train_data="evex_10_05_21_train" \
--dev_data="evex_10_05_21_dev" \
--test_data="evex_10_05_21_test" \
--model_name_or_path=${scibert_path} \
--question_types 0 1 2 57 3 4 5 58 6 7 8 59 9 10 11 60 18 19 20 63 21 22 23 64 42 43 53 71 49 50 51 52 72 \
--cache_dir=${cache_dir} \
--do_build_data=True \
--overwrite_cache=False \
--index_name="pubmed_detailed" \
--index_name_two="pubmed_sentences" \
--datefilter=True \
--retrieval_size=100 \
--retrieval_with_full_question=False \
--tagger="detailed" \
--entity_blinding=False \
--wandb=True
