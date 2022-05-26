#!/bin/bash
cache_dir="/glusterfs/dfs-gfs-dist/wangxida/masterarbeit/output/cache"
scibert_path="/glusterfs/dfs-gfs-dist/wangxida/masterarbeit/scibert_scivocab_uncased"
roberta_path="/glusterfs/dfs-gfs-dist/wangxida/masterarbeit/roberta-base-squad2" 

# question types split
# PTMS: 0 2 57 3 5 58 6 8 59 9 11 60 18 20 63 21 23 64
# OTHER: 42 53 49 51 72
# COMPLEX: 44 74
# ALL: 0 2 57 3 5 58 6 8 59 9 11 60 18 20 63 21 23 64 42 53 49 51 72 44 74

# Change NEGATIVE_POSITIVE_EXAMPLES_RATIO in /configs/__init__.py
# Use option -h or --help to get all available options

python run/run_qa.py \
--train_data="EVEX_18_01_21_train" \
--dev_data="EVEX_18_01_21_dev_1000" \
--test_data="EVEX_18_01_21_test_1000" \
--direct_data="BioNLP_New2" \
--negative_examples=False \
--model_name_or_path=${scibert_path} \
--question_types 44 74 \
--cache_dir=${cache_dir} \
--do_build_data=True \
--overwrite_cache=False \
--index="pubmed2" \
--retrieval_granularity="sentences" \
--datefilter=True \
--retrieval_size=1000 \
--retrieval_with_full_question=False \
--batch_size=100 \
--tagger="detailed" \
--entity_blinding=False \
--wandb=True
