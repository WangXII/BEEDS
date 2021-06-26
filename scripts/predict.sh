#!/bin/bash
model_dir="/glusterfs/dfs-gfs-dist/wangxida/masterarbeit/output/models/complex_04_03_21"
cache_dir="/glusterfs/dfs-gfs-dist/wangxida/masterarbeit/output/cache"

# Use option -h or --help to get all available options

CUDA_VISIBLE_DEVICES=2,3 python run/run_qa.py \
--output_dir=${model_dir} \
--cache_dir=${cache_dir} \
--cache_predictions=0 \
--predictions_suffix="_2" \
--do_predict=True \
--overwrite_cache=False \
--index_name="pubmed_sentences" \
--index_name_two="pubmed_detailed" \
--datefilter=False \
--retrieval_size=100 \
--max_bag_size=20 \
--max_seq_length=384 \
--tagger="detailed" \
--entity_blinding=False \
--crf=True \
--multiturn=False \
--use_simple_normalizer=False \
--wandb=False
