""" Command line argument parser with their default values """

import os
import sys

if os.name == 'nt':
    root_path = "/".join(os.path.realpath(__file__).split("\\")[:-2])
else:
    root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.append(root_path)

import argparse


def parse_arguments():
    ''' Return an ArgumentParser instance with the specified parameters '''

    parser = argparse.ArgumentParser()

    # Parameters
    parser.add_argument("--train_data", default="PHOSPHO_CAUSE_train",
                        type=str, help="The train data. Must contain exactly one of the three strings (train, dev, test).")
    parser.add_argument("--dev_data", default="PHOSPHO_CAUSE_dev",
                        type=str, help="The dev data. Must contain exactly one of the three strings (train, dev, test).")
    parser.add_argument("--test_data", default="PHOSPHO_CAUSE_test",
                        type=str, help="The test data. Must contain exactly one of the three strings (train, dev, test).")
    parser.add_argument("--question_types", default=[], nargs='+',
                        type=str, help="List of question types as numbers. Reference in /data_processing/datatypes.py.")

    parser.add_argument("--model_name_or_path", default="/glusterfs/dfs-gfs-dist/wangxida/masterarbeit/scibert_scivocab_uncased",
                        type=str, help="Path to pre-trained model or shortcut name")
    parser.add_argument("--output_dir", default="/glusterfs/dfs-gfs-dist/wangxida/masterarbeit/output_dir", type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--cache_dir", default="/glusterfs/dfs-gfs-dist/wangxida/masterarbeit/output/cache", type=str,
                        help="The output directory where models/nn outputs/examples are cached.")
    parser.add_argument("--predictions_suffix", default="", type=str,
                        help="Suffix for the predictions cache.")
    parser.add_argument("--predictions_dir", default="/glusterfs/dfs-gfs-dist/wangxida/masterarbeit/predictions", type=str,
                        help="The output directory where the model predictions will be written.")
    parser.add_argument("--visualize_preds", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Whether to visualize the predictions or to calculate the evaluation or evaluate the metrics.")
    parser.add_argument("--index_name", default="pubmed_sentences", type=str,
                        help="Name of the primary ElasticSearch index.")
    parser.add_argument("--index_name_two", default="pubmed_sentences", type=str,
                        help="Name of the secondary ElasticSearch index. If the string equals 'None', ignore retrieving from a second index.")

    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--limit_max_length", default=True, type=lambda x: (str(x).lower() == 'true'),
                        help="Whether to limit train examples to max seq length, or to split across several docs.")
    parser.add_argument("--doc_stride", default=64, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--do_build_data", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Whether to build the complete dataset (train, eval and predict).")
    parser.add_argument("--do_train", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Whether to run training.")
    parser.add_argument("--do_eval", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Whether to run predictions on the test set.")
    parser.add_argument("--evaluate_during_training", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Whether to run evaluation during training at each logging step.")
    parser.add_argument("--evaluate_train_set", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Whether to evaluate the training set during the training process.")
    parser.add_argument("--do_lower_case", default=True, type=lambda x: (str(x).lower() == 'true'),
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--debug", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Log debugging infos or not.")
    parser.add_argument("--entity_blinding", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Perform Entity Blinding or not.")
    parser.add_argument("--crf", default=True, type=lambda x: (str(x).lower() == 'true'),
                        help="Use CRF for sequence aggregation or not.")
    parser.add_argument("--multiturn", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Use answer from simple questions to feed into complex questions.")
    parser.add_argument("--multi_instance_learning", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Use multi instance learning or standard distant supervision.")
    parser.add_argument("--multi_instance_learning_neg", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Use multi instance learning or standard distant supervision for the negative examples.")
    parser.add_argument("--datefilter", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Filter documents of an earlier publication date (before 2013) than the EVEX baseline.")
    parser.add_argument("--use_simple_normalizer", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Use simple dictionary lookup as additional normalization tool.")
    parser.add_argument("--retrieval_size", default=100, type=int,
                        help="Retrieval size.")
    parser.add_argument("--retrieval_with_full_question", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Retrieve wuith full question or only question keywords.")

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="Hidden dropout probability.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.0, type=float,
                        help="Linear warmup over warmup_proportion times total_steps.")

    parser.add_argument("--tagger", default="simple", type=str,
                        help="Tagger 'simple' tags all entity mentions. Tagger 'detailed' only marks entity mentions near other trigger and entities.")
    parser.add_argument("--wandb", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Loggint with Weight and biases.")
    parser.add_argument("--logging_steps", type=int, default=149,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=-1,
                        help="If 1, then save checkpoint after each epoch. If -1, then do not save.")
    parser.add_argument("--eval_all_checkpoints", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending "
                             + "and ending with step number")
    parser.add_argument("--no_cuda", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Overwrite the content of the output directory")
    parser.add_argument("--overwrite_cache", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--cache_predictions", default=0, type=int,
                        help="If 0, do not cache predictions. If 1, cache predictions. If 2, load cached predictions.")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--fp16", default=True, type=lambda x: (str(x).lower() == 'true'),
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    return parser
