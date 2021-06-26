""" Utility IO functions for run.py """

from __future__ import absolute_import, division, print_function
import os
import sys

if os.name == 'nt':
    root_path = "/".join(os.path.realpath(__file__).split("\\")[:-2])
else:
    root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.append(root_path)

import logging
import random
import numpy as np
import torch

# from torch.utils.data import TensorDataset
from sklearn import preprocessing

from data_processing.retrieve_and_annotate import DataBuilder
from utils_ner import generate_examples, convert_examples_to_features
from run.dataset import DistantBertDataset

logger = logging.getLogger(__name__)


def load_and_cache_examples(args, model_helper, labels, pad_token_label_id, dataset, indra_statements, question_type,
                            cache=True, predict_bool=False):
    ''' Load and cache new datasets '''

    logger.info("Local rank {}".format(args.local_rank))
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training process the dataset and the others will use the cache
        logger.info("Waiting Rank {}".format(args.local_rank))
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.cache_dir + "/cached_features", "cached_{}_{}".format(dataset, question_type.name))
    cached_label_encoder = os.path.join(
        args.cache_dir + "/cached_features", "classes_{}_{}.npy".format(dataset, question_type.name))
    if os.path.exists(cached_features_file) and not args.overwrite_cache and cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        subject_label_encoder = preprocessing.LabelEncoder()
        subject_label_encoder.classes_ = np.load(cached_label_encoder)

    else:
        logger.info("Creating features for {}_{}".format(dataset, question_type.name))
        # logger.info("INDRA Statements")
        # logger.info(indra_statements)
        data_builder = DataBuilder(args.retrieval_size, model_helper, args.tagger, args.max_seq_length,
                                   args.limit_max_length, args.index_name, args.index_name_two, args.datefilter,
                                   args.retrieval_with_full_question)
        annotations = data_builder.generate_annotations(indra_statements, question_type, predict_bool=predict_bool)
        examples, subjects_list = generate_examples(annotations, question_type)
        logger.info("Number of generated Examples")
        logger.info(len(examples))
        features, subject_label_encoder = convert_examples_to_features(
            examples, labels, args.max_seq_length, args.doc_stride, args.retrieval_size, model_helper,
            pad_token=model_helper.tokenizer.convert_tokens_to_ids([model_helper.tokenizer.pad_token])[0],
            pad_token_segment_id=0, pad_token_label_id=pad_token_label_id,
            subject_list=subjects_list)
        if args.local_rank in [-1, 0]:
            # Don't save dynamic features from questions building on previous predictions
            if cache:
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(features, cached_features_file)
                np.save(cached_label_encoder, subject_label_encoder.classes_)

    if args.local_rank == 0:
        # Make sure only the first process in distributed training process the dataset and the others will use the cache
        logger.info("Waiting Rank {}".format(args.local_rank))
        torch.distributed.barrier()

    tensor_dataset = DistantBertDataset(features, args.entity_blinding)
    tensor_dataset.truncate(args.retrieval_size)

    return tensor_dataset, subject_label_encoder


def save_model(args, output_dir, model):
    ''' Save a model checkpoint '''

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Save a trained model and configuration using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    # Take care of distributed training
    if hasattr(model, "module"):
        model = model.module
    if "berta" in args.model_name_or_path:
        if hasattr(model.roberta, "module"):
            model.roberta.module.save_pretrained(output_dir)
        else:
            model.roberta.save_pretrained(output_dir)
    else:
        if hasattr(model.bert, "module"):
            model.bert.module.save_pretrained(output_dir)
        else:
            model.bert.save_pretrained(output_dir)

    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    logger.info("Saving model (or model checkpoint) to %s", output_dir)


def set_seed(args, n_gpu):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
