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

from data_processing.annotate_direct_data import DirectDataAnnotator
from data_processing.retrieve_and_annotate import DataBuilder
from utils_ner import generate_examples, convert_examples_to_features
from run.dataset import DistantBertDataset

logger = logging.getLogger(__name__)

class DataBuilderLoader:
    def __init__(self, args, model_helper, labels, pad_token_label_id):
        self.args = args
        self.model_helper = model_helper
        self.labels = labels
        self.pad_token_label_id = pad_token_label_id
        self.direct_annotator = None

    def load_and_cache_examples(self, dataset, indra_statements, question_type, cache=True, predict_bool=False):
        ''' Load and cache new distantly supervised datasets '''

        logger.info("Local rank {}".format(self.args.local_rank))
        if self.args.local_rank not in [-1, 0]:
            # Make sure only the first process in distributed training process the dataset and the others will use the cache
            logger.info("Waiting Rank {}".format(self.args.local_rank))
            torch.distributed.barrier()

        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            self.args.cache_dir + "/cached_features", "cached_{}_{}".format(dataset, question_type.name))
        cached_label_encoder = os.path.join(
            self.args.cache_dir + "/cached_features", "classes_{}_{}.npy".format(dataset, question_type.name))
        if os.path.exists(cached_features_file) and not self.args.overwrite_cache and cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
            subject_label_encoder = preprocessing.LabelEncoder()
            subject_label_encoder.classes_ = np.load(cached_label_encoder)

        else:
            logger.info("Creating features for {}_{}".format(dataset, question_type.name))
            # logger.info("INDRA Statements")
            # logger.info(indra_statements)
            data_builder = DataBuilder(self.args.retrieval_size, self.args.batch_size, self.model_helper, self.args.tagger, self.args.max_seq_length,
                                       self.args.limit_max_length, self.args.index, self.args.retrieval_granularity,
                                       self.args.datefilter, self.args.retrieval_with_full_question)
            annotations = data_builder.generate_annotations(indra_statements, question_type, predict_bool=predict_bool)
            examples, subjects_list = generate_examples(annotations, question_type)
            logger.info("Number of generated Examples")
            logger.info(len(examples))
            features, subject_label_encoder = convert_examples_to_features(
                examples, self.labels, self.args.max_seq_length, self.args.batch_size, self.model_helper,
                pad_token=self.model_helper.tokenizer.convert_tokens_to_ids([self.model_helper.tokenizer.pad_token])[0],
                pad_token_segment_id=0, pad_token_label_id=self.pad_token_label_id,
                subject_list=subjects_list)
            if self.args.local_rank in [-1, 0]:
                # Don't save dynamic features from questions building on previous predictions
                if cache:
                    logger.info("Saving features into cached file %s", cached_features_file)
                    torch.save(features, cached_features_file)
                    np.save(cached_label_encoder, subject_label_encoder.classes_)

        if self.args.local_rank == 0:
            # Make sure only the first process in distributed training process the dataset and the others will use the cache
            logger.info("Waiting Rank {}".format(self.args.local_rank))
            torch.distributed.barrier()

        tensor_dataset = DistantBertDataset(features, self.args.entity_blinding)
        tensor_dataset.truncate(self.args.batch_size)

        return tensor_dataset, subject_label_encoder

    def load_and_cache_direct_examples(self, dataset, question_type, datasplit, cache=True, add_negative_examples=True):
        ''' Load and cache new directly supervised datasets '''

        logger.info("Local rank {}".format(self.args.local_rank))
        if self.args.local_rank not in [-1, 0]:
            # Make sure only the first process in distributed training process the dataset and the others will use the cache
            logger.info("Waiting Rank {}".format(self.args.local_rank))
            torch.distributed.barrier()

        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            self.args.cache_dir + "/cached_features", "cached_{}_{}_{}".format(dataset, datasplit, question_type.name))
        cached_label_encoder = os.path.join(
            self.args.cache_dir + "/cached_features", "classes_{}_{}_{}.npy".format(dataset, datasplit, question_type.name))
        if os.path.exists(cached_features_file) and not self.args.overwrite_cache and cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
            subject_label_encoder = preprocessing.LabelEncoder()
            subject_label_encoder.classes_ = np.load(cached_label_encoder)

        else:
            logger.info("Creating features for {}_{}".format(dataset, question_type.name))
            if self.direct_annotator is None:
                self.direct_annotator = DirectDataAnnotator()
                self.direct_annotator.get_substrates()
                self.direct_annotator.get_all_relations(add_negative_examples)
                self.direct_annotator.generate_annotations(self.direct_annotator.relations_all)
                self.direct_annotator.get_stats()
            annotations = self.direct_annotator.get_datasplit_question_annotations(question_type, datasplit)
            examples, subjects_list = generate_examples(annotations, question_type)
            logger.info("Number of generated Examples")
            logger.info(len(examples))
            # Batch size is 1 for directly supervised examples
            features, subject_label_encoder = convert_examples_to_features(
                examples, self.labels, self.args.max_seq_length, 1, self.model_helper,
                pad_token=self.model_helper.tokenizer.convert_tokens_to_ids([self.model_helper.tokenizer.pad_token])[0],
                pad_token_segment_id=0, pad_token_label_id=self.pad_token_label_id,
                subject_list=subjects_list)
            if self.args.local_rank in [-1, 0]:
                # Don't save dynamic features from questions building on previous predictions
                if cache:
                    logger.info("Saving features into cached file %s", cached_features_file)
                    torch.save(features, cached_features_file)
                    np.save(cached_label_encoder, subject_label_encoder.classes_)

        if self.args.local_rank == 0:
            # Make sure only the first process in distributed training process the dataset and the others will use the cache
            logger.info("Waiting Rank {}".format(self.args.local_rank))
            torch.distributed.barrier()

        tensor_dataset = DistantBertDataset(features, self.args.entity_blinding)
        # tensor_dataset.truncate(args.batch_size)

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
