# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Helps creating question answering examples. """

from __future__ import absolute_import, division, print_function

import logging
from sklearn import preprocessing

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class DistantSupervisionExample(object):
    """ Contains a list of InputExample """

    def __init__(self):
        """Constructs a InputExample. List of size args.bag_size as we use Distant Supervision
        Args:
            words: list. The words of the sequence.
            blinded_words: list. The words of the sequence with entity blinding.
            labels: list. The labels for each word of the sequence. This should be specified for train and dev examples, but not for test examples.
            debug_labels: list. The debug_labels for each word of the sequence. This should be specified for train and dev examples, but not for test examples.
                Used for debugging and plotting the label scores.
            whitespaces: list. List for each token the bool whether a whitespace follow afterwards.
            positions: list of tuples. Tuple with start and end position of each token.
            question_end: int. The index where the question ends and the text span begins.
            subjects: list. Subjects of interest encoded in ints.
            pubmed_id: String. PubMed ID.
            question_id: list. Unique id for the example.
            question_type: list. Type of question asked
        """

        self.words_vec = []
        self.blinded_words_vec = []
        self.labels_vec = []
        self.debug_labels_vec = []
        self.whitespaces_vec = []
        self.positions_vec = []
        self.question_end_vec = []
        self.subjects_vec = []
        self.subject_lengths_vec = []
        self.pubmed_id_vec = []
        self.question_id_vec = []
        self.question_type_vec = []


class InputFeatureVectors(object):
    """A single set of features of data."""

    def __init__(self):
        self.input_ids = []
        self.input_ids_blinded = []
        self.input_mask = []
        self.segment_ids = []
        self.label_ids = []
        self.debug_label_ids = []
        self.whitespace_bools = []
        self.position_ids = []
        self.subjects = []
        self.pubmed_id = []
        self.question_id = []
        self.subject_lengths = []
        self.question_type = []


def generate_examples(annotation_data, question_type):
    examples = []
    logger.info("File answer statistics")
    logger.info(len(annotation_data))
    if len(annotation_data) > 0:
        logger.info(len(annotation_data[0]))
    all_subjects = []
    question_number = 0
    question_id = -1
    for question in annotation_data:
        # logger.info("Bag size in generate_examples()")
        # logger.info(len(question))
        question_example = DistantSupervisionExample()
        question_id = question_number * 100
        for answer_candidate in question:
            # logger.info(file)
            bool_question_end = False
            question_end = -1
            words = []
            blinded_words = []
            labels = []
            debug_labels = []
            whitespaces = []
            positions = []
            pubmed_id = int(answer_candidate[0].split("_")[0])  # Split combined Pubmed ID and Paragraph ID (and Sentence ID)
            question_id = question_id + 1
            subjects = answer_candidate[1]
            all_subjects.extend(subjects)
            answer_tokens = answer_candidate[2:]
            for i, token in enumerate(answer_tokens):
                if (token[0] == "[SEP]" or token[0] == "</s>") and not bool_question_end:
                    bool_question_end = True
                    question_end = i + 1
                words.append(token[0])
                blinded_words.append(token[6])
                labels.append(token[1])
                debug_labels.append(token[5])
                whitespaces.append(token[2])
                positions.append([token[3], token[4]])
            assert len(words) == len(blinded_words)
            question_example.words_vec.append(words)
            question_example.blinded_words_vec.append(blinded_words)
            question_example.labels_vec.append(labels)
            question_example.debug_labels_vec.append(debug_labels)
            question_example.whitespaces_vec.append(whitespaces)
            question_example.positions_vec.append(positions)
            question_example.question_end_vec.append(question_end)
            question_example.subjects_vec.append(subjects)
            question_example.subject_lengths_vec.append(len(subjects))
            question_example.pubmed_id_vec.append(pubmed_id)
            question_example.question_id_vec.append(question_id)
            question_example.question_type_vec.append(question_type.value)
            assert question_end != -1
        # logger.info("Sequence lengths")
        # logger.info([len(question_ex) for question_ex in question_example.words_vec])
        question_number += 1
        examples.append(question_example)
    return examples, all_subjects


def convert_examples_to_features(examples, label_list, max_seq_length, doc_stride, retrieval_size, model_helper,
                                 pad_token=0, pad_token_segment_id=0, pad_token_label_id=-1, sequence_a_segment_id=0,
                                 sequence_b_segment_id=1, mask_padding_with_zero=True, subject_list=[]):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}
    subject_label_encoder = preprocessing.LabelEncoder()
    subject_label_encoder.fit(subject_list)

    features = []
    for ds_sample in examples:
        sample_features = InputFeatureVectors()
        for ex_index in range(len(ds_sample.words_vec)):
            tokens = []
            blinded_tokens = []
            labels = []
            debug_labels = []
            whitespaces = []
            positions = []
            for word_tokens, label, blinded_word_tokens, debug_label, whitespace, position in zip(
                    ds_sample.words_vec[ex_index], ds_sample.labels_vec[ex_index], ds_sample.blinded_words_vec[ex_index], ds_sample.debug_labels_vec[ex_index],
                    ds_sample.whitespaces_vec[ex_index], ds_sample.positions_vec[ex_index]):
                tokens.append(word_tokens)
                blinded_tokens.append(blinded_word_tokens)
                # logger.warn(ds_sample.words_vec[ex_index])
                # logger.warn(ds_sample.blinded_words_vec[ex_index])
                # logger.warn(word_tokens)
                # logger.warn(blinded_word_tokens)
                labels.append(label_map[label])
                debug_labels.append(label_map[debug_label])
                whitespaces.append(whitespace)
                positions.append(position)

            assert len(tokens) == len(blinded_tokens)
            question_end = ds_sample.question_end_vec[ex_index]
            tokens_question = tokens[:question_end]
            tokens_document = tokens[question_end:]
            blinded_tokens_question = blinded_tokens[:question_end]
            blinded_tokens_document = blinded_tokens[question_end:]
            labels_question = labels[:question_end]
            labels_document = labels[question_end:]
            debug_labels_question = debug_labels[:question_end]
            debug_labels_document = debug_labels[question_end:]
            whitespaces_question = whitespaces[:question_end]
            whitespaces_document = whitespaces[question_end:]
            positions_question = positions[:question_end]
            positions_document = positions[question_end:]
            assert len(tokens_document) == len(blinded_tokens_document)

            max_document_length = max_seq_length - len(tokens_question)
            assert max_document_length > 0
            feature_length = max_document_length if len(tokens_document) > max_document_length else len(tokens_document)
            current_token_document = tokens_document[:feature_length]
            current_blinded_token_document = blinded_tokens_document[:feature_length]
            current_label_document = labels_document[:feature_length]
            current_debug_label_document = debug_labels_document[:feature_length]
            current_whitespace_document = whitespaces_document[:feature_length]
            current_position_document = positions_document[:feature_length]
            assert len(tokens_question) == len(blinded_tokens_question)
            assert len(current_token_document) == len(current_blinded_token_document)
            # logger.warn(current_token_document)
            # logger.warn(current_blinded_token_document)

            current_tokens = tokens_question + current_token_document
            input_ids = model_helper.tokenizer.convert_tokens_to_ids(current_tokens)
            current_blinded_tokens = blinded_tokens_question + current_blinded_token_document
            blinded_input_ids = model_helper.tokenizer.convert_tokens_to_ids(current_blinded_tokens)
            # logger.warning("Tokenized lengths {} {}".format(input_ids, blinded_input_ids))
            # logger.warning("Tokenized lengths {} {}".format(current_tokens, current_blinded_tokens))
            assert len(current_tokens) == len(current_blinded_tokens)
            segment_ids = [sequence_a_segment_id] * ds_sample.question_end_vec[ex_index] + [sequence_b_segment_id] * feature_length
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
            label_ids = labels_question + current_label_document
            debug_label_ids = debug_labels_question + current_debug_label_document
            whitespace_bools = whitespaces_question + current_whitespace_document
            position_ids = positions_question + current_position_document

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if padding_length > 0:
                input_ids += ([pad_token] * padding_length)
                blinded_input_ids += ([pad_token] * padding_length)
                input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
                segment_ids += ([pad_token_segment_id] * padding_length)
                label_ids += ([pad_token_label_id] * padding_length)
                debug_label_ids += ([pad_token_label_id] * padding_length)
                whitespace_bools += ([pad_token_label_id] * padding_length)
                position_ids += ([[pad_token_label_id, pad_token_label_id]] * padding_length)

            assert len(input_ids) == max_seq_length
            assert len(blinded_input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length
            assert len(whitespace_bools) == max_seq_length
            assert len(position_ids) == max_seq_length

            sample_features.input_ids.append(input_ids)
            sample_features.input_ids_blinded.append(blinded_input_ids)
            sample_features.input_mask.append(input_mask)
            sample_features.segment_ids.append(segment_ids)
            sample_features.label_ids.append(label_ids)
            sample_features.debug_label_ids.append(debug_label_ids)
            sample_features.whitespace_bools.append(whitespace_bools)
            sample_features.position_ids.append(position_ids)
            sample_features.subjects.append(subject_label_encoder.transform(ds_sample.subjects_vec[ex_index]))
            sample_features.subject_lengths.append(ds_sample.subject_lengths_vec[ex_index])
            sample_features.pubmed_id.append(ds_sample.pubmed_id_vec[ex_index])
            sample_features.question_id.append(ds_sample.question_id_vec[ex_index])
            sample_features.question_type.append(ds_sample.question_type_vec[ex_index])

        # Pad each sample_features to retrieval size
        while len(sample_features.input_ids) < retrieval_size:
            sample_features.input_ids.append([pad_token] * max_seq_length)
            sample_features.input_ids_blinded.append([pad_token] * max_seq_length)
            sample_features.input_mask.append([0 if mask_padding_with_zero else 1] * max_seq_length)
            sample_features.segment_ids.append([pad_token_segment_id] * max_seq_length)
            sample_features.label_ids.append([pad_token_label_id] * max_seq_length)
            sample_features.debug_label_ids.append([pad_token_label_id] * max_seq_length)
            sample_features.whitespace_bools.append([pad_token_label_id] * max_seq_length)
            sample_features.position_ids.append([[pad_token_label_id, pad_token_label_id]] * max_seq_length)
            sample_features.subjects.append(subject_label_encoder.transform(ds_sample.subjects_vec[0]))
            sample_features.subject_lengths.append(-1)
            sample_features.pubmed_id.append(-1)
            sample_features.question_id.append(-1)
            sample_features.question_type.append(-1)
        features.append(sample_features)

    return features, subject_label_encoder
