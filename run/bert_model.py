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
"""PyTorch BERT model. """

import logging
import torch
import math

from torch import nn
from torch.cuda.amp import autocast
from transformers import BertModel, BertPreTrainedModel
from torchcrf import CRF

from data_processing.datatypes import LABELS

logger = logging.getLogger(__name__)


class BertModelForQuestionAnswering(BertPreTrainedModel):
    r"""
        BertModelforQuestionAnswerTagging
    """
    def __init__(self, config, crf_bool):
        super(BertModelForQuestionAnswering, self).__init__(config)
        self.num_labels = config.num_labels
        self.crf_bool = crf_bool
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.crf = CRF(self.num_labels, True)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        with autocast():

            # Filter outputs with attention mask [0 * seq_length]
            pad_sequences = torch.any(attention_mask, 1)
            pad_seq_id = (pad_sequences.long() == 1).sum()
            batch_size = labels[:pad_seq_id].size()[0]
            seq_length = labels[:pad_seq_id].size()[1]

            input_ids = input_ids[:pad_seq_id]
            attention_mask = attention_mask[:pad_seq_id]
            token_type_ids = token_type_ids[:pad_seq_id]
            labels = labels[:pad_seq_id]

            outputs = self.bert(input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids
                                )
            sequence_output = self.dropout(outputs[0])
            logits = self.classifier(sequence_output)
            if torch.isnan(logits).any():
                logger.info(outputs)
                logger.info(sequence_output)
                logger.info(logits)

            if self.crf_bool:
                sequence_logits = self.crf(logits, labels, mask=attention_mask.type(torch.uint8), reduction='none')
                crf_sequence = self.crf.decode(logits, mask=attention_mask.type(torch.uint8))
            else:
                crf_sequence = [[-1] * seq_length] * batch_size
                # logits shape (batch_size, seq_length, num_tags)
                # labels shape (batch_size, seq_length)
                seq_numbers = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).repeat(batch_size, 1)  # (batch_size, seq_length)
                batch_numbers = torch.arange(batch_size, device=input_ids.device).unsqueeze(1).repeat(1, seq_length)  # (batch_size, seq_length)
                # Indexing with multi-dimensional arrays
                # https://docs.scipy.org/doc/numpy-1.13.0/user/basics.indexing.html
                label_logits = logits[batch_numbers, seq_numbers, labels]  # (batch_size, seq_length)
                normalizer_logits = torch.logsumexp(logits, dim=2)  # (batch_size, seq_length)
                label_logits_normalized = label_logits - normalizer_logits  # (batch_size, seq_length)
                label_logits_normalized = torch.where(
                    attention_mask == 1, label_logits_normalized,
                    torch.zeros(label_logits_normalized.size()).type_as(label_logits_normalized))
                sequence_logits = torch.sum(label_logits_normalized, 1)  # (batch_size, )

            crf_sequence = [sequence + [-1] * (seq_length - len(sequence)) for sequence in crf_sequence]  # Pad crf_sequence
            crf_sequence = torch.tensor(crf_sequence, device=input_ids.device)

            outputs = (sequence_logits, sequence_logits, logits, crf_sequence)

            return outputs


class ModelForDistantSupervision(nn.Module):
    """ Model for distantly supervised sequence labeling. One batch only contains examples belonging to exactly one multi-instance bag.
    """
    def __init__(self, config, multi_instance_bool, multi_instance_bool_neg, crf_bool, model_name):
        super(ModelForDistantSupervision, self).__init__()
        self.multi_instance_bool = multi_instance_bool
        self.multi_instance_bool_neg = multi_instance_bool_neg
        self.bert = BertModelForQuestionAnswering.from_pretrained(model_name, config=config, crf_bool=crf_bool)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        with autocast():
            sequence_logits, active_sequence_logits, logits, crf_sequence = self.bert(
                input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
            pad_sequences = torch.any(attention_mask, 1)
            pad_seq_id = (pad_sequences.long() == 1).sum()
            labels = labels[:pad_seq_id]

            # Get mask of positive and negative examples
            # Warning: Assumption that labels come from ['O', 'X', 'B', 'I'] and 'B' and 'I' are positive labels
            assert LABELS == ['O', 'X', 'B', 'I']
            pos_sequence_labels = torch.where(labels <= 1, 0, labels)
            pos_sequence_labels = torch.where(labels > 1, 1, pos_sequence_labels)
            pos_sequence_mask = torch.any(pos_sequence_labels, dim=1)
            # Likelihood numbers are negative, times float("inf") mask results in lowest likelihoods possible
            # pos_sequence_log_mask = torch.where(pos_sequence_mask == 0, float("inf"), pos_sequence_mask.double())

            neg_sequence_mask = ~pos_sequence_mask
            # Likelihood numbers are negative, times float("inf") mask results in lowest likelihoods possible
            # neg_sequence_log_mask = torch.where(neg_sequence_mask == 0, float("inf"), neg_sequence_mask.double())

            if pos_sequence_mask.any():
                # multi instance learning with at-least-once assumption
                if self.multi_instance_bool:
                    # logger.info(pos_sequence_mask)
                    # logger.info(sequence_logits)
                    # logger.info(sequence_logits[pos_sequence_mask])
                    # logger.info(torch.logsumexp(sequence_logits[pos_sequence_mask], 0))
                    positive_loss = - 1 * torch.logsumexp(sequence_logits[pos_sequence_mask], 0).unsqueeze(0)  # ( , )
                # normal distant supervision with noisy labeling assumption
                else:
                    positive_loss = torch.logsumexp(- 1 * sequence_logits[pos_sequence_mask], 0).unsqueeze(0)  # ( , )

            if neg_sequence_mask.any():
                # multi instance learning with at-least-once assumption
                if self.multi_instance_bool_neg:
                    negative_loss = - 1 * torch.logsumexp(sequence_logits[neg_sequence_mask], 0).unsqueeze(0)  # ( , )
                # normal distant supervision with noisy labeling assumption
                else:
                    negative_loss = torch.logsumexp(- 1 * sequence_logits[neg_sequence_mask], 0).unsqueeze(0)  # ( , )

            if pos_sequence_mask.any() and neg_sequence_mask.any():
                loss = positive_loss + negative_loss
            elif pos_sequence_mask.any():
                loss = positive_loss
            elif neg_sequence_mask.any():
                loss = negative_loss
            else:
                raise RuntimeError("Neither positive nor negative examples available")
            if math.isnan(loss.item()):
                logger.warn("Stats")
                logger.warn(input_ids)
                logger.warn(labels)
                logger.warn(attention_mask)
                logger.warn(token_type_ids)
                # logger.warn(labels[0])
                # logger.warn(labels[24])
                logger.warn(sequence_logits)
                logger.warn(pos_sequence_mask)
                if pos_sequence_mask.any():
                    logger.warn(positive_loss)
                logger.warn(neg_sequence_mask)
                if neg_sequence_mask.any():
                    logger.warn(negative_loss)
                logger.warn(loss)

            outputs = (loss, active_sequence_logits, logits, crf_sequence)

            return outputs


if __name__ == "__main__":
    # input_ids = torch.tensor([[12, 1, 3, 5, 6, 9],
    #                           [1, 3, 6, 9, 8, 0]])
    # attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1],
    #                                [1, 1, 1, 1, 1, 0]], dtype=torch.uint8)
    # token_type_ids = torch.tensor([[0, 0, 0, 0, 0, 0],
    #                                [0, 0, 0, 0, 0, 0]])
    # labels = torch.tensor([[0, 0, 0, 0, 0, 1],
    #                        [0, 0, 0, 1, 0, 0]])
    # label = torch.tensor([1], dtype=torch.float)

    # model = BertForQuestionAnswerTagging.from_pretrained('bert-base-uncased')
    # outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels, label=label)
    # print(outputs)

    # CRF test case 1, Marginal of length 1 at the start of the sequence
    num_tags = 3
    model = CRF(num_tags)
    seq_length = 2  # maximum sequence length in a batch
    batch_size = 1  # number of samples in the batch
    emissions = torch.randn(seq_length, batch_size, num_tags)
    attention_mask = torch.tensor([[1], [1]], dtype=torch.bool)

    out_seqs = model.decode(emissions, attention_mask)
    print('Out seqs: ', out_seqs)

    tags = torch.tensor([[2], [0]], dtype=torch.long)  # (seq_length, batch_size)
    seq_prob_1 = torch.exp(model(emissions, tags, reduction="none"))
    print("Full sequence probability: ", seq_prob_1)

    tags = torch.tensor([[2], [1]], dtype=torch.long)  # (seq_length, batch_size)
    seq_prob_2 = torch.exp(model(emissions, tags, reduction="none"))
    print("Full sequence probability: ", seq_prob_2)

    tags = torch.tensor([[2], [2]], dtype=torch.long)  # (seq_length, batch_size)
    seq_prob_3 = torch.exp(model(emissions, tags, reduction="none"))
    print("Full sequence probability: ", seq_prob_3)

    mask = torch.tensor([[1], [0]], dtype=torch.bool)
    tags = torch.tensor([[2], [0]], dtype=torch.long)  # (seq_length, batch_size)
    prob = model.conditional_probability(emissions, tags, mask)
    print("Marginal prob full sequence: ", prob)

    mg_prob = model.compute_marginal_probabilities(emissions, attention_mask).view(batch_size, seq_length, -1)
    mg_p = torch.nn.functional.softmax(mg_prob, dim=-1)
    print("MG prob: ", mg_prob)

    assert torch.allclose(seq_prob_1 + seq_prob_2 + seq_prob_3, prob)
    assert torch.allclose(prob.data, mg_prob[0, 0, 2])

    # CRF test case 2, Marginal of length 1 in the middle of the sequence
    num_tags = 2
    model = CRF(num_tags)
    seq_length = 3  # maximum sequence length in a batch
    batch_size = 1  # number of samples in the batch
    emissions = torch.randn(seq_length, batch_size, num_tags)
    attention_mask = torch.tensor([[1], [1], [1]], dtype=torch.bool)

    out_seqs = model.decode(emissions, attention_mask)
    print('Out seqs: ', out_seqs)

    tags = torch.tensor([[0], [0], [0]], dtype=torch.long)  # (seq_length, batch_size)
    seq_prob_1 = torch.exp(model(emissions, tags, reduction="none"))
    print("Full sequence probability: ", seq_prob_1)

    tags = torch.tensor([[0], [0], [1]], dtype=torch.long)  # (seq_length, batch_size)
    seq_prob_2 = torch.exp(model(emissions, tags, reduction="none"))
    print("Full sequence probability: ", seq_prob_2)

    tags = torch.tensor([[1], [0], [0]], dtype=torch.long)  # (seq_length, batch_size)
    seq_prob_3 = torch.exp(model(emissions, tags, reduction="none"))
    print("Full sequence probability: ", seq_prob_3)

    tags = torch.tensor([[1], [0], [1]], dtype=torch.long)  # (seq_length, batch_size)
    seq_prob_4 = torch.exp(model(emissions, tags, reduction="none"))
    print("Full sequence probability: ", seq_prob_4)

    mask = torch.tensor([[0], [1], [0]], dtype=torch.bool)
    tags = torch.tensor([[0], [0], [0]], dtype=torch.long)  # (seq_length, batch_size)
    prob = model.conditional_probability(emissions, tags, mask)
    print("Marginal prob full sequence: ", prob)

    mg_prob = model.compute_marginal_probabilities(emissions, attention_mask).view(batch_size, seq_length, -1)
    mg_p = torch.nn.functional.softmax(mg_prob, dim=-1)
    print("MG prob: ", mg_prob)

    assert torch.allclose(seq_prob_1 + seq_prob_2 + seq_prob_3 + seq_prob_4, prob)
    assert torch.allclose(prob.data, mg_prob[0, 1, 0])

    # CRF test case 3, Marginal of length 2 in the middle of the sequence
    num_tags = 2
    model = CRF(num_tags)
    seq_length = 4  # maximum sequence length in a batch
    batch_size = 1  # number of samples in the batch
    emissions = torch.randn(seq_length, batch_size, num_tags)
    attention_mask = torch.tensor([[1], [1], [1], [1]], dtype=torch.bool)

    out_seqs = model.decode(emissions, attention_mask)
    print('Out seqs: ', out_seqs)

    tags = torch.tensor([[0], [0], [0], [0]], dtype=torch.long)  # (seq_length, batch_size)
    seq_prob_1 = torch.exp(model(emissions, tags, reduction="none"))
    print("Full sequence probability: ", seq_prob_1)

    tags = torch.tensor([[0], [0], [0], [1]], dtype=torch.long)  # (seq_length, batch_size)
    seq_prob_2 = torch.exp(model(emissions, tags, reduction="none"))
    print("Full sequence probability: ", seq_prob_2)

    tags = torch.tensor([[1], [0], [0], [0]], dtype=torch.long)  # (seq_length, batch_size)
    seq_prob_3 = torch.exp(model(emissions, tags, reduction="none"))
    print("Full sequence probability: ", seq_prob_3)

    tags = torch.tensor([[1], [0], [0], [1]], dtype=torch.long)  # (seq_length, batch_size)
    seq_prob_4 = torch.exp(model(emissions, tags, reduction="none"))
    print("Full sequence probability: ", seq_prob_4)

    mask = torch.tensor([[0], [1], [1], [0]], dtype=torch.bool)
    tags = torch.tensor([[0], [0], [0], [0]], dtype=torch.long)  # (seq_length, batch_size)
    prob = model.conditional_probability(emissions, tags, mask)
    print("Marginal prob full sequence: ", prob)

    assert torch.allclose(seq_prob_1 + seq_prob_2 + seq_prob_3 + seq_prob_4, prob)
