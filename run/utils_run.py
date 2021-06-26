""" Utility functions for the neural network and tensor manipulation of run_mtqa.py """

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
import numpy as np
import scipy
import torch

from sqlalchemy import create_engine
from tqdm import trange
# from colorama import Style
from typing import List

from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
from torch._six import container_abcs
from transformers import get_linear_schedule_with_warmup

# from data_processing.datatypes import LABELS
from metrics.sequence_labeling import get_entities_with_names
from data_processing.nn_output_to_indra import get_db_xrefs, make_indra_statements
from data_processing.datatypes import PAD_TOKEN_LABEL_ID
from configs import PUBMED_EVIDENCE_ANNOTATIONS_DB

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def update_with_nn_output(inputs, output_seq, logits, preds, all_preds, out_label_ids, out_token_ids,
                          attention_masks, crf_bool):

    if out_label_ids is None:
        all_preds = logits.detach().cpu().numpy()
        if crf_bool:
            preds = output_seq.detach().cpu().numpy()
        else:
            preds = np.argmax(all_preds, axis=2)
        out_label_ids = inputs["labels"].detach().cpu().numpy()
        out_token_ids = inputs["input_ids"].detach().cpu().numpy()
        attention_masks = inputs["attention_mask"].detach().cpu().numpy()
    else:
        if crf_bool:
            all_preds = np.append(all_preds, logits.detach().cpu().numpy(), axis=0)
            preds = np.append(preds, output_seq.detach().cpu().numpy(), axis=0)
        else:
            logits_numpy = logits.detach().cpu().numpy()
            all_preds = np.append(all_preds, logits_numpy, axis=0)
            preds = np.append(preds, np.argmax(logits_numpy, axis=2), axis=0)
        out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
        out_token_ids = np.append(out_token_ids, inputs["input_ids"].detach().cpu().numpy(), axis=0)
        attention_masks = np.append(attention_masks, inputs["attention_mask"].detach().cpu().numpy(), axis=0)
    return preds.tolist(), all_preds, out_label_ids, out_token_ids, attention_masks


def update_metadata(input_pubmed_ids, input_subjects, input_whitespaces, input_positions, input_question_ids, input_subject_lengths, input_question_types,
                    input_debug_label_ids, input_blinded_token_ids,
                    pubmed_ids, subjects, whitespace_bools, position_ids, question_ids, subject_lengths, question_types, debug_label_ids, blinded_token_ids,
                    subject_label_encoder):
    if pubmed_ids is None:
        pubmed_ids = input_pubmed_ids.numpy()
        subject_lengths = input_subject_lengths.numpy()
        subjects = subject_label_encoder.inverse_transform(input_subjects.numpy().ravel())
        whitespace_bools = input_whitespaces.numpy()
        position_ids = input_positions.numpy()
        question_ids = input_question_ids.numpy()
        question_types = input_question_types.numpy()
        debug_label_ids = input_debug_label_ids.numpy()
        blinded_token_ids = input_blinded_token_ids.numpy()
    else:
        pubmed_ids = np.append(pubmed_ids, input_pubmed_ids.numpy(), axis=0)
        subject_lengths = np.append(subject_lengths, input_subject_lengths.numpy(), axis=0)
        subjects = np.append(subjects, subject_label_encoder.inverse_transform(input_subjects.numpy().ravel()), axis=0)
        whitespace_bools = np.append(whitespace_bools, input_whitespaces.numpy(), axis=0)
        position_ids = np.append(position_ids, input_positions.numpy(), axis=0)
        question_ids = np.append(question_ids, input_question_ids.numpy(), axis=0)
        question_types = np.append(question_types, input_question_types.numpy(), axis=0)
        debug_label_ids = np.append(debug_label_ids, input_debug_label_ids.numpy(), axis=0)
        blinded_token_ids = np.append(blinded_token_ids, input_blinded_token_ids.numpy(), axis=0)

    return pubmed_ids, subjects, whitespace_bools, position_ids, question_ids, subject_lengths, question_types, debug_label_ids, blinded_token_ids


def get_answer_probs(answer_list, db_refs, logits, attention_mask, model, args, answer_start_pos, groundtruth_bool=False):
    new_answer_list = []
    for i, answer in enumerate(answer_list):
        if groundtruth_bool is False:
            if answer[0].startswith("##"):  # Answers beginning in the middle of a token are ignored
                continue
            answer_tokens = answer[6]
            answer_labels = []
            probs = []
            start = answer_start_pos + answer[2]
            # end = answer_start_pos + answer[3] + 1
            # logger.info(answer)
            # logger.info(start)
            for j, token in enumerate(answer_tokens):
                if j == 0:
                    answer_labels.append('B')
                    probs.append(scipy.special.softmax(logits[start + j])[2])
                    # probs.append(logits[start + j][2])
                elif token.startswith("##"):
                    answer_labels.append('X')
                else:
                    answer_labels.append('I')
                    probs.append(scipy.special.softmax(logits[start + j])[3])
                    # probs.append(logits[start + j][3])
            # answer_label_ids = [LABELS.index(label) for label in answer_labels]
            # tags = torch.zeros(attention_mask.shape[0], 1)  # (seq_length, batch_size: 1)
            # if answer[3] - answer[2] != len(answer_label_ids) - 1:
            #     print(answer)
            #     print(len(answer_label_ids))
            # assert answer[3] - answer[2] == len(answer_label_ids) - 1
            # tags[start:end] = torch.tensor(answer_label_ids, dtype=torch.long, device=args.device).unsqueeze(1)
            # mask = torch.tensor(attention_mask, device=args.device).unsqueeze(1) - 1
            # mask[start:end, 0] = 1

            # emissions = torch.tensor(logits, device=args.device).unsqueeze(1).transpose(0, 1)
            # tags = tags.long().transpose(0, 1)
            # mask = mask.long().transpose(0, 1)

            # Calculate marginal probability of answer in CRF
            # with torch.no_grad():
            #     if (type(model) == torch.nn.DataParallel or type(model) == torch.nn.parallel.DistributedDataParallel) and args.local_rank in [-1, 0]:
            #         prob = model.module.crf.conditional_probability(emissions, tags, mask).cpu().item()
            #     else:
            #         prob = model.crf.conditional_probability(emissions, tags, mask).cpu().item()
            prob = np.mean(probs)
            # if prob < 0.1:
            #     print(probs)
            #     print(prob)
        else:
            prob = 1.0

        if db_refs is not None:
            for db_ref in db_refs[i]:
                answer_values = tuple(list(answer) + [db_ref] + [prob])
                new_answer_list.append(answer_values)
        else:  # Used for debugging and plotting main answer probs in wandb histogram
            answer_values = [prob]
            new_answer_list.append(answer_values)

    return new_answer_list


def highlight_text(tokens: List[str], answer_list: List[tuple]) -> str:
    # pubmed_text = Style.DIM
    pubmed_text = ""
    current_start = 0
    current_end = 0
    for word, _, chunk_start, chunk_end, _, _, _, db_ref, confidence in answer_list:
        current_start = current_end
        current_end = chunk_start
        pubmed_text += " ".join(tokens[current_start:current_end]).replace(" ##", "")
        current_start = current_end
        current_end = chunk_end + 1
        # answer_text = Style.NORMAL + Style.BRIGHT + " " + " ".join(tokens[current_start:current_end]).replace(" ##", "") \
        #     + " (" + db_ref[0] + " " + db_ref[1] + ", {:.4f}) ".format(confidence) + Style.NORMAL + Style.DIM
        answer_text = "====" + " " + " ".join(tokens[current_start:current_end]).replace(" ##", "") \
            + " (" + db_ref[0] + " " + db_ref[1] + ", {:.4f}) ".format(confidence) + "===="
        pubmed_text += answer_text
    current_start = current_end
    current_end = len(tokens)
    # pubmed_text += " ".join(tokens[current_start:current_end]).replace(" ##", "") + Style.NORMAL
    pubmed_text += " ".join(tokens[current_start:current_end]).replace(" ##", "")
    return pubmed_text


def extract_from_nn_output(labels, out_label_ids, preds, all_preds, out_token_ids, attention_masks, whitespace_bools, position_ids,
                           tokenizer, pubmed_ids, subjects, question_ids, subject_lengths, question_types, out_debug_label_ids, model, args):

    label_map = {i: label for i, label in enumerate(labels)}
    amount_of_questions = len(question_ids)
    question_ids = [i for i in range(len(question_ids))]
    logger.debug("Amount of questions: {}".format(amount_of_questions))
    out_label_dict = [[] for i in range(amount_of_questions)]
    out_debug_label_dict = [[] for i in range(amount_of_questions)]
    preds_dict = [[] for i in range(amount_of_questions)]
    token_dict = [[] for i in range(amount_of_questions)]
    whitespace_dict = [[] for i in range(amount_of_questions)]
    position_dict = [[] for i in range(amount_of_questions)]
    answer_start_dict = [-1 for i in range(amount_of_questions)]
    pubmed_list = ["" for i in range(amount_of_questions)]
    subject_list = [[] for i in range(amount_of_questions)]

    # Debug variables
    logger.debug("Subjects: {}".format(subjects))
    logger.debug("Subjects shape: {}".format(subjects.shape))
    logger.debug("Subjects len: {}".format(subject_lengths))
    logger.debug("Subjects len shape: {}".format(subject_lengths.shape))
    logger.debug("Out label shape: {}".format(out_label_ids.shape))
    logger.debug("Pubmed IDs shape: {}".format(pubmed_ids.shape))
    logger.debug("Attention masks shape: {}".format(attention_masks.shape))
    logger.debug("All logits shape: {}".format(all_preds.shape))
    logger.debug(type(all_preds))
    logger.debug(type(all_preds[0]))

    groundtruth = {}
    predictions = {}
    debug_scores = []

    subjects = subjects.reshape((out_label_ids.shape[0], -1))
    for i in range(out_label_ids.shape[0]):
        if question_ids[i] == -1:
            # Padding sequence
            continue
        pubmed_list[question_ids[i]] = pubmed_ids[i]
        subject_list[question_ids[i]] = subjects[i, :subject_lengths[i]]
        answer_started = False
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != PAD_TOKEN_LABEL_ID:
                if not answer_started:
                    answer_started = True
                    answer_start_dict[question_ids[i]] = j
                out_label_dict[question_ids[i]].append(label_map[out_label_ids[i][j]])
                out_debug_label_dict[question_ids[i]].append(label_map[out_debug_label_ids[i][j]])
                preds_dict[question_ids[i]].append(label_map[preds[i][j]])
                token_dict[question_ids[i]].append(tokenizer.convert_ids_to_tokens(out_token_ids[i][j].item()))
                whitespace_dict[question_ids[i]].append(whitespace_bools[i][j])
                position_dict[question_ids[i]].append(position_ids[i][j])

    # Python dicts retain order automatically since version 3.6
    # out_label_list = list(out_label_dict.values())
    # preds_list = list(preds_dict.values())

    pubmed_engine = create_engine(PUBMED_EVIDENCE_ANNOTATIONS_DB)
    for i in trange(len(out_label_dict), desc="Extract Entities"):
        if len(out_label_dict[i]) == 0:
            # Padding sequence
            continue
        groundtruth_list = get_entities_with_names(token_dict[i], out_label_dict[i], whitespace_dict[i], position_dict[i], question_types[i])
        answer_list = get_entities_with_names(token_dict[i], preds_dict[i], whitespace_dict[i], position_dict[i], question_types[i])

        # Add DB xrefs to groundtruth_list and answer_list if possible
        groundtruth_db_list = get_db_xrefs(groundtruth_list, pubmed_list[i], pubmed_engine, use_simple_normalizer=args.use_simple_normalizer)
        groundtruth_probs_list = get_answer_probs(groundtruth_list, groundtruth_db_list, all_preds[i], attention_masks[i], model, args, answer_start_dict[i])
        db_list = get_db_xrefs(answer_list, pubmed_list[i], pubmed_engine, use_simple_normalizer=args.use_simple_normalizer)
        answer_probs_list = get_answer_probs(answer_list, db_list, all_preds[i], attention_masks[i], model, args, answer_start_dict[i])

        # Extract highlighted Pubmed text
        pubmed_tokens = token_dict[i]
        groundtruth_text_highlighted = highlight_text(pubmed_tokens, groundtruth_probs_list)
        prediction_text_highlighted = highlight_text(pubmed_tokens, answer_probs_list)

        # debug_probs: Main answer probabilities to be later plotted in wandb histogram
        debug_list = get_entities_with_names(token_dict[i], out_label_dict[i], whitespace_dict[i], position_dict[i], question_types[i])
        debug_probs = get_answer_probs(debug_list, None, all_preds[i], attention_masks[i], model, args, answer_start_dict[i])
        if len(debug_probs) > 0:
            debug_scores.append(debug_probs)

        if pubmed_list[i] in groundtruth:
            groundtruth[pubmed_list[i]].append((subject_list[i], groundtruth_probs_list, groundtruth_text_highlighted))
        else:
            groundtruth[pubmed_list[i]] = [(subject_list[i], groundtruth_probs_list, groundtruth_text_highlighted)]
        if pubmed_list[i] in predictions:
            predictions[pubmed_list[i]].append((subject_list[i], answer_probs_list, prediction_text_highlighted))
        else:
            predictions[pubmed_list[i]] = [(subject_list[i], answer_probs_list, prediction_text_highlighted)]

    # logger.warn(groundtruth.keys())
    # debug_info = {}
    # for pmid, value in groundtruth.items():
    #     for infos in value:
    #         debug_info.setdefault(tuple(infos[0]), [])
    #         if len(infos[1]) > 0:
    #             debug_info[tuple(infos[0])].append((pmid, infos[1]))
    # debug_info_non_empty = [k for k, v in debug_info if len(v) > 0]
    # # logger.warn(debug_info)
    # # logger.warn(debug_info.keys())
    # logger.warn(len(debug_info_non_empty))
    # logger.warn(len(debug_info.keys()))
    # exit()

    # Make Indra Statements from the answers
    indra_preds = make_indra_statements(predictions)

    return groundtruth, predictions, out_label_dict, preds_dict, indra_preds, debug_scores


def initialize_loaders(args, dataset, optimizer, n_gpu):
    sampler = RandomSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=1, collate_fn=collate_fn, pin_memory=True)
    total = len(dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    warmup_steps = int(args.warmup_proportion * total)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total)
    return dataloader, scheduler


def collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        # Pad "subjects" tensor to tensor with largest number of subjects
        max_dimension = max([element.size(-1) for element in batch])
        # a_1 = batch[0].size()
        # a_2 = batch[1].size()
        # print(batch)
        # print(a_1)
        # print(a_2)
        batch = [torch.nn.functional.pad(element, [0, max_dimension - element.size(-1)]) for element in batch]
        # if a_1 != a_2:
        #     b_1 = batch[0].size()
        #     b_2 = batch[1].size()
        #     print(len(batch))
        #     print(a_1)
        #     print(a_2)
        #     print(b_1)
        #     print(b_2)
        #     exit()
        #     print(batch[0])
        #     print(batch)
        return torch.cat(batch, 0, out=out)
    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        return [collate_fn(samples) for samples in transposed]
    elif isinstance(elem, container_abcs.Mapping):
        return {key: collate_fn([d[key] for d in batch]) for key in elem}

    raise TypeError("collate_fn: batch must contain tensors or lists; found {}".format(elem_type))
