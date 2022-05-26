""" Converts the .a* files from the BioNLP challenges to our custom .iob format """

import os
import sys

if os.name == 'nt':
    root_path = "/".join(os.path.realpath(__file__).split("\\")[:-2])
else:
    root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.append(root_path)

import logging
import bisect
import numpy as np
import itertools

from data_processing.file_converter import match_current_answers, match_new_answers, end_matching, get_word_tokenization_position, apply_entity_blinding
from metrics.sequence_labeling import getEntities

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)


def tag_all_entities(context, answers, model_name_or_path):
    ''' Return all entities in the sequence tagged and the character positions of all sentences. '''

    matching = False
    matched_strings = {}
    tagged_answer = []
    sentenceBoundaries = []

    for w, word in enumerate(context):
        whitespace = False  # no whitespace before the token
        if (w >= 1) and word[1] > context[w - 1][2] + 1:
            whitespace = True
        word_continutation = False  # words does not start with two ##

        if word[0] == "." or word[0] == ";":
            sentenceBoundaries.append(word[1])

        non_continuation_words = ("_", "-", "/", "(", ")", "[", "]", ",", ".", ";", ":")
        if (word[0].startswith("##") and "berta" not in model_name_or_path) or ("berta" in model_name_or_path and not word[0].startswith("Ġ")
           and not word[0].endswith(non_continuation_words) and not context[w - 1][0].endswith(non_continuation_words)):
            word_continutation = True

        if not matching:
            if word_continutation:
                tagged_answer.append((word[0], 'X', False, word[1], word[2]))
            else:
                matching = match_new_answers(answers, matched_strings, word, matching, whitespace)
                end_matching(matched_strings, word, matching, whitespace, tagged_answer)
        else:
            if "berta" not in model_name_or_path:
                if word_continutation:
                    matching_string = word[0][2:]
                else:
                    matching_string = word[0]
            else:  # Roberta uses "Ġ" as space-encoding character before words instead of "##" before each subword
                matching_string = word[0]
            # Check if previous matching continues matching
            further_matching = False
            for answer, answer_prefix in matched_strings.items():
                if whitespace:
                    further_matching_string = answer_prefix[0] + " " + matching_string
                else:
                    further_matching_string = answer_prefix[0] + matching_string
                # Handling of Roberta Tokenizer
                if len(further_matching_string) > 0 and further_matching_string[0].startswith("Ġ"):
                    further_matching_string = further_matching_string[1:]
                further_matching_string = further_matching_string.replace("Ġ", " ")
                # Handling of Roberta Tokenizer (END)
                if answer.lower().startswith(further_matching_string.lower()) and answer != "":
                    further_matching = True
                # logger.info("Further matching string: {} {}".format(further_matching_string, further_matching))

            if (not further_matching) or (w == len(context) - 1):
                # Check if previous matching constitutes a complete answer
                complete_answers = []
                for answer, answer_prefix in matched_strings.items():
                    answer_item = answer_prefix[0]
                    if answer_item.startswith("Ġ"):
                        answer_item = answer_item[1:]
                    if answer.lower() == answer_item.lower() and answer != "":
                        # logger.info(answer_prefix)
                        complete_answers.append(answer_prefix)
                if len(complete_answers) == 0:
                    complete_answers.append(matched_strings[""])
                # Get longest matching answer and add its tags to our output
                matching_answer_tag = sorted(complete_answers, key=lambda x: len(x[0]), reverse=True)[0][1]

                # Append the matching answer
                tagged_answer.extend(matching_answer_tag)
                matched_strings = {}
                matching = False

                if word_continutation:
                    tagged_answer.append((word[0], 'X', False, word[1], word[2]))
                else:
                    matching = match_new_answers(answers, matched_strings, word, matching, whitespace)
                    end_matching(matched_strings, word, matching, whitespace, tagged_answer)

            else:
                match_current_answers(answers, matched_strings, word, matching, whitespace, word_continutation, matching_string)
                if not word_continutation:
                    match_new_answers(answers, matched_strings, word, matching, whitespace)
                end_matching(matched_strings, word, matching, whitespace, tagged_answer)

    return tagged_answer, sentenceBoundaries


def roberta_entity(entity, model_name_or_path):
    if "berta" in model_name_or_path:
        return (entity.replace(' ', 'Ġ')).lower()
    else:
        return entity


def insert_tmp_entity_index(entity, tmp_entity_index, answer_type, answer_number, model_name_or_path):
    if "berta" in model_name_or_path:
        tmp_entity_index[('Ġ' + entity.replace(' ', 'Ġ')).lower()] = (answer_type, answer_number)
        tmp_entity_index[(entity.replace(' ', 'Ġ')).lower()] = (answer_type, answer_number)
    else:
        tmp_entity_index[entity] = (answer_type, answer_number)


def identify_entities(tagged_answer, main_answers, other_answers, question_entities, model):
    ''' Returns list of tuple for the main answers ("AB", 1, 5), list of list of other answers and list of list question entities. '''

    entities = getEntities(tagged_answer)

    main_answer_pos = []
    other_answers_pos = [[] for i in range(len(other_answers))]
    question_entities_pos = [[] for i in range(len(question_entities))]

    tmp_entity_index = {}
    for entity in main_answers:
        insert_tmp_entity_index(entity, tmp_entity_index, "main_answer", 0, model)
    for i, lists in enumerate(other_answers):
        for entity in lists:
            if roberta_entity(entity, model) not in tmp_entity_index:
                insert_tmp_entity_index(entity, tmp_entity_index, "other_answer", i, model)
    for i, lists in enumerate(question_entities):
        for entity in lists:
            if roberta_entity(entity, model) not in tmp_entity_index:
                insert_tmp_entity_index(entity, tmp_entity_index, "question_answer", i, model)

    for entity in entities:
        if roberta_entity(entity[0], model) not in tmp_entity_index:
            continue
        answer_type = tmp_entity_index[roberta_entity(entity[0], model)][0]
        answer_index = tmp_entity_index[roberta_entity(entity[0], model)][1]
        if answer_type == "main_answer":
            main_answer_pos.append(entity)
        elif answer_type == "other_answer":
            other_answers_pos[answer_index].append(entity)
        elif answer_type == "question_answer":
            question_entities_pos[answer_index].append(entity)

    return main_answer_pos, other_answers_pos, question_entities_pos


def get_relevant_answers(main_answer_pos, other_answers_pos, question_ans_pos, sentence_boundaries, tagging_mode):
    ''' Return relevant answer belonging to question entities.
        All question entities must be in two sentences next to each other.
        Answer entities are determined by the nearest one in the same two sentences,
        or up to a further sentence for complexes/multi-turn questions.
    '''

    main_answers = sorted(main_answer_pos, key=lambda x: x[1])

    # TODO: Make maximum length of allowed nearby answer sentences a hyperparameters (current: 2)
    answers = [main_answer_pos] + other_answers_pos
    answers = [sorted(synonyms, key=lambda x: x[1]) for synonyms in answers]

    if tagging_mode == "simple":
        answers_with_corresponding_question_entities = [synonym for answer_ent in answers for synonym in answer_ent]
        answers_with_corresponding_question_entities = sorted(answers_with_corresponding_question_entities, key=lambda x: x[1])
        return answers_with_corresponding_question_entities, main_answers

    # For each sentence and each unique question entity, keep track where question entities are located
    sentence_index = [[[] for j in range(len(question_ans_pos))] for i in range(len(sentence_boundaries) + 1)]
    simple_index = [[False for j in range(len(question_ans_pos))] for i in range(len(sentence_boundaries) + 1)]
    for i, question_ent in enumerate(question_ans_pos):
        for answer_synonym in question_ent:
            pos = bisect.bisect_left(sentence_boundaries, answer_synonym[1])
            bisect.insort_left(sentence_index[pos][i], answer_synonym[1])
            simple_index[pos][i] = True
    simple_index = np.array(simple_index)

    logger.info("Question entity in sentence table:")
    logger.info(simple_index)
    logger.info("All answers:")
    logger.info(answers)

    answers_with_corresponding_question_entities = []
    for answer_ent in answers:
        current_sentence_bools = [False for i in range(len(sentence_boundaries) + 1)]
        # Always prioritize marking answers in same sentence as question triggers
        for synonym in answer_ent:
            pos = bisect.bisect_left(sentence_boundaries, synonym[1])
            bool_current = simple_index[pos:pos + 1, :].all(axis=1)[0]
            if bool_current:
                answers_with_corresponding_question_entities.append(synonym)
                current_sentence_bools[pos] = True
        # Mark answers over two sentences if no better answers found
        for synonym in answer_ent:
            pos = bisect.bisect_left(sentence_boundaries, synonym[1])
            bool_current = simple_index[pos:pos + 1, :].all(axis=1)[0]
            bool_current_next = simple_index[pos:pos + 2, :].any(axis=0).all()
            bool_last_current = simple_index[pos - 1:pos + 1, :].any(axis=0).all()
            logger.debug(synonym)
            logger.debug(pos)
            logger.debug(bool_current)
            logger.debug(bool_current_next)
            logger.debug(bool_last_current)
            if (not bool_current) and ((pos > 0 and not current_sentence_bools[pos - 1] and bool_last_current)
               or (pos < len(sentence_boundaries) and not current_sentence_bools[pos + 1] and bool_current_next)):
                answers_with_corresponding_question_entities.append(synonym)

    answers_with_corresponding_question_entities = sorted(answers_with_corresponding_question_entities, key=lambda x: x[1])

    return answers_with_corresponding_question_entities, main_answers


def tag_question_with_given_answers(context, context_blinded, answers, main_answers, question_length, max_seq_length,
                                    limit_max_length, model_name_or_path):
    ''' Tag question with given answer positions. Tagging all main answers is used for debugging purposes later.  '''

    # Tagged Answer Tuple (token, answer_tag, ws_bool, pos_start, pos_end, debug_tag, token_entity_blinded)

    tagged_answer = []
    at_least_one_answer = False
    sentence_pos = [0]
    # Variables for relevant answers
    t = 0  # Current Event/Answer Index
    state = 0  # 0 for 'O' and 'I', 1 for 'B'
    # Variables for debug answers (later used in wandb histogram)
    debug_t = 0  # Current Event/Answer Index
    debug_state = 0  # 0 for 'O' and 'I', 1 for 'B'
    first_answer_index_w = 0
    # try:
    #     assert len(context) == len(context_blinded)
    # except AssertionError:
    #     logger.warn(len(context))
    #     logger.warn(len(context_blinded))
    #     logger.warn(answers)
    #     logger.warn(main_answers)
    #     logger.warn(question_length)
    #     logger.warn(context)
    #     logger.warn(context_blinded)

    for w, word in enumerate(context):
        whitespace = False  # no whitespace before the token
        if (w >= 1) and word[1] > context[w - 1][2] + 1:
            whitespace = True

        if word[0] == ".":
            sentence_pos.append(w + 1)

        answer_tag = ''
        debug_tag = ''

        # Get answer tags
        non_continuation_words = ("_", "-", "/", "(", ")", "[", "]", ",", ".", ";", ":")
        if (word[0].startswith("##") and "berta" not in model_name_or_path) or ("berta" in model_name_or_path and not word[0].startswith("Ġ")
           and not word[0].endswith(non_continuation_words) and not context[w - 1][0].endswith(non_continuation_words)):
            answer_tag = 'X'
        elif len(answers) == 0:
            # no events for the protein, all tokens 'O'
            answer_tag = 'O'
        elif (word[1] >= int(answers[t][1])) and (word[2] < int(answers[t][2])) and (state == 1):
            answer_tag = 'I'
        elif (word[1] >= int(answers[t][1])) and (word[2] < int(answers[t][2])):
            at_least_one_answer = True
            answer_tag = 'B'
            state = 1
            if first_answer_index_w == -1:
                first_answer_index_w = w
        else:
            answer_tag = 'O'
            state = 0

        if len(answers) != 0 and (int(answers[t][2]) <= word[2] + 1) and (t < len(answers) - 1):
            t += 1
            state = 0

        # Get debug tags
        non_continuation_words = ("_", "-", "/", "(", ")", "[", "]", ",", ".", ";", ":")
        if (word[0].startswith("##") and "berta" not in model_name_or_path) or ("berta" in model_name_or_path and not word[0].startswith("Ġ")
           and not word[0].endswith(non_continuation_words) and not context[w - 1][0].endswith(non_continuation_words)):
            debug_tag = 'X'
        elif len(main_answers) == 0:
            # no events for the protein, all tokens 'O'
            debug_tag = 'O'
        elif (word[1] >= int(main_answers[debug_t][1])) and (word[2] < int(main_answers[debug_t][2])) and (debug_state == 1):
            debug_tag = 'I'
        elif (word[1] >= int(main_answers[debug_t][1])) and (word[2] < int(main_answers[debug_t][2])):
            debug_tag = 'B'
            debug_state = 1
        else:
            debug_tag = 'O'
            debug_state = 0

        if len(main_answers) != 0 and (int(main_answers[debug_t][2]) <= word[2] + 1) and (debug_t < len(main_answers) - 1):
            debug_t += 1
            debug_state = 0

        tagged_answer.append((word[0], answer_tag, whitespace, word[1], word[2], debug_tag, context_blinded[w][0]))

    # We need to truncate the sequence
    truncated = False
    if limit_max_length and (question_length + len(tagged_answer) + 1 > max_seq_length):
        truncated = True
        available_answer_len = max_seq_length - question_length - 1
        left_most_sentence_index = bisect.bisect_left(sentence_pos, first_answer_index_w - int(available_answer_len / 4))
        # print(current_sentence_pos)
        # print(left_most_sentence_index)
        if left_most_sentence_index == len(sentence_pos):
            left_most_sentence_index = len(sentence_pos) - 1
        answer_start = sentence_pos[left_most_sentence_index]
        answer_end = answer_start + available_answer_len
        if answer_end >= len(tagged_answer):
            answer_end = len(tagged_answer)
            answer_start = answer_end - available_answer_len
        tagged_answer = tagged_answer[answer_start:answer_end]

    return tagged_answer, at_least_one_answer, truncated


def tag_question_detailed(document_text, model_helper, main_answers, other_answers, question_ans, question_sub, question_str,
                          pubmed_id=-1, max_seq_length=384, limit_max_length=True, tagging_mode="detailed"):
    """ Tags all event arguments in a detailed way. Only tag answer entities next to question entities.
        If the trigger and substrate are in same two sentences, find nearest enzyme in the same sentence
        or in the sentence next to it. Otherwise skip marking the answer as correct.
    Parameters
    ----------
    document_text : str
        String of the document text.
    model_helper : ModelHelper
        ModelHelper containing both the AutoTokenizer and the model_name_or_path
    main_answers : list
        List of synonyms for the main answer entity to be processed
    other_answers : list of list
        List of synonyms for all other answers also valid in the multi-answer framework
    question_ans : list of list
        List of synonynms relevant for the question entities
    question_sub: list
        Subjects of the question. Example = ['CLIP1_SUBSTRATE_EGID_6249']
    question_str : list of str
        String of the question.
    pubmed_id: long
        PubMed ID of the corresponding question
    max_seq_length : int
        Maximum sequence length
    limit_max_length : bool
        Whether to truncate sequence to max_seq_length if over it
    tagging_mode: "simple" or "detailed"
        "simple" means all occurences of a protein are tagged, "detailed" means only the occurences near a trigger and question protein are tagged

    Returns
    -------
    list
        The annotated events by BERT in IOB format (token), whether a whitespace is after,
        and start and end positions of the tokens. Checking for maximum sequence length not done here
    """

    # Tokenize the text into a list of Tuples (token, start_index, end_index) and the question into list of token
    if "berta" in model_helper.model_name_or_path:
        question_str = " " + question_str
    question_tokens = model_helper.tokenizer.tokenize(question_str)
    context = get_word_tokenization_position(document_text, model_helper)

    # Apply entity blinding
    question_str_blinded, blinded_entity_dict, id_permutations = apply_entity_blinding(question_str, model_helper.tokenizer)
    question_tokens_blinded = model_helper.tokenizer.tokenize(question_str_blinded)
    context_blinded = get_word_tokenization_position(document_text, model_helper, True, blinded_entity_dict, id_permutations)

    # Assert same token lengths for entity blinding and not
    try:
        assert len(question_tokens) == len(question_tokens_blinded)
    except AssertionError:
        logger.warning(len(question_tokens))
        logger.warning(len(question_tokens_blinded))
        logger.warning(list(itertools.zip_longest(question_tokens, question_tokens_blinded)))
        raise AssertionError
    try:
        assert len(context) == len(context_blinded)
    except AssertionError:
        logger.warning(len(context))
        logger.warning(len(context_blinded))
        logger.warning(list(itertools.zip_longest(context, context_blinded)))
        raise AssertionError

    tagged_question = [pubmed_id, question_sub]
    tagged_question.append((model_helper.tokenizer.cls_token, 'O', False, -1, -1, 'O', model_helper.tokenizer.cls_token))
    for i, token in enumerate(question_tokens):
        tagged_question.append((token, 'O', False, -1, -1, 'O', question_tokens_blinded[i]))
    tagged_question.append((model_helper.tokenizer.sep_token, 'O', False, -1, -1, 'O', model_helper.tokenizer.sep_token))
    if "roberta" in model_helper.model_name_or_path:
        tagged_question.append((model_helper.tokenizer.sep_token, 'O', False, -1, -1, 'O', model_helper.tokenizer.sep_token))
    question_length = len(tagged_question)

    all_answers = main_answers + [item for answers in other_answers for item in answers] + [item for answers in question_ans for item in answers]
    all_answers = list(set(all_answers))
    sequence_all_tagged, sentence_boundaries = tag_all_entities(context, all_answers, model_helper.model_name_or_path)
    logger.info("Question answers:")
    logger.info(question_ans)
    logger.info("Context:")
    logger.info(context)
    logger.info("All entities in the sequence tagged:")
    logger.info(sequence_all_tagged)
    logger.info("Sentence boundaries:")
    logger.info(sentence_boundaries)
    main_answer_pos, other_answers_pos, question_ans_pos = identify_entities(sequence_all_tagged, main_answers, other_answers, question_ans,
                                                                             model_helper.model_name_or_path)
    logger.info("Main answer pos:")
    logger.info(main_answer_pos)
    logger.info("Other answer pos:")
    logger.info(other_answers_pos)
    logger.info("Question entity pos:")
    logger.info(question_ans_pos)
    answers, main_answers = get_relevant_answers(main_answer_pos, other_answers_pos, question_ans_pos, sentence_boundaries, tagging_mode)
    answers = list(set(answers))
    logger.info("Relevant answers:")
    logger.info(answers)

    # Find earliest occurence of substrate entity in question_str (on word basis, not character)
    # indexes = []
    # for substrate_synonym in question_ans[0]:
    #     substrate_tokens = model_helper.tokenizer.tokenize(substrate_synonym)
    #     np_tokens = np.array(question_tokens, dtype=object)
    #     candidate_indexes = np.where(np_tokens == substrate_tokens[0])[0]
    #     for ind in candidate_indexes:
    #         if question_tokens[ind: ind + len(substrate_tokens)] == substrate_tokens:
    #             indexes.append(ind)
    # if len(indexes) == 0:
    #     earliest_index = 0
    # else:
    #     earliest_index = min(indexes)

    tagged_answer, at_least_one_answer, truncated = tag_question_with_given_answers(
        context, context_blinded, answers, main_answers, question_length, max_seq_length, limit_max_length, model_helper.model_name_or_path)
    # logger.info("Final document tagged:")
    # logger.info(tagged_answer)
    # if at_least_one_answer:
    #     logger.warning(document_text)
    #     logger.warning(answers)

    question_length = len(tagged_question)
    tagged_question = tagged_question + tagged_answer
    tagged_question.append((model_helper.tokenizer.sep_token, 'O', False, -1, -1, 'O', model_helper.tokenizer.sep_token))

    logger.info("Length tagged sequence with question:")
    logger.info(len(tagged_question))
    # exit()

    return tagged_question, at_least_one_answer, question_length

def tag_question_directly_supervised(document_text, model_helper, answers, question_sub, question_str, pubmed_id=-1,
                                     max_length=384, limit_max_length=True):
    """ Tags all event arguments for a directly supervised question.
    Parameters
    ----------
    document_text : str
        String of the document text.
    model_helper : ModelHelper
        ModelHelper containing both the AutoTokenizer and the model_name_or_path
    answers : list of tuples
        Answers in the given document_text with start and end position
    question_sub: list
        Subjects of the question. Example = ['PHOSPHORYLATION_CAUSE', 'CLIP1_SUBSTRATE_EGID_6249']
    question_str : list of str
        String of the question.
    pubmed_id: long
        PubMed ID of the corresponding question
    max_length: int
        maximum sequence length
    limit_max_length: whether to limit maximum sequence length

    Returns
    -------
    list
        The annotated events by BERT in IOB format (token), whether a whitespace is after,
        and start and end positions of the tokens. No checking for maximum sequence length done here
    """

    # Tokenize the text into a list of Tuples (token, start_index, end_index) and the question into list of token
    if "berta" in model_helper.model_name_or_path:
        question_str = " " + question_str
    question_tokens = model_helper.tokenizer.tokenize(question_str)
    context = get_word_tokenization_position(document_text, model_helper)

    # Apply entity blinding
    question_str_blinded, blinded_entity_dict, id_permutations = apply_entity_blinding(question_str, model_helper.tokenizer)
    question_tokens_blinded = model_helper.tokenizer.tokenize(question_str_blinded)
    context_blinded = get_word_tokenization_position(document_text, model_helper, True, blinded_entity_dict, id_permutations)

    # Assert same token lengths for entity blinding and not
    try:
        assert len(question_tokens) == len(question_tokens_blinded)
    except AssertionError:
        logger.warning(len(question_tokens))
        logger.warning(len(question_tokens_blinded))
        logger.warning(list(itertools.zip_longest(question_tokens, question_tokens_blinded)))
        raise AssertionError
    try:
        assert len(context) == len(context_blinded)
    except AssertionError:
        logger.warning(len(context))
        logger.warning(len(context_blinded))
        logger.warning(list(itertools.zip_longest(context, context_blinded)))
        raise AssertionError

    tagged_question = [pubmed_id, question_sub]
    tagged_question.append((model_helper.tokenizer.cls_token, 'O', False, -1, -1, 'O', model_helper.tokenizer.cls_token))
    for i, token in enumerate(question_tokens):
        tagged_question.append((token, 'O', False, -1, -1, 'O', question_tokens_blinded[i]))
    tagged_question.append((model_helper.tokenizer.sep_token, 'O', False, -1, -1, 'O', model_helper.tokenizer.sep_token))
    if "roberta" in model_helper.model_name_or_path:
        tagged_question.append((model_helper.tokenizer.sep_token, 'O', False, -1, -1, 'O', model_helper.tokenizer.sep_token))
    question_length = len(tagged_question)

    tagged_answer, _, truncated = tag_question_with_given_answers(
        context, context_blinded, answers, answers, question_length, max_length, limit_max_length, model_helper.model_name_or_path)
    # logger.info("Final document tagged:")
    # logger.info(tagged_answer)
    # if at_least_one_answer:
    #     logger.warning(document_text)
    #     logger.warning(answers)
    if truncated:
        logger.info(document_text)
        logger.info(question_str)
        logger.info(answers)
        logger.info(tagged_answer)
        logger.info(pubmed_id)

    question_length = len(tagged_question)
    tagged_question = tagged_question + tagged_answer
    tagged_question.append((model_helper.tokenizer.sep_token, 'O', False, -1, -1, 'O', model_helper.tokenizer.sep_token))

    logger.info("Length tagged sequence with question:")
    logger.info(len(tagged_question))
    # exit()

    return tagged_question


# Main
if __name__ == "__main__":
    from data_processing.dict_of_triggers import TRIGGERS
    from configs import ModelHelper

    console = logging.StreamHandler()
    # add the handler to the root logger
    logger.addHandler(console)
    logger.setLevel(logging.INFO)

    model_helper = ModelHelper()

    test_document = "Pkb alpha-/- placentae also show significant reduction of hyperphosphorylation of PKB and endothelial nitric-oxide synthase."
    # annotation_sample, _, _ = tag_question_detailed(test_document, TOKENIZER, ["pkb alpha"], [[]], [TRIGGERS["Phosphorylation"]], [],
    #                                                 "?", max_seq_length=128, limit_max_length=True)

    test_document2 = "Extracellular signal-regulated kinase 2 (ERK2) is a serine/threonine protein kinase involved in many cellular programs," \
                     " such as cell proliferation, differentiation, motility and programed cell-death." \
                     " It is therefore considered an important target in the treatment of cancer." \
                     " In an effort to support biochemical screening and small molecule drug discovery," \
                     " we established a robust system to generate both inactive and active forms of ERK2 using insect expression system." \
                     " We report here, for the first time, that inactive ERK2 can be expressed and purified with 100%% homogeneity" \
                     " in the unphosphorylated form using insect system." \
                     " This resulted in a significant 20-fold yield improvement compared to that previously reported using bacterial expression system." \
                     " We also report a newly developed system to generate active ERK2 in insect cells through in vivo co-expression" \
                     " with a constitutively active MEK1 (S218D S222D)." \
                     " Isolated active ERK2 was confirmed to be doubly phosphorylated at the correct sites, T185 and Y187, in the activation loop of ERK2." \
                     " Both ERK2 forms, inactive and active, were well characterized by biochemical activity assay for their kinase function." \
                     " Inactive and active ERK2 were the two key reagents that enabled successful high through-put biochemical assay screen" \
                     " and structural drug discovery studies. \n"
    # annotation_sample, _ = tag_question_detailed(context2, ["t185"], [["y187"]], [["erk2"], TRIGGERS["Phosphorylation"]], ["erk2"],
    #                                              "?", max_seq_length=128, limit_max_length=True)

    test_doc_3 = "suggesting a normal function of the proto-oncogene c-erbA in erythropoiesis."
    # annotation_sample, _ = tag_question_detailed(context3, ["c-erba"], [["proto-oncogene c-akt"]], [["suggesting"]], [""],
    #                                              "?", max_seq_length=128, limit_max_length=True)

    # Tagging Complexes in the same sentence
    test_doc_4 = "The mTOR (mammalian target of rapamycin) protein kinase is an important regulator of cell growth and is a key target for therapeutic intervention "\
                 "in cancer. Two complexes of mTOR have been identified: complex 1 (mTORC1), consisting of mTOR, Raptor (regulatory associated protein of mTOR) "\
                 "and mLST8 (mammalian lethal with SEC13 protein 8) and complex 2 (mTORC2) consisting of mTOR, Rictor (rapamycin-insensitive companion of mTOR), "\
                 "Sin1 (stress-activated protein kinase-interacting protein 1), mLST8 and Protor-1 or Protor-2. Both complexes phosphorylate the hydrophobic motifs "\
                 "of AGC kinase family members: mTORC1 phosphorylates S6K (S6 kinase), whereas mTORC2 regulates phosphorylation of Akt, PKCalpha "\
                 "(protein kinase Calpha) and SGK1 (serum- and glucocorticoid-induced protein kinase 1)."

    # annotation_sample, bool_answer = tag_question_detailed(context4, ["mtor"], [["mlst8"]], [["pkcalpha"], TRIGGERS["Phosphorylation"]], ["pkcalpha"],
    #                                                        "?", max_seq_length=128, limit_max_length=True)

    # Tagging COmplexes that phosphorylate
    # test_doc_5 = "The signaling studies indicate that EPO modulates the growth state of the neurons in cultures without specifically reversing the inhibitory signaling "\
    #     "induced by anti-ganglioside Abs at the time points examined. This is supported by results showing that EPO's proregenerative effects in primary DRG cultures were " \
    #     "mediated via the activation of EPOR and its downstream signaling cascade involving JAK2/STAT5. It has been shown that binding of EPO to EPOR promotes the " \
    #     "phosphorylation of JAK2, the phosphorylated receptor sequentially activates several signal transduction proteins, including STAT5. The activated STAT protein " \
    #     "then binds to the promoters of specific genes in the nucleus and initiates transcription of those genes [34], [50]–[52]. Recent studies show that activation " \
    #     "of STAT5 is essential for the neurotrophic effects of EPO on neurite outgrowth [34], [53], which is consistent with our results. The exact signaling cascade " \
    #     "downstream of JAK/STAT, which promotes neurite outgrowth, is not completely characterized but some studies have suggested that PI3K/Akt pathway is partially " \
    #     "involved in EPO induced neuroprotective and neuoregenerative responses [24], [53]. We have previously shown that anti-ganglioside antibody-mediated activation of " \
    #     "small GTPase RhoA is an early signaling event (within 30 minutes) that induces inhibition of neurite outgrowth [17]. However, EPO did not directly alter the " \
    #     "anti-ganglioside antibody-mediated early activation of small GTPases RhoA at the time points examined (up to 30 minutes) indicating that modulation of small " \
    #     "GTPases (RhoA, Rac1, and Cdc42) is not an early signaling event underlying proregenerative effects of EPO in DRG neuronal cultures. Since small GTPases RhoA, Rac1, " \
    #     "and Cdc42 are considered critical and essential mediators of growth cone extension in neurons [35]–[39], it would not be surprising if these signaling molecules " \
    #     "were indirectly affected later in EPO-treated cultures as our phenotypic studies typically assess the neurite length after overnight treatment with Abs and/or EPO. " \
    #     "Alternatively, EPO could indirectly modulate the downstream effectors of RhoA. These signaling issues are beyond the scope of the current study." \

    test_doc_5 = "It has been shown that binding of EPO to EPOR promotes the " \
        "phosphorylation of JAK2, the phosphorylated receptor sequentially activates several signal transduction proteins, including STAT5." \

    # annotation_sample, _, _ = tag_question_detailed(
    #     test_doc_5, TOKENIZER, ["EPOR"],
    #     [['ckitligand', 'kitlg', 'stem cell factor', 'proto-oncogeneckit', '2.7.10.1',
    #       'v-kit hardy-zuckerman 4 feline sarcoma viral oncogene homolog', 'proto-oncogene ckit', 'proto-oncogene c-kit', 'proto-oncogenec-kit', 'EPOR', 'kit',
    #       'mast cell growth factor', 'scfr', 'scf', 'cd117', 'skitlg', 'p145 c-kit', 'protooncogene c-kit', 'p145 ckit', '2.7.10.', 'c-kit ligand', 'protooncogene ckit',
    #       'tyrosine-protein kinase kit', 'p145c-kit', 'ckit ligand', 'p145ckit', 'epo-r', 'pbt', 'protooncogeneckit', 'mgf', 'piebald trait protein', 'soluble kit ligand',
    #       'c-kitligand', 'protooncogenec-kit']],
    #     [['Janus kinase 2', 'JAK', 'JAK-2', 'JAK2', 'Jak', '2.7.10.', 'Januskinase', 'JAK-', '2.7.10.2', 'Janus kinase', 'Jak2'],
    #      ['Epoetin', 'EPO'], TRIGGERS["Phosphorylation"], TRIGGERS["Binding"]],
    #     ['PHOSPHORYLATION_COMPLEXCAUSE', 'JAK2_SUBSTRATE_EGID_3717', 'EPO_ENZYME_EGID_2056'],
    #     "What binds/is in complex with Epoetin, or EPO and causes PHOSPHORYLATION of JAK2, or Jak2?", max_seq_length=128, limit_max_length=True)

    # annotation_sample, _, _ = tag_question_detailed(
    #     test_doc_5, TOKENIZER, ["epor"],
    #     [['ckitligand', 'kitlg', 'stem cell factor', 'proto-oncogeneckit', '2.7.10.1',
    #       'v-kit hardy-zuckerman 4 feline sarcoma viral oncogene homolog', 'proto-oncogene ckit', 'proto-oncogene c-kit', 'proto-oncogenec-kit', 'epor', 'kit',
    #       'mast cell growth factor', 'scfr', 'scf', 'cd117', 'skitlg', 'p145 c-kit', 'protooncogene c-kit', 'p145 ckit', '2.7.10.', 'c-kit ligand', 'protooncogene ckit',
    #       'tyrosine-protein kinase kit', 'p145c-kit', 'ckit ligand', 'p145ckit', 'epo-r', 'pbt', 'protooncogeneckit', 'mgf', 'piebald trait protein', 'soluble kit ligand',
    #       'c-kitligand', 'protooncogenec-kit']],
    #     [['Janus kinase 2', 'JAK', 'JAK-2', 'JAK2', 'Jak', '2.7.10.', 'Januskinase', 'JAK-', '2.7.10.2', 'Janus kinase', 'Jak2'],
    #      ['Epoetin', 'epo'], TRIGGERS["Phosphorylation"], TRIGGERS["Binding"]],
    #     ['PHOSPHORYLATION_COMPLEXCAUSE', 'JAK2_SUBSTRATE_EGID_3717', 'EPO_ENZYME_EGID_2056'],
    #     "What binds/is in complex with Epoetin, or EPO and causes PHOSPHORYLATION of JAK2, or Jak2?", max_seq_length=128, limit_max_length=True)

    test_doc_6 = "CLIP-170/Restin belongs to a family of conserved microtubule (MT)-associated proteins, which are important for MT organization and functions. "\
        "CLIP-170 is a phosphoprotein and phosphorylation is thought to regulate the binding of CLIP-170 to MTs. However, little is known about the kinase(s) involved. " \
        "In this study, we show that FKBP12-rapamycin-associated protein (FRAP, also called mTOR/RAFT) interacts with CLIP-170. "\
        "CLIP-170 is phosphorylated in vivo at multiple sites, including rapamycin-sensitive and -insensitive sites, " \
        "and is phosphorylated by FRAP in vitro at the rapamycin-sensitive sites. "\
        "In addition, rapamycin inhibited the ability of CLIP-170 to bind to MTs. "\
        "Our observations suggest that multiple CLIP-170 kinases are involved in positive and negative control of CLIP-170, "\
        "and FRAP is a CLIP-170 kinase positively regulating the MT-binding behavior of CLIP-170."

    test_doc_6 = "In addition to facilitating mTOR phosphorylation of Akt, IQGAP1 interaction with mTOR may modulate its effects on the cytoskeleton. "\
        "mTOR phosphorylation of microtubule plus end binding (+TIP) protein CLIP-170 is required for the interaction of CLIP-170 and IQGAP1 in neurons [74]. "\
        "Knockdown of either CLIP-170 or IQGAP1 reduced the number of neuronal dendrites, which could be rescued by jasplakinolide-forced F-actin stabilization [74]. " \
        "Therefore, these data suggest that mTOR signaling in neurons coordinates microtubule and actin cytoskeletons by promoting the CLIP-170-IQGAP1 interaction."

    test_doc_6 = "The regulation of CLIP-170 activity appears to be rather complex. CLIP-170 is most likely phosphorylated by multiple kinases, "\
        "including FKBP12–rapamycin-associated protein (mTOR; Choi et al., 2002). Although phosphorylation by mTOR/FKBP 12–rapamycin-associated protein may stimulate "\
        "CLIP-170's microtubule binding, phosphorylation by other kinases may cause CLIP-170 to dissociate from microtubules (Rickard and Kreis 1996; Choi et al., 2002). "

    test_doc_6 = "Our observations suggest that the mechanism by which CDDO-Im disrupts Clip-170 may be through inhibition of the kinase activity of mTOR."

    annotation_sample, _, _ = tag_question_detailed(
        test_doc_6, model_helper.tokenizer, [],
        [['KIAA1303', 'TORC subunit LST', 'LST8', 'FRAP', 'MLST8', 'RAPTOR', 'Mammalian lethal with SEC13 protein', 'TORCsubunitLST',
          'FK506-binding protein 12-rapamycin complex-associated protein 1', 'Mechanistic target of rapamycin', 'Rapamycin and FKBP12 target 1',
          'Rapamycin and FKBP12 target', 'MLST', 'TORC subunitLST8', 'Rapamycintargetprotein', 'Rapamycintarget protein', 'LST', 'mLST', 'MTOR',
          'FK506-binding protein 12-rapamycin complex-associated protein', 'GBL', 'Rapamycin target protein', '2.7.11.', 'TORCsubunit LST8', 'TORC subunit LST8', 'Raptor',
          'RAPT1', 'mTOR', 'FKBP12-rapamycin complex-associated protein', 'Mammalian target of rapamycin', 'Protein GbetaL', 'RPTOR', 'TORC subunitLST', 'KIAA', 'RAFT1',
          'RAFT', 'FRAP1', 'ProteinGbetaL', 'TORCsubunitLST8', 'mLST8', 'G protein beta subunit-like', 'Gable', '2.7.11.1', 'Mammalian lethal with SEC13 protein 8', 'FRAP2',
          'TORCsubunit LST', 'p150 target of rapamycin (TOR)-scaffold protein', 'RAPT', 'Rapamycin target protein 1', 'Rapamycin targetprotein']],
        [['CLIP1', 'RSN', 'CLIP170', 'Restin', 'CYLN1', 'Cytoplasmic linker protein 1', 'Cytoplasmic linker protein 170 alpha-2', 'CLIP-170',
          'Reed-Sternberg intermediate filament-associated protein'], TRIGGERS["Phosphorylation"]],
        ['PHOSPHORYLATION_CAUSE', 'CLIP1_SUBSTRATE_EGID_6249'],
        "What causes phosphorylation of CLIP1, CYLN1, RSN, or Restin?", max_seq_length=384, limit_max_length=True)

    logger.info(annotation_sample)
    # logger.warn(bool_answer)
