""" Generates question and answer for input to the model """

import os
import sys

if os.name == 'nt':
    root_path = "/".join(os.path.realpath(__file__).split("\\")[:-2])
else:
    root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.append(root_path)

import logging
import copy
import random
import spacy

from configs import SPACY_NER_MODEL, DICT_NORMALIZER

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ner = spacy.load(SPACY_NER_MODEL)
random.seed(5)


def generate_question(entities, question_type):
    if question_type.name.endswith("_CAUSE"):
        modification_type = question_type.name.split("_")[0].lower()
        question_string = "What causes {} of ".format(modification_type)
    elif question_type.name.endswith("_SITE"):
        modification_type = question_type.name.split("_")[0].lower()
        question_string = "What are {} sites in ".format(modification_type)
    elif question_type.name.endswith("COMPLEX_PAIR"):
        question_string = "What binds/is in complex with "
    elif question_type.name.endswith("COMPLEX_MULTI"):
        question_string = "What binds/is in complex with "
        for i, entity in enumerate(entities[1]):
            if i + 1 == len(entities[1]) and i > 0:
                question_string += "or {} ".format(entity)
            elif i + 1 == len(entities[1]) and i == 0:
                question_string += "{} ".format(entity)
            else:
                question_string += "{}, ".format(entity)
        modification_type = question_type.name.split("_")[0].lower()
        question_string += "and {} ".format(modification_type)
    elif question_type.name.endswith("COMPLEXCAUSE"):
        question_string = "What binds/is in complex with "
        for i, entity in enumerate(entities[1]):
            if i + 1 == len(entities[1]) and i > 0:
                question_string += "or {} ".format(entity)
            elif i + 1 == len(entities[1]) and i == 0:
                question_string += "{} ".format(entity)
            else:
                question_string += "{}, ".format(entity)
        modification_type = question_type.name.split("_")[0].lower()
        question_string += "and causes {} of ".format(modification_type)
    elif question_type.name.endswith("COMPLEXSITE"):
        question_string = "Which sites does "
        for i, entity in enumerate(entities[1]):
            if i + 1 == len(entities[1]) and i > 0:
                question_string += "or {} ".format(entity)
            elif i + 1 == len(entities[1]) and i == 0:
                question_string += "{} ".format(entity)
            else:
                question_string += "{}, ".format(entity)
        modification_type = question_type.name.split("_")[0].lower()
        question_string += "regulate in {} of ".format(modification_type)
    for i, entity in enumerate(entities[0]):
        if i + 1 == len(entities[0]) and i > 0:
            question_string += "or {}?".format(entity)
        elif i + 1 == len(entities[0]) and i == 0:
            question_string += "{}?".format(entity)
        else:
            question_string += "{}, ".format(entity)

    return question_string


def apply_entity_blinding(doc_string, tokenizer, blinded_entity_dict=None, id_permutation=None):
    spacy_doc = ner(doc_string)
    new_string = doc_string
    if id_permutation is None:
        id_permutation = list(range(50))
        random.shuffle(id_permutation)
    if blinded_entity_dict is None:
        blinded_entity_dict = {}
    ent_ids = []
    # print(spacy_doc.ents)
    for e in reversed(spacy_doc.ents):  # reversed to not modify the offsets of other entities when substituting
        # Edge Case where for instance substring "°C" of "37°C" is deteted as entity and leads to different lengths in blinded and non-blinded tokens
        if not((e.start_char == 0 or doc_string[e.start_char] == ' ' or doc_string[e.start_char - 1] == ' ')
                and (e.end_char == len(doc_string) or doc_string[e.end_char - 1] == ' ' or doc_string[e.end_char] == ' ')):
            continue
        if e in blinded_entity_dict:
            protein_id = blinded_entity_dict[e]
        elif (DICT_NORMALIZER is not None) and (e.text.lower() in DICT_NORMALIZER):
            db_id = DICT_NORMALIZER[e.text.lower()][0]
            if db_id in blinded_entity_dict:
                protein_id = blinded_entity_dict[db_id]
            else:
                if not id_permutation:  # Empty list
                    # print(doc_string)
                    # print(spacy_doc.ents)
                    # print(len(set(spacy_doc.ents)))
                    # # Ordered Set from ent_ids, least recently used ent_id is recycled, relevant for docs bigger than max_seq_length
                    # print(ent_ids)
                    ent_ids = list(dict.fromkeys(ent_ids))
                    protein_id = ent_ids.pop(0)
                    # print(protein_id)
                    # raise Exception("More than 50 proteins in document!")
                else:
                    protein_id = id_permutation.pop(0)
                blinded_entity_dict[db_id] = protein_id
                blinded_entity_dict[e] = protein_id
        else:
            if not id_permutation:  # Empty list
                # print(doc_string)
                # print(spacy_doc.ents)
                # print(len(set(spacy_doc.ents)))
                # print(ent_ids)
                # Ordered Set from ent_ids, least recently used ent_id is recycled, relevant for docs bigger than max_seq_length
                ent_ids = list(dict.fromkeys(ent_ids))
                protein_id = ent_ids.pop(0)
                # print(ent_ids)
                # raise Exception("More than 50 proteins in document!")
            else:
                protein_id = id_permutation.pop(0)
            blinded_entity_dict[e] = protein_id
        ent_ids.append(protein_id)

        start = e.start_char
        end = start + len(e.text)
        # print(e.text)
        # print(tokenizer.tokenize(e.text.strip()))
        entity_text = e.text
        if entity_text[-1] == " ":
            entity_text = entity_text[:-1]
        e_length = len(tokenizer.tokenize(" " + entity_text))
        if e_length >= 1:
            if new_string[start] == " ":
                blinded_entity_str = " <protein" + str(protein_id) + ">"
            else:
                blinded_entity_str = "<protein" + str(protein_id) + ">"
            for i in range(e_length - 1):
                blinded_entity_str += "<##prot>"
            if new_string[end - 1] == " ":
                blinded_entity_str += " "
            # new_string = new_string[:start] + e.label_ + new_string[end:]
            new_string = new_string[:start] + blinded_entity_str + new_string[end:]
        else:
            logger.warning("Entity length < 1")
            logger.warning(e)

    return new_string, blinded_entity_dict, id_permutation


def get_word_tokenization_position(doc_string, model_helper, blind_doc=False, blinded_entity_dict=None, id_permutation=None):
    """ Reads the file line by line and returns start and end character for each token
    Parameters
    ----------
    doc_str : str
        Document content as a string
    blind_doc : bool
        Whether to blind entities or not
    blinded_entity_dict : dict
        entity string to blinded entity IDs
    id_permutation : list
        list of integer permutation between 0 and 50

    Returns
    -------
    list
        List of all tokens as the following list: [token, start_character, end_character]
    """

    words = []
    index = 0
    doc_string = " " + doc_string
    tokens = model_helper.tokenizer.tokenize(doc_string)
    if blind_doc:
        doc_string_blinded, _, _ = apply_entity_blinding(doc_string, model_helper.tokenizer, blinded_entity_dict, id_permutation)
        tokens_blinded = model_helper.tokenizer.tokenize(doc_string_blinded)
        try:
            assert len(tokens) == len(tokens_blinded)
        except AssertionError:
            logger.warn(tokens)
            logger.warn(tokens_blinded)
            logger.warn(doc_string)
            logger.warn(doc_string_blinded)
            logger.warn(list(zip(tokens, tokens_blinded)))
            logger.warn(len(tokens))
            logger.warn(len(tokens_blinded))
            raise AssertionError("Length of Tokens and blinded Tokens is not the same")
    tokens_index = 0
    current_token = model_helper.lower_string(tokens[tokens_index])
    if current_token[:2] == '##' and "berta" not in model_helper.model_name_or_path:
        current_token = current_token[2:]
    current_token_position = 0
    matching = False
    start_i = 0
    end_i = 0
    candidate_token = ""
    sub_token = False
    if "berta" in model_helper.model_name_or_path:
        doc_string = "".join(tokens)
    for i, char in enumerate(doc_string, index):
        if current_token_position >= len(current_token):
            logger.warn(current_token)
            logger.warn(current_token_position)
            logger.warn(char)
            logger.warn(i)
            logger.warn(doc_string)
        if model_helper.lower_string(char) == current_token[current_token_position]:
            if not matching:
                start_i = i
                matching = True
            candidate_token = candidate_token + model_helper.lower_string(char)
            current_token_position += 1
            if candidate_token == current_token:
                if sub_token:
                    current_token = '##' + current_token
                end_i = i
                if not blind_doc:
                    words.append([current_token, start_i, end_i])
                else:
                    words.append([tokens_blinded[tokens_index], start_i, end_i])
                matching = False
                candidate_token = ""
                current_token_position = 0
                tokens_index += 1
                if tokens_index != len(tokens):
                    current_token = tokens[tokens_index]
                    sub_token = False
                    if current_token[:2] == '##' and "berta" not in model_helper.model_name_or_path:
                        current_token = current_token[2:]
                        sub_token = True
        else:  # char != current_token[current_token_position]
            matching = False
            candidate_token = ""
            current_token_position = 0
    index = i + 1
    return words


def match_new_answers(answers, matched_strings, word, matching, whitespace):
    matching_string = word[0]
    if matching_string[0].startswith("Ġ"):
        matching_string = matching_string[1:]
    matching_string = matching_string.replace("Ġ", " ")
    # Handling of Roberta Tokenizer (END)
    for answer in answers:
        # if answer.startswith(matching_string) or matching_string.startswith(answer):
        if answer.lower().startswith(matching_string.lower()):
            matched_strings[answer] = ["", []]
            matched_strings[answer][0] = matching_string
            if "" in matched_strings:
                # Use because we are further matching
                matched_strings[answer][1] = copy.deepcopy(matched_strings[""][1])
            matched_strings[answer][1].append((word[0], 'B', whitespace, word[1], word[2]))
            matching = True
    # logger.warn(" MatchNewAnswers   {}".format(matched_strings))
    return matching


def match_current_answers(answers, matched_strings, word, matching, whitespace, word_continutation, matching_string):
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
            # logger.warn(answer.lower())
            # logger.warn(further_matching_string.lower())
            matched_strings[answer][0] = further_matching_string
            if word_continutation:
                matched_strings[answer][1].append((word[0], 'X', whitespace, word[1], word[2]))
            else:
                matched_strings[answer][1].append((word[0], 'I', whitespace, word[1], word[2]))
        elif answer != "":
            matched_strings[answer][1].append((word[0], 'O', whitespace, word[1], word[2]))
    # logger.warn(" MatchCurrentAnswers   {}".format(matched_strings))


def end_matching(matched_strings, word, matching, whitespace, tagged_question):
    if matching:
        # Add a zero answer throughout the matching step
        if "" in matched_strings:
            matched_strings[""][1].append((word[0], 'O', whitespace, word[1], word[2]))
        else:
            matched_strings[""] = ["", [(word[0], 'O', whitespace, word[1], word[2])]]
    else:
        tagged_question.append((word[0], 'O', whitespace, word[1], word[2]))
