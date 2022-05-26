""" Annotate directly supervised examples from BioNLP  """

import os
import re
import sys

if os.name == 'nt':
    root_path = "/".join(os.path.realpath(__file__).split("\\")[:-2])
else:
    root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.append(root_path)

import logging
import pandas as pd

from baseline.bionlp_standoff import BIONLP_QUESTION_MAPPER, PTMS, BaselineEventDict
from configs import BIONLP_CACHE, OWL_LIST, OWL_STATEMENTS, ModelHelper
from data_processing.biopax_to_retrieval import IndraDataLoader
from data_processing.datatypes import QUESTION_TYPES, GENE_INFO_ENGINE, GeneIDToNames, Question
from data_processing.file_converter import generate_question
from data_processing.question_tagger import tag_question_directly_supervised
from util.build_document_corpus import get_pmid_from_pmcid_mapper

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class DirectDataAnnotator:
    ''' Create examples for directly supervised data from BioNLP '''

    def __init__(self, model_helper=ModelHelper(), max_length=384, limit_max_length=True):
        self.training_question_subtrates = {}
        self.evaluation_question_subtrates = {}
        self.test_question_subtrates = {}
        self.training_subtrates = set()
        self.evaluation_subtrates = set()
        self.test_subtrates = set()
        self.pmcid_mapper = None
        self.model_helper = model_helper
        self.max_length = max_length
        self.limit_max_length = limit_max_length
        if GeneIDToNames.lookup_table is None:
            GeneIDToNames.initialize()

        self.relations = None
        self.relations_negative = None
        self.relations_all = None
        self.annotations = None
        self.annotation_dict = None

    def initialize_pmcid_mapper(self):
        _, self.pmcid_mapper = get_pmid_from_pmcid_mapper()

    def get_substrates(self):
        self.training_question_subtrates, self.training_subtrates = self.get_datasplit_substrates("train")
        self.evaluation_question_subtrates, self.evaluation_subtrates = self.get_datasplit_substrates("eval")
        self.test_question_subtrates, self.test_subtrates = self.get_datasplit_substrates("test")

    def get_datasplit_substrates(self, mode):
        ''' Get list of EntrezGene IDs representing event substrates in their respective data splits '''
        question_types = [Question(question) for question in QUESTION_TYPES]
        # question_types = [Question.PHOSPHORYLATION_CAUSE, Question.ACETYLATION_CAUSE, Question.UBIQUITINATION_CAUSE,
        #                   Question.EXPRESSION_CAUSE, Question.INHIBEXPRESSION_CAUSE, Question.STATECHANGE_CAUSE]
        # question_types = [Question.PHOSPHORYLATION_CAUSE]
        question_substrate_dict = {}
        substrate_set = set()
        for question_type in question_types:
            logger.info("Current Mode and Question Type: {} {}".format(mode, question_type.name))
            _, event_dict = IndraDataLoader.get_dataset(
                biopax_owl_strings=OWL_LIST, use_cache=True, mode=mode, question_type=question_type, biopax_model_str=OWL_STATEMENTS)
            for agents, events in event_dict.items():
                subject_agents = IndraDataLoader.get_all_indra_agents(events[0], agents)
                for agent in subject_agents:
                    # print(agent.db_refs)
                    if "UP" in agent.db_refs:  # Transform Uniprot ID to EntrezGene ID
                        df = pd.read_sql_query(('SELECT GeneID FROM up_to_geneid2 WHERE "UniProtKB-AC" = "{}" AND "NCBI-taxon" = "9606"'
                                                'LIMIT 1').format(agent.db_refs["UP"]), GENE_INFO_ENGINE)
                        if len(df) > 0:
                            eg_id_number = df["GeneID"][0].split(";")[0]
                            if eg_id_number != "":
                                question_substrate_dict.setdefault(question_type, set())
                                question_substrate_dict[question_type].add(eg_id_number)
                                if not question_type.name.endswith("COMPLEX") and not question_type.name != "COMPLEX_MULTI":
                                    substrate_set.add(eg_id_number)

        return question_substrate_dict, substrate_set

    def map_to_pid_question_types(self):
        ''' Returns dict of all BioNLP events mapped to their PID Question type ((Question, EGID), (File_Name, BioNLP event))
        '''

        bionlp_dict = BaselineEventDict(BIONLP_CACHE)
        bionlp_dict.load_event_dict()
        self.relations = {}

        for _, values in bionlp_dict.bionlp_db.items():
            for value in values:
                event_type = ""
                if len(value) == 14 or len(value) == 11 or len(value) == 10:  # Protein Complex Interaction 14 (PTMS), 11 (OTHER), 10 (COMPLEX_MULTI)
                    event_type = value[8]
                    # regulation_type = value[9]
                elif len(value) == 8:  # COMPLEX_PAIR
                    event_type = value[6]
                elif len(value) == 9:  # Protein Site Description
                    event_type = value[4]
                file_name = value[-1]

                if event_type in BIONLP_QUESTION_MAPPER and len(value) in [11, 14] and value[0] not in ["", "XX"]:
                    theme_id = value[0]
                    cause_ids = (value[2], value[4])
                    custom_event_type = BIONLP_QUESTION_MAPPER[event_type]

                    for i, cause_id in enumerate(cause_ids):
                        if cause_id in ["", "XX", theme_id]:
                            continue
                        negation = value[6]
                        # Our model does not look for negations so we also do not look for negations in the baseline,
                        # only for EXPRESSION_CAUSE
                        if True:  # negation != 1:
                            # Add _CAUSE events
                            # print(cause_id)
                            if negation == 1 and custom_event_type == "EXPRESSION":
                                custom_event_type = "INHIBEXPRESSION"
                            question_type = custom_event_type + "_CAUSE"
                            if i == 0:
                                dict_value = value[0:4] + value[6:-1]
                            else:
                                dict_value = value[0:2] + value[4:-1]
                            rel_file = self.relations.setdefault((question_type, theme_id), {})
                            rel_file_list = rel_file.setdefault(file_name, [])
                            rel_file_list.append(dict_value)

                            # Add _COMPLEXSITE events for PTMs
                            if len(value) == 14 and custom_event_type != "STATECHANGE":
                                res = value[-4]
                                if res != "":
                                    question_type = custom_event_type + "_COMPLEXSITE"
                                    rel_file = self.relations.setdefault((question_type, theme_id), {})
                                    rel_file_list = rel_file.setdefault(file_name, [])
                                    rel_file_list.append(dict_value)

                            # Add _COMPLEXCAUSE events
                            if i == 0:
                                other_cause_id = cause_ids[1]
                                dict_value = value[:-1]
                            else:
                                other_cause_id = cause_ids[0]
                                dict_value = value[0:2] + value[4:6] + value[2:4] + value[6:-1]
                            if not (other_cause_id == "" or other_cause_id == theme_id):
                                question_type = custom_event_type + "_COMPLEXCAUSE"
                                rel_file = self.relations.setdefault((question_type, theme_id), {})
                                rel_file_list = rel_file.setdefault(file_name, [])
                                rel_file_list.append(dict_value)

                # Add _SITE events for PTMs
                if event_type in BIONLP_QUESTION_MAPPER and len(value) == 9:
                    theme_id = value[0]
                    res = value[5]
                    dict_value = value[:-1]
                    if res != "":
                        question_type = BIONLP_QUESTION_MAPPER[event_type] + "_SITE"
                        rel_file = self.relations.setdefault((question_type, theme_id), {})
                        rel_file_list = rel_file.setdefault(file_name, [])
                        rel_file_list.append(dict_value)

                # Add COMPLEX_PAIR events
                if event_type in BIONLP_QUESTION_MAPPER and len(value) == 8:
                    theme_id = value[0]
                    dict_value = value[:-1]
                    question_type = BIONLP_QUESTION_MAPPER[event_type] + "_PAIR"
                    rel_file = self.relations.setdefault((question_type, theme_id), {})
                    rel_file_list = rel_file.setdefault(file_name, [])
                    rel_file_list.append(dict_value)

                # Add COMPLEX_MULTI events
                if event_type in BIONLP_QUESTION_MAPPER and len(value) == 10:
                    theme_id = value[0]
                    theme_id_2 = value[2]
                    dict_value = value[:-1]
                    question_type = BIONLP_QUESTION_MAPPER[event_type] + "_MULTI"
                    rel_file = self.relations.setdefault((question_type, theme_id, theme_id_2), {})
                    rel_file_list = rel_file.setdefault(file_name, [])
                    rel_file_list.append(dict_value)

        return self.relations

    def map_entity_trigger_pairs(self, add_negative_examples=True):
        ''' Returns dict of all entity trigger or complex trigger BioNLP pairs mapped to their PID Question type ((Question, EGID), (File_Name, BioNLP event))
        '''

        bionlp_dict = BaselineEventDict(BIONLP_CACHE)
        bionlp_dict.load_event_questions_dict()
        bionlp_dict.load_event_dict()
        self.relations_negative = {}

        if not add_negative_examples:
            return self.relations_negative

        # Get single-turn "negative" examples
        for _, values in bionlp_dict.bionlp_pairs_db.items():
            for value in values:
                event_type = ""
                if len(value) == 8:  # Complex
                    event_type = value[6]
                else:
                    event_type = value[8]

                if event_type in BIONLP_QUESTION_MAPPER and value[0] not in ["", "XX"]:
                    theme_id = value[0]
                    cause_ids = (value[2], value[4])
                    custom_event_type = BIONLP_QUESTION_MAPPER[event_type]

                    # _COMPLEXCAUSE questions are ignored here

                    question_type = custom_event_type + "_CAUSE"
                    file_name = value[-1]
                    dict_value = value[0:2] + value[4:-1]
                    rel_file = self.relations_negative.setdefault((question_type, theme_id), {})
                    rel_file_list = rel_file.setdefault(file_name, [])
                    rel_file_list.append(dict_value)
                    if custom_event_type == "EXPRESSION":
                        question_type = "INHIB" + custom_event_type + "_CAUSE"
                        rel_file = self.relations_negative.setdefault((question_type, theme_id), {})
                        rel_file_list = rel_file.setdefault(file_name, [])
                        rel_file_list.append(dict_value)

                    # Add _SITE events for PTMs
                    if event_type in PTMS:
                        question_type = custom_event_type + "_SITE"
                        dict_value = value[0:2] + value[6:-1]
                        rel_file = self.relations_negative.setdefault((question_type, theme_id), {})
                        rel_file_list = rel_file.setdefault(file_name, [])
                        rel_file_list.append(dict_value)

                    # Add COMPLEX_PAIR events
                    if event_type in BIONLP_QUESTION_MAPPER and len(value) == 8:
                        theme_id = value[0]
                        dict_value = value[:-1]
                        question_type = BIONLP_QUESTION_MAPPER[event_type] + "_PAIR"
                        rel_file = self.relations.setdefault((question_type, theme_id), {})
                        rel_file_list = rel_file.setdefault(file_name, [])
                        rel_file_list.append(dict_value)

        # Get multi-turn "negative" examples
        for _, values in bionlp_dict.bionlp_db.items():
            for value in values:
                event_type = ""
                if len(value) == 14:
                    event_type = value[8]
                elif len(value) == 8:
                    event_type = value[6]

                if len(value) == 14 and event_type in PTMS and event_type in BIONLP_QUESTION_MAPPER and value[0] not in ["", "XX"]:
                    theme_id = value[0]
                    cause_ids = (value[2], value[4])
                    custom_event_type = BIONLP_QUESTION_MAPPER[event_type]

                    for i, cause_id in enumerate(cause_ids):
                        if cause_id in ["", "XX", theme_id]:
                            continue
                        question_type = custom_event_type + "_COMPLEXSITE"
                        file_name = value[-1]
                        if i == 0:
                            dict_value = value[0:4] + value[6:-1]
                        else:
                            dict_value = value[0:2] + value[4:-1]
                        rel_file = self.relations_negative.setdefault((question_type, theme_id, cause_id), {})
                        rel_file_list = rel_file.setdefault(file_name, [])
                        rel_file_list.append(dict_value)

                if event_type in BIONLP_QUESTION_MAPPER and len(value) == 8:  # Add COMPLEX_MULTI events
                    participant_1_id = value[0]
                    participant_2_id = value[2]
                    dict_value = value[:4] + ("", ("", "-1", "-1")) + value[4:-1]
                    question_type = BIONLP_QUESTION_MAPPER[event_type] + "_MULTI"
                    rel_file = self.relations.setdefault((question_type, participant_1_id, participant_2_id), {})
                    rel_file_list = rel_file.setdefault(file_name, [])
                    rel_file_list.append(dict_value)
                    # print(dict_value)

        return self.relations_negative

    def combine_negative_and_positive_relations(self, add_unknown_substrates=True):
        self.relations_all = {}
        for key_event, value_dict in self.relations_negative.items():
            for key_file_name, value_event_body in value_dict.items():
                if key_event in self.relations and key_file_name in self.relations[key_event]:
                    continue
                elif key_event in self.relations and key_file_name not in self.relations[key_event]:
                    rel_file = self.relations_all.setdefault(key_event, {})
                    rel_file_list = rel_file.setdefault(key_file_name, [])
                    rel_file_list.extend(value_event_body)
                elif add_unknown_substrates:  # key_event not in self.relations
                    rel_file = self.relations_all.setdefault(key_event, {})
                    rel_file_list = rel_file.setdefault(key_file_name, [])
                    rel_file_list.extend(value_event_body)
        for key_event, value_dict in self.relations.items():
            for key_file_name, value_event_body in value_dict.items():
                if len(value_event_body) > 0 and key_event[1] != "XX":
                    rel_file = self.relations_all.setdefault(key_event, {})
                    rel_file_list = rel_file.setdefault(key_file_name, [])
                    rel_file_list.extend(value_event_body)

        return self.relations_all

    def get_all_relations(self, add_negative_examples=True, add_unknown_substrates=True):
        self.map_to_pid_question_types()
        self.map_entity_trigger_pairs(add_negative_examples)
        return self.combine_negative_and_positive_relations(add_unknown_substrates)

    def generate_annotations(self, relations):
        ''' Returns dict of all annotated BioNLP events mapped to their PID Question type ((Question, EGID), (File_Name, Annotated))
        '''
        lengths = []
        self.annotation_dict = {}
        for key_event, value_dict in relations.items():
            annotation_files = self.annotation_dict.setdefault(key_event, {})
            for key_file_name, value_event_body in value_dict.items():
                if len(value_event_body) > 0:
                    annotation_files[key_file_name] = self.annotate_example(key_event[0], key_file_name, value_event_body)
                    lengths.append(len(annotation_files[key_file_name]))
        # lengths_under_max_length = [length for length in lengths if length <= self.max_length + 2]
        # lengths_over_max_length = [length for length in lengths if length > self.max_length + 2]
        # print(lengths)
        # print(lengths_under_max_length)
        # print(lengths_over_max_length)
        logger.debug("Number of examples: {}".format(len(lengths)))
        # print(len(lengths_under_max_length))
        # print(len(lengths_over_max_length))

    def annotate_example(self, question_type, file_name, event_list):
        ''' Returns tagged BioNLP event for list of events from one file and one question type and one substrate
            Example question_type = 'STATECHANGE_CAUSE'
            Example file_name = '/glusterfs/dfs-gfs-dist/wangxida/bionlp_datasets/BioNLP-ST_2013_PC_development_data/PMID-16571800.a2'
            or just 'PMID-16571800.a2'
            Example event_list = [('7157', ('p53', '95', '98'), '472', ('ATM', '205', '208'), 0, 0, 'Regulation', 'Regulation'),
            ('7157', ('p53', '669', '672'), '1869', ('E2F1', '631', '635'), 0, 0, 'Regulation', 'Positive_regulation')]
        '''

        # Generate question
        substrate_entrez_gene_id = [event_list[0][0], event_list[0][2]]
        substrate_names = [set(), set()]
        for event in event_list:  # Collect explicit substrate metions in given text
            substrate_names[0].add(event[1][0])
            if question_type.endswith("COMPLEXCAUSE") or question_type.endswith("COMPLEXSITE") or question_type.endswith("COMPLEX_MULTI"):
                substrate_names[1].add(event[3][0])

        if substrate_entrez_gene_id[0] in GeneIDToNames.lookup_table:
            substrate_names[0].add(GeneIDToNames.lookup_table[substrate_entrez_gene_id[0]][0])
            for substrate_synonym in GeneIDToNames.lookup_table[substrate_entrez_gene_id[0]][1]:
                if len(substrate_synonym) == 1:
                    continue
                if len(substrate_names[0]) >= 6:
                    break
                substrate_names[0].add(substrate_synonym)
        if question_type.endswith(("COMPLEXCAUSE", "COMPLEXSITE", "COMPLEX_MULTI")) and substrate_entrez_gene_id[1] in GeneIDToNames.lookup_table:
            substrate_names[1].add(GeneIDToNames.lookup_table[substrate_entrez_gene_id[1]][0])
            for substrate_synonym in GeneIDToNames.lookup_table[substrate_entrez_gene_id[1]][1]:
                if len(substrate_synonym) == 1:
                    continue
                if len(substrate_names[1]) >= 6:
                    break
                substrate_names[1].add(substrate_synonym)
        substrate_names = [sorted(substrate_names[0], key=lambda x: len(x)), sorted(substrate_names[1], key=lambda x: len(x))]
        question_sub = [question_type, substrate_names[0][0] + "_SUBSTRATE_EGID_" + str(substrate_entrez_gene_id[0])]
        if len(substrate_names[1]) > 0:
            question_sub.append(substrate_names[1][0] + "_ENZYME_EGID_" + str(substrate_entrez_gene_id[1]))
        # if question_type.endswith(("COMPLEX_MULTI")):
        #     print(len(question_sub))
        question_str = generate_question(substrate_names, Question[question_type])

        # Gather all answers
        answer_set = set()
        for event in event_list:
            if question_type.endswith("SITE"):
                answer_info = event[-1]
            elif question_type.endswith("COMPLEXCAUSE") or question_type.endswith("COMPLEX_MULTI"):
                answer_info = event[5]
            else:
                answer_info = event[3]
            answer_info = list(answer_info)
            # print(answer_info)
            # print(event)
            answer_info[1] = int(answer_info[1]) + 1  # Adjust start position
            answer_info[2] = int(answer_info[2]) + 1  # Adjust end position
            if answer_info[1] > 0:  # not a negative examples
                answer_set.add(tuple(answer_info))
        answer_list = sorted(answer_set, key=lambda x: x[1])

        # Open corresponding text file
        current_file_content = ""
        with open(file_name[:-3] + ".txt", "r", encoding="utf-8", errors="replace") as f:
            txt_lines = f.readlines()
            for line in txt_lines:
                current_file_content += line
        # Parse Annotation File Name
        if self.pmcid_mapper is None:
            self.initialize_pmcid_mapper()
        short_file_name = file_name.split("/")[-1]
        file_name_parts = re.split("[-.]", short_file_name)
        if file_name_parts[0] == "PMC":
            pmid = self.pmcid_mapper[file_name_parts[1]]
        elif file_name_parts[0] == "PMID":
            pmid = file_name_parts[1]
        else:
            raise ValueError("Cannot infer Pubmed ID from %s", file_name)

        # Tag question
        logger.debug(current_file_content)
        logger.debug(answer_list)
        logger.debug(question_sub)
        logger.debug(question_str)
        logger.debug(file_name)
        self.annotations = tag_question_directly_supervised(current_file_content, self.model_helper, answer_list, question_sub, question_str, pmid,
                                                            self.max_length, self.limit_max_length)
        return self.annotations

    def get_datasplit_question_annotations(self, question_type, datasplit):
        datasplit_annotations = []
        for key, value in self.annotation_dict.items():
            question_type_string = key[0]
            substrate_eg_id = key[1]
            if question_type.name != question_type_string:
                continue
            if datasplit == "train" and substrate_eg_id not in self.training_subtrates and (substrate_eg_id in self.evaluation_subtrates
               or substrate_eg_id in self.test_subtrates):  # Not mappable Entrez Gene IDs are assigned to the training set
                continue
            elif datasplit == "eval" and substrate_eg_id not in self.evaluation_subtrates:
                continue
            elif datasplit == "test" and substrate_eg_id not in self.test_subtrates:
                continue
            for file_name, annotation in value.items():
                datasplit_annotations.append([annotation])  # Treat each directly supervised examples as its own set
        return datasplit_annotations

    def get_stats(self):
        # Get statistics about the question types
        length_train_total = 0
        length_eval_total = 0
        length_test_total = 0
        length_total = 0
        strings_list = ["".ljust(42), "TRAIN".ljust(8), "EVAL".ljust(8), "TEST".ljust(8), "ALL".ljust(8)]
        logger.info("".join(strings_list))
        for number in QUESTION_TYPES:
            tmp_annotation_train = self.get_datasplit_question_annotations(Question(number), "train")
            tmp_annotation_eval = self.get_datasplit_question_annotations(Question(number), "eval")
            tmp_annotation_test = self.get_datasplit_question_annotations(Question(number), "test")
            question_name = Question(number).name
            length_train = len(tmp_annotation_train)
            length_eval = len(tmp_annotation_eval)
            length_test = len(tmp_annotation_test)
            length_question = length_train + length_eval + length_test
            length_train_total += length_train
            length_eval_total += length_eval
            length_test_total += length_test
            length_total += length_question
            strings_list = ["Stats for {}:".format(question_name).ljust(42), "{}".format(length_train).ljust(8), "{}".format(length_eval).ljust(8),
                            "{}".format(length_test).ljust(8), "{}".format(length_question).ljust(8)]
            logger.info("".join(strings_list))
        strings_list = ["Total:".ljust(42), "{}".format(length_train_total).ljust(8), "{}".format(length_eval_total).ljust(8),
                        "{}".format(length_test_total).ljust(8), "{}".format(length_total).ljust(8)]
        logger.info("".join(strings_list))

# Main
if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    annotator = DirectDataAnnotator()

    # Get size of data sets
    # #annotator.get_substrates()
    # print("  EntrezGene IDs of Substrates:")
    # print(annotator.training_subtrates)
    # print(len(annotator.training_subtrates))
    # print(annotator.evaluation_subtrates)
    # print(len(annotator.evaluation_subtrates))
    # print(annotator.test_subtrates)
    # print(len(annotator.test_subtrates))
    # print("  EntrezGene IDs in different datasplits and their intersections:")
    # print(annotator.training_subtrates.intersection(annotator.evaluation_subtrates))
    # print(annotator.training_subtrates.intersection(annotator.test_subtrates))
    # print(annotator.evaluation_subtrates.intersection(annotator.test_subtrates))

    # Convert BioNLP data
    # bionlp_dict = BaselineEventDict(BIONLP_CACHE)
    # bionlp_dict.load_event_dict()
    # for key, value in bionlp_dict.bionlp_db.items():
    #     print(key, value)

    # print("  Sample of BioNLP events in our custom question types:")
    relations = annotator.map_to_pid_question_types()
    # for i, (key, value) in enumerate(relations.items()):
    #     print(key, value)
    #     if i > 10:
    #         break

    # Add negative examples
    negative_relations = annotator.map_entity_trigger_pairs()

    # Logging and information purposes
    number_positive_questions = 0
    number_positive_question_answers = 0
    number_positive_not_in_negative = 0
    number_positive_not_in_negative_files = 0
    number_sanity_check = 0
    number_negative_questions_known_substrate = 0
    number_negative_questions_unknown_substrate = 0
    for key_event, value_dict in negative_relations.items():
        for key_file_name, value_event_body in value_dict.items():
            if key_event in relations and key_file_name in relations[key_event]:
                number_sanity_check += 1
            elif key_event in relations and key_file_name not in relations[key_event]:
                number_negative_questions_known_substrate += 1
            else:  # key_event not in relations
                number_negative_questions_unknown_substrate += 1
    for key_event, value_dict in relations.items():
        for key_file_name, value_event_body in value_dict.items():
            if len(value_event_body) > 0 and key_event[1] != "XX":
                number_positive_questions += 1
                number_positive_question_answers += len(value_event_body)
                if key_event not in negative_relations:
                    number_positive_not_in_negative += 1
                    # print("  ", key_event)
                elif key_file_name not in negative_relations[key_event]:
                    number_positive_not_in_negative_files += 1
                    # print(key_event)
                    # print(relations[key_event])
                    # print(negative_relations[key_event])

    print("Number of positive questions: {}".format(number_positive_questions))
    print("Number of positive question answers: {}".format(number_positive_question_answers))
    print("Number of positive answers not in negative: {}".format(number_positive_not_in_negative))
    print("Number of positive answers not in negative files: {}".format(number_positive_not_in_negative_files))
    print("Number of sanity check: {}".format(number_sanity_check))
    print("Number of negative questions with known substrate: {}".format(number_negative_questions_known_substrate))
    print("Number of negative questions with unknown substrate: {}".format(number_negative_questions_unknown_substrate))

    all_relations = annotator.get_all_relations()
    # annotator.generate_annotations(all_relations)

    # Get annotations
    annotator.generate_annotations(relations)
    annotator.get_stats()
    annotator.generate_annotations(all_relations)
    annotator.get_stats()
