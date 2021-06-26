''' Evaluate MAP Scores of simple EVEX Baseline '''

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
import random

from tqdm import tqdm
from sklearn.metrics import PrecisionRecallDisplay

from metrics.sklearn_revised import average_precision_score, precision_recall_curve
from data_processing.datatypes import Question, get_db_id, QUESTION_TYPES, QUESTION_TYPES_EVEX, get_subject_list
from data_processing.biopax_to_retrieval import IndraDataLoader
from configs import PID_MODEL_FAMILIES, STANDOFF_CACHE, STANDOFF_CACHE_SIMPLE
from baseline.evex_standoff import BaselineEventDict, EVEX_QUESTION_MAPPER

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# TODO: Verfiy that higher confidence values are more reliable
# TODO: Further distinguish Refined Types (Regulation, Indirect_regulation, Catalysis of phosphorylation)
# and Refined Polarities (Positive, Negative, Unspecified). Now, treat all as the same type. Ignore negations for now.


class KnowledgeBaseEvaluator:

    def __init__(self, groundtruth_dict=None, predictions_dict=None, mode="eval", standoff=False):
        self.mode = mode
        self.indra_dict, self.indra_dict_verbose = self._get_indra_relations(self.mode)
        self.model_groundtruth_dict = groundtruth_dict
        self.model_predictions_dict = predictions_dict
        self.model_indra_dict = None
        if groundtruth_dict is not None:
            self.model_indra_dict = self._get_event_intersection(self.indra_dict, self.model_groundtruth_dict)
        if standoff:
            self.evex_dict = self._get_all_evex_standoff_relations()
        else:
            self.evex_dict = self._get_all_evex_relations(use_cache=True)

    @classmethod
    def _get_all_evex_standoff_relations(cls):
        ''' Returns dict of all relations in EVEX statement set like {"PHOSPHORYLATION": {1: {2: 0.123}}},
            with question_type, EntrezGeneID of the Theme, EntrezGeneID of the Cause and the confidence in the relation.
            The higher the value, the more confident is the prediction.
        '''

        evex_dict = BaselineEventDict(standoff_cache=STANDOFF_CACHE)
        evex_dict.load_event_dict()
        relations = {}

        for key, values in evex_dict.evex_db.items():
            for value in values:
                event_type = ""
                _, protein_id = key.split("_", 1)
                if len(value) == 10 or len(value) == 8:  # Protein Complex Interaction 10 (PTMS), 8 (OTHER)
                    event_type = value[6]
                    # regulation_type = value[7]
                elif len(value) == 7:  # Protein Site Description
                    event_type = value[4]
                elif len(value) == 6:  # Complex
                    event_type = value[5]

                if event_type in EVEX_QUESTION_MAPPER and len(value) in [8, 10] and value[0] != "":
                    theme_id = ("EGID", value[0])
                    cause_ids = (("EGID", value[1]), ("EGID", value[2]))

                    for i, cause_id in enumerate(cause_ids):
                        if cause_id[1] == "" or cause_id == theme_id:
                            continue
                        confidence_score = float(value[3])
                        # negation = value[4]
                        # Our model does not look for negations so we also do not look for negations in the baseline
                        if True:  # negation != 1:
                            # Add _CAUSE events
                            question_type = EVEX_QUESTION_MAPPER[event_type] + "_CAUSE"
                            modification = relations.setdefault(question_type, {})
                            themes = modification.setdefault(theme_id, {})
                            themes.setdefault(cause_id, [0, 0])
                            if (cause_id in themes and confidence_score > themes[cause_id][0]) or (cause_id not in themes):
                                themes[cause_id][0] = confidence_score
                            themes[cause_id][1] += 1
                            # Add pairs for the general STATECHANGE_CAUSE question type
                            statechange = relations.setdefault("STATECHANGE_CAUSE", {})
                            themes = statechange.setdefault(theme_id, {})
                            themes.setdefault(cause_id, [0, 0])
                            if (cause_id in themes and confidence_score > themes[cause_id][0]) or (cause_id not in themes):
                                themes[cause_id][0] = confidence_score
                            themes[cause_id][1] += 1

                            # Add _COMPLEXSITE events for PTMs
                            if len(value) == 10 and EVEX_QUESTION_MAPPER[event_type] != "STATECHANGE":
                                question_type = EVEX_QUESTION_MAPPER[event_type] + "_COMPLEXSITE"
                                complex_site = relations.setdefault(question_type, {})
                                res_pos = ("SITE", value[8] + value[9])
                                if res_pos[1] != "":
                                    themes = complex_site.setdefault((theme_id, cause_id), {})
                                    themes.setdefault(res_pos, [0, 0])
                                    if (res_pos in themes and confidence_score > themes[res_pos][0]) or (res_pos not in themes):
                                        themes[res_pos][0] = confidence_score
                                    themes[res_pos][1] += 1

                            # Add _COMPLEXCAUSE events
                            if i == 0:
                                other_cause_id = cause_ids[1]
                            else:
                                other_cause_id = cause_ids[0]
                            if not (other_cause_id[1] == "" or other_cause_id == theme_id):
                                question_type = EVEX_QUESTION_MAPPER[event_type] + "_COMPLEXCAUSE"
                                complex_modification = relations.setdefault(question_type, {})
                                themes = complex_modification.setdefault((theme_id, cause_id), {})
                                themes.setdefault(other_cause_id, [0, 0])
                                if (other_cause_id in themes and confidence_score > themes[other_cause_id][0]) or (other_cause_id not in themes):
                                    themes[other_cause_id][0] = confidence_score
                                themes[other_cause_id][1] += 1
                                # Add pairs for the general STATECHANGE_COMPLEXCAUSE question type
                                complex_statechange = relations.setdefault("STATECHANGE_COMPLEXCAUSE", {})
                                themes = complex_statechange.setdefault((theme_id, cause_id), {})
                                themes.setdefault(other_cause_id, [0, 0])
                                if (other_cause_id in themes and confidence_score > themes[other_cause_id][0]) or (other_cause_id not in themes):
                                    themes[other_cause_id][0] = confidence_score
                                themes[other_cause_id][1] += 1

                # Add _SITE events for PTMs
                if event_type in EVEX_QUESTION_MAPPER and len(value) == 7:
                    theme_id = ("EGID", value[0])
                    confidence_score = float(value[1])
                    question_type = EVEX_QUESTION_MAPPER[event_type] + "_SITE"
                    site = relations.setdefault(question_type, {})
                    res_pos = ("SITE", value[5] + value[6])
                    if res_pos[1] != "":
                        themes = site.setdefault((theme_id), {})
                        themes.setdefault(res_pos, [0, 0])
                        if (res_pos in themes and confidence_score > themes[res_pos][0]) or (res_pos not in themes):
                            themes[res_pos][0] = confidence_score
                        themes[res_pos][1] += 1

        return relations

    @classmethod
    def _get_indra_relations(cls, mode):
        ''' Returns dict of all relations in INDRA statement set like {"PHOSPHORYLATION_CAUSE": {1: {2: 0.123}}},
            with question_type, EntrezGeneID of the Theme, EntrezGeneID of the Cause and the confidence in the relation.
            The higher the value, the more confident is the prediction.
        '''

        relations = {}
        relations_verbose = {}
        question_types = [Question(number) for number in QUESTION_TYPES]
        # question_types = [Question.ACTIVATION_CAUSE]
        # question_types = [Question.PHOSPHORYLATION_CAUSE]
        logger.info("    **** Loading INDRA events")
        for question_type in tqdm(question_types, desc="Iterate over all question types and load from groundtruth"):
            stats_logging = False
            if question_type.name in ["PHOSPHORYLATION_CAUSE", "PHOSPHORYLATION_COMPLEXCAUSE"] or logger.level == logging.DEBUG:
                stats_logging = True
                logger.info(question_type)
            _, event_dict = IndraDataLoader.get_dataset(mode=mode, question_type=question_type, biopax_model_str=PID_MODEL_FAMILIES)
            subjects = list(event_dict.keys())
            counter_pairs = 0
            counter_pairs_with_ids = 0
            # for i, subject in enumerate(tqdm(subjects, desc="Load INDRA groundtruth")):
            for i, subject in enumerate(subjects):
                statements = event_dict[subject]
                subject_agents = IndraDataLoader.get_all_indra_agents(statements[0], subject)
                _, unique_answer_agents = IndraDataLoader.get_unique_args_statements(subject, statements, question_type)

                # relations_verbose used for visualization
                subject_list = tuple(get_subject_list(question_type.name, subject_agents))
                relations_verbose.setdefault(subject_list, {})

                # Change DEPHOSPHORYLATION to PHOSPHORYLATION and INHIBEXPRESSION to EXPRESSION for evaluation purposes
                if question_type.name.startswith("DE"):
                    question_type = Question[question_type.name[2:]]
                elif question_type.name.startswith("INHIBEXPRESSION"):
                    question_type = Question[question_type.name[5:]]
                elif question_type.name.startswith("ACTIVATION"):
                    question_type = Question["STATECHANGE" + "_" + question_type.name.split("_")[-1]]
                elif question_type.name.startswith("INHIBITION"):
                    question_type = Question["STATECHANGE" + "_" + question_type.name.split("_")[-1]]

                if question_type.name.endswith("_CAUSE"):
                    theme_agent = subject_agents[0]
                    theme_id = get_db_id(theme_agent)
                    for cause_agent in unique_answer_agents:
                        cause_id = get_db_id(cause_agent)
                        counter_pairs += 1
                        if theme_id[1] != "##" and cause_id[1] != "##":
                            modification = relations.setdefault(question_type.name, {})
                            themes = modification.setdefault(theme_id, {})
                            themes[cause_id] = [1, 1]
                            counter_pairs_with_ids += 1
                            relations_verbose[subject_list].setdefault(cause_id, {})
                elif question_type.name.endswith("_COMPLEXCAUSE"):
                    theme_ids = []
                    themes = []
                    for subject_agent in subject_agents:
                        theme_agent = subject_agent
                        theme_id = get_db_id(theme_agent)
                        theme_ids.append(theme_id)
                    if "##" in theme_ids:
                        continue
                    for cause_agent in unique_answer_agents:
                        cause_id = get_db_id(cause_agent)
                        counter_pairs += 1
                        if cause_id[1] != "##":
                            complex_modification = relations.setdefault(question_type.name, {})
                            themes = complex_modification.setdefault(tuple(theme_ids), {})
                            themes[cause_id] = [1, 1]
                            counter_pairs_with_ids += 1
                            relations_verbose[subject_list].setdefault(cause_id, {})
                elif question_type.name.endswith("_SITE"):
                    theme_agent = subject_agents[0]
                    theme_id = get_db_id(theme_agent)
                    # logger.info(unique_answer_agents)
                    for res_pos_str in unique_answer_agents:
                        if theme_id[1] != "##" and res_pos_str != "":
                            modification = relations.setdefault(question_type.name, {})
                            themes = modification.setdefault(theme_id, {})
                            themes[("SITE", res_pos_str)] = [1, 1]
                            counter_pairs += 1
                            relations_verbose[subject_list].setdefault(("SITE", res_pos_str), {})
                elif question_type.name.endswith("_COMPLEXSITE"):
                    theme_ids = []
                    themes = []
                    for subject_agent in subject_agents:
                        theme_agent = subject_agent
                        theme_id = get_db_id(theme_agent)
                        theme_ids.append(theme_id)
                        themes.append(theme_id[1])
                    if "##" in themes:
                        continue
                    for res_pos_str in unique_answer_agents:
                        if res_pos_str != "":
                            complex_site = relations.setdefault(question_type.name, {})
                            themes = complex_site.setdefault(tuple(theme_ids), {})
                            themes[("SITE", res_pos_str)] = [1, 1]
                            counter_pairs += 1
                            relations_verbose[subject_list].setdefault(("SITE", res_pos_str), {})

            if stats_logging:
                logger.info("Pairs (INDRA Groundtruth)")
                logger.info(counter_pairs)
                logger.info("Pairs with DB identifiers (INDRA Groundtruth)")
                logger.info(counter_pairs_with_ids)

        return relations, relations_verbose

    @classmethod
    def _get_event_intersection(cls, dict_1, dict_2):
        ''' Transform relation dict to relation list. '''

        dict_intersection = {}

        for question_type in dict_1:
            if question_type in dict_2:
                dict_intersection[question_type] = {}
                for themes, cause_set in dict_1[question_type].items():
                    if themes in dict_2[question_type]:
                        dict_intersection[question_type][themes] = {}
                        for cause, confidence in dict_1[question_type][themes].items():
                            if cause in dict_2[question_type][themes]:
                                dict_intersection[question_type][themes][cause] = confidence

        return dict_intersection

    @classmethod
    def _get_relation_list(cls, relation_dict, question_type="PHOSPHORYLATION_CAUSE"):
        ''' Transform relation dict to relation list. '''

        relation_list = []

        if question_type not in [" All", " Simple", " Complex"]:
            if question_type in relation_dict:
                for themes, cause_set in relation_dict[question_type].items():
                    for cause, confidence in cause_set.items():
                        # if question_type == "PHOSPHORYLATION_COMPLEXSITE":
                        #     print(themes)
                        #     print(cause)
                        if cause[1] != "##":
                            # themes_int = tuple([theme for theme in themes]) if isinstance(themes, tuple) else themes
                            answer = cause
                            relation_list.append((themes, answer, float(confidence[0]), confidence[1]))
        else:
            for tmp_question_type in relation_dict.keys():
                if question_type == " Simple" and "COMPLEX" in tmp_question_type:
                    continue
                elif question_type == " Complex" and "COMPLEX" not in tmp_question_type:
                    continue
                for themes, cause_set in relation_dict[tmp_question_type].items():
                    for cause, confidence in cause_set.items():
                        if cause[1] != "##":
                            themes_question = tuple([tmp_question_type] + [theme for theme in themes]) if isinstance(themes, tuple) else themes
                            answer = cause
                            relation_list.append((themes_question, answer, float(confidence[0]), confidence[1]))

        return relation_list

    @classmethod
    def _compare_relations(cls, predictions, groundtruth, min_confidence=None):
        ''' Compare relations and match relations with the same entities. Sort ascending by
            relation tuples where the values are proteins in form of their Entrez Gene IDs.
            Set predictions which are not in groundtruth to new minimum value. Add new
            groundtruth for unmatched predictions with binary label 0.
        '''

        y_true = []
        y_scores = []

        # logger.info(predictions[0])
        # logger.info(groundtruth[0])
        predictions_sorted = sorted(predictions, key=lambda x: (x[0], x[1], x[2]))
        groundtruth_sorted = sorted(groundtruth, key=lambda x: (x[0], x[1]))

        logger.debug("# of Protein pairs (Predicted): {}".format(len(predictions_sorted)))
        logger.debug("# of Protein pairs (Groundtruth): {}".format(len(groundtruth_sorted)))

        if predictions_sorted:
            logger.info("Predictions sample: {}".format(predictions_sorted[0:10]))
            logger.info("Groundtruth sample: {}".format(groundtruth_sorted[0:10]))
            logger.debug("Maximum confidence score: {}".format(max(predictions_sorted, key=lambda x: x[2])[2]))
            logger.debug("Minimal confidence score: {}".format(min(predictions_sorted, key=lambda x: x[2])[2]))
            if min_confidence is None:
                min_confidence = min(predictions_sorted, key=lambda x: x[2])[2] - 1
                logger.debug("Confidence score for unseen groundtruth Protein Pairs: {}".format(min_confidence))
        else:
            logger.info("No predictions")

        if len(groundtruth_sorted) == 0:
            return np.array([0]), np.array([0])

        question_entities_groundtruth = set()
        question_entities_prediction = set()

        groundtruth_set = set()
        baseline_set = set()
        last_groundtruth_theme = -1

        i = 0
        j = 0
        while i < len(groundtruth_sorted) or j < len(predictions_sorted):
            if i >= len(groundtruth_sorted):
                groundtruth = groundtruth_sorted[-1]
            else:
                groundtruth = groundtruth_sorted[i]
            if len(predictions_sorted) == 0:
                prediction = groundtruth_sorted[-1]
            elif j >= len(predictions_sorted):
                prediction = predictions_sorted[-1]
            else:
                prediction = predictions_sorted[j]

            question_entities_groundtruth.add(groundtruth[0])
            question_entities_prediction.add(prediction[0])

            # Relation tuple in predictions but not in groundtruth
            if (prediction[0] < groundtruth[0] and j < len(predictions_sorted)) \
                    or (prediction[0] > groundtruth[0] and i >= len(groundtruth_sorted)) \
                    or (prediction[0] == groundtruth[0] and prediction[1] < groundtruth[1]):
                if prediction[0] in [groundtruth[0], last_groundtruth_theme]:  # Ask questions from given themes. Themes not in groundtruth are ignored
                    y_true.append(0)
                    y_scores.append(prediction[2])
                    baseline_set.add((prediction[0], prediction[1]))
                if prediction[0] == groundtruth[0] and prediction[1] < groundtruth[1] and j >= len(predictions_sorted):
                    i += 1
                else:
                    j += 1
            # Relation tuple in groundtruth but not in prediction
            elif (prediction[0] < groundtruth[0] and j >= len(predictions_sorted)) \
                    or (prediction[0] > groundtruth[0] and i < len(groundtruth_sorted)) \
                    or (prediction[0] == groundtruth[0] and prediction[1] > groundtruth[1]):
                # if (prediction[0] == groundtruth[0]):  # Ask questions from given themes. Themes not in groundtruth are ignored
                y_true.append(1)
                y_scores.append(min_confidence)
                if prediction[0] == groundtruth[0] and prediction[1] > groundtruth[1] and i >= len(groundtruth_sorted):
                    j += 1
                else:
                    i += 1
                last_groundtruth_theme = groundtruth[0]  # Handle special case with lookbehind memory
                groundtruth_set.add((groundtruth[0], groundtruth[1]))
            # Relation tuple in both predictions and groundtruth
            elif (prediction[0] == groundtruth[0]) and (prediction[1] == groundtruth[1]):
                y_true.append(1)
                y_scores.append(prediction[2])
                i += 1
                j += 1
                last_groundtruth_theme = groundtruth[0]  # Handle special case with lookbehind memory
                groundtruth_set.add((prediction[0], prediction[1]))
                baseline_set.add((prediction[0], prediction[1]))
            else:
                raise RuntimeError("This branch can not happen!")

        intersection_question_entities = question_entities_groundtruth.intersection(question_entities_prediction)
        # prediction_only_question_entities = question_entities_prediction.difference(question_entities_groundtruth)
        # groundtruth_only_question_entities = question_entities_groundtruth.difference(question_entities_prediction)
        # logger.info("Question Entities in both Groundtruth and Prediction: {}".format(intersection_question_entities))
        # logger.info("Question Entities only in Prediction: {}".format(prediction_only_question_entities))
        # logger.info("Question Entities only in Groundtruth: {}".format(groundtruth_only_question_entities))
        logger.info("# Question Subjects (Prediction): {} (All)".format(len(question_entities_prediction)))
        logger.info("# Question Subjects (Groundtruth): {}".format(len(question_entities_groundtruth)))
        logger.info("# Question Subjects (Groundtruth and Prediction): {}".format(len(intersection_question_entities)))

        # number_correct = len(groundtruth_set & baseline_set)
        # number_true = len(groundtruth_set)
        # number_pred = len(baseline_set)

        # recall_score = number_correct / number_true if number_true > 0 else 0
        # precision_score = number_correct / number_pred if number_pred > 0 else 0
        # logger.info("-- Recall (Assertion): {}".format(recall_score))
        # logger.info("-- Precision (Assertion): {}".format(precision_score))

        return np.array(y_true), np.array(y_scores)

    @classmethod
    def _get_performance_scores(cls, groundtruth_list, baseline_list, optional_list=None):
        question_entity_set = set()
        groundtruth_set = set()
        baseline_set = set()
        evidence_counts_dict = {}
        evidence_counts = 0
        for example in groundtruth_list:
            groundtruth_set.add((example[0], example[1]))
            question_entity_set.add(example[0])
        for example in baseline_list:
            if example[0] in question_entity_set:
                baseline_set.add((example[0], example[1]))
                evidence_counts_dict.setdefault(example[3], 0)
                evidence_counts_dict[example[3]] += 1
                evidence_counts += example[3]

        number_correct = len(groundtruth_set & baseline_set)
        number_true = len(groundtruth_set)
        number_pred = len(baseline_set)
        logger.info("Evidence Counts Histogram")
        logger.info(sorted(list(evidence_counts_dict.items()), key=lambda x: x[0]))
        logger.info("# Question Answer Pairs (Prediction - Evidence): {}".format(evidence_counts))
        logger.info("# Question Answer Pairs (Prediction): {}".format(number_pred))
        logger.info("# Question Answer Pairs (Groundtruth): {}".format(number_true))
        logger.info("# Question Answer Pairs (Groundtruth and Prediction): {}".format(number_correct))
        # logger.debug("-- Correct predictions")
        # logger.debug(groundtruth_set & baseline_set)

        if optional_list is not None:
            optional_set = set()
            for example in optional_list:
                if example[0] in question_entity_set:
                    optional_set.add((example[0], example[1]))
            found_in_all_set = (groundtruth_set & optional_set) & (groundtruth_set & baseline_set)
            logger.info("# Question Answer Pairs (Groundtruth, Prediction and Optional): {}".format(len(found_in_all_set)))
            only_in_optional_set = (groundtruth_set & optional_set) - (groundtruth_set & baseline_set)
            logger.info("# Question Answer Pairs (Groundtruth and Optional, but not Prediction): {}".format(len(only_in_optional_set)))
            if len(only_in_optional_set) > 10:
                sample_size = 10
            else:
                sample_size = len(only_in_optional_set)
            logger.info("# Sample (Groundtruth and Optional, but not Prediction): {}".format(random.sample(only_in_optional_set, sample_size)))

        recall_score = number_correct / number_true if number_true > 0 else 0
        precision_score = number_correct / number_pred if number_pred > 0 else 0
        f1_score = 2 * precision_score * recall_score / (precision_score + recall_score) if precision_score + recall_score > 0 else 0

        return recall_score, precision_score, f1_score, number_pred

    def calculate_average_precision(self, groundtruth_dict, predictions_dict, question_type="PHOSPHORYLATION_CAUSE",
                                    min_confidence=None, save_bool=False, optional_dict=None):
        ''' Calculate average precision from a knowledge base perspective. '''

        if (question_type not in groundtruth_dict.keys() or question_type not in predictions_dict.keys()) and \
                question_type not in [" All", " Simple", " Complex"]:
            return -0.01, -0.01, -0.01, -0.01, 0, "None"

        groundtruth_list = self._get_relation_list(groundtruth_dict, question_type)
        predictions_list = self._get_relation_list(predictions_dict, question_type)
        optional_list = self._get_relation_list(optional_dict, question_type) if optional_dict is not None else None

        y_true, y_scores = self._compare_relations(predictions_list, groundtruth_list, min_confidence)
        recall_score, precision_score, f1_score, number_pred = self._get_performance_scores(groundtruth_list, predictions_list, optional_list)

        try:
            average_precision = average_precision_score(y_true, y_scores)
            # precision, recall, threshold = precision_recall_curve(y_true, y_scores)
            # pr_display = PrecisionRecallDisplay(precision=precision, recall=recall).plot()
            # pr_display.figure_.savefig(root_path + "/notes/plot.png")
        except (IndexError, TypeError):
            average_precision = 0

        logger.info("Question type: {}".format(question_type))
        logger.info("Average precision: {}".format(average_precision))
        logger.info("Recall: {}".format(recall_score))
        logger.info("Precision: {}".format(precision_score))
        logger.info("F1: {}".format(f1_score))
        logger.info("-----------------------------------------------------------------------------")

        return average_precision, recall_score, precision_score, f1_score, number_pred, question_type


# Main
if __name__ == "__main__":
    # relations = get_all_evex_relations(use_cache=True)
    # relations = get_all_evex_standoff_relations()
    # logger.info("EVEX Predictions for EGID 729359: {}".format(relations["PHOSPHORYLATION_CAUSE"]['729359']))
    # logger.info("EVEX Predictions for EGIDs: {}".format(relations["PHOSPHORYLATION_CAUSE"]))
    logger.setLevel(logging.INFO)
    kb_evaluator = KnowledgeBaseEvaluator(mode="test", standoff=True)
    question_types = [Question(number).name for number in QUESTION_TYPES_EVEX]
    question_types.append(" All")
    question_types.append(" Simple")
    question_types.append(" Complex")
    # question_types = [Question.PHOSPHORYLATION_CAUSE.name]
    for question_type in question_types:
        kb_evaluator.calculate_average_precision(kb_evaluator.indra_dict, kb_evaluator.evex_dict, question_type)
