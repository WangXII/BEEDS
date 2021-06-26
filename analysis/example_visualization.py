""" Visualize sentences and predictions from neural networks. """

import os
import sys

if os.name == 'nt':
    root_path = "/".join(os.path.realpath(__file__).split("\\")[:-2])
else:
    root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.append(root_path)

from typing import List, Tuple, Set

import logging
import numpy as np
import random

# from data_processing.biopax_to_retrieval import IndraDataLoader
# from data_processing.datatypes import Question
from configs import NN_CACHE  # , PID_MODEL_FAMILIES
import metrics.eval as util

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ExampleVisualizer():

    def __init__(self, cached_file=None, groundtruth=None, predictions=None,
                 indra_event_dict=None, groundtruth_type="annotations"):
        self.cached_file = cached_file
        self.groundtruth_by_pubmed_id = groundtruth
        self.predictions_by_pubmed_id = predictions
        self.groundtruth_by_entities = indra_event_dict
        self.groundtruth_type = groundtruth_type
        self._load_cached_file()

    def _load_cached_file(self):
        if self.cached_file is not None:
            npzcache = np.load(self.cached_file, allow_pickle=True)
            groundtruth, predictions, _, _, _ = [npzcache[array] for array in npzcache.files]
            self.groundtruth_by_pubmed_id = [groundtruth.item()]
            self.predictions_by_pubmed_id = [predictions.item()]
        self.predictions_by_entities = util.convert_answers_from_docs_to_entities(self.predictions_by_pubmed_id, bool_return_docs=True)
        if self.groundtruth_type == "annotations":
            self.groundtruth_by_entities = util.convert_answers_from_docs_to_entities(self.groundtruth_by_pubmed_id, bool_return_docs=True)

    def _group_by_key(self, pair_list):
        entity_dict = {}
        for entity_0, entity_1 in pair_list:
            entity_dict.setdefault(entity_0, [])
            entity_dict[entity_0].append(entity_1)
        return entity_dict

    def visualize_entities(self):
        ground_qa_pairs, pred_qa_pairs = self.get_entity_questions_answers()
        intersection_qa_pairs = pred_qa_pairs & ground_qa_pairs
        ground_qa_dict = self._group_by_key(ground_qa_pairs)
        pred_qa_dict = self._group_by_key(pred_qa_pairs)
        intersection_qa_dict = self._group_by_key(intersection_qa_pairs)
        row_format = "{:>60}" + "{:>15}" * 3
        logger.info("Number of subjects: {}".format(len(ground_qa_dict)))
        logger.info(row_format.format("Question subject", "# Groundtruth", "# Predictions", "# Common"))
        total_groundtruth = 0
        total_predictions = 0
        total_intersection = 0
        for substrate in ground_qa_dict:
            substrate_str = " ".join(substrate[1:])
            row_list = [substrate_str, len(ground_qa_dict[substrate])]
            total_groundtruth += len(ground_qa_dict[substrate])
            if substrate in pred_qa_dict:
                row_list.append(len(pred_qa_dict[substrate]))
                total_predictions += len(pred_qa_dict[substrate])
            else:
                row_list.append(0)
            if substrate in intersection_qa_dict:
                row_list.append(len(intersection_qa_dict[substrate]))
                total_intersection += len(intersection_qa_dict[substrate])
            else:
                row_list.append(0)
            logger.info(row_format.format(*row_list))
        logger.info(row_format.format("TOTAL", total_groundtruth, total_predictions, total_intersection))

    def visualize_predictions(self, mode="all", number=10, sampling_method="top", question_subject=None, evidence_preds=False):
        """ Visualizes neural network predictions based on the confidence.
        Parameters
        ----------
        mode: str
            "all" chooses from (predictions)
            "true_positives" chooses from (predictions & groundtruth)
            "false_positives" chooses from (predictions - predictions & groundtruth)
            "false_negatives" chooses from (groundtruth - predictions & groundtruth)
        number : int
            Number of examples to visualize
        smpling_method : str
            "top" takes the highest confidence examples.
            "random" takes random examples.
        """

        ground_qa_pairs, pred_qa_pairs = self.get_entity_questions_answers()
        if mode == "all":
            qa_pairs = pred_qa_pairs
            qa_pairs_with_evidence = self.get_pairs_with_evidence(qa_pairs, self.predictions_by_entities, question_subject)
            evidence_list = self.get_list_of_evidence(qa_pairs, self.predictions_by_entities, question_subject)
        elif mode == "true_positives":
            qa_pairs = pred_qa_pairs & ground_qa_pairs
            qa_pairs_with_evidence = self.get_pairs_with_evidence(qa_pairs, self.predictions_by_entities, question_subject)
            evidence_list = self.get_list_of_evidence(qa_pairs, self.predictions_by_entities, question_subject)
        elif mode == "false_positives":
            qa_pairs = pred_qa_pairs - (pred_qa_pairs & ground_qa_pairs)
            qa_pairs_with_evidence = self.get_pairs_with_evidence(qa_pairs, self.predictions_by_entities, question_subject)
            evidence_list = self.get_list_of_evidence(qa_pairs, self.predictions_by_entities, question_subject)
        elif mode == "false_negatives":
            qa_pairs = ground_qa_pairs - (pred_qa_pairs & ground_qa_pairs)
            qa_pairs_with_evidence = self.get_pairs_with_evidence(qa_pairs, self.groundtruth_by_entities, question_subject)
            evidence_list = self.get_list_of_evidence(qa_pairs, self.groundtruth_by_entities, question_subject)
        else:
            raise ValueError("Wrong mode string!")

        # if sampling_method == "top":
        #     sorted_qa_pairs = sorted(qa_pairs_with_evidence, key=lambda x: x[1][0], reverse=True)  # Sort by descending confidence
        # else:  # Sampling method is random
        sorted_qa_pairs = random.sample(qa_pairs_with_evidence, len(qa_pairs_with_evidence))
        if sampling_method == "random":
            top_pairs = sorted_qa_pairs[:number]
        else:  # Sampling method is top
            event_dict = {}
            top_pairs = []
            for pair in sorted_qa_pairs:
                event_type = pair[0][0][0]
                event_dict.setdefault(event_type, 0)
                if event_dict[event_type] <= number:
                    top_pairs.append(pair)
                    event_dict[event_type] += 1
        logger.info(" ------------------------------------------------------------ ")
        logger.info(" RELATION PREDICTIONS ")
        # logger.info("Maximum Confidence: {}".format(sorted_qa_pairs[0][1][0]))
        # logger.info("Minimum Confidence: {}".format(sorted_qa_pairs[-1][1][0]))
        for i, pair in enumerate(top_pairs):
            logger.info("  **** Example {} ****".format(i))
            logger.info("Substrate: {}".format(pair[0][0]))
            logger.info("Kinase: {}".format(pair[0][1]))
            logger.info("")
            for j, evidence in enumerate(pair[1]):
                logger.info(" Evidence Number {}".format(j))
                logger.info("Synonyms: {}, Confidence: {}, PubMed ID: {}".format(evidence[0], evidence[1], evidence[2]))
                logger.info(" Text Evidence: {}".format(evidence[3]))
                logger.info("")

        if evidence_preds is True:
            logger.info(" ------------------------------------------------------------ ")
            logger.info(" EVIDENCE PREDICTIONS ")
            if sampling_method == "top":
                evidences = sorted(evidence_list, key=lambda x: x[1][0], reverse=True)  # Sort by descending confidence
            else:  # Sampling method is random
                evidences = random.sample(evidence_list, len(evidence_list))
            top_pairs = evidences[:number]
            for i, pair in enumerate(top_pairs):
                logger.info("  **** Example {} ****".format(i))
                logger.info("Substrate: {}".format(pair[0]))
                logger.info("Kinase: {} (Synonyms: {}, Confidence: {}, PubMed ID: {})".format(pair[1], pair[2], pair[3], pair[4]))
                logger.info("")
                logger.info("Text Evidence: {}".format(pair[5]))
                logger.info("")

        return sorted_qa_pairs

    def get_list_of_evidence(self, qa_pairs, events_by_entities, substrate_tuple=None):
        qa_pairs_with_evidence = []
        for substrate in events_by_entities:
            if substrate_tuple is None or substrate_tuple == substrate:
                for kinase, docs in events_by_entities[substrate].items():
                    if (substrate, kinase) in qa_pairs and kinase[1] != "##":
                        for pubmed_id, answers in docs.items():
                            for confidence, text in answers[1]:
                                qa_pairs_with_evidence.append((substrate, kinase, answers[0], confidence, pubmed_id, text))
        return qa_pairs_with_evidence

    def get_pairs_with_evidence(self, qa_pairs, events_by_entities, substrate_tuple=None):
        qa_pairs_with_evidence = {}
        for substrate in events_by_entities:
            if substrate_tuple is None or substrate_tuple == substrate:
                for kinase, docs in events_by_entities[substrate].items():
                    if (substrate, kinase) in qa_pairs and kinase[1] != "##":
                        qa_pairs_with_evidence.setdefault((substrate, kinase), [])
                        for pubmed_id, answers in docs.items():
                            for confidence, text in answers[1]:
                                # logger.info(answers)
                                # logger.info(qa_pairs_with_evidence[(substrate, kinase)])
                                # logger.info(answers[1])
                                # logger.info(confidence)
                                qa_pairs_with_evidence[(substrate, kinase)].append((answers[0], confidence, pubmed_id, text))
        qa_pairs_with_evidence = list(qa_pairs_with_evidence.items())
        return qa_pairs_with_evidence

    def visualize_entity(self, entrez_gene_id: Tuple):
        logger.info("Results for Entity ID {}".format(entrez_gene_id))
        if entrez_gene_id in self.groundtruth_by_entities:
            logger.info("  Groundtruth Answers:")
            for object_id, doc_list in self.groundtruth_by_entities[entrez_gene_id].items():
                logger.info("Database ID {}".format(object_id))
                pubmed_ids = []
                for pubmed_id, answer_list in doc_list.items():
                    pubmed_ids.append(str(pubmed_id))
                logger.info("  Pubmed IDs: " + " ".join(pubmed_ids))
            logger.info("  List of Answer Database IDs: {}".format(self.groundtruth_by_entities[entrez_gene_id].keys()))
        if entrez_gene_id in self.predictions_by_entities:
            logger.info("Predictions:")
            logger.info("  List of Answer Database IDs: {}".format(self.predictions_by_entities[entrez_gene_id].keys()))
            for object_id, doc_list in self.predictions_by_entities[entrez_gene_id].items():
                logger.info("Answer Database ID: {}".format(object_id))
                # for i, (pubmed_id, answer_list) in enumerate(doc_list.items()):
                #     if i >= 3:
                #         break
                #     logger.info("  Pubmed ID: {}".format(pubmed_id))
                #     for confidence, text in answer_list:
                #         logger.info("  Confidence: {}".format(confidence))
                #         logger.info("    " + text)

    def get_entity_questions(self) -> Tuple[List, List]:
        return list(self.groundtruth_by_entities.keys()), list(self.predictions_by_entities.keys())

    def get_entity_questions_answers(self) -> Tuple[Set, Set]:
        groundtruth_question_answer_pair = \
            set([(subject_id, object_id) for subject_id in self.groundtruth_by_entities for object_id in self.groundtruth_by_entities[subject_id]])
        prediction_question_answer_pair = \
            set([(subject_id, object_id) for subject_id in self.predictions_by_entities for object_id in self.predictions_by_entities[subject_id]])
        return groundtruth_question_answer_pair, prediction_question_answer_pair


if __name__ == "__main__":
    visualizer = ExampleVisualizer(cached_file=NN_CACHE + "/processed_output_question_0_1.npz")
    visualizer.visualize_entities()
    # groundtruth_subjects, predictions_subjects = visualizer.get_entity_questions()
    # logger.info(groundtruth_subjects)
    # logger.info()
    # logger.info(predictions_subjects)
    # visualizer.visualize_entities(('PHOSPHORYLATION_CAUSE', 'RBBP8_SUBSTRATE_EGID_5932'))
    # visualizer.visualize_entities(('PHOSPHORYLATION_CAUSE', 'GSK3B_SUBSTRATE_EGID_2932'))
    # visualizer.visualize_entities(('PHOSPHORYLATION_CAUSE', 'MAP2K1_SUBSTRATE_EGID_5604'))
    visualizer.visualize_entity(('PHOSPHORYLATION_COMPLEXCAUSE', 'MAPK8_SUBSTRATE_EGID_5599'))
