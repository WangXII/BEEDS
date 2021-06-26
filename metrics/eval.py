''' Evaluation scripts for the distant supervision '''

import os
import sys

if os.name == 'nt':
    root_path = "/".join(os.path.realpath(__file__).split("\\")[:-2])
else:
    root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.append(root_path)

# from sklearn.metrics import average_precision_score
from data_processing.datatypes import Question
from configs import CACHE

import baseline.evex_baseline as baseline
import analysis.example_visualization as visualization

# import math
# import numpy as np
import logging
import pickle

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# TODO rework for all question types and error analysis
def compare_entities_between_modes():
    with open(CACHE + "/predictions_train.pickle", 'rb') as handle:
        kb_phospho_train = pickle.load(handle)

    with open(CACHE + "/predictions_eval.pickle", 'rb') as handle:
        kb_phospho_dev = pickle.load(handle)

    causes_train_truth = set()
    causes_train_predict = set()
    for substrate, causes in kb_phospho_train.items():
        causes_train_truth.update([cause for cause in causes[0]])
        causes_train_predict.update([cause for cause in causes[1]])

    causes_dev_truth = set()
    causes_dev_predict = set()
    for substrate, causes in kb_phospho_dev.items():
        causes_dev_truth.update([cause for cause in causes[0]])
        causes_dev_predict.update([cause for cause in causes[1]])

    # Train Truth/Predict vs Dev Truth/Predict
    logger.info("   Train Predict")
    logger.info(causes_train_predict)
    logger.info(len(causes_train_predict))
    logger.info("   Train Truth")
    logger.info(causes_train_truth)
    logger.info(len(causes_train_truth))
    logger.info("   Dev Predict")
    logger.info(causes_dev_predict)
    logger.info(len(causes_dev_predict))
    logger.info("   Dev Truth")
    logger.info(causes_dev_truth)
    logger.info(len(causes_dev_truth))

    logger.info("   Unique for Train Predict (vs Dev Truth)")
    logger.info(causes_train_predict - causes_dev_truth)
    logger.info(len(causes_train_predict - causes_dev_truth))
    logger.info("   Unique for Train Predict (vs Train Truth)")
    logger.info(causes_train_predict - causes_train_truth)
    logger.info(len(causes_train_predict - causes_train_truth))
    logger.info("   Unique for Train Truth (vs Train Predict)")
    logger.info(causes_train_truth - causes_train_predict)
    logger.info(len(causes_train_truth - causes_train_predict))
    logger.info("   Unique for Dev Truth (vs Train Predict)")
    logger.info(causes_dev_truth - causes_train_predict)
    logger.info(len(causes_dev_truth - causes_train_predict))
    logger.info("   Intersection Dev Truth and Train Predict")
    logger.info(causes_dev_truth.intersection(causes_train_predict))
    logger.info(len(causes_dev_truth.intersection(causes_train_predict)))

    logger.info("   Unique for Dev Predict (vs Train Truth)")
    logger.info(causes_dev_predict - causes_train_truth)
    logger.info(len(causes_dev_predict - causes_train_truth))
    logger.info("   Unique for Dev Predict (vs Train Predict)")
    logger.info(causes_dev_predict - causes_train_predict)
    logger.info(len(causes_dev_predict - causes_train_predict))
    logger.info("   Unique for Dev Predict (vs Train Predict) but found in Dev Truth")
    logger.info((causes_dev_predict - causes_train_predict).intersection(causes_dev_truth))
    logger.info(len((causes_dev_predict - causes_train_predict).intersection(causes_dev_truth)))
    logger.info("   Intersection Dev Predict and Train Predict")
    logger.info(causes_dev_predict.intersection(causes_train_predict))
    logger.info(len(causes_dev_predict.intersection(causes_train_predict)))


def convert_answers_from_docs_to_entities(dicts_by_pubmed_id, bool_normalized=True, bool_return_docs=False, debug_groundtruth=False):
    index = 7 if bool_normalized else 0  # used to access db_ids index in answers
    # Get groundtruth confidence scores (best possible)
    answer_dict = {}
    for dict_by_pubmed_id in dicts_by_pubmed_id:
        for doc_id, pubmed_content in dict_by_pubmed_id.items():
            for tuples in pubmed_content:
                doc_text = tuples[2]
                for answer in tuples[1]:
                    subject = tuple(tuples[0])
                    if subject not in answer_dict:
                        answer_dict[subject] = {}
                    if debug_groundtruth is True:
                        prob_score = 1.0
                    else:
                        prob_score = answer[8]
                    object_id = answer[index]
                    if not bool_return_docs:
                        if object_id not in answer_dict[subject]:
                            answer_dict[subject][object_id] = [0, 0]
                        if prob_score > answer_dict[subject][object_id][0]:
                            answer_dict[subject][object_id][0] = prob_score
                        answer_dict[subject][object_id][1] += 1
                    else:
                        answer_dict[subject].setdefault(object_id, {})
                        answer_dict[subject][object_id].setdefault(doc_id, [[], []])
                        answer_dict[subject][object_id][doc_id][0].append(answer[0])
                        answer_dict[subject][object_id][doc_id][1].append((prob_score, doc_text))

    if bool_return_docs:
        return answer_dict

    # Use same average precision calculation like the Evex baseline
    # TODO: Adjust for different question types
    entity_dict = {}
    for theme in answer_dict.keys():
        # logger.warn(theme)
        # input("Continue")
        if theme[0] in Question.__members__:
            question_type = str(theme[0])
            # DEPHOSPHORYLATION to PHOSPHORYLATION for evaluation purposes
            if question_type.startswith("DE"):
                question_type = question_type[2:]
            elif question_type.startswith("INHIBEXPRESSION"):
                question_type = question_type[5:]
            elif question_type.startswith("ACTIVATION"):
                question_type = "STATECHANGE" + "_" + question_type.split("_")[-1]
            elif question_type.startswith("INHIBITION"):
                question_type = "STATECHANGE" + "_" + question_type.split("_")[-1]
            question_types = [question_type]
            if not question_type.startswith("STATECHANGE") and not question_type.endswith("SITE"):
                question_types.append("STATECHANGE" + "_" + question_type.split("_")[-1])
            for current_question_type in question_types:
                if current_question_type not in entity_dict:
                    entity_dict[current_question_type] = {}
            theme_ids = []
            bool_numeric = True
            for theme_subject in theme[1:]:
                theme_id = theme_subject.split("_")[-1]
                if theme_id.isnumeric():
                    theme_id = ("EGID", theme_id)
                elif theme_id != "##":
                    theme_id = ("CHEBI", theme_id)
                elif theme_id == "##":
                    theme_id = ("##", theme_id)
                    bool_numeric = False
                else:
                    raise ValueError("Unexpected theme id {}".format(theme_id))
                theme_ids.append(theme_id)
            if len(theme_ids) == 1:
                theme_ids = theme_ids[0]
            elif len(theme_ids) > 1:
                theme_ids = tuple(theme_ids)
            if bool_numeric:
                for cause, confidence in answer_dict[theme].items():
                    if cause[0] in ["EGID", "CHEBI", "SITE"]:
                        for current_question_type in question_types:
                            theme_in_dict = entity_dict[current_question_type].setdefault(theme_ids, {})
                            if cause not in theme_in_dict:
                                theme_in_dict[cause] = [0, 0]
                            if theme_in_dict[cause][0] < confidence[0]:
                                theme_in_dict[cause][0] = confidence[0]
                            theme_in_dict[cause][1] = confidence[1]

    return entity_dict, answer_dict


def visualize(groundtruth, predictions, indra_event_dict=None, top=True):
    visualizer = visualization.ExampleVisualizer(groundtruth=groundtruth, predictions=predictions, indra_event_dict=indra_event_dict,
                                                 groundtruth_type="indra")
    # Table with occurrences of all entities
    # visualizer.visualize_entities()
    question_subject = None
    # question_subject = ('PHOSPHORYLATION_CAUSE', 'MAPK8_SUBSTRATE_EGID_5599')
    # visualizer.visualize_entity(question_subject)
    if top:
        # logger.info("All predictions")
        # visualizer.visualize_predictions(mode="all")
        # logger.info("True positives")
        # visualizer.visualize_predictions(mode="true_positives", question_subject=question_subject)
        logger.info("False positives - Random (Analysis of new predictions)", )
        visualizer.visualize_predictions(mode="false_positives", number=50, sampling_method="top", question_subject=question_subject, evidence_preds=False)
        # logger.info("False negatives")
        # visualizer.visualize_predictions(mode="false_negatives")
        logger.info(" ===================================================================== ")
        logger.info("All - Random (Estimation of Precision) ")
        visualizer.visualize_predictions(mode="all", number=250, sampling_method="random", question_subject=question_subject, evidence_preds=False)
    else:
        groundtruth_subjects, predictions_subjects = visualizer.get_entity_questions()
        logger.info("Groundtruth Subjects")
        logger.info(groundtruth_subjects)
        logger.info("Prediction Subjects")
        logger.info(predictions_subjects)
        visualizer.visualize_entity(predictions_subjects[0])


def get_average_precision(groundtruth, predictions, mode, use_db_ids=False, visualize_bool=False):
    ''' Calculate average precision for Query and its answers. One score for one question type. '''

    groundtruth_dict, _ = convert_answers_from_docs_to_entities(groundtruth, use_db_ids, debug_groundtruth=True)
    prediction_dict, _ = convert_answers_from_docs_to_entities(predictions, use_db_ids)

    kb_evaluator = baseline.KnowledgeBaseEvaluator(groundtruth_dict, prediction_dict, mode=mode, standoff=True)
    question_types = list(groundtruth_dict.keys())
    if len(question_types) == 0:
        question_types = list(prediction_dict.keys())
    if len(question_types) > 0:  # Result for all questions together
        question_types.append(" All")
        question_types.append(" Simple")
        question_types.append(" Complex")

    logger.info(" Analysis of Predictions ")
    if len(predictions) > 0 and visualize_bool:
        visualize(groundtruth, predictions, kb_evaluator.indra_dict_verbose)

    logger.info("____")
    logger.info("    **** Best possible results (Groundtruth INDRA vs Groundtruth Annotations)")
    for question_type in question_types:
        logger.info("Question type: {}".format(question_type))
        logger.info("Groundtruth INDRA vs Groundtruth Annotations Results")
        kb_evaluator.calculate_average_precision(kb_evaluator.indra_dict, kb_evaluator.model_groundtruth_dict, question_type)

    logger.info("____")
    logger.info("    **** Best possible results (Groundtruth Annotations vs Predictions Annotations)")
    for question_type in question_types:
        logger.info("Question type: {}".format(question_type))
        logger.info("Groundtruth Annotations vs Predictions Annotations Results")
        kb_evaluator.calculate_average_precision(kb_evaluator.model_indra_dict, kb_evaluator.model_predictions_dict, question_type)

    logger.info("____")
    logger.info("    **** EVEX results (Groundtruth Annotations vs EVEX Annotations)")
    for question_type in question_types:
        logger.info("Question type: {}".format(question_type))
        logger.info("Groundtruth Annotations vs EVEX Annotations Results")
        kb_evaluator.calculate_average_precision(kb_evaluator.model_indra_dict, kb_evaluator.evex_dict, question_type,
                                                 optional_dict=kb_evaluator.model_predictions_dict)

    logger.info("____")
    logger.info("    **** Prediction results (Groundtruth INDRA vs Predictions Annotations)")
    for question_type in question_types:
        logger.info("Question type: {}".format(question_type))
        logger.info("Groundtruth INDRA vs Predictions Annotations Results")
        yield kb_evaluator.calculate_average_precision(kb_evaluator.indra_dict, kb_evaluator.model_predictions_dict, question_type)


if __name__ == "__main__":
    import numpy as np
    from configs import NN_CACHE
    # compare_entities_between_modes()
    npzcache = np.load(NN_CACHE + "/processed_output_depth__10.npz", allow_pickle=True)
    groundtruth, predictions, _, _, _ = [npzcache[array] for array in npzcache.files]
    groundtruth_by_pubmed_id = groundtruth.item()
    predictions_by_pubmed_id = predictions.item()
    groundtruth_by_entities, _ = convert_answers_from_docs_to_entities([groundtruth_by_pubmed_id], debug_groundtruth=True)
    predictions_by_entities, _ = convert_answers_from_docs_to_entities([predictions_by_pubmed_id])
    logger.info(groundtruth_by_entities.keys())
    dictionary = groundtruth_by_entities['PHOSPHORYLATION_COMPLEXCAUSE']
    sample = {k: dictionary[k] for k in list(dictionary)[:2]}
    logger.info(sample)
