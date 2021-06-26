""" Evaluate Retrieval performance. """

import os
import sys

if os.name == 'nt':
    root_path = "/".join(os.path.realpath(__file__).split("\\")[:-2])
else:
    root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.append(root_path)

from typing import Dict, List, Tuple

import numpy as np

from data_processing.biopax_to_retrieval import IndraDataLoader
from data_processing.retrieve_and_annotate import get_subject_list
from data_processing.datatypes import Question, get_db_id
from configs import NN_CACHE
from metrics.eval import convert_answers_from_docs_to_entities


class RetrievalEvaluater():

    def __init__(self, cached_file: str = NN_CACHE + "/processed_output_depth_0.npz",
                 pid_mode: str = "eval", pid_questions: Question = Question.PHOSPHORYLATION_CAUSE):
        self.cached_file = cached_file
        self.pid_data = IndraDataLoader.get_dataset(mode=pid_mode, question_type=pid_questions)
        self._load_cached_file()

    def _load_cached_file(self):
        npzcache = np.load(self.cached_file, allow_pickle=True)
        groundtruth, _, _, _, _ = [npzcache[array] for array in npzcache.files]
        self.groundtruth_by_pubmed_id = groundtruth.item()
        self.groundtruth_by_entities = convert_answers_from_docs_to_entities(self.groundtruth_by_pubmed_id, bool_return_docs=True)

    def create_event_dict(self, question_type=Question.PHOSPHORYLATION_CAUSE) -> Dict:
        self.event_dict: Dict = {}

        # PID knowledge base
        _, event_dict = IndraDataLoader.get_dataset(mode="eval", question_type=question_type)
        for subject, indra_statements in event_dict.items():
            _, object_agents = IndraDataLoader.get_unique_args_statements(subject, indra_statements, question_type)
            all_agents = IndraDataLoader.get_all_indra_agents(indra_statements[0], subject)
            question_type_and_subject = tuple(get_subject_list(question_type.name, all_agents))
            self.event_dict.setdefault(question_type_and_subject, [[], []])
            object_ids = sorted(set([get_db_id(object_agent) for object_agent in object_agents]), key=lambda x: x[1])
            self.event_dict[question_type_and_subject][0] = object_ids

        # Retrieval results
        for question_type_and_subject, object_infos in self.groundtruth_by_entities.items():
            if question_type.name != question_type_and_subject[0]:
                continue
            object_ids = sorted(self.groundtruth_by_entities[question_type_and_subject].keys())
            self.event_dict.setdefault(question_type_and_subject, [[], []])
            self.event_dict[question_type_and_subject][1] = sorted(set(object_ids), key=lambda x: x[1])

        return self.event_dict

    def get_event_dict(self):
        for subject, objects in self.event_dict.items():
            print(subject)
            print(objects[0])
            print(objects[1])
            print()

    def get_entity_keys(self) -> Tuple[List]:
        return self.groundtruth_by_entities.keys()


if __name__ == "__main__":
    evaluator = RetrievalEvaluater()
    groundtruth_subjects = evaluator.get_entity_keys()
    print(groundtruth_subjects)
    evaluator.create_event_dict()
    evaluator.get_event_dict()
    # visualizer.visualize_entities(('PHOSPHORYLATION_CAUSE', 'RBBP8_SUBSTRATE_EGID_5932'))
    # visualizer.visualize_entities(('PHOSPHORYLATION_CAUSE', 'GSK3B_SUBSTRATE_EGID_2932'))
    # visualizer.visualize_entities(('PHOSPHORYLATION_CAUSE', 'MAP2K1_SUBSTRATE_EGID_5604'))
