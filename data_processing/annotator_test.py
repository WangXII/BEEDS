''' Example retrieval for some phosphorylation events '''

import os
import sys

if os.name == 'nt':
    root_path = "/".join(os.path.realpath(__file__).split("\\")[:-2])
else:
    root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.append(root_path)

# import json

from data_processing.biopax_to_retrieval import IndraDataLoader
from data_processing.retrieve_and_annotate import annotate_example
from data_processing.datatypes import Question
from data_processing.dict_of_triggers import get_chebi_names
from elasticsearch import Elasticsearch

chebi_dict = get_chebi_names()
print(chebi_dict["16618"])

# with open(root_path + '/cache/retrieval_statistics', 'r') as handle:
#     retrieval_stats = json.load(handle)

# pairs_with_answer = 0
# pairs = 0
# for i, substrate in enumerate(retrieval_stats):
#     for number_answers in substrate["number_answer_docs"]:
#         if number_answers > 0:
#             pairs_with_answer += 1
#         pairs += 1
# print("Number Phoshorylation Control Pairs")
# print(pairs)
# print("Number Of Phoshorylation Control Pairs with Evidence")
# print(pairs_with_answer)
# print(phospho_dict.keys())

question_type = Question.PHOSPHORYLATION_COMPLEXCAUSE
family_member_bool = True
retrieval_size = 1000
bag_size = 100
es_index_name = "pubmed_detailed"
example_answer_index = 0
# substrate_condition = (i == len(event_dict.keys()) - 1)


def substrate_condition(x):
    # return True
    # return x[0] == "JAK2"
    return x == ("JAK2", "EPO")
# substrate_condition = (i == 8)


_, event_dict = IndraDataLoader.get_dataset(use_cache=True, mode="all", question_type=question_type)

breaking = False
es = Elasticsearch()

for substrate in event_dict.keys():
    found_answer = 0
    if substrate_condition(substrate):
        for i in range(len(event_dict[substrate])):
            found_answer += 1
            if found_answer <= example_answer_index:
                continue
            print("Question Subject")
            print(substrate)
            print("Question type")
            print(question_type)
            print()
            print("Corresponding Indra Events")
            print(event_dict[substrate][i])
            print()
            print("Event Infos")
            print(vars(event_dict[substrate][i]))
            print()
            _, unique_answer_agents = IndraDataLoader.get_unique_args_statements(substrate, event_dict[substrate], question_type)
            print("Answer Infos")
            print(unique_answer_agents)
            print()
            annotations, answer_stats = annotate_example(
                substrate, event_dict[substrate], question_type, es, retrieval_size, bag_size,
                index_name=es_index_name, index_name_two=es_index_name, benchmark=True)
            # print(answer_stats["number_answer_docs"])
            annotation_pubmed_ids = [pubmed_id[0] for pubmed_id in annotations[0]]
            print(annotation_pubmed_ids)
            # print(annotations[0][0])
            for pubmed_id in annotations[1]:
                if pubmed_id[0] == "22046448":
                    print(pubmed_id)
            print(len(annotations[0]))
            print(len(annotations[1]))
            print(len(annotations[2]))
            print(len(annotations))
            breaking = True
            if breaking:
                break
    if breaking:
        break
