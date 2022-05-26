''' Utility functions to convert neural network outputs back to INDRA statements for subsequent questions '''

import os
import sys

if os.name == 'nt':
    root_path = "/".join(os.path.realpath(__file__).split("\\")[:-2])
else:
    root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.append(root_path)

import indra.statements.statements as statements
import pandas as pd
import requests
import pickle

from mygene import MyGeneInfo
from tqdm import tqdm
from sqlalchemy import create_engine
from sqlitedict import SqliteDict
from unidecode import unidecode

from data_processing.dict_of_triggers import get_chebi_ids
from data_processing.datatypes import parse_residue_position_string, convert_gene_to_human_gene_id, updateAllNames, QUESTION_TO_BIOPAX_TYPE, Question
from data_processing.biopax_to_retrieval import IndraDataLoader
from configs import DICT_NORMALIZER, EVENT_SUBSTRATES_COMPLEX_MULTI, PUBMED_DOC_ANNOTATIONS_DB, GENE_INFO_DB, PUBMED_EVIDENCE_ANNOTATIONS_DB, TQDM_DISABLE

import logging
# import re
# import pickle

logger = logging.getLogger(__name__)

CHEBI_IDS = get_chebi_ids()

PUBTATOR_DOC_NORMALIZER = SqliteDict(PUBMED_DOC_ANNOTATIONS_DB, tablename='pubtator_normalizer', flag='r')

class NameToDBID:
    pubmed_engine = create_engine(PUBMED_EVIDENCE_ANNOTATIONS_DB)
    mg = MyGeneInfo()
    mg_cache = {}

    @classmethod
    def get_db_xrefs(cls, answer_list, pubmed_id, use_simple_normalizer=False):
        # Try to map to human gene ids (NCBI Tax 9606)
        new_answer_list = []
        for answer in answer_list:
            question_type = Question(answer[1])
            answer_lower = answer[0].lower()
            if question_type.name.endswith("SITE"):
                residue, position = parse_residue_position_string(answer_lower)
                if residue == "":
                    db_ids = [("##", "##")]
                else:
                    db_ids = [("SITE", residue + position)]
            else:
                sql_query = 'SELECT * FROM pubmed_annotations WHERE PubMedID = {} AND "Start Position" = {} AND "End Position" = {} LIMIT 1'.format(
                    pubmed_id, answer[4], answer[5])
                df = pd.read_sql_query(sql_query, cls.pubmed_engine)
                pubtator_doc_key = str(pubmed_id) + "_" + answer_lower
                special_symbols = " !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~'"
                if len(df) > 0 and df["Type"][0] == "Gene":
                    db_ids = [("EGID", convert_gene_to_human_gene_id(db_id)) for db_id in df["DB Identifier"][0].split(";")]
                elif pubtator_doc_key in PUBTATOR_DOC_NORMALIZER:
                    db_ids = [("EGID", convert_gene_to_human_gene_id(db_id[1])) for db_id in PUBTATOR_DOC_NORMALIZER[pubtator_doc_key] if db_id[1].isdigit()]
                elif answer_lower in CHEBI_IDS and answer_lower[0] not in special_symbols:  # Check CheBI
                    db_ids = [("CHEBI", db_id) for db_id in CHEBI_IDS[answer_lower]]
                elif answer_lower in cls.mg_cache:  # Check MyGeneInfo Cache
                    db_ids = [("EGID", cls.mg_cache[answer_lower])]
                else:
                    # TODO Uncomment if querying MyGeneInfo offers good performance
                    # try:  # Try MyGeneInfo
                    #     answer_curated = unidecode(answer_lower.replace("/", " ").replace("+", "").replace("(", "").replace(")", "").replace("-", "")
                    #                                            .replace("[", "").replace("]", "").replace("{", "").replace("}", "").lower())
                    #     res = cls.mg.query(answer_curated, size=1, fields='entrezgene', species='human')['hits']
                    # except requests.exceptions.HTTPError:
                    #     raise ValueError("Invalid string: %s", answer_lower)
                    # if not res:  # Try other species
                    #     res = cls.mg.query(answer_curated, size=1, fields='entrezgene')['hits']
                    # if res and 'entrezgene' in res[0]:
                    #     cls.mg_cache[answer_lower] = res[0]['entrezgene']
                    #     db_ids = [("EGID", res[0]['entrezgene'])]
                    # elif answer_lower in DICT_NORMALIZER and use_simple_normalizer and answer_lower[0] not in special_symbols:
                    if answer_lower in DICT_NORMALIZER and use_simple_normalizer and answer_lower[0] not in special_symbols:
                        db_ids = [("EGID", db_id) for db_id in DICT_NORMALIZER[answer_lower]]
                    else:
                        db_ids = [("##", "##")]
            new_answer_list.append(db_ids)
        return new_answer_list

def make_indra_statements(answer_list_with_db_refs):
    gene_info_engine = create_engine(GENE_INFO_DB)
    with open(EVENT_SUBSTRATES_COMPLEX_MULTI, 'rb') as handle:
        complex_pair_substrates = pickle.load(handle)
    indra_statements = {}

    if TQDM_DISABLE:
        logger.info("Iterate over PMID to create INDRA statements")
    for _, pmid in tqdm(answer_list_with_db_refs.items(), desc="Iterate over PMID to create INDRA statements", disable=TQDM_DISABLE):
        for doc in pmid:
            # logger.info(doc)
            # logger.info(doc[0])
            # input("Continue with any key ")
            # logger.info(doc[0].tolist())
            question_type_str = doc[0][0]
            question_type = QUESTION_TO_BIOPAX_TYPE[question_type_str]
            indra_statements.setdefault(question_type_str, {})
            if question_type is None:
                continue
            subject_list = doc[0][1:]
            answer_list = doc[1]
            dict_key = []
            indra_agents = []
            first_subject_name = ""
            for i, subject in enumerate(subject_list):
                # Example Subject TBC1D4_SUBSTRATE_EGID_9882
                subject_parts = subject.split("_")
                name = subject_parts[0]
                all_names = set()
                all_names.add(name)
                if i == 0:
                    first_subject_name = name
                all_names, _, db_id = updateAllNames(all_names, name, subject_parts[2], subject_parts[3], gene_info_engine)
                indra_agents.append(statements.Agent(name, list(all_names), db_refs=db_id))
                dict_key.append(name)

            if not question_type_str.endswith(("COMPLEX_PAIR", "COMPLEX_MULTI")):
                sub_attr, enz_attr = IndraDataLoader.get_attribute_strings(question_type)

            if question_type_str.endswith(("COMPLEX_PAIR", "COMPLEX_MULTI")) and (len(answer_list) > 0) and first_subject_name in complex_pair_substrates:
                answer_names = set()
                for answer in answer_list:
                    kwargs = {}
                    kwargs["members"] = []
                    for subject_agent in indra_agents:
                        kwargs["members"].append(subject_agent)
                    name = answer[0]
                    all_names = set()
                    all_names, name, db_id = updateAllNames(all_names, name, answer[7][0], answer[7][1], gene_info_engine)
                    if name not in answer_names:
                        answer_names.add(name)
                        indra_agent = statements.Agent(name, list(all_names), db_refs=db_id)
                        kwargs["members"].append(indra_agent)
                        dict_key_answer = tuple(sorted(dict_key + [name]))
                        indra_event = question_type(**kwargs)
                        indra_statements[question_type_str].setdefault(dict_key_answer, [])
                        indra_statements[question_type_str][dict_key_answer].append(indra_event)

            elif question_type_str.endswith("SITE") and (len(answer_list) > 0):
                # Parse Residue/Position String
                for answer in answer_list:
                    kwargs = {}
                    kwargs[sub_attr] = indra_agents[0]
                    if len(indra_agents) > 1:  # Build complexes for the COMPLEX question types
                        kwargs[enz_attr] = indra_agents[1]
                        indra_agents[1].bound_conditions = []
                        for i in range(2, len(indra_agents)):
                            indra_agents[1].bound_conditions.append(statements.BoundCondition(indra_agents[i]))
                    else:
                        generic_agent = statements.Agent("#PROTEIN", [])
                        kwargs[enz_attr] = generic_agent
                    residue, position = parse_residue_position_string(answer[0])
                    kwargs["residue"] = residue
                    kwargs["position"] = position
                    if residue != "":
                        indra_event = question_type(**kwargs)
                        dict_key_answer = tuple(dict_key)
                        if dict_key_answer not in indra_statements[question_type_str]:
                            indra_statements[question_type_str][dict_key_answer] = []
                        indra_statements[question_type_str][dict_key_answer].append(indra_event)

            elif (not question_type_str.endswith("SITE")) and (not question_type_str.endswith(("COMPLEX_PAIR", "COMPLEX_MULTI"))) and (len(answer_list) > 0):
                # logger.info(answer_list)
                answer_names = set()
                for answer in answer_list:
                    kwargs = {}
                    kwargs[sub_attr] = indra_agents[0]
                    name = answer[0]
                    all_names = set()
                    all_names, name, db_id = updateAllNames(all_names, name, answer[7][0], answer[7][1], gene_info_engine)
                    if name not in answer_names:
                        answer_names.add(name)
                        indra_agent = statements.Agent(name, list(all_names), db_refs=db_id)
                        if len(indra_agents) == 1:
                            kwargs[enz_attr] = indra_agent
                            kwargs[enz_attr].bound_conditions = []
                        else:  # Build complexes for the COMPLEX question types
                            kwargs[enz_attr] = indra_agents[1]
                            kwargs[enz_attr].bound_conditions = []
                            for i in range(2, len(indra_agents)):
                                kwargs[enz_attr].bound_conditions.append(statements.BoundCondition(indra_agents[i]))
                            kwargs[enz_attr].bound_conditions.append(statements.BoundCondition(indra_agent))
                        dict_key_answer = tuple(dict_key + [name])

                        indra_event = question_type(**kwargs)
                        # logger.info(" ")
                        # logger.info(indra_event)
                        # logger.info(question_type_str)
                        # logger.info(dict_key_answer)
                        indra_statements[question_type_str].setdefault(dict_key_answer, [])
                        indra_statements[question_type_str][dict_key_answer].append(indra_event)
                        # logger.info(indra_statements)
    # logger.warn(indra_statements)
    # input("Continue with any key ")

    return indra_statements


if __name__ == "__main__":
    print(list(CHEBI_IDS.items())[0:3])
