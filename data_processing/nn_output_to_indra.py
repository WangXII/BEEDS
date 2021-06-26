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

from tqdm import tqdm
from sqlalchemy import create_engine
from sqlitedict import SqliteDict
from data_processing.dict_of_triggers import get_chebi_ids, get_chebi_names
from data_processing.datatypes import parse_residue_position_string, convert_gene_to_human_gene_id, QUESTION_TO_BIOPAX_TYPE, Question
from data_processing.biopax_to_retrieval import IndraDataLoader
from util.build_ncbi_gene_id_db import get_gene_id_to_names
from configs import DICT_NORMALIZER, PUBMED_DOC_ANNOTATIONS_DB, GENE_INFO_DB

import logging
# import re
# import pickle

logger = logging.getLogger(__name__)

CHEBI_IDS = get_chebi_ids()
CHEBI_NAMES = get_chebi_names()
logger.info("Loading GENE_ID_TO_NAMES")
GENE_ID_TO_NAMES = get_gene_id_to_names()
PUBTATOR_DOC_NORMALIZER = SqliteDict(PUBMED_DOC_ANNOTATIONS_DB, tablename='pubtator_normalizer', flag='r')


def get_db_xrefs(answer_list, pubmed_id, pubmed_engine, use_simple_normalizer=False):
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
            df = pd.read_sql_query(sql_query, pubmed_engine)
            pubtator_doc_key = str(pubmed_id) + "_" + answer_lower
            special_symbols = " !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~'"
            if len(df) > 0 and df["Type"][0] == "Gene":
                db_ids = [("EGID", convert_gene_to_human_gene_id(db_id)) for db_id in df["DB Identifier"][0].split(";")]
            elif pubtator_doc_key in PUBTATOR_DOC_NORMALIZER:
                db_ids = [("EGID", convert_gene_to_human_gene_id(db_id[1])) for db_id in PUBTATOR_DOC_NORMALIZER[pubtator_doc_key] if db_id[1].isdigit()]
            elif answer_lower in DICT_NORMALIZER and use_simple_normalizer and answer_lower[0] not in special_symbols:
                db_ids = [("EGID", db_id) for db_id in DICT_NORMALIZER[answer_lower]]
            elif answer_lower in CHEBI_IDS and answer_lower[0] not in special_symbols:  # Check CheBI
                db_ids = [("CHEBI", db_id) for db_id in CHEBI_IDS[answer_lower]]
            else:
                db_ids = [("##", "##")]
        new_answer_list.append(db_ids)
    return new_answer_list


def updateAllNames(all_names, name, db, identifier, gene_info_engine):
    if db == "EGID" and identifier in GENE_ID_TO_NAMES:  # Gene Info NCBI
        # Alternative database
        # sql_query = 'SELECT Symbol, Synonyms, description, Symbol_from_nomenclature_authority, Full_name_from_nomenclature_authority, ' \
        #             'Other_designations FROM pubmed_annotations WHERE GeneID = "{}" LIMIT 1'.format(identifier)
        # df = pd.read_sql_query(sql_query, gene_info_engine)
        # for col in df.columns:
        #     if df[col][0] != "-" and col in ["Symbol", "description", "Symbol_from_nomenclature_authority", "Full_name_from_nomenclature_authority"]:
        #         all_names.add(df[col][0])
        #     elif df[col][0] != "-" and col in ["Synonyms", "Other_designations"]:
        #         all_names.update(df[col][0].split("|"))
        # name = df["Symbol"][0]
        all_names.update(GENE_ID_TO_NAMES[identifier][1])
        name = GENE_ID_TO_NAMES[identifier][0]
    elif db == "CHEBI" and identifier in CHEBI_NAMES:  # CheBI
        # identifier = re.sub("[^0-9]", "", identifier)
        all_names.update(CHEBI_NAMES[identifier])
        name = CHEBI_NAMES[identifier][0]

    # No matching DB ref
    if db == "##" or identifier == "##":
        db_id = {}
    else:
        db_id = {db: identifier}

    return all_names, name, db_id


def make_indra_statements(answer_list_with_db_refs):
    gene_info_engine = create_engine(GENE_INFO_DB)
    indra_statements = {}

    for _, pmid in tqdm(answer_list_with_db_refs.items(), desc="Iterate over PMID to create INDRA statements"):
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
            for i, subject in enumerate(subject_list):
                # Example Subject TBC1D4_SUBSTRATE_EGID_9882
                subject_parts = subject.split("_")
                name = subject_parts[0]
                all_names = set()
                all_names, _, db_id = updateAllNames(all_names, name, subject_parts[2], subject_parts[3], gene_info_engine)
                indra_agents.append(statements.Agent(name, list(all_names), db_refs=db_id))
                dict_key.append(name)

            sub_attr, enz_attr = IndraDataLoader.get_attribute_strings(question_type)

            if question_type_str.endswith("SITE") and (len(answer_list) > 0):
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

            elif (not question_type_str.endswith("SITE")) and (len(answer_list) > 0):
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
