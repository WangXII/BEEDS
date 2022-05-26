''' Collection of utility constants and datatypes used across the project '''

from __future__ import absolute_import, division, print_function
import os
import sys

if os.name == 'nt':
    root_path = "/".join(os.path.realpath(__file__).split("\\")[:-2])
else:
    root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.append(root_path)

import copy
import re
import pandas as pd
from enum import Enum
from sqlalchemy import create_engine

import indra.statements.statements as statements

from configs import GENE_INFO_DB
from data_processing.dict_of_triggers import TRIGGERS, AMINO_ACIDS, AMINO_ACIDS_ABBREVIATIONS, AMINO_ACIDS_STRINGS, get_chebi_names
from util.build_ncbi_gene_id_db import get_gene_id_to_names

import logging

logger = logging.getLogger(__name__)

CHEBI_NAMES = get_chebi_names()
GENE_INFO_ENGINE = create_engine(GENE_INFO_DB)

LABELS = ['O', 'X', 'B', 'I']
PAD_TOKEN_LABEL_ID = -1
CHEBI_DICT = get_chebi_names()


class Question(Enum):
    # Modifications
    PHOSPHORYLATION_CAUSE = 0
    PHOSPHORYLATION_COMPLEXCAUSE = 1
    PHOSPHORYLATION_SITE = 2
    DEPHOSPHORYLATION_CAUSE = 3
    DEPHOSPHORYLATION_COMPLEXCAUSE = 4
    DEPHOSPHORYLATION_SITE = 5

    ACETYLATION_CAUSE = 6
    ACETYLATION_COMPLEXCAUSE = 7
    ACETYLATION_SITE = 8
    DEACETYLATION_CAUSE = 9
    DEACETYLATION_COMPLEXCAUSE = 10
    DEACETYLATION_SITE = 11

    HYDROXYLATION_CAUSE = 12
    HYDROXYLATION_COMPLEXCAUSE = 13
    HYDROXYLATION_SITE = 14
    DEHYDROXYLATION_CAUSE = 15
    DEHYDROXYLATION_COMPLEXCAUSE = 16
    DEHYDROXYLATION_SITE = 17

    UBIQUITINATION_CAUSE = 18
    UBIQUITINATION_COMPLEXCAUSE = 19
    UBIQUITINATION_SITE = 20
    DEUBIQUITINATION_CAUSE = 21
    DEUBIQUITINATION_COMPLEXCAUSE = 22
    DEUBIQUITINATION_SITE = 23

    GLYCOSYLATION_CAUSE = 24
    GLYCOSYLATION_COMPLEXCAUSE = 25
    GLYCOSYLATION_SITE = 26
    DEGLYCOSYLATION_CAUSE = 27
    DEGLYCOSYLATION_COMPLEXCAUSE = 28
    DEGLYCOSYLATION_SITE = 29

    PALMITOYLATION_CAUSE = 30
    PALMITOYLATION_COMPLEXCAUSE = 31
    PALMITOYLATION_SITE = 32
    DEPALMITOYLATION_CAUSE = 33
    DEPALMITOYLATION_COMPLEXCAUSE = 34
    DEPALMITOYLATION_SITE = 35

    SUMOYLATION_CAUSE = 36
    SUMOYLATION_COMPLEXCAUSE = 37
    SUMOYLATION_SITE = 38
    DESUMOYLATION_CAUSE = 39
    DESUMOYLATION_COMPLEXCAUSE = 40
    DESUMOYLATION_SITE = 41

    # Other
    EXPRESSION_CAUSE = 42
    EXPRESSION_COMPLEXCAUSE = 43
    COMPLEX_PAIR = 44
    COMPLEX_MULTI = 74
    TRANSPORT_CAUSE = 45  # TODO: Implement in INDRA
    TRANSPORT_COMPLEXCAUSE = 46  # TODO: Implement in INDRA
    TRANSPORT_FROM = 47  # TODO: Implement in INDRA
    TRANSPORT_TO = 48  # TODO: Implement in INDRA
    ACTIVATION_CAUSE = 49
    ACTIVATION_COMPLEXCAUSE = 50
    INHIBITION_CAUSE = 51
    INHIBITION_COMPLEXCAUSE = 52
    INHIBEXPRESSION_CAUSE = 53
    GEF_CAUSE = 54
    GAP_CAUSE = 55
    CONVERSION_PRODUCT = 56  # TODO: To be implemented in biopax_to_retrieval.py

    # Comples Site Questions
    PHOSPHORYLATION_COMPLEXSITE = 57
    DEPHOSPHORYLATION_COMPLEXSITE = 58
    ACETYLATION_COMPLEXSITE = 59
    DEACETYLATION_COMPLEXSITE = 60
    HYDROXYLATION_COMPLEXSITE = 61
    DEHYDROXYLATION_COMPLEXSITE = 62
    UBIQUITINATION_COMPLEXSITE = 63
    DEUBIQUITINATION_COMPLEXSITE = 64
    GLYCOSYLATION_COMPLEXSITE = 65
    DEGLYCOSYLATION_COMPLEXSITE = 66
    PALMITOYLATION_COMPLEXSITE = 67
    DEPALMITOYLATION_COMPLEXSITE = 68
    SUMOYLATION_COMPLEXSITE = 69
    DESUMOYLATION_COMPLEXSITE = 70

    # Other Question Types
    INHIBEXPRESSION_COMPLEXCAUSE = 71
    STATECHANGE_CAUSE = 72
    STATECHANGE_COMPLEXCAUSE = 73


TRIGGER_WORDS = {
    "PHOSPHORYLATION_CAUSE": TRIGGERS["Phosphorylation"],
    "PHOSPHORYLATION_COMPLEXCAUSE": TRIGGERS["Phosphorylation"],
    "PHOSPHORYLATION_SITE": TRIGGERS["Phosphorylation"],
    "PHOSPHORYLATION_COMPLEXSITE": TRIGGERS["Phosphorylation"],
    "DEPHOSPHORYLATION_CAUSE": TRIGGERS["Dephosphorylation"],
    "DEPHOSPHORYLATION_COMPLEXCAUSE": TRIGGERS["Dephosphorylation"],
    "DEPHOSPHORYLATION_SITE": TRIGGERS["Dephosphorylation"],
    "DEPHOSPHORYLATION_COMPLEXSITE": TRIGGERS["Dephosphorylation"],
    "ACETYLATION_CAUSE": TRIGGERS["Acetylation"],
    "ACETYLATION_COMPLEXCAUSE": TRIGGERS["Acetylation"],
    "ACETYLATION_SITE": TRIGGERS["Acetylation"],
    "ACETYLATION_COMPLEXSITE": TRIGGERS["Acetylation"],
    "DEACETYLATION_CAUSE": TRIGGERS["Deacetylation"],
    "DEACETYLATION_COMPLEXCAUSE": TRIGGERS["Deacetylation"],
    "DEACETYLATION_SITE": TRIGGERS["Deacetylation"],
    "DEACETYLATION_COMPLEXSITE": TRIGGERS["Deacetylation"],
    "HYDROXYLATION_CAUSE": TRIGGERS["Hydroxylation"],
    "HYDROXYLATION_COMPLEXCAUSE": TRIGGERS["Hydroxylation"],
    "HYDROXYLATION_SITE": TRIGGERS["Hydroxylation"],
    "HYDROXYLATION_COMPLEXSITE": TRIGGERS["Hydroxylation"],
    "DEHYDROXYLATION_CAUSE": TRIGGERS["Dehydroxylation"],
    "DEHYDROXYLATION_COMPLEXCAUSE": TRIGGERS["Dehydroxylation"],
    "DEHYDROXYLATION_SITE": TRIGGERS["Dehydroxylation"],
    "DEHYDROXYLATION_COMPLEXSITE": TRIGGERS["Dehydroxylation"],
    "UBIQUITINATION_CAUSE": TRIGGERS["Ubiquitination"],
    "UBIQUITINATION_COMPLEXCAUSE": TRIGGERS["Ubiquitination"],
    "UBIQUITINATION_SITE": TRIGGERS["Ubiquitination"],
    "UBIQUITINATION_COMPLEXSITE": TRIGGERS["Ubiquitination"],
    "DEUBIQUITINATION_CAUSE": TRIGGERS["Deubiquitination"],
    "DEUBIQUITINATION_COMPLEXCAUSE": TRIGGERS["Deubiquitination"],
    "DEUBIQUITINATION_SITE": TRIGGERS["Deubiquitination"],
    "DEUBIQUITINATION_COMPLEXSITE": TRIGGERS["Deubiquitination"],
    "GLYCOSYLATION_CAUSE": TRIGGERS["Glycosylation"],
    "GLYCOSYLATION_COMPLEXCAUSE": TRIGGERS["Glycosylation"],
    "GLYCOSYLATION_SITE": TRIGGERS["Glycosylation"],
    "GLYCOSYLATION_COMPLEXSITE": TRIGGERS["Glycosylation"],
    "DEGLYCOSYLATION_CAUSE": TRIGGERS["Deglycosylation"],
    "DEGLYCOSYLATION_COMPLEXCAUSE": TRIGGERS["Deglycosylation"],
    "DEGLYCOSYLATION_SITE": TRIGGERS["Deglycosylation"],
    "DEGLYCOSYLATION_COMPLEXSITE": TRIGGERS["Deglycosylation"],
    "PALMITOYLATION_CAUSE": TRIGGERS["Palmitoylation"],
    "PALMITOYLATION_COMPLEXCAUSE": TRIGGERS["Palmitoylation"],
    "PALMITOYLATION_SITE": TRIGGERS["Palmitoylation"],
    "PALMITOYLATION_COMPLEXSITE": TRIGGERS["Palmitoylation"],
    "DEPALMITOYLATION_CAUSE": TRIGGERS["Depalmitoylation"],
    "DEPALMITOYLATION_COMPLEXCAUSE": TRIGGERS["Depalmitoylation"],
    "DEPALMITOYLATION_SITE": TRIGGERS["Depalmitoylation"],
    "DEPALMITOYLATION_COMPLEXSITE": TRIGGERS["Depalmitoylation"],
    "SUMOYLATION_CAUSE": TRIGGERS["Sumoylation"],
    "SUMOYLATION_COMPLEXCAUSE": TRIGGERS["Sumoylation"],
    "SUMOYLATION_SITE": TRIGGERS["Sumoylation"],
    "SUMOYLATION_COMPLEXSITE": TRIGGERS["Sumoylation"],
    "DESUMOYLATION_CAUSE": TRIGGERS["Desumoylation"],
    "DESUMOYLATION_COMPLEXCAUSE": TRIGGERS["Desumoylation"],
    "DESUMOYLATION_SITE": TRIGGERS["Desumoylation"],
    "DESUMOYLATION_COMPLEXSITE": TRIGGERS["Desumoylation"],
    "EXPRESSION_CAUSE": TRIGGERS["Gene_expression"],
    "EXPRESSION_COMPLEXCAUSE": TRIGGERS["Gene_expression"],
    "INHIBEXPRESSION_CAUSE": TRIGGERS["Gene_expression"],
    "INHIBEXPRESSION_COMPLEXCAUSE": TRIGGERS["Gene_expression"],
    "COMPLEX_PAIR": TRIGGERS["Binding"],
    "TRANSPORT_CAUSE": TRIGGERS["Transport"],
    "TRANSPORT_COMPLEXCAUSE": TRIGGERS["Transport"],
    "TRANSPORT_FROM": TRIGGERS["Transport"],
    "TRANSPORT_TO": TRIGGERS["Transport"],
    "ACTIVATION_CAUSE": TRIGGERS["Positive_regulation"],
    "ACTIVATION_COMPLEXCAUSE": TRIGGERS["Positive_regulation"],
    "INHIBITION_CAUSE": TRIGGERS["Negative_regulation"],
    "INHIBITION_COMPLEXCAUSE": TRIGGERS["Negative_regulation"],
    "STATECHANGE_CAUSE": TRIGGERS["Positive_regulation"] + TRIGGERS["Negative_regulation"] + TRIGGERS["Regulation"],
    "STATECHANGE_COMPLEXCAUSE": TRIGGERS["Positive_regulation"] + TRIGGERS["Negative_regulation"] + TRIGGERS["Regulation"],
    "COMPLEX_MULTI": TRIGGERS["Binding"]
}


QUESTION_TO_BIOPAX_TYPE = {
    "PHOSPHORYLATION_CAUSE": statements.Phosphorylation,
    "PHOSPHORYLATION_COMPLEXCAUSE": statements.Phosphorylation,
    "PHOSPHORYLATION_SITE": statements.Phosphorylation,
    "PHOSPHORYLATION_COMPLEXSITE": statements.Phosphorylation,
    "DEPHOSPHORYLATION_CAUSE": statements.Dephosphorylation,
    "DEPHOSPHORYLATION_COMPLEXCAUSE": statements.Dephosphorylation,
    "DEPHOSPHORYLATION_SITE": statements.Dephosphorylation,
    "DEPHOSPHORYLATION_COMPLEXSITE": statements.Dephosphorylation,
    "ACETYLATION_CAUSE": statements.Acetylation,
    "ACETYLATION_COMPLEXCAUSE": statements.Acetylation,
    "ACETYLATION_SITE": statements.Acetylation,
    "ACETYLATION_COMPLEXSITE": statements.Acetylation,
    "DEACETYLATION_CAUSE": statements.Deacetylation,
    "DEACETYLATION_COMPLEXCAUSE": statements.Deacetylation,
    "DEACETYLATION_SITE": statements.Deacetylation,
    "DEACETYLATION_COMPLEXSITE": statements.Deacetylation,
    "HYDROXYLATION_CAUSE": statements.Hydroxylation,
    "HYDROXYLATION_COMPLEXCAUSE": statements.Hydroxylation,
    "HYDROXYLATION_SITE": statements.Hydroxylation,
    "HYDROXYLATION_COMPLEXSITE": statements.Hydroxylation,
    "DEHYDROXYLATION_CAUSE": statements.Dehydroxylation,
    "DEHYDROXYLATION_COMPLEXCAUSE": statements.Dehydroxylation,
    "DEHYDROXYLATION_SITE": statements.Dehydroxylation,
    "DEHYDROXYLATION_COMPLEXSITE": statements.Dehydroxylation,
    "UBIQUITINATION_CAUSE": statements.Ubiquitination,
    "UBIQUITINATION_COMPLEXCAUSE": statements.Ubiquitination,
    "UBIQUITINATION_SITE": statements.Ubiquitination,
    "UBIQUITINATION_COMPLEXSITE": statements.Ubiquitination,
    "DEUBIQUITINATION_CAUSE": statements.Deubiquitination,
    "DEUBIQUITINATION_COMPLEXCAUSE": statements.Deubiquitination,
    "DEUBIQUITINATION_SITE": statements.Deubiquitination,
    "DEUBIQUITINATION_COMPLEXSITE": statements.Deubiquitination,
    "GLYCOSYLATION_CAUSE": statements.Glycosylation,
    "GLYCOSYLATION_COMPLEXCAUSE": statements.Glycosylation,
    "GLYCOSYLATION_SITE": statements.Glycosylation,
    "GLYCOSYLATION_COMPLEXSITE": statements.Glycosylation,
    "DEGLYCOSYLATION_CAUSE": statements.Deglycosylation,
    "DEGLYCOSYLATION_COMPLEXCAUSE": statements.Deglycosylation,
    "DEGLYCOSYLATION_SITE": statements.Deglycosylation,
    "DEGLYCOSYLATION_COMPLEXSITE": statements.Deglycosylation,
    "PALMITOYLATION_CAUSE": statements.Palmitoylation,
    "PALMITOYLATION_COMPLEXCAUSE": statements.Palmitoylation,
    "PALMITOYLATION_SITE": statements.Palmitoylation,
    "PALMITOYLATION_COMPLEXSITE": statements.Palmitoylation,
    "DEPALMITOYLATION_CAUSE": statements.Depalmitoylation,
    "DEPALMITOYLATION_COMPLEXCAUSE": statements.Depalmitoylation,
    "DEPALMITOYLATION_SITE": statements.Depalmitoylation,
    "DEPALMITOYLATION_COMPLEXSITE": statements.Depalmitoylation,
    "SUMOYLATION_CAUSE": statements.Sumoylation,
    "SUMOYLATION_COMPLEXCAUSE": statements.Sumoylation,
    "SUMOYLATION_SITE": statements.Sumoylation,
    "SUMOYLATION_COMPLEXSITE": statements.Sumoylation,
    "DESUMOYLATION_CAUSE": statements.Desumoylation,
    "DESUMOYLATION_COMPLEXCAUSE": statements.Desumoylation,
    "DESUMOYLATION_SITE": statements.Desumoylation,
    "DESUMOYLATION_COMPLEXSITE": statements.Desumoylation,
    "EXPRESSION_CAUSE": statements.IncreaseAmount,
    "EXPRESSION_COMPLEXCAUSE": statements.IncreaseAmount,
    "INHIBEXPRESSION_CAUSE": statements.DecreaseAmount,
    "INHIBEXPRESSION_COMPLEXCAUSE": statements.DecreaseAmount,
    "COMPLEX_PAIR": statements.Complex,
    # "TRANSPORT_CAUSE": TRIGGERS["Transport"],
    # "TRANSPORT_COMPLEXCAUSE": TRIGGERS["Transport"],
    # "TRANSPORT_FROM": TRIGGERS["Transport"],
    # "TRANSPORT_TO": TRIGGERS["Transport"],
    "ACTIVATION_CAUSE": statements.Activation,
    "ACTIVATION_COMPLEXCAUSE": statements.Activation,
    "INHIBITION_CAUSE": statements.Inhibition,
    "INHIBITION_COMPLEXCAUSE": statements.Inhibition,
    "GEF_CAUSE": statements.Gef,
    "GAP_CAUSE": statements.Gap,
    # "CONVERSION_PRODUCT": statements.Conversion,
    "STATECHANGE_CAUSE": statements.Phosphorylation,
    "STATECHANGE_COMPLEXCAUSE": statements.Phosphorylation,
    "COMPLEX_MULTI": statements.Complex,
}


QUESTION_TYPES = [0, 2, 57, 3, 5, 58, 6, 8, 59, 9, 11, 60, 18, 20, 63, 21, 23, 64, 42, 53, 49, 51, 72, 44, 74]
QUESTION_TYPES_EVEX = [0, 2, 57, 6, 8, 59, 18, 20, 63, 42, 72, 44, 74]
COMPLEX_TYPES_MAPPING = {57: 0, 58: 3, 59: 6, 60: 9, 63: 18, 64: 21, 74: 44}
STATECHANGE_BIOPAX_TYPES = [
    statements.Phosphorylation, statements.Ubiquitination, statements.Dephosphorylation, statements.Acetylation,
    statements.Deubiquitination, statements.Methylation, statements.Sumoylation, statements.Deacetylation,
    statements.Demethylation, statements.Desumoylation, statements.Hydroxylation, statements.Palmitoylation,
    statements.Glycosylation, statements.Deglycosylation, statements.Activation, statements.Inhibition,
    statements.IncreaseAmount, statements.DecreaseAmount, statements.Gef, statements.Gap]


def shorten_synonym_list(synonyms, machine_reading=True, retrieval="standard"):
    ''' Make canonical forms of gene names. Remove whitespaces and hyphens, e.g., ESR-1 to esr1. '''
    # machine_reading bool actually denotes a retrieval bool
    shortened_list = set()
    if len(synonyms) < 5 and not machine_reading:
        return list(synonyms)
    if retrieval == "relaxed":
        synonym_expanded = []
        for synonym in synonyms:
            numbers_truncated = synonym.rstrip("0123456789 ")
            if len(numbers_truncated) >= 3:
                synonym_expanded.append(numbers_truncated)
            # GSK3B to GSK3, Remove Greek Letters in the end
            greekletters = ["alpha", "eta", "beta", "gamma", "delta", "epsilon", "zeta", "theta", "iota", "kappa",
                            "lambda", "mu", "nu", "xi", "omikron", "pi", "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega"]
            if synonym.endswith(tuple(greekletters)):
                longest_suffix = ""
                for letter in greekletters:
                    if synonym.endswith(letter):
                        longest_suffix = letter
                length = -1 * len(longest_suffix)
                suffix_truncated = synonym[:length].rstrip(" -")
                synonym_expanded.append(suffix_truncated)
        synonyms = synonyms + synonym_expanded

    for synonym in synonyms:
        insert = True
        # Shortest prefix for synonym list, not used for retrieval!
        if not machine_reading:
            for comparison in synonyms:
                if (synonym.startswith(comparison) or synonym.endswith(comparison)) and comparison != synonym:
                    insert = False
                # MAPK 2 equals MAPK2
                elif (len(synonym) > 2) and (re.sub(" |-", "", synonym) == comparison) and comparison != synonym:
                    insert = False
        # Remove synonym if it starts with a number or consiting of more than 3 words
        if synonym == "" or synonym[0].isdigit() or len(synonym.split(" ")) > 3 or len(synonym) > 25:
            insert = False
        if machine_reading or insert:
            shortened_list.add(synonym)

        # Extra synonyms only used for retrieval
        # For GSK-3 beta, add both GSK3beta and GSK-3beta and GSK3 beta
        if machine_reading and insert:
            synonym_expanded = []
            for i, character in enumerate(synonym):
                if i == 0:
                    synonym_expanded.append(character)
                elif character != " " and character != "-":
                    for candidate in range(len(synonym_expanded)):
                        synonym_expanded[candidate] += character
                else:
                    expanded_copy = copy.deepcopy(synonym_expanded)
                    for candidate in range(len(synonym_expanded)):
                        expanded_copy.append(synonym_expanded[candidate])
                        expanded_copy[candidate] += character
                    synonym_expanded = expanded_copy
            for expanded in synonym_expanded:
                shortened_list.add(expanded)
    return list(shortened_list)


def get_chemical_synonyms(agent, retrieval=True):
    synonyms = [agent.name]
    if agent.db_refs["CHEBI"].startswith("CHEBI:"):
        chebi_id = agent.db_refs["CHEBI"][6:]
        synonyms = CHEBI_DICT[chebi_id]
    if retrieval or len(synonyms) <= 5:
        return synonyms
    else:
        return sorted(synonyms, key=lambda x: len(x))[:5]


def get_residue_and_position_list(residue, position):
    residue_long = AMINO_ACIDS_ABBREVIATIONS[residue]
    residue_short = AMINO_ACIDS[residue_long][0]
    # residue Y, 38 results in ["Tyrosine 38", "Tyrosine38", "Tyrosine(38)", "Tyrosine-38", "Tyr 38", "Y 38"] etc
    residues = [residue, residue_short, residue_long]
    results = []
    for form in residues:
        results.append(form + " " + position)
        results.append(form + position)
        results.append(form + "(" + position + ")")
        results.append(form + "-" + position)
    return results


def parse_residue_position_string(res_pos_string):
    res_candidate = re.sub('[^a-zA-Z]', '', res_pos_string).lower()
    pos_candidate = re.sub('[^0-9]', '', res_pos_string).lower()
    if res_candidate in AMINO_ACIDS_STRINGS.keys():
        return AMINO_ACIDS_STRINGS[res_candidate], pos_candidate
    else:
        return "", ""


class GeneIDToNames:
    lookup_table = None  # Needs to be initialized with function initialize_geneid_to_names()

    @classmethod
    def initialize(cls):
        logger.info("Loading GENE_ID_TO_NAMES")
        cls.lookup_table = get_gene_id_to_names()


def updateAllNames(all_names, name, db, identifier, gene_info_engine):
    if GeneIDToNames.lookup_table is None:
        GeneIDToNames.initialize()
    if db == "EGID" and identifier in GeneIDToNames.lookup_table:  # Gene Info NCBI
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
        all_names.update(GeneIDToNames.lookup_table[identifier][1])
        name = GeneIDToNames.lookup_table[identifier][0]
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


def synonym_expansion(indra_agent, machine_reading=True, retrieval="standard"):
    if len(indra_agent.all_names) == 0 and "CHEBI" in indra_agent.db_refs:
        return get_chemical_synonyms(indra_agent)
    else:
        names = indra_agent.all_names
        names.append(indra_agent.name)
        if "EGID" in indra_agent.db_refs:
            names, _, _ = updateAllNames(set(names), indra_agent.name, "EGID", indra_agent.db_refs["EGID"], None)
        elif "EGID" not in indra_agent.db_refs and "UP" in indra_agent.db_refs:
            eg_id = get_db_id(indra_agent)
            if eg_id[0] == "EGID":
                names, _, _ = updateAllNames(set(names), indra_agent.name, eg_id[0], eg_id[1], None)
        return shorten_synonym_list(list(names), machine_reading=machine_reading, retrieval=retrieval)


def get_subject_list(question_type, indra_subject_list):
    subject_list = []
    subject_list.append(question_type)
    # Get GENE_ID Entrez_Gene
    # TODO: Handle multiple Entrez_Gene IDs
    for i, subject_agent in enumerate(indra_subject_list):
        db_id = get_db_id(subject_agent)
        if i == 0:
            role = "SUBSTRATE"
        else:
            role = "ENZYME"
        agent_string = subject_agent.name + "_" + role + "_" + db_id[0] + "_" + db_id[1]
        subject_list.append(agent_string)
    return subject_list


def get_db_id(agent):
    if "EGID" in agent.db_refs:
        db_id = ("EGID", agent.db_refs["EGID"])
    elif "UP" in agent.db_refs:
        df = pd.read_sql_query(('SELECT GeneID FROM up_to_geneid2 WHERE "UniProtKB-AC" = "{}" AND "NCBI-taxon" = "9606"'
                                'LIMIT 1').format(agent.db_refs["UP"]), GENE_INFO_ENGINE)
        if len(df) == 0:
            id_number = ""
        else:
            id_number = df["GeneID"][0].split(";")[0]
        if id_number == "":
            db_id = ("##", "##")
        else:
            db_id = ("EGID", id_number)
    elif "CHEBI" in agent.db_refs:
        id_number = agent.db_refs["CHEBI"]
        if id_number.startswith("CHEBI:"):
            id_number = id_number[6:]
        db_id = ("CHEBI", id_number)
    else:
        db_id = ("##", "##")
    return db_id


def convert_gene_to_human_gene_id(gene_id):
    if gene_id == "":
        return gene_id

    df = pd.read_sql_query('''SELECT "HID", "Taxonomy_ID", "Gene_ID" FROM homologene
                           WHERE "Gene_ID" = {}'''.format(gene_id), GENE_INFO_ENGINE)
    if len(df.head()) != 1:
        return gene_id
    hid = df["HID"][0]

    df = pd.read_sql_query('''SELECT "HID", "Taxonomy_ID", "Gene_ID" FROM homologene
                           WHERE "HID" = {} AND "TAXONOMY_ID" = 9606'''.format(hid), GENE_INFO_ENGINE)
    if len(df.head()) != 1:
        return gene_id
    return df["Gene_ID"][0]


def parse_subject_strings(list_subject_strings):
    theme_ids = []
    for theme_subject in list_subject_strings[1:]:
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
    return theme_ids


if __name__ == "__main__":
    print(Question.PHOSPHORYLATION_CAUSE.name)

    name = "AKT1"
    all_names = []
    db_id = {"EGID": "207"}
    indra_agent = statements.Agent(name, list(all_names), db_refs=db_id)
    print(synonym_expansion(indra_agent, retrieval="relaxed"))
