" Read and process BioNLP Standoff Files. Forked from evex_standoff.py. Using MyGeneInfo for protein/gene lookup "

import os
import sys

if os.name == 'nt':
    root_path = "/".join(os.path.realpath(__file__).split("\\")[:-2])
else:
    root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.append(root_path)

import itertools
import logging
import pandas as pd
import re
import requests

from itertools import permutations
from sqlitedict import SqliteDict
from tqdm import tqdm
from mygene import MyGeneInfo
from unidecode import unidecode
from sqlalchemy.engine import create_engine

from configs import BIONLP_DIR, BIONLP_CACHE, PUBMED_EVIDENCE_ANNOTATIONS_DB
from data_processing.nn_output_to_indra import PUBTATOR_DOC_NORMALIZER
from data_processing.datatypes import convert_gene_to_human_gene_id, parse_residue_position_string
from util.build_document_corpus import get_pmid_from_pmcid_mapper

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PTMS = ["Dephosphorylation", "Phosphorylation", "Dehydroxylation", "Hydroxylation", "Deubiquitination", "Ubiquitination",
        "Demethylation", "Methylation", "Deacetylation", "Acetylation", "Deglycosylation", "Glycosylation"]

OTHER = ["Gene_expression", "Transcription", "Translation", "Localization", "Protein catabolism", "Transport", "Degradation", "Dissociation"]

REGULATIONS = ["Regulation", "Negative_regulation", "Positive_regulation", "Activation", "Inactivation"]

ALL_EVENT_TYPES = PTMS + OTHER + REGULATIONS

BIONLP_QUESTION_MAPPER = {"Gene_expression": "EXPRESSION", "Transcription": "EXPRESSION", "Translation": "EXPRESSION", "Phosphorylation": "PHOSPHORYLATION",
                          "Dephosphorylation": "DEPHOSPHORYLATION", "Acetylation": "ACETYLATION", "Deacetylation": "DEACETYLATION",
                          "Ubiquitination": "UBIQUITINATION", "Deubiquitination": "DEUBIQUITINATION", "Regulation": "STATECHANGE",
                          "Negative_regulation": "STATECHANGE", "Positive_regulation": "STATECHANGE", "Activation": "STATECHANGE",
                          "Inactivation": "STATECHANGE", "Binding": "COMPLEX"}


class AnnotationReader():
    def __init__(self):
        self.entities = {}
        self.events = {}
        self.current_ann_file = ""
        self.current_pmid = ""
        self.pubmed_engine = create_engine(PUBMED_EVIDENCE_ANNOTATIONS_DB)
        self.pmcid_mapper = None
        self.mg = MyGeneInfo()
        self.mg_cache = {}

    def initialize_pmcid_mapper(self):
        _, self.pmcid_mapper = get_pmid_from_pmcid_mapper()

    def natural_language_to_entrezgene(self, string, start, end):
        eg_id = None
        string = unidecode(string.replace("/", " ").replace("+", "").replace("(", "").replace(")", "")
                                 .replace("[", "").replace("]", "").replace("{", "").replace("}", "").lower())

        sql_query = 'SELECT * FROM pubmed_annotations WHERE PubMedID = {} AND "Start Position" = {} AND "End Position" = {} LIMIT 1'.format(
                    self.current_pmid, start, end)
        df = pd.read_sql_query(sql_query, self.pubmed_engine)
        pubtator_doc_key = str(self.current_pmid) + "_" + string
        # 1 First try Pubmed Text Span Normalization
        if len(df) > 0 and df["Type"][0] == "Gene":
            db_ids = [convert_gene_to_human_gene_id(db_id) for db_id in df["DB Identifier"][0].split(";")]
            if len(db_ids) > 0:
                eg_id = db_ids[0]
        # 2 Try Pubmed Document Normalization
        elif pubtator_doc_key in PUBTATOR_DOC_NORMALIZER:
            db_ids = [convert_gene_to_human_gene_id(db_id[1]) for db_id in PUBTATOR_DOC_NORMALIZER[pubtator_doc_key] if db_id[1].isdigit()]
            if len(db_ids) > 0:
                eg_id = db_ids[0]
        # 3 Try lookup MyGeneInfo cache and or MyGeneInfo directly
        elif string in self.mg_cache:
            eg_id = self.mg_cache[string]
        else:
            try:
                res = self.mg.query(string, size=1, fields='entrezgene', species='human')['hits']
            except requests.exceptions.HTTPError:
                raise ValueError("Invalid string: %s", string)
            if not res:  # Try other species
                res = self.mg.query(string, size=1, fields='entrezgene')['hits']
            if res and 'entrezgene' in res[0]:
                self.mg_cache[string] = res[0]['entrezgene']
                eg_id = res[0]['entrezgene']
            # else:  # Return None
        return eg_id

    def read(self, ann_file, entities, events):
        if self.pmcid_mapper is None:
            self.initialize_pmcid_mapper()
        self.entities = entities
        self.events = events
        self.current_ann_file = ann_file
        # Parse Annotation File Name
        file_name = self.current_ann_file.split("/")[-1]
        # print(file_name)
        file_name_parts = re.split("[-.]", file_name)
        if file_name_parts[0] == "PMC":
            self.current_pmid = self.pmcid_mapper[file_name_parts[1]]
        elif file_name_parts[0] == "PMID":
            self.current_pmid = file_name_parts[1]
        else:
            raise ValueError("Cannot infer Pubmed ID from %s", self.current_ann_file)

        with open(self.current_ann_file, "r") as f:
            ann_lines = f.readlines()
        for line in ann_lines:
            self.__process_line(line)

        return self.entities, self.events

    def __process_line(self, line):
        line_elements = line.split("\t")
        line_id = line_elements[0]
        # if len(line_elements) == 1:
        #     print(line)
        line_type_infos = line_elements[1].split(" ")
        line_type_infos[-1] = line_type_infos[-1].rstrip()
        line_value = line_elements[2].rstrip() if len(line_elements) > 2 else None

        if line_id.startswith("T"):  # Trigger or GGP
            self.entities[line_id] = {}
            self.entities[line_id]["Type"] = line_type_infos[0]
            self.entities[line_id]["Start Position"] = line_type_infos[1]
            self.entities[line_id]["End Position"] = line_type_infos[2]
            self.entities[line_id]["Value"] = line_value
            self.entities[line_id]["References"] = {}
            if line_type_infos[0] in ["Protein", "Gene_or_gene_product", "Entity", "Complex"]:  # Ignore "Simple_chemical" and "Cellular_component"
                entrez_gene_id = self.natural_language_to_entrezgene(line_value, line_type_infos[1], line_type_infos[2])
                if entrez_gene_id is not None:
                    self.entities[line_id]["References"]["EG"] = entrez_gene_id

        elif line_id.startswith("E"):  # Event
            event_infos = line_type_infos[0].split(":", 1)
            event_type = event_infos[0]
            event_trigger = event_infos[1]
            self.events[line_id] = {}
            self.events[line_id]["Type"] = event_type
            self.events[line_id]["Trigger"] = event_trigger
            self.events[line_id]["Speculation"] = 0
            self.events[line_id]["Negation"] = 0
            self.events[line_id]["Confidence #"] = 0
            if event_type == "Negative_regulation":
                self.events[line_id]["Negation"] = 1
            for argument in line_type_infos[1:]:
                arg_infos = argument.split(":", 1)
                if len(arg_infos) != 2:  # Happens in Pathway events with no participants
                    break
                # In Binding events for BioNLP datasets the themes and other arguments may have number suffixes
                # TODO: Include handling of numbered suffixes matching different themes and sites together
                arg_type = arg_infos[0].rstrip("0123456789")
                arg_value = arg_infos[1]
                if arg_type in self.events[line_id] and isinstance(self.events[line_id][arg_type], str):
                    value = self.events[line_id][arg_type]
                    self.events[line_id][arg_type] = [value, arg_value]
                elif arg_type in self.events[line_id]:
                    self.events[line_id][arg_type].append(arg_value)
                else:
                    self.events[line_id][arg_type] = arg_value

        elif line_id.startswith("R"):  # Relation
            event_arg1 = line_type_infos[1].split(":", 1)[1]
            event_arg2 = line_type_infos[2].split(":", 1)[1]
            self.events[line_id] = {}
            self.events[line_id]["Type"] = line_type_infos[0]
            self.events[line_id]["Arg1"] = event_arg1
            self.events[line_id]["Arg2"] = event_arg2

        elif line_id.startswith("M"):  # Event/Relation Modifier Confidence
            mod_type = line_type_infos[0]
            ref_id = line_type_infos[1]
            if mod_type in ["Speculation", "Negation"]:
                self.events[ref_id][mod_type] = 1
            else:
                logger.error("Encountered unknown input while parsing annotation files: %s", line_elements)
                exit()

        # elif line_id.startswith("#"):  # Event/Relation Confidence
        #     ref_id = line_type_infos[1]
        #     if ref_id[-1] not in ["N", "S"]:
        #         self.events[ref_id]["Confidence #"] = line_value

        elif line_id.startswith("*"):  # Equivalence of two entities
            pass

        else:
            logger.error("Encountered unknown input while parsing annotation files: %s", line_elements)
            exit()


class BaselineEventDict():
    def __init__(self, standoff_cache=BIONLP_CACHE):
        self.bionlp_db = None
        self.bionlp_pairs_db = None
        self._doc_entities = None
        self._doc_events = None
        self._annotation_reader = AnnotationReader()
        self.standoff_cache = standoff_cache

    def build_event_dict(self, debug=False):
        self.bionlp_db = SqliteDict(self.standoff_cache, tablename='bionlp_events', flag='w', autocommit=False)
        for directory in tqdm(os.listdir(BIONLP_DIR)):
            dir_full_name = os.path.join(BIONLP_DIR, directory)
            if os.path.isdir(dir_full_name):
                for pubmed_file in tqdm(os.listdir(dir_full_name)):
                    pubmed_file_full_name = os.path.join(BIONLP_DIR, directory, pubmed_file)
                    if pubmed_file.endswith(".a1"):
                        self.extract_events(pubmed_file_full_name, debug)
                self.bionlp_db.commit()
        self.bionlp_db.close()

    def load_event_dict(self):
        self.bionlp_db = SqliteDict(self.standoff_cache, tablename='bionlp_events', flag='r', autocommit=False)

    def __parse_ent(self, entity_id):
        ''' Get NCBIGeneID if human gene/protein from the entity dict. '''
        ncbi_gene_id = "XX"
        entity = self._doc_entities[entity_id]
        entity_name = self._doc_entities[entity_id]["Value"]
        entity_start = self._doc_entities[entity_id]["Start Position"]
        entity_end = self._doc_entities[entity_id]["End Position"]
        if "EG" in entity["References"]:
            ncbi_gene_id = entity["References"]["EG"]
        return ncbi_gene_id, (entity_name, entity_start, entity_end)

    def __parse_site(self, entity_id):
        ''' Parse Residue and Position for PTMs. '''
        entity = self._doc_entities[entity_id]["Value"]
        site_start = self._doc_entities[entity_id]["Start Position"]
        site_end = self._doc_entities[entity_id]["End Position"]
        res_pos = parse_residue_position_string(entity)
        return res_pos[0], res_pos[1], (entity, site_start, site_end)

    def __parse_sub_complex_event(self, theme_ref_id):
        ''' Parse sub event with entity or complex by EGIDs of all its members '''
        theme_id = complex_id = event_type = res = pos = ""
        theme_name = complex_name = site_info = ("", "-1", "-1")
        if theme_ref_id.startswith("E"):
            event = self._doc_events[theme_ref_id]
            event_type = event["Type"]
            if (event_type in PTMS) or (event_type in OTHER):
                if "Theme" in event:
                    theme_id, theme_name = self.__parse_ent(event["Theme"])
                if (event_type in PTMS) and ("Site" in event):
                    sites = event["Site"]
                    if not isinstance(sites, list):
                        sites = [sites]
                    for site in sites:
                        res, pos, site_info = self.__parse_site(site)
                        yield theme_id, theme_name, complex_id, complex_name, event_type, res, pos, site_info
                else:
                    yield theme_id, theme_name, complex_id, complex_name, event_type, res, pos, site_info
            elif event_type == "Binding" and "Theme" in event:
                for complex_pair in self.__parse_binding_pairs(event["Theme"]):
                    theme_id = complex_pair[0][0]
                    theme_name = complex_pair[0][1]
                    complex_id = complex_pair[1][0]
                    complex_name = complex_pair[1][1]
                    yield theme_id, theme_name, complex_id, complex_name, event_type, res, pos, site_info
            # Simple Regulation events and no nested Regulation event structures
            elif event_type in REGULATIONS and "Theme" in event and event["Theme"].startswith("T"):
                theme_id, theme_name = self.__parse_ent(event["Theme"])
                yield theme_id, theme_name, complex_id, complex_name, event_type, res, pos, site_info
        elif theme_ref_id.startswith("T"):
            theme_id, theme_name = self.__parse_ent(theme_ref_id)
            event_type = "Regulation"
            yield theme_id, theme_name, complex_id, complex_name, event_type, res, pos, site_info

    def __parse_binding_pairs(self, ent_list, length=2):
        if not isinstance(ent_list, list):
            ent_list = [ent_list]
        binding_ent = set()
        for ent in ent_list:
            ent_id, ent_name = self.__parse_ent(ent)
            if ent_id != "":
                binding_ent.add((ent_id, ent_name))
        binding_ent = list(binding_ent)
        binding_pairs = permutations(binding_ent, length)
        return binding_pairs

    def __parse_event(self, event, ann_file):
        ''' Return events in iterator of (key, value) form.
            Binding Events have the form ("Binding_$EGID1", [(EGID1, EGID2, Confidence, Negation, Speculation, "Binding"), ...])
        '''
        db_key = db_value = None
        if event["Type"] in REGULATIONS:
            # Handle Protein and Complex Arguments
            if ("Theme" in event) and ("Cause" in event):
                for theme_id, theme_name, _, _, theme_event_type, res, pos, site_info in self.__parse_sub_complex_event(event["Theme"]):
                    for cause_id_0, cause_name_0, cause_id_1, cause_name_1, _, _, _, _ in self.__parse_sub_complex_event(event["Cause"]):
                        if (cause_id_0 != "") and (theme_id != "") and (theme_event_type not in REGULATIONS):
                            db_key = theme_event_type + "_" + theme_id + "_" + ann_file
                            if theme_event_type in PTMS:
                                db_value = (theme_id, theme_name, cause_id_0, cause_name_0, cause_id_1, cause_name_1, event["Negation"],
                                            event["Speculation"], theme_event_type, event["Type"], res, pos, site_info, ann_file)
                                yield db_key, db_value
                            elif theme_event_type in OTHER:
                                # if event["Negation"] == 1:
                                #     print(theme_event_type)
                                db_value = (theme_id, theme_name, cause_id_0, cause_name_0, cause_id_1, cause_name_1, event["Negation"],
                                            event["Speculation"], theme_event_type, event["Type"], ann_file)
                                yield db_key, db_value
                            db_key_2 = "Regulation_" + theme_id + "_" + ann_file  # For comparison to question type STATECHANGE_CAUSE
                            db_value_2 = (theme_id, theme_name, cause_id_0, cause_name_0, cause_id_1, cause_name_1, event["Negation"],
                                          event["Speculation"], "Regulation", event["Type"], ann_file)
                            yield db_key_2, db_value_2
                        elif (cause_id_0 != "") and (theme_id != "") and (theme_event_type in REGULATIONS):
                            db_key = theme_event_type + "_" + theme_id + "_" + ann_file
                            db_value = (theme_id, theme_name, cause_id_0, cause_name_0, cause_id_1, cause_name_1, event["Negation"],
                                        event["Speculation"], theme_event_type, event["Type"], ann_file)
                            yield db_key, db_value

        if event["Type"] in PTMS:
            # Handle Site Arguments
            if ("Theme" in event) and ("Site" in event):
                theme_id, theme_name = self.__parse_ent(event["Theme"])
                sites = event["Site"]
                if not isinstance(sites, list):
                    sites = [sites]
                for site in sites:
                    res, pos, site_info = self.__parse_site(site)
                    if theme_id != "":
                        db_key = event["Type"] + "_" + theme_id + "_" + ann_file
                        db_value = (theme_id, theme_name, event["Negation"], event["Speculation"], event["Type"], res, pos, site_info, ann_file)
                        yield db_key, db_value
        # Handle Cause Arguments (Relevant for simple Regulations and in Pathway Curation where Cause arguments can be directly in PTMs)
        # Cause here can only be a molecule
        if ("Theme" in event) and ("Cause" in event) and event["Theme"].startswith("T") and event["Cause"].startswith("T"):
            theme_id, theme_name = self.__parse_ent(event["Theme"])
            cause_id, cause_name = self.__parse_ent(event["Cause"])
            if (cause_id != "") and (theme_id != ""):
                db_key = event["Type"] + "_" + theme_id + "_" + ann_file
                db_value = (theme_id, theme_name, cause_id, cause_name, "", "", event["Negation"],
                            event["Speculation"], event["Type"], "Regulation", ann_file)
                yield db_key, db_value
                if event["Type"] in PTMS or event["Type"] in OTHER:
                    db_key_2 = "Regulation_" + theme_id + "_" + ann_file  # For comparison to question type STATECHANGE_CAUSE
                    db_value_2 = (theme_id, theme_name, cause_id, cause_name, "", ("", "-1", "-1"), event["Negation"],
                                  event["Speculation"], "Regulation", "Regulation", ann_file)
                    yield db_key_2, db_value_2

        elif event["Type"] == "Binding":
            if "Theme" not in event:  # E.g. Binding events only with a product
                return
                yield
            else:
                # Binding Pairs
                binding_pairs = self.__parse_binding_pairs(event["Theme"])
                for binding_pair in binding_pairs:
                    # binding_pair has the list of tuples (binding_entity_id, binding_entity_name)
                    if len(binding_pair) == 2 and binding_pair[0][0] != "" and binding_pair[1][0] != "":  # ID not equal ""
                        db_key = event["Type"] + "_" + binding_pair[0][0] + "_" + ann_file
                        db_value = (binding_pair[0][0], binding_pair[0][1], binding_pair[1][0], binding_pair[1][1], event["Negation"],
                                    event["Speculation"], event["Type"], ann_file)
                        yield db_key, db_value
                # Binding Triples
                binding_complexes = self.__parse_binding_pairs(event["Theme"], 3)
                for binding_complex in binding_complexes:
                    # binding_complex has the list of tuples (binding_entity_id, binding_entity_name)
                    if len(binding_complex) == 3 and binding_complex[0][0] != "" and binding_complex[1][0] != "" and binding_complex[2][0] != "":
                        db_key = event["Type"] + "_" + binding_complex[0][0] + "_" + binding_complex[1][0] + "_" + ann_file
                        db_value = (binding_complex[0][0], binding_complex[0][1], binding_complex[1][0], binding_complex[1][1],
                                    binding_complex[2][0], binding_complex[2][1], event["Negation"], event["Speculation"], event["Type"], ann_file)
                        yield db_key, db_value

    def __parse_entity_trigger_pair(self, ann_file):
        ''' Return trigger entity pairs in iterator of (key, value) form. Events here only include potential substrate entities.
            Only includes potential Site multi-turn questions, Complex regulators of a modification (COMPLEXCAUSE question) is not included here.
        '''
        entrez_gene_entities = set()  # List of (Entity Name, EGID) where EGID may occur multiple times
        event_types = set()
        for _, entity in self._doc_entities.items():
            if entity["Type"] in ["Protein", "Gene_or_gene_product", "Entity", "Complex"] and "EG" in entity["References"]:
                entrez_gene_entities.add((entity["Value"], entity["References"]["EG"]))
            elif entity["Type"] in ALL_EVENT_TYPES:
                event_types.add(entity["Type"])

        for event_type, entity in itertools.product(event_types, entrez_gene_entities):
            entity_name = entity[0]
            entity_id = entity[1]
            db_key = event_type + "_" + entity_id + "_" + ann_file
            # _CAUSE, _SITE, _COMPLEXSITE and COMPLEX_PAIR question types
            if event_type in PTMS:
                db_value = (entity_id, (entity_name, "-1", "-1"), "", ("", "-1", "-1"), "", ("", "-1", "-1"), 0, 0, event_type,
                            "Regulation", "", "", ("", "-1", "-1"), ann_file)
            elif event_type == "Binding":
                db_value = (entity_id, (entity_name, "-1", "-1"), "", ("", "-1", "-1"), 0, 0, event_type, ann_file)
            else:
                db_value = (entity_id, (entity_name, "-1", "-1"), "", ("", "-1", "-1"), "", ("", "-1", "-1"), 0, 0, event_type,
                            "Regulation", ann_file)
            yield db_key, db_value
            if event_type not in REGULATIONS and event_type != "Binding":
                db_key_2 = "Regulation_" + entity_id + "_" + ann_file  # For comparison to question type STATECHANGE_CAUSE
                db_value_2 = (entity_id, (entity_name, "-1", "-1"), "", ("", "-1", "-1"), "", ("", "-1", "-1"), 0, 0, "Regulation", "Regulation", ann_file)
                yield db_key_2, db_value_2

        # if "Binding" in event_types:
        #     # COMPLEX_MULTI question type
        #     for entity_1, entity_2 in itertools.product(entrez_gene_entities, entrez_gene_entities):
        #         entity_name_1 = entity_1[0]
        #         entity_id_1 = entity_1[1]
        #         entity_name_2 = entity_2[0]
        #         entity_id_2 = entity_2[1]
        #         db_key = event_type + "_" + entity_id_1 + "_" + entity_id_2 + "_" + ann_file
        #         db_value = (entity_id_1, (entity_name_1, "-1", "-1"), entity_id_2, (entity_name_2, "-1", "-1"), "", ("", "-1", "-1"),
        #                     0, 0, event_type, ann_file)
        #         yield db_key, db_value

    def extract_events(self, ann_file, debug=False):
        self._doc_entities = {}
        self._doc_events = {}
        self._doc_entities, self._doc_events = self._annotation_reader.read(ann_file, self._doc_entities, self._doc_events)  # Read a1. file
        a2_ann_file = ann_file[:-3] + ".a2"
        self._doc_entities, self._doc_events = self._annotation_reader.read(a2_ann_file, self._doc_entities, self._doc_events)
        if debug:
            print(self._doc_entities)
            print(self._doc_events)
            print(ann_file)
            self.bionlp_db = {}
        for event_id, infos in self._doc_events.items():
            if event_id.startswith("R"):  # Relation
                continue
            for db_key, db_value in self.__parse_event(infos, a2_ann_file):
                # print(db_key, db_value)
                if db_key is not None:
                    val = self.bionlp_db.setdefault(db_key, [])
                    val.append(db_value)
                    self.bionlp_db[db_key] = val
        if debug:
            print(self.bionlp_db.items())

    def build_trigger_entity_pair_dict(self, debug=False):
        ''' Includes trigger entity pairs denoting all possible events, including positive and negative events.
            Negative examples have to be removed in a post-processing step. '''
        self.bionlp_pairs_db = SqliteDict(self.standoff_cache, tablename='bionlp_trigger_entity_pairs', flag='w', autocommit=False)
        for directory in tqdm(os.listdir(BIONLP_DIR)):
            dir_full_name = os.path.join(BIONLP_DIR, directory)
            if os.path.isdir(dir_full_name):
                for pubmed_file in tqdm(os.listdir(dir_full_name)):
                    pubmed_file_full_name = os.path.join(BIONLP_DIR, directory, pubmed_file)
                    if pubmed_file.endswith(".a1"):
                        self.extract_entity_trigger_product(pubmed_file_full_name, debug)
                self.bionlp_pairs_db.commit()
        self.bionlp_pairs_db.close()

    def load_event_questions_dict(self):
        ''' Includes trigger entity pairs denoting all possible events, including positive and negative events.
            Negative examples have to be removed in a post-processing step. '''
        self.bionlp_pairs_db = SqliteDict(self.standoff_cache, tablename='bionlp_trigger_entity_pairs', flag='r', autocommit=False)

    def extract_entity_trigger_product(self, ann_file, debug=False):
        ''' Extracts cartesian product and triggers for one annotation file '''
        self._doc_entities = {}
        self._doc_events = {}
        self._doc_entities, self._doc_events = self._annotation_reader.read(ann_file, self._doc_entities, self._doc_events)  # Read a1. file
        a2_ann_file = ann_file[:-3] + ".a2"
        self._doc_entities, self._doc_events = self._annotation_reader.read(a2_ann_file, self._doc_entities, self._doc_events)
        if debug:
            print(self._doc_entities)
            print(self._doc_events)
            print(ann_file)
            self.bionlp_pairs_db = {}
        for db_key, db_value in self.__parse_entity_trigger_pair(a2_ann_file):
            # print(db_key, db_value)
            if db_key is not None:
                val = self.bionlp_pairs_db.setdefault(db_key, [])
                val.append(db_value)
                self.bionlp_pairs_db[db_key] = val
        if debug:
            print(self.bionlp_pairs_db.items())


# Main
if __name__ == "__main__":
    bionlp_dict = BaselineEventDict(BIONLP_CACHE)
    bionlp_dict.build_event_dict(debug=False)
    # bionlp_dict.extract_events("/glusterfs/dfs-gfs-dist/wangxida/bionlp_datasets/BioNLP-ST-2013_GE_devel_data_rev3/PMC-3062687-00-TIAB.a1", debug=True)
    # bionlp_dict.extract_events("/glusterfs/dfs-gfs-dist/wangxida/bionlp_datasets/BioNLP-ST_2013_PC_development_data/PMID-16571800.a1", debug=True)
    # bionlp_dict.extract_events("/glusterfs/dfs-gfs-dist/wangxida/bionlp_datasets/BioNLP-ST-2013_GE_devel_data_rev3/PMC-1920263-02-MATERIALS_AND_METHODS.a1", debug=True)
    # evex_dict.extract_events("/glusterfs/dfs-gfs-dist/wangxida/evex/standoff-annotation/version-0.1/files_pmc_000034/batch_pmc_003312/pmc_2709114.ann", debug=True)

    # bionlp_dict.extract_entity_trigger_product("/glusterfs/dfs-gfs-dist/wangxida/bionlp_datasets/BioNLP-ST_2013_PC_development_data/"
    #                                            + "PMID-16571800.a1", debug=True)
    # bionlp_dict.extract_entity_trigger_product("/glusterfs/dfs-gfs-dist/wangxida/bionlp_datasets/BioNLP-ST_2013_PC_development_data/"
    #                                            + "PMID-16131838.a1", debug=True)
    # bionlp_dict.build_trigger_entity_pair_dict(debug=False)
