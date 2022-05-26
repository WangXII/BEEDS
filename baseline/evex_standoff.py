"Read and process EVEX Standoff Files"

import os
import sys

if os.name == 'nt':
    root_path = "/".join(os.path.realpath(__file__).split("\\")[:-2])
else:
    root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.append(root_path)

import logging

from itertools import permutations
from sqlitedict import SqliteDict
from tqdm import tqdm
from configs import STANDOFF_CACHE_PMIDS, STANDOFF_DIR, STANDOFF_CACHE, STANDOFF_CACHE_SIMPLE, SIMPLE_NORMALIZER_DB
from data_processing.datatypes import parse_residue_position_string, convert_gene_to_human_gene_id

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PTMS = ["Dephosphorylation", "Phosphorylation", "Dehydroxylation", "Hydroxylation", "Deubiquitination", "Ubiquitination",
        "Demethylation", "Methylation", "Deacetylation", "Acetylation", "Deglycosylation", "Glycosylation"]

OTHER = ["Gene_expression", "Transcription", "Localization", "Protein_catabolism", "DNA_demethylation", "DNA_methylation"]

REGULATIONS = ["Regulation", "Negative_regulation", "Positive_regulation", "Catalysis"]

EVEX_QUESTION_MAPPER = {"Gene_expression": "EXPRESSION", "Transcription": "EXPRESSION", "Phosphorylation": "PHOSPHORYLATION",
                        "Dephosphorylation": "DEPHOSPHORYLATION", "Acetylation": "ACETYLATION", "Deacetylation": "DEACETYLATION",
                        "Ubiquitination": "UBIQUITINATION", "Deubiquitination": "DEUBIQUITINATION", "Regulation": "STATECHANGE",
                        "Negative_regulation": "STATECHANGE", "Positive_regulation": "STATECHANGE", "Catalysis": "STATECHANGE",
                        "Binding": "COMPLEX"}


class AnnotationReader():
    def __init__(self):
        self.entities = {}
        self.events = {}
        self.ann_file = ""

    def read(self, ann_file):
        self.entities = {}
        self.events = {}
        self.ann_file = ann_file
        with open(ann_file, "r") as f:
            ann_lines = f.readlines()
        for line in ann_lines:
            self.__process_line(line)
        return self.entities, self.events

    def __process_line(self, line):
        line_elements = line.split("\t")
        line_id = line_elements[0]
        line_type_infos = line_elements[1].split(" ")
        line_type_infos[-1] = line_type_infos[-1].rstrip()
        line_value = line_elements[2].rstrip() if len(line_elements) > 2 else None

        if line_id.startswith("T"):  # Trigger or GGP
            self.entities[line_id] = {}
            self.entities[line_id]["Type"] = line_type_infos[0]
            if line_type_infos[0] == "GGP":
                self.entities[line_id]["References"] = {}
            self.entities[line_id]["Start Position"] = line_type_infos[1]
            self.entities[line_id]["End Position"] = line_type_infos[2]
            self.entities[line_id]["Value"] = line_value

        elif line_id.startswith("N"):  # Reference for GGP
            ref_id = line_type_infos[1]
            db_infos = line_type_infos[2].split(":", 1)
            db_name = db_infos[0]
            db_id = db_infos[1]
            self.entities[ref_id]["References"][db_name] = db_id

        elif line_id.startswith("E"):  # Event
            event_infos = line_type_infos[0].split(":", 1)
            event_type = event_infos[0]
            event_trigger = event_infos[1]
            self.events[line_id] = {}
            self.events[line_id]["Type"] = event_type
            self.events[line_id]["Trigger"] = event_trigger
            self.events[line_id]["Speculation"] = 0
            self.events[line_id]["Negation"] = 0
            for argument in line_type_infos[1:]:
                arg_infos = argument.split(":", 1)
                arg_type = arg_infos[0]
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

        elif line_id.startswith("A"):  # Event/Relation Modifier Confidence
            mod_type = line_type_infos[0]
            ref_id = line_type_infos[1]
            if mod_type == "Confidence":
                self.events[ref_id][mod_type] = line_type_infos[2]
            elif mod_type in ["Speculation", "Negation"]:
                self.events[ref_id][mod_type] = 1

        elif line_id.startswith("#"):  # Event/Relation Confidence
            ref_id = line_type_infos[1]
            if ref_id[-1] not in ["N", "S"]:
                self.events[ref_id]["Confidence #"] = line_value

        else:
            print(line_elements)
            exit()


class BaselineEventDict():
    def __init__(self, standoff_cache=STANDOFF_CACHE, use_simple_normalizer=False):
        self.evex_db = None
        self._doc_entities = None
        self._doc_events = None
        self._annotation_reader = AnnotationReader()
        self.dict_normalizer = SqliteDict(SIMPLE_NORMALIZER_DB, tablename='simple_normalizer', flag='r', autocommit=False)
        self.standoff_cache = standoff_cache
        self.use_simple_normalizer = use_simple_normalizer

    def __del__(self):
        self.dict_normalizer.close()

    def get_full_path(self, article_id):
        for directory in tqdm(os.listdir(STANDOFF_DIR)):
            dir_full_name = os.path.join(STANDOFF_DIR, directory)
            if os.path.isdir(dir_full_name):
                for batch_dir in tqdm(os.listdir(dir_full_name)):
                    batch_dir_full_name = os.path.join(STANDOFF_DIR, directory, batch_dir)
                    if os.path.isdir(batch_dir_full_name):
                        for pubmed_file in os.listdir(batch_dir_full_name):
                            pubmed_file_full_name = os.path.join(STANDOFF_DIR, directory, batch_dir, pubmed_file)
                            if pubmed_file.endswith(article_id):
                                return pubmed_file_full_name

    def build_event_dict(self):
        self.evex_db = SqliteDict(self.standoff_cache, tablename='evex_events', flag='w', autocommit=False)
        for directory in tqdm(os.listdir(STANDOFF_DIR)):
            dir_full_name = os.path.join(STANDOFF_DIR, directory)
            if os.path.isdir(dir_full_name):
                for batch_dir in tqdm(os.listdir(dir_full_name)):
                    batch_dir_full_name = os.path.join(STANDOFF_DIR, directory, batch_dir)
                    if os.path.isdir(batch_dir_full_name):
                        for pubmed_file in os.listdir(batch_dir_full_name):
                            pubmed_file_full_name = os.path.join(STANDOFF_DIR, directory, batch_dir, pubmed_file)
                            if pubmed_file.endswith(".ann"):
                                self.extract_events(pubmed_file_full_name)
                self.evex_db.commit()
        self.evex_db.close()

    def load_event_dict(self):
        self.evex_db = SqliteDict(self.standoff_cache, tablename='evex_events', flag='r', autocommit=False)

    def __parse_ent(self, entity_id):
        ''' Get NCBIGeneID if human gene/protein from the entity dict. '''
        ncbi_gene_id = ""
        entity = self._doc_entities[entity_id]
        entity_name = entity["Value"].lower()
        entity_start = entity["Start Position"]
        entity_end = entity["End Position"]
        if "EG" in entity["References"]:
            ncbi_gene_id = entity["References"]["EG"]
        elif self.use_simple_normalizer and entity_name in self.dict_normalizer:
            ncbi_gene_id = self.dict_normalizer[entity_name][0]
        ncbi_tax_id = ""
        if "NCBITaxon" in entity["References"]:
            ncbi_tax_id = entity["References"]["NCBITaxon"]
        if ncbi_tax_id == "9606":
            return ncbi_gene_id, (entity_name, entity_start, entity_end)
        else:
            # Homologene Support
            updated_gene_id = convert_gene_to_human_gene_id(ncbi_gene_id)
            return updated_gene_id, (entity_name, entity_start, entity_end)

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
                    theme_id, theme_name = complex_pair[0]
                    complex_id, complex_name = complex_pair[1]
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
        ''' Return events in iterator of key, value) form.
            Binding Events have the form ("Binding_$EGID1", [(EGID1, EGID2, Confidence, Negation, Speculation, "Binding"), ...])
        '''
        db_key = db_value = None
        # ATTENTION: In EVEX, only regulation events can have causes! That is why it is sufficient to only filter for them here (They use the GENIA event definitions)
        # For the bionlp_standoff.py file, this is not sufficient as PTMs themselves (not Gene Expressions) can have causes attached to them
        if event["Type"] in REGULATIONS:
            # print(event)
            # Handle Protein and Complex Arguments
            if ("Theme" in event) and ("Cause" in event):
                for theme_id, theme_name, _, _, theme_event_type, res, pos, site_info in self.__parse_sub_complex_event(event["Theme"]):
                    for cause_id_0, cause_name_0, cause_id_1, cause_name_1, _, _, _, _ in self.__parse_sub_complex_event(event["Cause"]):
                        if (cause_id_0 != "") and (theme_id != "") and (theme_event_type not in REGULATIONS):
                            db_key = theme_event_type + "_" + theme_id
                            if cause_id_0 != "" and theme_id != "" and theme_event_type in PTMS:
                                db_value = (theme_id, theme_name, cause_id_0, cause_name_0, cause_id_1, cause_name_1, event["Confidence #"], event["Negation"],
                                            event["Speculation"], theme_event_type, event["Type"], res, pos, site_info, ann_file)
                                yield db_key, db_value
                            elif cause_id_0 != "" and theme_id != "" and theme_event_type in OTHER:
                                db_value = (theme_id, theme_name, cause_id_0, cause_name_0, cause_id_1, cause_name_1, event["Confidence #"], event["Negation"],
                                            event["Speculation"], theme_event_type, event["Type"], ann_file)
                                yield db_key, db_value
                            db_key_2 = "Regulation_" + theme_id  # For comparison to question type STATECHANGE_CAUSE
                            db_value_2 = (theme_id, theme_name, cause_id_0, cause_name_0, cause_id_1, cause_name_1, event["Confidence #"], event["Negation"],
                                          event["Speculation"], "Regulation", event["Type"], ann_file)
                            yield db_key_2, db_value_2
                        elif (cause_id_0 != "") and (theme_id != "") and (theme_event_type in REGULATIONS):
                            db_key = theme_event_type + "_" + theme_id
                            db_value = (theme_id, theme_name, cause_id_0, cause_name_0, cause_id_1, cause_name_1, event["Confidence #"], event["Negation"],
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
                        db_key = event["Type"] + "_" + theme_id
                        db_value = (theme_id, theme_name, event["Confidence #"], event["Negation"], event["Speculation"], event["Type"],
                                    res, pos, site_info, ann_file)
                        yield db_key, db_value
        # Handle Cause Arguments (Relevant for simple Regulations)
        # Cause here can only be a molecule
        if ("Theme" in event) and ("Cause" in event) and event["Theme"].startswith("T") and event["Cause"].startswith("T"):
            theme_id, theme_name = self.__parse_ent(event["Theme"])
            cause_id, cause_name = self.__parse_ent(event["Cause"])
            if (cause_id != "") and (theme_id != ""):
                db_key = event["Type"] + "_" + theme_id
                db_value = (theme_id, theme_name, cause_id, cause_name, "", ("", "-1", "-1"), event["Confidence #"], event["Negation"],
                            event["Speculation"], event["Type"], "Regulation", ann_file)
                yield db_key, db_value

        elif event["Type"] == "Binding":
            binding_pairs = self.__parse_binding_pairs(event["Theme"])
            for binding_pair in binding_pairs:
                if len(binding_pair) == 2 and binding_pair[0][0] != "" and binding_pair[1][0] != "":
                    db_key = event["Type"] + "_" + binding_pair[0][0]
                    db_value = (binding_pair[0][0], binding_pair[0][1], binding_pair[1][0], binding_pair[1][1],
                                event["Confidence #"], event["Negation"], event["Speculation"], ann_file)
                    yield db_key, db_value
            binding_triples = self.__parse_binding_pairs(event["Theme"], 3)
            for binding_triple in binding_triples:
                if len(binding_triple) == 3 and binding_triple[0][0] != "" and binding_triple[1][0] != "" and binding_triple[2][0] != "":
                    db_key = event["Type"] + "_" + binding_triple[0][0] + "_" + binding_triple[1][0]
                    db_value = (binding_triple[0][0], binding_triple[0][1], binding_triple[1][0], binding_triple[1][1], binding_triple[2][0], binding_triple[2][1],
                                event["Confidence #"], event["Negation"], event["Speculation"], ann_file)
                    yield db_key, db_value

    def extract_events(self, ann_file, debug=False):
        self._doc_entities, self._doc_events = self._annotation_reader.read(ann_file)
        if debug:
            print(self._doc_entities)
            print(self._doc_events)
            print(ann_file)
            self.evex_db = {}
        for event_id, infos in self._doc_events.items():
            if event_id.startswith("R"):  # Relation
                continue
            for db_key, db_value in self.__parse_event(infos, ann_file):
                # print(db_key, db_value)
                if db_key is not None:
                    val = self.evex_db.setdefault(db_key, [])
                    val.append(db_value)
                    self.evex_db[db_key] = val
        if debug:
            print(self.evex_db.items())


# Main
if __name__ == "__main__":
    evex_dict = BaselineEventDict(STANDOFF_CACHE_PMIDS, True)
    evex_dict.build_event_dict()
    # evex_dict.load_event_dict()
    # evex_dict.extract_events("/glusterfs/dfs-gfs-dist/wangxida/evex/standoff-annotation/version-0.1/files_pubmed_000027/batch_pubmed_002623/pmc_29065.ann", debug=True)
    # evex_dict.extract_events("/glusterfs/dfs-gfs-dist/wangxida/evex/standoff-annotation/version-0.1/files_pmc_000034/batch_pmc_003312/pmc_2709114.ann", debug=True)
    # print(evex_dict.get_full_path("pmc_2709114.ann"))
    # evex_dict.extract_events("/glusterfs/dfs-gfs-dist/wangxida/evex/standoff-annotation/version-0.1/files_pmc_000027/batch_pmc_002678/pmc_1166548.ann", debug=True)
    evex_dict.dict_normalizer.close()
