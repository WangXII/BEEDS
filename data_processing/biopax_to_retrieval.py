''' Retrieve Events from Biopax database and prepare queries for them. '''

import os
import sys

if os.name == 'nt':
    root_path = "/".join(os.path.realpath(__file__).split("\\")[:-2])
else:
    root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.append(root_path)

import pickle
import resource
import random
import logging

from sqlitedict import SqliteDict
from tqdm import tqdm

import indra.sources.biopax as biopax
import indra.statements.statements as bp

from configs import EVENT_SUBSTRATES_COMPLEX_MULTI, OWL_LIST, OWL_STATEMENTS, TRAIN_TEST_SPLIT_SEED, TRAIN_SPLIT, DEV_SPLIT, PFAM_DB, TQDM_DISABLE
from data_processing.datatypes import Question, QUESTION_TO_BIOPAX_TYPE, STATECHANGE_BIOPAX_TYPES

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)


class IndraDataLoader:
    ''' Loads and extracts data from BioPAX/OWL files into the INDRA format.
        Provides function for manipulation of biomolecular events in the INDRA format.
    '''

    @classmethod
    def get_dataset(cls, biopax_owl_strings=OWL_LIST, use_cache=True, mode="train", question_type=Question.PHOSPHORYLATION_CAUSE, biopax_model_str=None):
        ''' Returns list and dict of INDRA statements.
        Parameters
        ----------
        biopax_owl_strings: list of str
            List of OWL file names to be processed
        use_cache : bool
            Process biopax_owl or load dataset from cache
        mode : str
            Choose different data split from ["train", "eval", "test", "all"]
        question_type : Question
            Question type for which the data should be generated
        biopax_model_str : str
            file name as a string for the corresponding and cached BiopaxProcessor

        Returns
        -------
        list
            List of indra.statements.statements
        dict
            Dict of (str or tuple(str) of canonical protein theme, indra.statements.statements)
        '''
        logger.info("Get dataset for question type {} and split {}".format(question_type, mode))

        if biopax_model_str is None:
            biopax_model_str = PID_MODEL_FAMILIES

        if not use_cache:
            biopax_model_statements = []
            for biopax_owl in biopax_owl_strings:
                biopax_model = biopax.process_owl(biopax_owl)
                biopax_model_statements += biopax_model.statements
            # eliminate_exact_duplicates() from INDRA class BiopaxProcessor
            biopax_model_statements = list({stmt.get_hash(shallow=False, refresh=True): stmt for stmt in biopax_model_statements}.values())
            with open(biopax_model_str, 'wb') as handle:
                # max_rec = 0x100000
                # # May segfault without this line. 0x100 is a guess at the size of each stack frame.
                # resource.setrlimit(resource.RLIMIT_STACK, [0x100 * max_rec, resource.RLIM_INFINITY])
                # sys.setrecursionlimit(max_rec)
                pickle.dump(biopax_model_statements, handle)
        else:
            with open(biopax_model_str, 'rb') as handle:
                # biopax_model_statements = pickle.load(handle).statements
                biopax_model_statements = pickle.load(handle)

        # logger.info(type(biopax_model))
        # logger.info(type(biopax_model.statements))

        try:
            biopax_type = QUESTION_TO_BIOPAX_TYPE[question_type.name]
        except KeyError:
            logger.warn("{} not implemented yet".format(question_type.name))
            return None, None

        implemented_classes = (bp.Modification, bp.RegulateAmount, bp.RegulateActivity, bp.Gap, bp.Gef, bp.Conversion, bp.Complex)
        if not issubclass(biopax_type, implemented_classes):
            logger.WARN("{} not implemented yet".format(question_type))
            return None, None

        events = []
        events_substrates = set()

        # Collect some statistics about different kinds of phosphorylation_argument
        # All phosphorylations have enzyme and substrate in INDRA syntax
        events_control_pairs = set()
        # events_by_complexes = []

        for statement in tqdm(biopax_model_statements, disable=TQDM_DISABLE):
            # Split event substrates along statements of all different question types
            current_type = type(statement)
            if current_type != bp.Complex:
                try:
                    substrate_attr, enzyme_attr = cls.get_attribute_strings(current_type)
                    substrate = getattr(statement, substrate_attr)
                    enzyme = getattr(statement, enzyme_attr)
                    if "UP" not in substrate.db_refs:  # For now, ignore substrates without Uniprot IDs
                        continue
                        # events_substrates.add((substrate.name, ""))
                    else:
                        events_substrates.add((substrate.name, substrate.db_refs["UP"]))
                except NotImplementedError:
                    continue
                if isinstance(statement, biopax_type) or (question_type == Question.STATECHANGE_CAUSE and type(statement) in STATECHANGE_BIOPAX_TYPES) \
                        or (question_type == Question.STATECHANGE_COMPLEXCAUSE and type(statement) in STATECHANGE_BIOPAX_TYPES):
                    events.append(statement)
                    if (substrate.name, enzyme.name) not in events_control_pairs and substrate.name != enzyme.name:
                        events_control_pairs.add((substrate.name, enzyme.name))
                    # if len(enzyme.bound_conditions) > 0:
                    #     events_by_complexes.append(statement)
                    for complex_part in enzyme.bound_conditions:
                        if (substrate.name, complex_part.agent.name) not in events_control_pairs \
                                and substrate.name != complex_part.agent.name \
                                and substrate.name[:-1] != complex_part.agent.name[:-1]:  # For Protein Families, e.g., AKT1 and AKT2
                            events_control_pairs.add((substrate.name, complex_part.agent.name))
        for statement in tqdm(biopax_model_statements, disable=TQDM_DISABLE):
            # Split event substrates along statements of all different question types
            current_type = type(statement)
            if current_type == bp.Complex:  # Only use Complex substrates which are substrates above
                # TODO: See comment below
                # if isinstance(statement, biopax_type):
                #     events.append(statement)
                statement_added = False
                for i, member in enumerate(statement.members):
                    # TODO: Add all Complex proteins as potential substrates
                    # For now, ignore proteins without mention in another event type (uncomment following lines) because run time too long
                    # if member.name not in events_substrates:
                    #     if "UP" not in member.db_refs:
                    #         events_substrates.add((member.name, ""))
                    #     else:
                    #         events_substrates.add((member.name, member.db_refs["UP"]))
                    # if isinstance(statement, biopax_type):
                    #     for j, complex_member in enumerate(statement.members):
                    #         if i >= j or member.name == complex_member.name:
                    #             continue
                    #         # Complexes may include two members of the same family, e.g., AKT1 and AKT2, or even the same protein twice
                    #         pair_name = tuple(sorted([member.name, complex_member.name]))
                    #         if pair_name not in events_control_pairs:
                    #             events_control_pairs.add(pair_name)
                    if isinstance(statement, biopax_type):
                        if "UP" not in member.db_refs:  # For now, ignore substrates without Uniprot IDs
                            # member_name = (member.name, "")
                            continue
                        else:
                            member_name = (member.name, member.db_refs["UP"])
                        if member_name in events_substrates and not statement_added:
                            events.append(statement)
                            statement_added = True
                        for j, complex_member in enumerate(statement.members):
                            if i >= j or member.name == complex_member.name:
                                continue
                            if "UP" not in complex_member.db_refs:  # For now, ignore substrates without Uniprot IDs
                                # Because they can also be CHEBI IDs
                                # complex_member_name = (complex_member.name, "")
                                continue
                            else:
                                complex_member_name = (complex_member.name, complex_member.db_refs["UP"])
                            if complex_member_name in events_substrates and not statement_added:
                                events.append(statement)
                                statement_added = True
                            # Complexes may include two members of the same family, e.g., AKT1 and AKT2, or even the same protein twice
                            pair_name = tuple(sorted([member.name, complex_member.name]))
                            if pair_name not in events_control_pairs:
                                events_control_pairs.add(pair_name)
                            # events_by_complexes.append(statement)

        # Output statistics
        logger.info("Number {} INDRA events".format(question_type))
        logger.info(len(events))
        # logger.info(events[:10])
        # logger.info("Number {} INDRA events with Complex causes".format(question_type))
        # logger.info(len(events_by_complexes))
        logger.info("Protein Pairs {} as causes".format(question_type))
        logger.info(len(events_control_pairs))
        # logger.info(events_control_pairs)  # There are some chemical and other stuff in here, that is where the bigger number comes from

        events_all = []
        events_train = []
        events_dev = []
        events_test = []

        events_all_dict = {}
        events_train_dict = {}
        events_dev_dict = {}
        events_test_dict = {}

        # Protein families
        substrate_names = set()
        substrates_without_db_id = set()
        substrates_with_multiple_db_ids = set()
        substrate_pfam_dict = {}
        pfam_db = SqliteDict(PFAM_DB, tablename='pfam_dict', flag='r')
        protein_families = {}
        for substrate in events_substrates:
            substrate_names.add(substrate[0].lower())
            if substrate[0].startswith("MAPK"):
                dict_entry = protein_families.setdefault("MAPK", [])
            elif substrate[0].startswith("MAP2K"):
                dict_entry = protein_families.setdefault("MAP2K", [])
            elif substrate[1] in pfam_db:
                dict_entry = protein_families.setdefault(tuple(sorted(pfam_db[substrate[1]])), [])
                substrate_pfam_dict.setdefault(substrate[0], set())
                substrate_pfam_dict[substrate[0]].add(tuple(sorted(pfam_db[substrate[1]])))
            else:  # If no fitting family name or no DB ID
                substrate_pfam_dict.setdefault(substrate[0], set())
                continue
            dict_entry.append((substrate[0],))
        for name, pfam_ids in substrate_pfam_dict.items():
            if len(pfam_ids) == 0:  # Have not seen the name in any other substrate
                dict_entry = protein_families.setdefault(name, [])
                dict_entry.append((name,))
                substrates_without_db_id.add(name)
            elif len(pfam_ids) > 1:  # Same name but multiple, different db ids and pfam families
                substrates_with_multiple_db_ids.add(name)
        family_names = list(sorted(protein_families.items(), key=lambda x: (len(x[1]), sorted(x[1]))))
        # logger.info("First 100 family names")
        # logger.info(family_names[:100])

        events_subjects = []
        events_subjects_0 = []
        events_subjects_1 = []
        events_subjects_2 = []

        # Train Test Data Split / 60 10 30
        # Use seed to ensure reproducibility and no data leakage between simple and complex question types!
        random.Random(TRAIN_TEST_SPLIT_SEED).shuffle(family_names)
        train_dev_split = int(TRAIN_SPLIT * len(events_substrates))
        dev_test_split = int(DEV_SPLIT * len(events_substrates))
        # logger.info(len(events_substrates))
        for family, unsorted_family_members in family_names:
            family_members = sorted(unsorted_family_members)
            events_subjects += family_members
            if len(events_subjects_0) + len(events_subjects_1) >= dev_test_split:
                events_subjects_2 += family_members
            elif len(events_subjects_0) >= train_dev_split:
                events_subjects_1 += family_members
            else:
                events_subjects_0 += family_members
        # logger.info(len(events_subjects_0))
        # logger.info(len(events_subjects_1))
        # logger.info(events_subjects_1[:25])
        # logger.info(len(events_subjects_2))
        # logger.info(events_subjects_2[:25])
        # exit()
        # logger.info(events_subjects_0[:10])

        # Artificially limit the amount of examples for _COMPLEX_PAIR and _COMPLEX_MULTI
        # because there are too many examples in both training and test set
        if question_type.name.endswith(("_CAUSE", "_SITE")):
            events_subjects_train = events_subjects_0
            events_subjects_dev = events_subjects_1
            events_subjects_test = events_subjects_2
        elif question_type.name.endswith(("_COMPLEXCAUSE", "_COMPLEXSITE")):
            events_subjects_train = []
            events_subjects_dev = []
            events_subjects_test = []
            substrates_train = events_subjects_0
            substrates_dev = events_subjects_1
            substrates_test = events_subjects_2
            for substrate, complex_kinase in events_control_pairs:
                substrate_tuple = (substrate,)
                if substrate_tuple in substrates_train:
                    events_subjects_train.append((substrate, complex_kinase))
                elif substrate_tuple in substrates_dev:
                    events_subjects_dev.append((substrate, complex_kinase))
                elif substrate_tuple in substrates_test:
                    events_subjects_test.append((substrate, complex_kinase))
                else:
                    logger.warn(substrate)
                    logger.warn(events_control_pairs)
                    logger.warn(events_subjects)
                    raise NotImplementedError
        elif question_type.name.endswith("COMPLEX_PAIR"):
            events_subjects_train = events_subjects_0  # Keep here because during training both COMPLEX_PAIR and COMPLEX_MULTI questions are independent
            events_subjects_dev = events_subjects_1[:len(events_subjects_1)]
            events_subjects_test = events_subjects_2[:len(events_subjects_2)]
        elif question_type.name.endswith("COMPLEX_MULTI"):
            events_subjects_train = []
            events_subjects_dev = []
            events_subjects_test = []
            substrates_train = events_subjects_0[len(events_subjects_0) // 2:len(events_subjects_0)]
            substrates_dev = events_subjects_1[:len(events_subjects_1) // 16]
            substrates_test = events_subjects_2[:len(events_subjects_2) // 1]
            # For later storage and multi-turn question handling
            cached_substrates = list(zip(*substrates_train))[0] + list(zip(*substrates_dev))[0] + list(zip(*substrates_test))[0]
            with open(EVENT_SUBSTRATES_COMPLEX_MULTI, 'wb') as handle:
                pickle.dump(cached_substrates, handle)
            # Handling end
            for complex_participant_1, complex_participant_2 in events_control_pairs:
                comp_1_tuple = (complex_participant_1,)
                comp_2_tuple = (complex_participant_2,)
                if comp_1_tuple in substrates_train and comp_2_tuple in substrates_train:
                    events_subjects_train.append((complex_participant_1, complex_participant_2))
                elif comp_1_tuple in substrates_dev or comp_2_tuple in substrates_dev:
                    events_subjects_dev.append((complex_participant_1, complex_participant_2))
                elif comp_1_tuple in substrates_test or comp_2_tuple in substrates_test:
                    events_subjects_test.append((complex_participant_1, complex_participant_2))
                # TODO: Uncomment if we include all potential complex substrates and not only these one with another event
                # else:
                #     events_subjects_test.append((complex_participant_1, complex_participant_2))
            events_subjects = events_subjects_train + events_subjects_dev + events_subjects_test
        else:
            raise NotImplementedError

        question_subjects_all = set()
        question_subjects_train = set()
        question_subjects_dev = set()
        question_subjects_test = set()
        for event in tqdm(events, disable=TQDM_DISABLE):
            for subject in cls.get_subjects(event, question_type):
                if subject in events_subjects:
                    events_all.append(event)
                    events_all_dict.setdefault(subject, []).append(event)
                    question_subjects_all.add(subject)
                if subject in events_subjects_train:
                    events_train.append(event)
                    events_train_dict.setdefault(subject, []).append(event)
                    question_subjects_train.add(subject)
                elif subject in events_subjects_dev:
                    events_dev.append(event)
                    events_dev_dict.setdefault(subject, []).append(event)
                    question_subjects_dev.add(subject)
                elif subject in events_subjects_test:
                    events_test.append(event)
                    events_test_dict.setdefault(subject, []).append(event)
                    question_subjects_test.add(subject)
                else:
                    logger.debug("Subject that is ignored: {}".format(subject))

        logger.info("Whole set: Number of events, Number of subjects")
        logger.info(len(events_all))
        logger.info(len(events_all_dict))
        logger.info("Train Set")
        logger.info(len(events_train))
        logger.info(len(events_train_dict))
        logger.info("Dev Set")
        logger.info(len(events_dev))
        logger.info(len(events_dev_dict))
        logger.info("Test Set")
        logger.info(len(events_test))
        logger.info(len(events_test_dict))

        # logger.info(len(events_subjects_0))
        # logger.info(len(substrates_train))
        # logger.info(len(events_subjects_train))
        # logger.info(events_subjects_train[:10])
        # logger.info(len(events_subjects_1))
        # logger.info(len(substrates_dev))
        # logger.info(substrates_dev)
        # logger.info(len(events_subjects_dev))
        # logger.info(events_subjects_dev[:10])
        # logger.info(len(events_subjects_2))
        # logger.info(len(substrates_test))
        # logger.info(substrates_test)
        # logger.info(len(events_subjects_test))
        # logger.info(events_subjects_test[:10])
        # logger.info(len(events_subjects))

        logger.info("Total number of substrates (Multiple DB IDs possible)")
        logger.info(len(events_substrates))
        logger.info("Total number of substrates (just unique names)")
        logger.info(len(substrate_names))
        logger.info("Substrates without DB IDs/Pfam memberships")
        logger.info(len(substrates_without_db_id))
        logger.info(list(substrates_without_db_id)[:20])
        logger.info("Substrates with multiple DB IDs/Pfam memberships")
        logger.info(len(substrates_with_multiple_db_ids))
        logger.info(list(substrates_with_multiple_db_ids))
        logger.info("Dev split")
        logger.info(train_dev_split)
        logger.info("Test split")
        logger.info(dev_test_split)
        logger.info("{} number of substrates".format(question_type))
        logger.info(len(question_subjects_all))
        logger.info("{} number of TRAIN substrates".format(question_type))
        logger.info(len(question_subjects_train))
        logger.info("{} number of DEV substrates".format(question_type))
        logger.info(len(question_subjects_dev))
        logger.info("{} number of TEST substrates".format(question_type))
        logger.info(len(question_subjects_test))

        if mode == "all":
            return events_all, events_all_dict
        elif mode == "train":
            return events_train, events_train_dict
        elif mode == "eval":
            return events_dev, events_dev_dict
        else:
            return events_test, events_test_dict

    @classmethod
    def get_custom_dataset(cls, question_type):
        ''' Get custom dataset from user input. '''
        from data_processing.datatypes import GeneIDToNames
        if GeneIDToNames.lookup_table is None:
            GeneIDToNames.initialize()
        try:
            biopax_type = QUESTION_TO_BIOPAX_TYPE[question_type.name]
        except KeyError:
            logger.warn("{} not implemented yet".format(question_type.name))
            return None

        implemented_classes = (bp.Modification, bp.RegulateAmount, bp.RegulateActivity, bp.Gap, bp.Gef, bp.Conversion, bp.Complex)
        if not issubclass(biopax_type, implemented_classes):
            logger.WARN("{} not implemented yet".format(question_type))
            return None

        event_dict = {}
        print("Inputting event substrates")
        while True:
            kwargs = {}
            gene_input = input("Enter substrate Gene ID or name. \nEnter 'x' to finish: \n")
            if gene_input == "x":
                break
            gene_synonyms = []
            if gene_input in GeneIDToNames.lookup_table:
                gene_symbol = GeneIDToNames.lookup_table[gene_input][0]
                gene_synonyms = list(GeneIDToNames.lookup_table[gene_input][1])
            event_subject = (gene_symbol,)
            kwargs["sub"] = bp.Agent(gene_symbol, gene_synonyms)

            if question_type.name.endswith(("_COMPLEXCAUSE", "_COMPLEXSITE", "COMPLEX_MULTI")):
                kinase_input = input("Input kinase Gene ID or name: ")
                kinase_synonyms = []
                if kinase_input in GeneIDToNames.lookup_table:
                    kinase_symbol = GeneIDToNames.lookup_table[kinase_input][0]
                    kinase_synonyms = list(GeneIDToNames.lookup_table[kinase_input][1])
                    kwargs["enz"] = bp.Agent(kinase_symbol, kinase_synonyms)
                    event_subject = (gene_symbol, kinase_symbol)
            else:
                kwargs["enz"] = bp.Agent("ENZYM")

            indra_event = biopax_type(**kwargs)
            event_dict[event_subject] = [indra_event]

        return event_dict

    @classmethod
    def get_all_indra_agents(cls, indra_statement, names=None):
        ''' Extractes all INDRA agent from one INDRA statement. '''

        all_agents = []
        agents = []
        agent_names = []
        if type(indra_statement) != bp.Complex:
            substrate_attr, enzyme_attr = cls.get_attribute_strings(type(indra_statement))
            substrate_agent = getattr(indra_statement, substrate_attr)
            enzyme_agent = getattr(indra_statement, enzyme_attr)
            all_agents += [substrate_agent, enzyme_agent]
            for bound_condition in substrate_agent.bound_conditions + enzyme_agent.bound_conditions:
                all_agents.append(bound_condition.agent)
        else:
            for member in indra_statement.members:
                all_agents.append(member)
        if names is None:
            agents = all_agents
        else:
            for agent in all_agents:
                if agent.name in names and agent.name not in agent_names:
                    agents.append(agent)
                    agent_names.append(agent.name)
        return agents

    @staticmethod
    def get_unique_args_statements(subjects, indra_statements, question_type):
        ''' Get unique arguments for list of indra statements '''

        statements_with_unique_args = []
        argument_set = set()
        argument_agents_list = []
        args = []
        args_agents = []
        args_statements = []
        biopax_type = QUESTION_TO_BIOPAX_TYPE[question_type.name]
        if biopax_type != bp.Complex:
            _, enzyme_attr = IndraDataLoader.get_attribute_strings(biopax_type)
            for statement in indra_statements:
                if question_type in (Question.STATECHANGE_CAUSE, Question.STATECHANGE_COMPLEXCAUSE):
                    _, enzyme_attr = IndraDataLoader.get_attribute_strings(type(statement))
                enzyme_agent = getattr(statement, enzyme_attr)
                if question_type.name.endswith("_CAUSE"):
                    if enzyme_agent.name not in args:
                        args.append(enzyme_agent.name)
                        args_statements.append(statement)
                        args_agents.append(enzyme_agent)
                    if len(enzyme_agent.bound_conditions) > 0:
                        for binding_arg in enzyme_agent.bound_conditions:
                            if binding_arg.agent.name not in args:
                                args.append(binding_arg.agent.name)
                                args_statements.append(statement)
                                args_agents.append(binding_arg.agent)
                elif question_type.name.endswith("SITE"):  # Used for both _SITE and _COMPLEXSITE question types
                    if statement.residue is not None and statement.position is not None and statement.residue + statement.position not in args:
                        args.append(statement.residue + statement.position)
                        args_statements.append(statement)
                        args_agents.append(statement.residue + statement.position)
                elif question_type.name.endswith("COMPLEXCAUSE"):
                    enzymes = [enzyme_agent] + [complex_partner.agent for complex_partner in enzyme_agent.bound_conditions]
                    for enzym in enzymes:
                        if enzym.name not in args and enzym.name not in subjects:
                            args.append(enzym.name)
                            args_statements.append(statement)
                            args_agents.append(enzym)
        else:  # bp.Complex
            for statement in indra_statements:
                for member in statement.members:
                    if member.name not in args and member.name not in subjects:
                        args.append(member.name)
                        args_statements.append(statement)
                        args_agents.append(member)

        for i, arg in enumerate(args):
            if arg not in argument_set:
                argument_set.add(arg)
                statements_with_unique_args.append(args_statements[i])
                argument_agents_list.append(args_agents[i])

        return statements_with_unique_args, argument_agents_list

    @classmethod
    def get_subjects(cls, event, question_type):
        subjects = []
        # Firstly, handle "COMPLEX_PAIR" and "COMPLEX_MULTI" question types
        if question_type.name.endswith("COMPLEX_PAIR"):
            for member in event.members:
                subjects.append((member.name,))
        elif question_type.name.endswith("COMPLEX_MULTI"):
            for i, member in enumerate(event.members):
                for j, complex_member in enumerate(event.members):
                    if i < j and member.name != complex_member.name:
                        subjects.append(tuple(sorted([member.name, complex_member.name])))
        # Secondly, handle other question types
        else:
            substrate_attr, enzyme_attr = cls.get_attribute_strings(type(event))
            substrate_agent = getattr(event, substrate_attr)
            enzyme_agent = getattr(event, enzyme_attr)
            if question_type.name.endswith(("_CAUSE", "_SITE")):
                substrates = [substrate_agent.name]  # + [complex_partner.agent.name for complex_partner in substrate_agent.bound_conditions]
                for substrate in substrates:
                    subjects.append((substrate,))
            elif question_type.name.endswith(("COMPLEXCAUSE", "COMPLEXSITE")):
                substrates = [substrate_agent.name]  # + [complex_partner.agent.name for complex_partner in substrate_agent.bound_conditions]
                for substrate in substrates:
                    enzyms = [enzyme_agent.name] + [complex_partner.agent.name for complex_partner in enzyme_agent.bound_conditions]
                    if len(enzyms) > 0:
                        for enzym in enzyms:
                            subjects.append((substrate, enzym))
        return subjects

    @staticmethod
    def get_attribute_strings(biopax_type):
        if issubclass(biopax_type, bp.Modification):
            substrate_attr = "sub"
            enzyme_attr = "enz"
        elif issubclass(biopax_type, (bp.RegulateAmount, bp.RegulateActivity)):
            substrate_attr = "obj"
            enzyme_attr = "subj"
        elif issubclass(biopax_type, bp.Gap):
            substrate_attr = "ras"
            enzyme_attr = "gap"
        elif issubclass(biopax_type, bp.Gef):
            substrate_attr = "ras"
            enzyme_attr = "gef"
        elif issubclass(biopax_type, bp.Conversion):
            substrate_attr = "subj"
            enzyme_attr = "subj"  # TODO: Adjust attribute values
        elif issubclass(biopax_type, bp.Complex):
            raise TypeError("bp.Complex")
        else:
            raise NotImplementedError
        return substrate_attr, enzyme_attr


if __name__ == "__main__":
    from data_processing.datatypes import QUESTION_TYPES
    from configs import NETPATH_OWL, PANTHER_OWL, REACTOME_OWL, NETPATH_MODEL, PANTHER_MODEL, REACTOME_MODEL, PID_MODEL, PID_OWL, PID_MODEL_FAMILIES
    # from configs import PID_MODEL_EXPANDED

    logger.setLevel(logging.INFO)
    # logger.info(IndraDataLoader.get_attribute_strings(bp.Activation))
    # exit()
    question_types = [Question(question) for question in QUESTION_TYPES]
    # question_types = [Question.PHOSPHORYLATION_CAUSE, Question.ACETYLATION_CAUSE, Question.UBIQUITINATION_CAUSE,
    #                   Question.EXPRESSION_CAUSE, Question.INHIBEXPRESSION_CAUSE, Question.STATECHANGE_CAUSE]
    question_types = [Question.COMPLEX_MULTI]
    # question_types = [Question.STATECHANGE_CAUSE, Question.STATECHANGE_COMPLEXCAUSE]
    # question_types = [Question.PHOSPHORYLATION_CAUSE]
    # question_types = [Question.PHOSPHORYLATION_COMPLEXSITE]
    # question_types = [Question.EXPRESSION_CAUSE]
    # question_types = [Question.CONVERSION_PRODUCT]
    for i, question_type in enumerate(question_types):
        event_list, event_dict = IndraDataLoader.get_dataset(
            biopax_owl_strings=OWL_LIST, use_cache=True, mode="test", question_type=question_type, biopax_model_str=OWL_STATEMENTS)
        # event_list, event_dict = IndraDataLoader.get_dataset(
        #     biopax_owl_strings=[PID_OWL], use_cache=True, mode="test", question_type=question_type, biopax_model_str=PID_MODEL_FAMILIES)
        # print(list(event_dict.keys()))
        # break
        # print(len(event_dict[("STAT1",)]))
        # print(len(event_dict[("STAT2",)]))
        # print(len(event_dict[("STAT3",)]))
        # print(len(event_dict[("STAT4",)]))
        # print(len(event_dict[("STAT5A",)]))
        # print(len(event_dict[("STAT5B",)]))

    # event_list, event_dict = IndraDataLoader.get_dataset(use_cache=True, mode="all", question_type=Question.PHOSPHORYLATION_COMPLEXCAUSE, biopax_model_str = PID_MODEL)
    # event_list2, event_dict2 = IndraDataLoader.get_dataset(use_cache=True, mode="all", question_type=Question.PHOSPHORYLATION_COMPLEXCAUSE, biopax_model_str = PID_MODEL_FAMILIES)
    # subjects = set(event_dict.keys())
    # subjects_2 = set(event_dict2.keys())
    # logger.info(subjects)
    # logger.info(len(subjects))
    # logger.info(len(subjects_2))
    # print()
    # logger.info(subjects - subjects_2)
    # logger.info(subjects_2 - subjects)
    # print(len(event_dict[("STAT1",)]))
    # print(len(event_dict[("STAT2",)]))
    # print(len(event_dict[("STAT3",)]))
    # print(len(event_dict[("STAT4",)]))
    # print(len(event_dict[("STAT5A",)]))
    # print(len(event_dict[("STAT5B",)]))
    # logger.info(event_list[:3])
    # logger.info(vars(event_list[0]))
    # logger.info(vars(event_list[1]))
    # logger.info(vars(event_list[2]))
