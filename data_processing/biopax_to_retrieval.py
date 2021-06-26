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
import random
import logging

from sqlitedict import SqliteDict

import indra.sources.biopax as biopax
import indra.statements.statements as bp

from configs import PID_OWL, PID_MODEL_FAMILIES, TRAIN_TEST_SPLIT_SEED, TRAIN_SPLIT, DEV_SPLIT, PFAM_DB
from data_processing.datatypes import Question, QUESTION_TO_BIOPAX_TYPE, STATECHANGE_BIOPAX_TYPES

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)


class IndraDataLoader:
    ''' Loads and extracts data from BioPAX/OWL files into the INDRA format.
        Provides function for manipulation of biomolecular events in the INDRA format.
    '''

    @classmethod
    def get_dataset(cls, biopax_owl=PID_OWL, use_cache=True, mode="train", question_type=Question.PHOSPHORYLATION_CAUSE,
                    biopax_model_str=None, add_family_members=True):
        ''' Returns list and dict of INDRA statements.
        Parameters
        ----------
        use_cache : bool
            Process biopax_owl or load dataset from cache
        mode : str
            Choose different data split from ["train", "eval", "test", "all"]
        question_type : Question
            Question type for which the data should be generated
        biopax_model_str : str
            file name as a string for the corresponding and cached BiopaxProcessor
        add_family_members : bool
            Add family members of protein, e.g., Pkb alpha and Pkb beta, if biopax_model is None

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
            biopax_model = biopax.process_owl(biopax_owl, add_family_members)
            with open(biopax_model_str, 'wb') as handle:
                pickle.dump(biopax_model, handle)
        else:
            with open(biopax_model_str, 'rb') as handle:
                biopax_model = pickle.load(handle)

        # logger.info(type(biopax_model))
        # logger.info(type(biopax_model.statements))

        try:
            biopax_type = QUESTION_TO_BIOPAX_TYPE[question_type.name]
        except KeyError:
            logger.warn("{} not implemented yet".format(question_type.name))
            return None, None

        implemented_classes = (bp.Modification, bp.RegulateAmount, bp.RegulateActivity, bp.Gap, bp.Gef, bp.Conversion)
        if not issubclass(biopax_type, implemented_classes):
            logger.WARN("{} not implemented yet".format(question_type))
            return None, None

        events = []
        events_substrates = set()

        # Collect some statistics about different kinds of phosphorylation_argument
        # All phosphorylations have enzyme and substrate in INDRA syntax
        events_control_pairs = set()
        events_by_complexes = []

        substrate_attr, enzyme_attr = cls.get_attribute_strings(biopax_type)

        for statement in biopax_model.statements:
            # Split event substrates along statements of all different question types
            current_type = type(statement)
            try:
                substrate_attr, enzyme_attr = cls.get_attribute_strings(current_type)
                substrate = getattr(statement, substrate_attr)
                enzyme = getattr(statement, enzyme_attr)
                if substrate.name not in events_substrates:
                    if "UP" not in substrate.db_refs:
                        events_substrates.add((substrate.name, ""))
                    else:
                        events_substrates.add((substrate.name, substrate.db_refs["UP"]))
            except NotImplementedError:
                continue
            if isinstance(statement, biopax_type) or (question_type == Question.STATECHANGE_CAUSE and type(statement) in STATECHANGE_BIOPAX_TYPES) \
                    or (question_type == Question.STATECHANGE_COMPLEXCAUSE and type(statement) in STATECHANGE_BIOPAX_TYPES):
                events.append(statement)
                if (substrate.name, enzyme.name) not in events_control_pairs \
                        and substrate.name != enzyme.name:
                    events_control_pairs.add((substrate.name, enzyme.name))
                if len(enzyme.bound_conditions) > 0:
                    events_by_complexes.append(statement)
                for complex_part in enzyme.bound_conditions:
                    if (substrate.name, complex_part.agent.name) not in events_control_pairs \
                            and substrate.name != complex_part.agent.name \
                            and substrate.name[:-1] != complex_part.agent.name[:-1]:  # For Protein Families, e.g., AKT1 and AKT2
                        events_control_pairs.add((substrate.name, complex_part.agent.name))

        # Output statistics
        logger.info("Number {} INDRA events".format(question_type))
        logger.info(len(events))
        logger.info("Number {} INDRA events with Complex causes".format(question_type))
        logger.info(len(events_by_complexes))
        logger.info("Protein Pairs {} as causes".format(question_type))
        logger.info(len(events_control_pairs))

        events_all = []
        events_train = []
        events_dev = []
        events_test = []

        events_all_dict = {}
        events_train_dict = {}
        events_dev_dict = {}
        events_test_dict = {}

        # Protein families
        pfam_db = SqliteDict(PFAM_DB, tablename='pfam_dict', flag='r')
        protein_families = {}
        for substrate in events_substrates:
            if substrate[0].startswith("MAPK"):
                dict_entry = protein_families.setdefault("MAPK", [])
            elif substrate[0].startswith("MAP2K"):
                dict_entry = protein_families.setdefault("MAP2K", [])
            elif substrate[1] in pfam_db:
                dict_entry = protein_families.setdefault(tuple(sorted(pfam_db[substrate[1]])), [])
            else:
                dict_entry = protein_families.setdefault(substrate[0], [])
            dict_entry.append((substrate[0],))
        family_names = list(sorted(protein_families.items(), key=lambda x: (len(x[1]), sorted(x[1]))))
        # logger.info(family_names[:10])

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
        for family, family_members in family_names:
            events_subjects += family_members
            if len(events_subjects_0) + len(events_subjects_1) >= dev_test_split:
                events_subjects_2 += family_members
            elif len(events_subjects_0) >= train_dev_split:
                events_subjects_1 += family_members
            else:
                events_subjects_0 += family_members
        # logger.info(len(events_subjects_0))
        # logger.info(len(events_subjects_1))
        # logger.info(len(events_subjects_2))
        # exit()
        # logger.info(events_subjects_0[:10])

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
        else:
            raise NotImplementedError

        question_subjects_all = set()
        question_subjects_train = set()
        question_subjects_dev = set()
        question_subjects_test = set()
        for event in events:
            for potential_subject in cls.get_subjects(event, question_type):
                cls.update_modification_event_dict(potential_subject, event, events_all, events_all_dict)
                question_subjects_all.add(potential_subject)
                if potential_subject in events_subjects_train:
                    cls.update_modification_event_dict(potential_subject, event, events_train, events_train_dict)
                    question_subjects_train.add(potential_subject)
                elif potential_subject in events_subjects_dev:
                    cls.update_modification_event_dict(potential_subject, event, events_dev, events_dev_dict)
                    question_subjects_dev.add(potential_subject)
                elif potential_subject in events_subjects_test:
                    cls.update_modification_event_dict(potential_subject, event, events_test, events_test_dict)
                    question_subjects_test.add(potential_subject)
                else:
                    logger.debug("Subject that is ignored: {}".format(potential_subject))

        logger.info("Whole set")
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

        logger.info("Total number of substrates")
        logger.info(len(events_substrates))
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
        from data_processing.nn_output_to_indra import GENE_ID_TO_NAMES
        try:
            biopax_type = QUESTION_TO_BIOPAX_TYPE[question_type.name]
        except KeyError:
            logger.warn("{} not implemented yet".format(question_type.name))
            return None

        implemented_classes = (bp.Modification, bp.RegulateAmount, bp.RegulateActivity, bp.Gap, bp.Gef, bp.Conversion)
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
            if gene_input in GENE_ID_TO_NAMES:
                gene_symbol = GENE_ID_TO_NAMES[gene_input][0]
                gene_synonyms = list(GENE_ID_TO_NAMES[gene_input][1])
            event_subject = (gene_symbol,)
            kwargs["sub"] = bp.Agent(gene_symbol, gene_synonyms)

            if question_type.name.endswith(("_COMPLEXCAUSE", "_COMPLEXSITE")):
                kinase_input = input("Input kinase Gene ID or name: ")
                kinase_synonyms = []
                if kinase_input in GENE_ID_TO_NAMES:
                    kinase_symbol = GENE_ID_TO_NAMES[kinase_input][0]
                    kinase_synonyms = list(GENE_ID_TO_NAMES[kinase_input][1])
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

        substrate_attr, enzyme_attr = cls.get_attribute_strings(type(indra_statement))
        substrate_agent = getattr(indra_statement, substrate_attr)
        enzyme_agent = getattr(indra_statement, enzyme_attr)
        all_agents = []
        agents = []
        agent_names = []
        all_agents += [substrate_agent, enzyme_agent]
        for bound_condition in substrate_agent.bound_conditions + enzyme_agent.bound_conditions:
            all_agents.append(bound_condition.agent)
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
        biopax_type = QUESTION_TO_BIOPAX_TYPE[question_type.name]
        _, enzyme_attr = IndraDataLoader.get_attribute_strings(biopax_type)
        for statement in indra_statements:
            if question_type in (Question.STATECHANGE_CAUSE, Question.STATECHANGE_COMPLEXCAUSE):
                _, enzyme_attr = IndraDataLoader.get_attribute_strings(type(statement))
            enzyme_agent = getattr(statement, enzyme_attr)
            if question_type.name.endswith("_CAUSE"):
                args.append(enzyme_agent.name)
                args_agents.append(enzyme_agent)
                if len(enzyme_agent.bound_conditions) > 0:
                    for binding_arg in enzyme_agent.bound_conditions:
                        if binding_arg.agent.name not in args:
                            args.append(binding_arg.agent.name)
                            args_agents.append(binding_arg.agent)
            elif question_type.name.endswith("SITE"):  # Used for both _SITE and _COMPLEXSITE question types
                if statement.residue is not None and statement.position is not None:
                    args.append(statement.residue + statement.position)
                    args_agents.append(statement.residue + statement.position)
            elif question_type.name.endswith("COMPLEXCAUSE"):
                enzymes = [enzyme_agent] + [complex_partner.agent for complex_partner in enzyme_agent.bound_conditions]
                for enzym in enzymes:
                    if enzym.name not in args and enzym.name not in subjects:
                        args.append(enzym.name)
                        args_agents.append(enzym)

        for i, arg in enumerate(args):
            if arg not in argument_set:
                argument_set.add(arg)
                statements_with_unique_args.append(statement)
                argument_agents_list.append(args_agents[i])

        return statements_with_unique_args, argument_agents_list

    @classmethod
    def get_subjects(cls, event, question_type):
        substrate_attr, enzyme_attr = cls.get_attribute_strings(type(event))
        substrate_agent = getattr(event, substrate_attr)
        enzyme_agent = getattr(event, enzyme_attr)
        subjects = []
        if question_type.name.endswith(("_CAUSE", "_SITE")):
            substrates = [substrate_agent.name]  # + [complex_partner.agent.name for complex_partner in substrate_agent.bound_conditions]
            for substrate in substrates:
                subjects.append((substrate,))
        elif question_type.name.endswith(("COMPLEXCAUSE", "COMPLEXSITE")):
            substrates = [substrate_agent.name]  # + [complex_partner.agent.name for complex_partner in substrate_agent.bound_conditions]
            for substrate in substrates:
                enzyms = [enzyme_agent.name] + [complex_partner.agent.name for complex_partner in enzyme_agent.bound_conditions]
                if len(enzyms) > 1:
                    for enzym in enzyms:
                        subjects.append((substrate, enzym))
        return subjects

    @staticmethod
    def update_modification_event_dict(subject, event, event_list, event_dict):
        event_list.append(event)
        if subject not in event_dict:
            event_dict[subject] = []
        event_dict[subject].append(event)

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
        else:
            raise NotImplementedError
        return substrate_attr, enzyme_attr


if __name__ == "__main__":
    from data_processing.datatypes import QUESTION_TYPES
    # from configs import PANTHER_OWL, PANTHER_MODEL
    # from configs import NETPATH_OWL, NETPATH_MODEL
    # from configs import PID_MODEL_EXPANDED

    logger.setLevel(logging.INFO)
    # logger.info(IndraDataLoader.get_attribute_strings(bp.Activation))
    # exit()
    question_types = [Question(question) for question in QUESTION_TYPES]
    # question_types = [Question.PHOSPHORYLATION_CAUSE, Question.ACETYLATION_CAUSE, Question.UBIQUITINATION_CAUSE,
    #                   Question.EXPRESSION_CAUSE, Question.INHIBEXPRESSION_CAUSE, Question.STATECHANGE_CAUSE]
    question_types = [Question.PHOSPHORYLATION_CAUSE]
    # question_types = [Question.EXPRESSION_CAUSE]
    # question_types = [Question.CONVERSION_PRODUCT]
    for i, question_type in enumerate(question_types):
        event_list, event_dict = IndraDataLoader.get_dataset(
            biopax_owl=PID_OWL, use_cache=True, mode="test", question_type=question_type, biopax_model_str=PID_MODEL_FAMILIES)
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
