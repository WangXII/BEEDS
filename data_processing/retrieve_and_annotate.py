""" Get documents for INDRA(BioPAX) examples and annotate them  """

import os
import sys

if os.name == 'nt':
    root_path = "/".join(os.path.realpath(__file__).split("\\")[:-2])
else:
    root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.append(root_path)

import logging
# import json
import copy
# import random
import plotille

from tqdm import tqdm
from elasticsearch import Elasticsearch

import indra.statements.statements as bp
from data_processing.biopax_to_retrieval import IndraDataLoader
from data_processing.datatypes import Question, TRIGGER_WORDS, synonym_expansion, get_residue_and_position_list, get_subject_list
from data_processing.file_converter import generate_question
from data_processing.question_tagger import tag_question_detailed
from configs import DOC_STRIDE

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
es_logger = logging.getLogger("elasticsearch")
es_logger.setLevel(logging.WARNING)


def chunks(tagged_answer, max_seq_length, question_length, doc_stride=DOC_STRIDE):
    """Yield successive n-sized chunks from lst."""
    answer_length = max_seq_length - question_length
    for i, doc_index in enumerate(range(question_length - doc_stride, len(tagged_answer), answer_length)):
        if i == 0:
            doc_index += doc_stride
        yield tagged_answer[:question_length] + tagged_answer[doc_index:doc_index + answer_length]


class Retriever:
    """ Retrieve documents from an Elasticsearch Index. """

    def __init__(self, es_instance, primary_index, secondary_index, retrieval_size, datefilter=False,
                 retrieval_with_full_question=False):
        self.es_instance = es_instance
        self.primary_index = primary_index
        self.secondary_index = secondary_index
        self.retrieval_size = retrieval_size
        self.datefilter = datefilter
        self.retrieval_with_full_question = retrieval_with_full_question

    def retrieve_documents(self, subject_agents, question_type, primary_index_bool, retrieval_mode="standard", question=""):
        ''' Retrieve documents for a question or question answer pair (in case of training data)
        Parameters
        ----------
        subject_agents : list of indra.statement.Agent
            Theme of the question will be extracted from here, e.g., the substrate for Phosphorylation
        '''

        if primary_index_bool:
            index_name = self.primary_index
        else:
            index_name = self.secondary_index

        # Find documents relevenant to the question (Same that are found during prediction time)
        proteins = []
        for subject_agent in subject_agents:
            proteins.append(synonym_expansion(subject_agent, retrieval=retrieval_mode))
        triggers = []
        triggers.append(TRIGGER_WORDS[question_type.name])
        if question_type.name.endswith(("COMPLEXCAUSE", "COMPLEXSITE")):
            triggers.append(TRIGGER_WORDS["COMPLEX_PARTICIPANT"])
        arguments = {"must": proteins + triggers}

        match_phrase = True

        if self.retrieval_with_full_question:
            arguments = {"must": [[question]]}
            match_phrase = False

        query_body = self.build_es_query(arguments, datefilter=self.datefilter, match_phrase=match_phrase)
        result_question = self.es_instance.search(body=query_body, index=index_name, size=self.retrieval_size)

        # Store found document IDs for further processing
        # For training time, find documents relevant for the answer from the stored docs
        # Use them for the distant supervision sample
        doc_ids_questions, pubmed_id_questions, number_hits_questions = self.get_retrieval_result_stats(result_question)

        question_entities = proteins + triggers

        logger.debug("Subjects: ")
        logger.debug(subject_agents)
        logger.debug(question)
        logger.debug("Query body: ")
        logger.debug(query_body)

        logger.debug("Retrieval statistics")
        logger.debug(result_question)
        logger.debug("Pubmed IDs (question entities only): ")
        logger.debug(pubmed_id_questions)
        logger.debug("Number hits (question entities only): ")
        logger.debug(number_hits_questions)
        if logger.level == logging.DEBUG:
            input("Press any key to continue ")

        return result_question, pubmed_id_questions, number_hits_questions, question_entities

    @staticmethod
    def build_pubmed_es_query(pubmed_id, paragraph_id):
        request_body = {
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"name": pubmed_id}},
                        {"term": {"paragraph_id": paragraph_id}}
                    ]
                }
            }
        }
        return request_body

    @staticmethod
    def build_es_query(args, ids=None, datefilter=False, match_phrase=True):
        """ Builds Elastic Search Queries from the provided arguments
        Parameters
        ----------
        args : dict of list
            List of list. For instance, [["PKB alpha", "RAC", "PKB", "Protein kinase B", "AKT1"]]
        ids : list of document ids, used for refinement retrievals
        datefilter: bool
            Filter documents of an earlier publication date (before 2013) than the EVEX baseline

        Returns
        -------
        dict
            Returns HTTP Request Body for the corresponding ElasticSearch Query
        """

        request_body = {
            "query": {
                "bool": {
                    "must": [  # TO BE ADDED FROM THE ARGUMENTS
                    ]
                }
            }
        }

        argument = {
            "bool": {
                "should": [  # TO BE ADDED FROM THE ARGUMENTS
                ]
            }
        }

        if match_phrase:
            query_term = "match_phrase"
        else:
            query_term = "match"
        # For the primary arguments in the INDRA statement
        for list_arg in args["must"]:
            current_argument = copy.deepcopy(argument)
            for synonym in list_arg:
                # If there is shorter
                current_argument["bool"]["should"].append({query_term: {"text": synonym}})
            request_body["query"]["bool"]["must"].append(current_argument)

        # For other controllers in complexes (deprecated!)
        if "should" in args:
            request_body["query"]["bool"]["should"] = []
            for list_arg in args["should"]:
                current_argument = copy.deepcopy(argument)
                for synonym in list_arg:
                    current_argument["bool"]["should"].append({query_term: {"text": synonym}})
                request_body["query"]["bool"]["should"].append(current_argument)

        # Filter on subset of doc ids
        # https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-ids-query.html
        if ids is not None:
            request_body["query"]["bool"]["filter"] = {}
            request_body["query"]["bool"]["filter"]["ids"] = {}
            request_body["query"]["bool"]["filter"]["ids"]["values"] = []
            for doc_id in ids:
                request_body["query"]["bool"]["filter"]["ids"]["values"].append(doc_id)

        if datefilter is True:
            if "filter" not in request_body["query"]["bool"]:
                request_body["query"]["bool"]["filter"] = {}
            request_body["query"]["bool"]["filter"]["range"] = {}
            request_body["query"]["bool"]["filter"]["range"]["year"] = {}
            request_body["query"]["bool"]["filter"]["range"]["year"]["lt"] = 2013

        return request_body

    @staticmethod
    def get_retrieval_result_stats(result):
        doc_ids = []
        pubmed_ids = []
        for hit in result['hits']['hits']:
            document_text = hit['_source']['text']
            pubmed_id = str(hit['_source']['name'])
            par_id = str(hit['_source']['paragraph_id'])
            # if 'sentence_id' in hit['_source']:
            #     # sentence_id = str(hit['_source']['sentence_id'])
            #     sentence_id = "0"
            # else:
            #     sentence_id = "-1"
            # if pubmed_id != "" and len(document_text) <= 100000:  # Set limit to less than 100000 characters
            #     doc_ids.append(hit['_id'])
            #     pubmed_ids.append(pubmed_id + "_" + par_id + "_" + sentence_id)
            if pubmed_id != "" and len(document_text) <= 100000:  # Set limit to less than 100000 characters
                doc_ids.append(hit['_id'])
                pubmed_ids.append(pubmed_id + "_" + par_id)
        number_hits = result['hits']['total']['value']
        return doc_ids, pubmed_ids, number_hits


class DataBuilder:
    ''' Build question answer datasets with IndraDataloader and a Retriever. '''

    def __init__(self, retrieval_size, model_helper, tagger="simple", max_seq_length=384, limit_max_length=True,
                 primary_index="pubmed_sentences", secondary_index="pubmed_detailed", datefilter=False,
                 retrieval_with_full_question=False):
        self.tagger = tagger
        self.max_seq_length = max_seq_length
        self.limit_max_length = limit_max_length
        self.model_helper = model_helper
        self.tokenizer = model_helper.tokenizer

        self.retrieval_size = retrieval_size
        self.primary_index = primary_index
        self.secondary_index = secondary_index
        self.datefilter = datefilter
        self.retrieval_with_full_question = retrieval_with_full_question
        self.es = Elasticsearch(timeout=90, retry_on_timeout=True)
        self.retriever = Retriever(self.es, self.primary_index, self.secondary_index, self.retrieval_size,
                                   self.datefilter, self.retrieval_with_full_question)

    def generate_annotations(self, indra_statements, question_type, predict_bool=False):
        """ Create question/answer pairs for indra_statements of given question_type
        Parameters
        ----------
        indra_statements : list of list
            INDRA statements in the INDRA format, could be gold INDRA statements or predicted ones

        Returns
        -------
        list
            The annotated files for given question by BERT in IOB format
            Checking for maximum sequence length not done here
        """

        all_annotations = []
        subjects = list(indra_statements.keys())  # [0:2]
        logger.debug("Number Substrates")
        logger.debug(len(subjects))
        all_answer_stats = []

        for i, subject in enumerate(tqdm(subjects)):
            statements = indra_statements[subject]
            annotations, answer_stats = self.annotate_example(
                subject, statements, question_type, predict_bool)
            # INDRA Statement with no retrieved documents are skipped
            # annotations is a list with annotations with either 0 docs, or (bag_size) docs
            if answer_stats:
                all_answer_stats.append(answer_stats)

            if len(annotations) != 0:
                all_annotations.append(annotations)

        self.output_stats(all_answer_stats, question_type)

        return all_annotations

    def output_stats(self, all_answer_stats, question_type):
        # Number of INDRA statements
        all_statements = 0
        unique_statements = 0
        # Number of substrate kinase pairs
        pairs = 0
        pairs_with_docs = 0
        relaxed = 0
        more_docs_in_relaxed_mode = 0
        # Number of substrates
        substrate_with_docs = 0
        substrate_relaxed_mode = 0
        # Number of retrieved docs
        number_docs = 0
        # Number of tagged sequences
        tagged_questions = 0
        tagged_answers = 0
        answer_docs_histogram = []
        for i, stats in enumerate(all_answer_stats):
            pairs += stats["number_pairs"]
            unique_statements += stats["number_unique_statements"]
            all_statements += stats["statements"]
            if stats["number_question_docs"] > 0:
                number_docs += stats["number_question_docs"]
                answer_docs_histogram.append(stats["number_question_docs"])
                substrate_with_docs += 1
                pairs_with_docs += stats["number_pairs"]
            if stats["relaxed_mode"] > 0:
                substrate_relaxed_mode += 1
                relaxed += stats["relaxed_mode"]
            more_docs_in_relaxed_mode += stats["more_docs_in_relaxed_mode"]
            tagged_questions += stats["tagged_questions"]
            tagged_answers += stats["tagged_answers"]

        logger.info("Retrieval Stats:")
        print("Number of statements (total)")
        print(all_statements)
        print("Number of unique statements")
        print(unique_statements)

        print("{} Pairs".format(question_type.name))
        print(pairs)
        print("{} Pairs with possible retrieved answers".format(question_type.name))
        print(pairs_with_docs)
        print("Pair retrieval with relaxed constraints")
        print(relaxed)
        print("Pairs more docs in relaxed mode")
        print(more_docs_in_relaxed_mode)

        print("Number substrate entities")
        print(len(all_answer_stats))
        print("Substrates with retrieved docs")
        print(substrate_with_docs)

        print("Number of retrieved docs")
        print(number_docs)

        print("Tagged questions (total)")
        print(tagged_questions)
        print("Tagged questions with answers (total)")
        print(tagged_answers)

        print("Histogram of answer docs")
        print(answer_docs_histogram)
        if len(answer_docs_histogram) > 10:
            print(plotille.histogram(answer_docs_histogram, height=20))

    def annotate_example(self, subject, indra_statements, question_type, predict_bool=False):
        ''' Generate question and answer annotations for subject given indra_statements and question_type.
        Parameters
        ----------
        subject : tuple of str
            Entities which make up the whole subject
        indra_statement : list of indra.statement
            All statements share the same substrate (e.g., for phosphorylations)
        question_type: Question
            Question type to build data for
        predict_bool : bool
            If True, no answers are known, just retrieve and annotate for the question entities

        Returns
        -------
        list
            The annotated files for given question by BERT in IOB format
            Checking for maximum sequence length not done here
        '''

        subject_entities = []
        subject_agents = IndraDataLoader.get_all_indra_agents(indra_statements[0], subject)
        if len(subject_agents) != len(subject):
            logger.warn("Mismatch during preprocessing INDRA statements: Subject lengths do not add up!")
            logger.warn(subject_agents)
            logger.warn(subject)
            logger.warn(indra_statements[0])
            # Used in special case:
            # Phosphorylation(LCK(mods: (phosphorylation, Y, 505), bound: [JAK1, True], bound: [IL2RB, True], bound: [SOCS1, True]),
            # JAK1(bound: [LCK, True], bound: [IL2RB, True]), Y, 1023)
            annotations = []
            stats = {}
            return annotations, stats

        for agent in subject_agents:
            synonyms = synonym_expansion(agent, machine_reading=False)
            # Restrict number of entity mentions in a question
            if len(synonyms) > 5:
                synonyms = sorted(synonyms, key=lambda x: len(x))[:5]
                if agent.name not in synonyms:
                    synonyms.append(agent.name)  # Max six synonyms possible
            subject_entities.append(synonyms)

        if predict_bool:
            statements_with_unique_args = indra_statements
            unique_answer_agents = [None]
        else:
            statements_with_unique_args, unique_answer_agents = IndraDataLoader.get_unique_args_statements(subject, indra_statements, question_type)

        # Initialize some logging variables
        answers = set()
        question_ents = []
        answers_list = []
        all_doc_ids = []
        all_docs = []
        switched_to_relaxed_mode = 0
        more_docs_in_relaxed_mode = 0
        subject_list = get_subject_list(question_type.name, subject_agents)
        question = generate_question(subject_entities, question_type)

        # Get answer entities
        for i, answer_agent in enumerate(unique_answer_agents):
            # Used in special cases with complexes, e.g.,
            # Phosphorylation(MAPKAPK2(mods: (phosphorylation, T, 338), (phosphorylation, T, 334), (phosphorylation, S, 272),
            # (phosphorylation, T, 222), (phosphorylation, T, 25), bound: [MAPK14, True]), MAPK14(), S, 272)
            # Phosphorylation(LCK(mods: (phosphorylation, Y, 505), bound: [JAK1, True], bound: [IL2RB, True], bound: [SOCS1, True]),
            # JAK1(bound: [LCK, True], bound: [IL2RB, True]), Y, 1023)
            if isinstance(answer_agent, bp.Agent) and subject_agents[0].name == answer_agent.name:
                continue
            # Used for finetuning the first retrieval answer set
            if answer_agent is not None:  # Used during predictions:
                if question_type.name.endswith("CAUSE"):
                    # Includes both "COMPLEXCAUSE" and "_CAUSE" question types
                    answer = synonym_expansion(answer_agent, retrieval="relaxed")
                    # answer = ['mek', 'mekk1', 'mekk7', 'mekk', 'mekk3', 'cdc', 'mekk2', 'mekk4']
                elif question_type.name.endswith("SITE"):
                    # Split string, e.g., "Y138" to "Y" and "138"
                    answer = get_residue_and_position_list(answer_agent[:1], answer_agent[1:])
                answer_entities = answer
            else:
                answer_entities = []
            answers = answers.union(set(answer_entities))
            answers_list.append(answer_entities)

        # Conduct retrieval and get question entities
        result, doc_ids_questions, number_hits_questions, question_entities = \
            self.retriever.retrieve_documents(subject_agents, question_type, True, "standard", question)
        all_doc_ids += doc_ids_questions
        all_docs.append(result)
        if self.secondary_index != "None":
            result, doc_ids_questions, _, _ = \
                self.retriever.retrieve_documents(subject_agents, question_type, False, "standard", question)
            all_doc_ids += doc_ids_questions
            all_docs.append(result)
        if number_hits_questions < self.retrieval_size:
            switched_to_relaxed_mode += 1
            result, doc_ids_questions, new_hits_questions, question_entities = \
                self.retriever.retrieve_documents(subject_agents, question_type, True, "relaxed", question)
            all_doc_ids += doc_ids_questions
            all_docs.append(result)
            if self.secondary_index != "None":
                result, doc_ids_questions, _, _ = \
                    self.retriever.retrieve_documents(subject_agents, question_type, False, "relaxed", question)
                all_doc_ids += doc_ids_questions
                all_docs.append(result)
            if new_hits_questions > number_hits_questions:
                more_docs_in_relaxed_mode += 1
            question_ents = question_entities
        if question_ents == []:
            question_ents = question_entities

        all_doc_ids = list(dict.fromkeys(all_doc_ids))
        all_docs_dict = {}
        sentence_dict = {}
        for doc_results in all_docs:
            for hit in doc_results['hits']['hits']:
                if 'sentence_id' in hit['_source']:
                    pubmed_id = str(hit['_source']['name']) + "_" + str(hit['_source']['paragraph_id'])
                    document_text = hit['_source']['text']
                    all_docs_dict.setdefault(pubmed_id, "")
                    sentence_dict.setdefault(pubmed_id, document_text)
                else:
                    pubmed_id = str(hit['_source']['name']) + "_" + str(hit['_source']['paragraph_id'])
                    document_text = hit['_source']['text']
                    all_docs_dict[pubmed_id] = document_text

        # Retrieve paragraph for retrieved sentences
        for i in range(len(all_docs_dict)):
            doc_id, doc_text = list(all_docs_dict.items())[i]
            if doc_text == "":
                pubmed_id, paragraph_id = doc_id.split("_", 1)
                query_body = self.retriever.build_pubmed_es_query(pubmed_id, paragraph_id)
                result_question = self.retriever.es_instance.search(body=query_body, index=self.secondary_index, size=self.retrieval_size)
                for hit in result_question['hits']['hits']:
                    retrieved_doc_id = str(hit['_source']['name']) + "_" + str(hit['_source']['paragraph_id'])
                    # logger.info(pubmed_id)
                    # logger.info(paragraph_id)
                    # logger.info(retrieved_doc_id)
                    assert retrieved_doc_id == doc_id
                    document_text = hit['_source']['text']
                    all_docs_dict[doc_id] = document_text
                # if all_docs_dict[doc_id] == "":
                #     all_docs_dict[doc_id] = sentence_dict[pubmed_id]

        if len(unique_answer_agents) == 0:
            annotations, nb_questions, nb_answers = [[], 0, 0]
        else:
            annotations, nb_questions, nb_answers = self.build_model_bags(
                answers, question_ents, answers_list, all_doc_ids, all_docs_dict, subject_list, question)

        stats = {}
        stats["number_pairs"] = len(unique_answer_agents)
        stats["number_question_docs"] = len(all_doc_ids)
        stats["statements"] = len(indra_statements)
        stats["number_unique_statements"] = len(statements_with_unique_args)
        stats["relaxed_mode"] = switched_to_relaxed_mode
        stats["more_docs_in_relaxed_mode"] = more_docs_in_relaxed_mode
        stats["tagged_questions"] = nb_questions
        stats["tagged_answers"] = nb_answers

        return annotations, stats

    def build_model_bags(self, answers, question_entities, answers_list, all_doc_ids,
                         all_docs_dict, subject_list, question):
        annotations = []

        answers = [[self.model_helper.lower_string(answer) for answer in answers]]
        question_entities = [[self.model_helper.lower_string(entity) for entity in entities] for entities in question_entities]
        number_questions = 0
        number_answers = 0

        # Add all examples for evaluation
        logger.debug("Answers: {}".format(answers))
        logger.debug("Question entities: {}".format(question_entities))
        logger.debug("Question: {}".format(question))
        logger.debug("Subjects lists: {}".format(subject_list))
        logger.debug("Answers list: {}".format(answers_list))
        logger.debug("All documents: {}".format(all_doc_ids))
        subjects = subject_list
        if len(all_doc_ids) > self.retrieval_size:
            all_doc_ids = all_doc_ids[:self.retrieval_size]
        for pubmed_id in all_doc_ids:
            document_text = all_docs_dict[pubmed_id]

            tagged_sequence, at_least_one_answer, question_length = tag_question_detailed(
                document_text, self.model_helper, [], answers, question_entities, subjects, question, pubmed_id,
                max_seq_length=float('inf'), limit_max_length=self.limit_max_length, tagging_mode=self.tagger)
            if at_least_one_answer:
                number_answers += 1
            number_questions += 1
            annotations.append(tagged_sequence)

        return annotations, number_questions, number_answers


# Main
if __name__ == "__main__":
    from configs import PID_MODEL_FAMILIES, ModelHelper

    question_types = [question for question in Question]
    question_types = [Question.PHOSPHORYLATION_COMPLEXCAUSE]
    question_types = [Question.PHOSPHORYLATION_CAUSE]

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)
    biopax_logger = logging.getLogger("data_processing.biopax_to_retrieval")
    biopax_logger.setLevel(logging.WARN)
    question_tagger_logger = logging.getLogger("data_processing.question_tagger")
    question_tagger_logger.setLevel(logging.WARN)
    logger.setLevel(logging.INFO)

    model_helper = ModelHelper()

    prior_events = None
    data_builder = DataBuilder(100, model_helper, tagger="detailed", primary_index="pubmed_detailed", secondary_index="pubmed_sentences",
                               datefilter=False, retrieval_with_full_question=False)
    for i, question_type in enumerate(question_types):
        _, event_dict = IndraDataLoader.get_dataset(mode="train", question_type=question_type, biopax_model_str=PID_MODEL_FAMILIES)
        # data_builder.generate_annotations(dict(list(event_dict.items())[:1]), question_type)
        data_builder.generate_annotations(event_dict, question_type)
