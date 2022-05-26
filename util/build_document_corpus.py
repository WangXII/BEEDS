""" Extract abstracts from PubTator bioconcepts2pubtatorcentral.offset to specified directory
    and build Lucene Index in the ElasticSearch Folder
"""

import sys
import os

if os.name == 'nt':
    root_path = "/".join(os.path.realpath(__file__).split("\\")[:-2])
else:
    root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.append(root_path)

import logging

logging.basicConfig(level=logging.WARNING, datefmt="%Y/%m/%d %H:%M:%S",
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

import re
import parse
import lxml.etree as ET
import multiprocessing_logging

from nltk import tokenize
from pathlib import Path
from tqdm import tqdm
from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
from sqlitedict import SqliteDict
from multiprocessing import Manager, Pool, Process
from itertools import repeat
# from elasticsearch import Elasticsearch

from configs import PUBMED_META_DB, PMC_TO_PMID_MAPPER, PMC_DOCS_DIRECTORY, PUBMED_META_RAW_FILES_DIC, PUBMED_META_XSLT, PUBMED_FILES_PUBTATOR, TQDM_DISABLE

class IndexWriter:
    def __init__(self, index_name="pubmed_detailed", overwrite_index=False, index_batch_size=10_000):
        self.pubmed_meta_db = SqliteDict(PUBMED_META_DB, tablename='pubmed_meta', flag='r', autocommit=False)
        self.index_batch_size = index_batch_size
        self.document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index=index_name)
        if overwrite_index:
            self.document_store.delete_documents(index=index_name)
        self.documents = []
        self.documents_processed = 0

    def add_docs_to_index(self, text, pubmed_id, par_id, sentence_id=None):
        # Uncomment when indexing PubMed Central
        # self.pubmed_meta_db = SqliteDict(PUBMED_META_DB, tablename='pubmed_meta', flag='r', autocommit=False)
        if sentence_id is None:
            if pubmed_id not in self.pubmed_meta_db:
                self.documents.append({"content": text, "meta": {"name": pubmed_id, "paragraph_id": par_id, "sentence_id": -1, "year": 9999}})
            else:
                self.documents.append({"content": text, "meta": {"name": pubmed_id, "paragraph_id": par_id, "sentence_id": -1,
                                                                 "year": int(self.pubmed_meta_db[pubmed_id][0])}})
        else:
            if pubmed_id not in self.pubmed_meta_db:
                self.documents.append({"content": text, "meta": {"name": pubmed_id, "paragraph_id": par_id, "sentence_id": sentence_id, "year": 9999}})
            else:
                self.documents.append({"content": text, "meta": {"name": pubmed_id, "paragraph_id": par_id, "sentence_id": sentence_id,
                                                                 "year": int(self.pubmed_meta_db[pubmed_id][0])}})
        self.documents_processed += 1
        if self.documents_processed == self.index_batch_size:
            self.flush_docs_cache()
            self.documents = []
            self.documents_processed = 0

    def initialize_sql_dict(self):
        self.pubmed_meta_db = SqliteDict(self.db_path, tablename='pubmed_meta', flag='r', autocommit=False)

    def flush_docs_cache(self):
        try:
            self.document_store.write_documents(self.documents)
        except Exception as err:
            logger.error(err)


def get_pmid_from_pmcid_mapper():
    file_path = PMC_TO_PMID_MAPPER
    string_mapper = {}
    pmcid_mapper = {}
    format_strings = ["{journal} {year:.4} {month:3} {day:.2}", "{journal} {year:.4} {month:3}", "{journal} {year:.4}", "{journal}"]
    with open(file_path, "r") as f:
        txt_lines = f.readlines()
        for line in tqdm(txt_lines, disable=TQDM_DISABLE):
            content = line.split('\t')
            if len(content) != 5:
                continue
            r = None
            i = 0
            journal_date, other = content[1].split(";", maxsplit=1)
            other_list = other.split(":", maxsplit=1)
            if len(other_list) == 2:
                volume, pages = other_list
            else:
                volume = other_list[0]
                pages = None
            while r is None:
                r = parse.parse(format_strings[i], journal_date)
                if r is not None:
                    break
                i += 1
                if i == len(format_strings) and journal_date == "":
                    break
                elif i == len(format_strings) and journal_date != "":
                    print(content)
                    exit()
            if i == 0:
                string_id = "_".join([substring.strip(".") for substring in r["journal"].split(" ")] + [r["year"], r["month"], r["day"]])
            elif i == 1:
                string_id = "_".join([substring.strip(".") for substring in r["journal"].split(" ")] + [r["year"], r["month"]])
            elif i == 2:
                string_id = "_".join([substring.strip(".") for substring in r["journal"].split(" ")] + [r["year"]])
            elif i == 3:
                string_id = "_".join([substring.strip(".") for substring in r["journal"].split(" ")])
            else:
                string_id = ""
            string_id += "_" + volume
            if pages is not None:
                string_id += "_" + pages
            pmcid = content[2][3:]
            pmid = content[3][5:]
            string_mapper[string_id] = pmid
            pmcid_mapper[pmcid] = pmid

    return string_mapper, pmcid_mapper


def add_pubmed_central_parallel(stringid_mapper, pmcid_mapper, index_name="pubmed_detailed", overwrite_index=False,
                                num_workers=1, queue_size=None):
    path = PMC_DOCS_DIRECTORY
    num_workers = num_workers
    manager = Manager()
    if queue_size is None:
        max_queue_size = 100 * num_workers
    else:
        max_queue_size = queue_size
    work_queue = manager.Queue(maxsize=max_queue_size)

    # Start for workers
    pool = []
    for _ in range(num_workers):
        p = FullTextsConsumer(work_queue, index_name, overwrite_index, path, stringid_mapper, pmcid_mapper)
        p.start()
        pool.append(p)

    for i, directory in enumerate(tqdm(os.listdir(path))):
        if i <= 3000:
            continue
        dir_full_name = os.path.join(path, directory)
        if os.path.isdir(dir_full_name):
            for pubmed_file in tqdm(os.listdir(dir_full_name)):
                pubmed_file_name = os.path.join(path, directory, pubmed_file)
                if pubmed_file_name.endswith(".txt"):
                    pmc_name = pubmed_file[3:-4]
                    file_name = pubmed_file[:-4]
                    if pmc_name in pmcid_mapper:
                        pmid = pmcid_mapper[pmc_name]
                    elif file_name in stringid_mapper:
                        pmid = stringid_mapper[file_name]
                    else:
                        continue
                    if pmid == "":
                        continue
                    work_queue.put((pubmed_file_name, pmid))

    for _ in pool:
        work_queue.put(None)
        logger.info("Send poison pill...")

    for i, p in enumerate(pool):
        p.join()
        logger.info("Process {} has joined...".format(i))

    logger.info("Finished processing all Pubmed Central documents!")


class FullTextsConsumer(Process):
    def __init__(self, task_queue, index_name="pubmed", overwrite_index=False, path="", stringid_mapper=None, pmcid_mapper=None):
        Process.__init__(self)
        self.task_queue = task_queue
        self.index_name = index_name
        self.overwrite_index = overwrite_index
        self.path = path
        self.stringid_mapper = stringid_mapper
        self.pmcid_mapper = pmcid_mapper

    def run(self):
        logger.info("Starting process...")
        self.doc_writer = IndexWriter(index_name=self.index_name, overwrite_index=self.overwrite_index)
        while True:
            pubmed_files = self.task_queue.get()
            # Poision pill
            if pubmed_files is None:
                self.doc_writer.flush_docs_cache()
                logger.info("Shutting down process...")
                return
            pubmed_file_name = pubmed_files[0]
            pmid = pubmed_files[1]
            text_body_bool = False
            current_paragraph = ""
            paragraph_count = 1
            with open(pubmed_file_name, "r", encoding="utf-8", errors="replace") as f:
                txt_lines = f.readlines()
                for line in txt_lines:
                    if line == "==== Body\n":
                        text_body_bool = True
                    elif line == "==== Refs\n":
                        text_body_bool = False
                    if text_body_bool:
                        if line.strip() == "" and current_paragraph != "":
                            self.doc_writer.add_docs_to_index(current_paragraph.replace("\n", "").replace("\t", ""), int(pmid), paragraph_count)
                            sentences = tokenize.sent_tokenize(current_paragraph)
                            # print(sentences)
                            # print(len(sentences[0]))
                            # raise ValueError
                            for sentence_id, sentence in enumerate(sentences):
                                # Sentences longer than 1000 characters are not more precise than paragraphs
                                if len(sentence) > 10 and len(sentence) < 1000:
                                    self.doc_writer.add_docs_to_index(sentence.replace("\n", "").replace("\t", ""), int(pmid),
                                                                      paragraph_count, sentence_id=sentence_id)
                            current_paragraph = ""
                            paragraph_count += 1
                        else:
                            current_paragraph += line
                if current_paragraph.strip() != "":
                    self.doc_writer.add_docs_to_index(current_paragraph.replace("\n", "").replace("\t", ""), int(pmid), paragraph_count)
                    sentences = tokenize.sent_tokenize(current_paragraph)
                    for sentence_id, sentence in enumerate(sentences):
                        # Sentences longer than 1000 characters are not more precise than paragraphs
                        if len(sentence) > 10 and len(sentence) < 1000:
                            self.doc_writer.add_docs_to_index(sentence.replace("\n", "").replace("\t", ""), int(pmid),
                                                              paragraph_count, sentence_id=sentence_id)
            # self.doc_writer.flush_docs_cache()
            # logger.info("Finished indexing directory {}!".format(directory))


def extract_abstract(line):
    if re.match(r"^[0-9]*\|a\|", line):
        content = line.split("|a|", 1)
        pubmed_id = content[0]
        abstract = content[1]
        return pubmed_id, abstract
    else:
        return ()


def build_pubmed_date_dict():
    path = PUBMED_META_RAW_FILES_DIC
    db_path = PUBMED_META_DB
    xsl_filename = PUBMED_META_XSLT
    pubmed_meta_db = SqliteDict(db_path, tablename='pubmed_meta', flag='w', autocommit=False)
    with tqdm(os.scandir(path)) as it:
        for entry in it:
            if entry.name.endswith(".xml") and entry.is_file():
                dom = ET.parse(path + "/" + entry.name)
                xslt = ET.parse(xsl_filename)
                transform = ET.XSLT(xslt)
                newdom = str(transform(dom)).split('\n')
                for pubmed_id_dates in newdom:
                    if pubmed_id_dates != "":
                        pubmed_id, year, month = [x.strip() for x in pubmed_id_dates.split(',')]
                        if year == "":
                            year = "9999"
                        if month == "":
                            month = "Jan"
                        pubmed_meta_db[pubmed_id] = (year, month)
            pubmed_meta_db.commit()
    pubmed_meta_db.close()


class AbstractsConsumer(Process):
    def __init__(self, task_queue, index_name="pubmed", overwrite_index=False, index_batch_size=10_000):
        Process.__init__(self)
        self.task_queue = task_queue
        self.index_name = index_name
        self.overwrite_index = overwrite_index
        self.index_batch_size = index_batch_size

    def run(self):
        logger.info("Starting process...")
        self.doc_writer = IndexWriter(index_name=self.index_name, overwrite_index=self.overwrite_index, index_batch_size=self.index_batch_size)
        while True:
            line = self.task_queue.get()
            # Poision pill
            if line is None:
                self.doc_writer.flush_docs_cache()
                logger.info("Shutting down process...")
                return
            abstract = extract_abstract(line)
            if len(abstract) == 2 and abstract[1] != "\n":
                pubmed_id = abstract[0]
                self.doc_writer.add_docs_to_index(abstract[1].replace("\n", "").replace("\t", ""), int(pubmed_id), 0)
                sentences = tokenize.sent_tokenize(abstract[1])
                # print(sentences)
                # print(len(sentences[0]))
                # exit()
                for sentence_id, sentence in enumerate(sentences):
                    if len(sentence) < 1000:  # Sentences longer than 1000 characters are not more precise than paragraphs
                        self.doc_writer.add_docs_to_index(sentence.replace("\n", "").replace("\t", ""), int(pubmed_id), 0, sentence_id=sentence_id)


def build_pubmed_index(index_name="pubmed_detailed", overwrite_index=False, index_batch_size=10_000, num_workers=1, queue_size=None):
    document_path = PUBMED_FILES_PUBTATOR
    # /home/tmp/wangxida/elasticsearch-7.9.2
    # https://stackoverflow.com/questions/11196367/processing-single-file-from-multiple-processes
    # https://stackoverflow.com/questions/11196438/what-does-multiprocessing-process-init-self-do
    num_workers = num_workers
    manager = Manager()
    if queue_size is None:
        max_queue_size = 100 * num_workers
    else:
        max_queue_size = queue_size
    work_queue = manager.Queue(maxsize=max_queue_size)

    # Start for workers
    pool = []
    for _ in range(num_workers):
        p = AbstractsConsumer(work_queue, index_name, overwrite_index, index_batch_size)
        p.start()
        pool.append(p)

    with tqdm(total=Path(document_path).stat().st_size) as pbar:
        with open(document_path) as infile:
            line = infile.readline()
            while line:
                pbar.update(infile.tell() - pbar.n)
                work_queue.put(line)
                line = infile.readline()

    for _ in pool:
        work_queue.put(None)
        logger.info("Send poison pill...")

    for i, p in enumerate(pool):
        p.join()
        logger.info("Process {} has joined...".format(i))

    logger.info("Processed all Pubmed Abstracts!")


def count_sentences_pubmed():
    document_path = PUBMED_FILES_PUBTATOR
    total_sentences = 0
    with tqdm(total=Path(document_path).stat().st_size) as pbar:
        with open(document_path) as infile:
            line = infile.readline()
            while line:
                pbar.update(infile.tell() - pbar.n)
                abstract = extract_abstract(line)
                if len(abstract) == 2 and abstract[1] != "\n":
                    sentences = tokenize.sent_tokenize(abstract[1])
                    total_sentences += len(sentences)
                line = infile.readline()
    return total_sentences


# Main
if __name__ == "__main__":
    # Build PubMed Date Mapping
    # db_path = "/glusterfs/dfs-gfs-dist/wangxida/pubmed_meta.sqlite"
    # pubmed_meta_db = SqliteDict(db_path, tablename='pubmed_meta', flag='r', autocommit=False)
    # for i, item in enumerate(pubmed_meta_db.items()):
    #     if i > 3:
    #         break
    #     print(item)
    # build_pubmed_date_dict()

    logger.setLevel(logging.INFO)
    multiprocessing_logging.install_mp_handler()
    logger.info("Start indexing")

    # Build Pubmed Index (on paragraph and sentence basis)
    # build_pubmed_index("pubmed2", overwrite_index=False, num_workers=16)

    # Build PubMed Central Index (on paragraph and sentence basis)
    stringid_mapper, pmcid_mapper = get_pmid_from_pmcid_mapper()
    add_pubmed_central_parallel(stringid_mapper, pmcid_mapper, index_name="pubmed2", num_workers=16)

    # Debug wrong mapping from paragraph ids
    # with open("/glusterfs/dfs-gfs-dist/wangxida/pubmed_central/J_Cell_Biol/PMC2137816.txt", "r", encoding="utf-8", errors="replace") as f:
    #     text_body_bool = False
    #     current_paragraph = ""
    #     paragraph_count = 1
    #     txt_lines = f.readlines()
    #     for i, line in enumerate(txt_lines):
    #         if line == "==== Body\n":
    #             text_body_bool = True
    #         elif line == "==== Refs\n":
    #             text_body_bool = False
    #         if text_body_bool:
    #             if line.strip() == "":
    #                 if paragraph_count >= 0 and paragraph_count < 4000:
    #                     print("  Add paragraph")
    #                     print(paragraph_count)
    #                     print(current_paragraph)
    #                 sentences = tokenize.sent_tokenize(current_paragraph)
    #                 # print(sentences)
    #                 # print(len(sentences[0]))
    #                 # raise ValueError
    #                 for sentence_id, sentence in enumerate(sentences):
    #                     if len(sentence) > 10 and len(sentence) < 1000 and paragraph_count >= 0 and paragraph_count < 4000:
    #                         print("  Add sentence")
    #                         print(paragraph_count, sentence_id)
    #                         print(sentence)
    #                 current_paragraph = ""
    #                 paragraph_count += 1
    #             else:
    #                 current_paragraph += line
    #         if paragraph_count == 5:
    #             break
