""" Extract abstracts from PubTator bioconcepts2pubtatorcentral.offset to specified directory
    and build Lucene Index in the ElasticSearch Folder
"""

import os
import re
import parse
import lxml.etree as ET

from nltk import tokenize
from pathlib import Path
from tqdm import tqdm
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from sqlitedict import SqliteDict
from multiprocessing import Pool
from itertools import repeat
# from elasticsearch import Elasticsearch

from configs import PUBMED_META_DB, PMC_TO_PMID_MAPPER, PMC_DOCS_DIRECTORY, PUBMED_META_RAW_FILES_DIC, PUBMED_META_XSLT, PUBMED_FILES_PUBTATOR

import logging

# logging.basicConfig(filename='/glusterfs/dfs-gfs-dist/wangxida/myapp.log', level=logging.DEBUG,
#                     format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)


class IndexWriter:
    def __init__(self, index_name="pubmed_detailed", overwrite_index=False, index_batch_size=100000):
        self.db_path = PUBMED_META_DB
        self.index_batch_size = index_batch_size
        self.pubmed_meta_db = SqliteDict(self.db_path, tablename='pubmed_meta', flag='r', autocommit=False)
        self.document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index=index_name)
        if overwrite_index:
            self.document_store.delete_all_documents(index=index_name)
        self.documents = []
        self.documents_processed = 0

    def add_docs_to_index(self, text, pubmed_id, par_id, sentence_id=None):
        if sentence_id is None:
            if pubmed_id not in self.pubmed_meta_db:
                self.documents.append({"text": text, "meta": {"name": pubmed_id, "paragraph_id": par_id, "month": "Jan", "year": "9999"}})
            else:
                self.documents.append({"text": text, "meta": {"name": pubmed_id, "paragraph_id": par_id,
                                                              "month": self.pubmed_meta_db[pubmed_id][1], "year": self.pubmed_meta_db[pubmed_id][0]}})
        else:
            if pubmed_id not in self.pubmed_meta_db:
                self.documents.append({"text": text, "meta": {"name": pubmed_id, "paragraph_id": par_id, "sentence_id": sentence_id, "year": 9999}})
            else:
                self.documents.append({"text": text, "meta": {"name": pubmed_id, "paragraph_id": par_id, "sentence_id": sentence_id,
                                                              "year": int(self.pubmed_meta_db[pubmed_id][0])}})
        self.documents_processed += 1
        if self.documents_processed == self.index_batch_size:
            self.flush_docs_cache()
            self.documents = []
            self.documents_processed = 0

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
        for line in tqdm(txt_lines):
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


# For multiprocessing https://chriskiehl.com/article/parallelism-in-one-line
# https://stackoverflow.com/questions/41920124/multiprocessing-use-tqdm-to-display-a-progress-bar

def add_pubmed_central(stringid_mapper, pmcid_mapper):
    path = PMC_DOCS_DIRECTORY
    number_files = 0
    index_writer = IndexWriter(overwrite_index=True)
    pbar = tqdm(os.listdir(path), desc="Full texts processed: {}. Docs written: {}.".format(number_files, index_writer.documents_processed))
    for directory in pbar:
        pbar.set_description(desc="Full texts processed: {}. Docs written: {}.".format(number_files, index_writer.documents_processed))
        number_files = add_pubmed_central_file(path, directory, index_writer, number_files, stringid_mapper, pmcid_mapper)
    index_writer.flush_docs_cache()


def add_pubmed_central_parallel(stringid_mapper, pmcid_mapper):
    path = PMC_DOCS_DIRECTORY
    number_files = 0
    index_writer = None
    directories = []
    for directory in os.listdir(path):
        directories.append(directory)
    inputs = list(zip(repeat(path), directories, repeat(index_writer), repeat(number_files), repeat(stringid_mapper), repeat(pmcid_mapper)))
    with Pool() as pool:
        _ = pool.starmap(add_pubmed_central_file, tqdm(inputs, total=len(inputs)))


def add_pubmed_central_file(path, directory, index_writer, number_files, stringid_mapper, pmcid_mapper):
    if index_writer is None:
        index_writer = IndexWriter(index_name="pubmed_sentences", overwrite_index=False, index_batch_size=1000000)
    par_docs_bool = False
    text_body_bool = False
    dir_full_name = os.path.join(path, directory)
    if os.path.isdir(dir_full_name):
        for pubmed_file in os.listdir(dir_full_name):
            number_files += 1
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
                with open(pubmed_file_name, "r", encoding="utf-8", errors="replace") as f:
                    text_body_bool = False
                    current_paragraph = ""
                    paragraph_count = 1
                    txt_lines = f.readlines()
                    for line in txt_lines:
                        if line == "==== Body\n":
                            text_body_bool = True
                        elif line == "==== Refs\n":
                            text_body_bool = False
                        if text_body_bool:
                            if line.strip() == "":
                                if par_docs_bool:
                                    index_writer.add_docs_to_index(current_paragraph, pmid, str(paragraph_count))
                                else:
                                    sentences = tokenize.sent_tokenize(current_paragraph)
                                    # print(sentences)
                                    # print(len(sentences[0]))
                                    # raise ValueError
                                    for sentence_id, sentence in enumerate(sentences):
                                        if len(sentence) > 10 and len(sentence) < 1000:  # Sentences longer than 1000 characters are not more precise than paragraphs
                                            index_writer.add_docs_to_index(sentence, int(pmid), paragraph_count, sentence_id=sentence_id)
                                current_paragraph = ""
                                paragraph_count += 1
                            else:
                                current_paragraph += line
                    if current_paragraph.strip() != "":
                        if par_docs_bool:
                            index_writer.add_docs_to_index(current_paragraph, pmid, str(paragraph_count))
                        else:
                            sentences = tokenize.sent_tokenize(current_paragraph)
                            for sentence_id, sentence in enumerate(sentences):
                                if len(sentence) > 10 and len(sentence) < 1000:  # Sentences longer than 1000 characters are not more precise than paragraphs
                                    index_writer.add_docs_to_index(sentence, int(pmid), paragraph_count, sentence_id=sentence_id)
    index_writer.flush_docs_cache()
    return number_files


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


def build_pubmed_index(index_name="pubmed_detailed", index_batch_size=100000, par_docs_bool=True):
    document_path = PUBMED_FILES_PUBTATOR
    # /home/tmp/wangxida/elasticsearch-7.9.2

    doc_writer = IndexWriter(index_name=index_name, overwrite_index=False, index_batch_size=index_batch_size)
    with tqdm(total=Path(document_path).stat().st_size) as pbar:
        with open(document_path) as infile:
            line = infile.readline()
            while line:
                pbar.update(infile.tell() - pbar.n)
                abstract = extract_abstract(line)
                if len(abstract) == 2 and abstract[1] != "\n":
                    pubmed_id = abstract[0]
                    if par_docs_bool:
                        doc_writer.add_docs_to_index(abstract[1], pubmed_id, "0")
                    else:
                        sentences = tokenize.sent_tokenize(abstract[1])
                        # print(sentences)
                        # print(len(sentences[0]))
                        # exit()
                        for sentence_id, sentence in enumerate(sentences):
                            if len(sentence) < 1000:  # Sentences longer than 1000 characters are not more precise than paragraphs
                                doc_writer.add_docs_to_index(sentence, int(pubmed_id), 0, sentence_id=sentence_id)
                line = infile.readline()

            doc_writer.flush_docs_cache()


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

    # Build Pubmed Index (on paragraph basis)
    # build_pubmed_index()

    # Build Pubmed Index (on sentence basis)
    # build_pubmed_index(index_name="pubmed_sentences", index_batch_size=1000000, par_docs_bool=False)
    print(count_sentences_pubmed())

    # Build PubMed Central Index (on paragraph basis)
    # stringid_mapper, pmcid_mapper = get_pmid_from_pmcid_mapper()
    # add_pubmed_central(stringid_mapper, pmcid_mapper)

    # Build PubMed Central Index (on sentence basis)
    # stringid_mapper, pmcid_mapper = get_pmid_from_pmcid_mapper()
    # add_pubmed_central_parallel(stringid_mapper, pmcid_mapper)
