''' Read PubTator Annotations into a SQLite Database for further processing '''

from pathlib import Path
from sqlalchemy import create_engine
from sqlitedict import SqliteDict
# from sqlalchemy.engine import reflection
from tqdm import tqdm

import pandas as pd
import re
# import sqlalchemy

from configs import PUBMED_FILES_PUBTATOR, PUBMED_DOC_ANNOTATIONS, PUBMED_DOC_ANNOTATIONS_DB, PUBMED_EVIDENCE_ANNOTATIONS_DB


def build_pubmed_doc_annotations():
    document_path = PUBMED_DOC_ANNOTATIONS
    db_path = PUBMED_DOC_ANNOTATIONS_DB
    normalizer_db = SqliteDict(db_path, tablename='pubtator_normalizer', flag='w', autocommit=False)
    with tqdm(total=Path(document_path).stat().st_size) as pbar:
        with open(document_path) as infile:
            line = infile.readline()
            chunk = []
            i = 0
            while line:
                pbar.update(infile.tell() - pbar.n)
                line = infile.readline()

                line_data = line.split('\t', maxsplit=4)
                if len(line_data) > 1:
                    pmid, type_class, concept_id, mentions, resource = line_data
                    mentions = mentions.split("|")
                    for mention in mentions:
                        if mention != "":
                            i += 1
                            chunk.append([pmid, type_class, concept_id, mention])
                            pubmed_gene = pmid + "_" + mention.lower()
                            val = normalizer_db.setdefault(pubmed_gene, [])
                            if (type_class, concept_id) not in normalizer_db[pubmed_gene]:
                                val.append((type_class, concept_id))
                                normalizer_db[pubmed_gene] = val
                if (not line) or i >= 100000:
                    normalizer_db.commit()
                    i = 0
    normalizer_db.close()


def build_pubmed_annotations_db():
    document_path = PUBMED_FILES_PUBTATOR
    disk_engine = create_engine(PUBMED_EVIDENCE_ANNOTATIONS_DB)
    with tqdm(total=Path(document_path).stat().st_size) as pbar:
        with open(document_path) as infile:
            line = infile.readline()
            chunk = []
            while line:
                pbar.update(infile.tell() - pbar.n)
                line = infile.readline()
                # Matched Abstract
                if re.match(r"^[0-9]*\|a\|", line):
                    continue
                # Matched Title
                elif re.match(r"^[0-9]*\|t\|", line):
                    continue
                # Empty Line
                elif line == "\n":
                    continue

                values = line.split('\t')
                values[-1] = values[-1].strip()
                chunk.append(values)
                # print(line)
                # print(values)
                if len(chunk) == 100000:
                    df = pd.DataFrame(data=chunk, columns=['pubmed_id', 'start_position', 'end_position', 'word', 'type', 'db_identifier'])
                    df[['pubmed_id', 'start_position', 'end_position']] = df[['pubmed_id', 'start_position', 'end_position']].apply(pd.to_numeric)
                    df.set_index(['pubmed_id', 'start_position', 'end_position'], inplace=True)
                    df.to_sql('pubmed_annotations', disk_engine, if_exists='append')
                    chunk = []

            # Final chunk
            df = pd.DataFrame(data=chunk, columns=['pubmed_id', 'start_position', 'end_position', 'word', 'type', 'db_identifier'])
            df[['pubmed_id', 'start_position', 'end_position']] = df[['pubmed_id', 'start_position', 'end_position']].apply(pd.to_numeric)
            df.set_index(['pubmed_id', 'start_position', 'end_position'], inplace=True)
            df.to_sql('pubmed_annotations', disk_engine, if_exists='append')
            chunk = []


# Main
if __name__ == "__main__":
    disk_engine = create_engine(PUBMED_EVIDENCE_ANNOTATIONS_DB)

    # # Create index
    # meta = sqlalchemy.MetaData()
    # meta.reflect(bind=disk_engine)
    # flows = meta.tables['pubmed_annotations']
    # # alternative of retrieving the table from meta:
    # # flows = sqlalchemy.Table('flows', meta, autoload=True, autoload_with=engine)

    # my_index = sqlalchemy.Index('evidence_index', flows.columns.get('PubMedID'), flows.columns.get('Start Position'), flows.columns.get('End Position'))
    # my_index.create(bind=disk_engine)

    # # lets confirm it is there
    # inspector = reflection.Inspector.from_engine(disk_engine)
    # print(inspector.get_indexes('pubmed_annotations'))

    df = pd.read_sql_query('SELECT * FROM pubmed_annotations '
                           'WHERE PubMedID = 19923418 AND "Start Position" = 50 AND "End Position" = 56 LIMIT 3', disk_engine)

    print(df)

    build_pubmed_doc_annotations()
