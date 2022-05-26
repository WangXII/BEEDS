''' Read PubTator Annotations into a SQLite Database for further processing '''

import os
import sys

if os.name == 'nt':
    root_path = "/".join(os.path.realpath(__file__).split("\\")[:-2])
else:
    root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.append(root_path)

from pathlib import Path
from sqlalchemy import create_engine
from sqlitedict import SqliteDict
from tqdm import tqdm

import pandas as pd
import pickle

from configs import GENE_INFO_DB, GENE_NAMES_CACHE, NCBI_GENE_INFO_FILE, GENE_ID_TO_NAMES_DB, LOAD_ID_GENE_NAMES_BOOL


def build_ncbi_gene_id_db():
    document_path = NCBI_GENE_INFO_FILE
    disk_engine = create_engine(GENE_INFO_DB)
    index = 0
    with tqdm(total=Path(document_path).stat().st_size) as pbar:
        with open(document_path) as infile:
            line = infile.readline()
            chunk = []
            while line:
                pbar.update(infile.tell() - pbar.n)
                line = infile.readline()

                values = line.split('\t')
                values[-1] = values[-1].strip()
                chunk.append(values)
                # print(line)
                # print(values)
                columns = ["tax_id", "GeneID", "Symbol", "LocusTag", "Synonyms", "dbXrefs", "chromosome", "map_location", "description", "type_of_gene",
                           "Symbol_from_nomenclature_authority", "Full_name_from_nomenclature_authority", "Nomenclature_status", "Other_designations",
                           "Modification_date", "Feature_type"]
                important_columns = ["tax_id", "GeneID", "Symbol", "Synonyms", "dbXrefs", "description", "Symbol_from_nomenclature_authority",
                                     "Full_name_from_nomenclature_authority", "Nomenclature_status", "Other_designations"]
                if len(chunk) == 100000:
                    # df = pd.DataFrame(data=chunk, columns=columns)
                    # df[["tax_id", "GeneID"]] = df[["tax_id", "GeneID"]].apply(pd.to_numeric)
                    # df = df[important_columns]
                    # df.index += index
                    # df.to_sql('pubmed_annotations', disk_engine, if_exists='append')
                    chunk = []

                index += 1

            # Final chunk
            df = pd.DataFrame(data=chunk, columns=columns)
            df[["tax_id", "GeneID"]] = df[["tax_id", "GeneID"]].apply(pd.to_numeric)
            df = df[important_columns]
            df.index += index
            df.to_sql('pubmed_annotations', disk_engine, if_exists='append')
            chunk = []


def get_gene_id_to_names(use_cache=True, load_bool=True):
    load_bool = LOAD_ID_GENE_NAMES_BOOL
    if not load_bool:
        return {}
    if use_cache:
        with open(GENE_ID_TO_NAMES_DB, 'rb') as handle:
            gene_id_to_names = pickle.load(handle)
        return gene_id_to_names

    document_path = NCBI_GENE_INFO_FILE
    gene_id_to_names = {}
    with tqdm(total=Path(document_path).stat().st_size) as pbar:
        with open(document_path) as infile:
            line = infile.readline()
            while line:
                pbar.update(infile.tell() - pbar.n)
                line = infile.readline()

                values = line.split('\t')
                if len(values) <= 1:
                    break
                values[-1] = values[-1].strip()

                # columns = ["tax_id", "GeneID", "Symbol", "LocusTag", "Synonyms", "dbXrefs", "chromosome", "map_location", "description", "type_of_gene",
                #            "Symbol_from_nomenclature_authority", "Full_name_from_nomenclature_authority", "Nomenclature_status", "Other_designations",
                #            "Modification_date", "Feature_type"]
                synonyms = values[2:3] + values[4].split('|') + values[8:9] + values[10:11] + values[11:12] + values[13].split('|')
                synonyms = (values[2], set(synonyms))
                gene_id_to_names[values[1]] = synonyms

    with open(GENE_ID_TO_NAMES_DB, 'wb') as handle:
        pickle.dump(gene_id_to_names, handle)

    return gene_id_to_names


def dump_into_sqlite_dict():
    with open(GENE_ID_TO_NAMES_DB, 'rb') as handle:
        gene_id_to_names = pickle.load(handle)
    gene_names_sql_dict = SqliteDict(GENE_NAMES_CACHE, tablename='gene_id_to_names', flag='w', autocommit=False)
    gene_names_sql_dict.update(gene_id_to_names.items())
    gene_names_sql_dict.commit()
    gene_names_sql_dict.close()

def load_into_sqlite_dict():
    return SqliteDict(GENE_NAMES_CACHE, tablename='gene_id_to_names', flag='r', autocommit=False)


# Main
if __name__ == "__main__":
    dump_into_sqlite_dict()

    # Build data
    # build_ncbi_gene_id_db()

    # disk_engine = create_engine(GENE_INFO_DB)

    # # Create index
    # import sqlalchemy
    # from sqlalchemy.engine import reflection
    # meta = sqlalchemy.MetaData()
    # meta.reflect(bind=disk_engine)
    # flows = meta.tables['pubmed_annotations']
    # # alternative of retrieving the table from meta:
    # # flows = sqlalchemy.Table('flows', meta, autoload=True, autoload_with=engine)

    # my_index = sqlalchemy.Index('gene_info_index', flows.columns.get('GeneID'))
    # my_index.drop(bind=disk_engine)
    # my_index.create(bind=disk_engine)

    # # lets confirm it is there
    # inspector = reflection.Inspector.from_engine(disk_engine)
    # print(inspector.get_indexes('pubmed_annotations'))

    # df = pd.read_sql_query('SELECT GeneID, Symbol, Synonyms, description FROM pubmed_annotations '
    #                        'WHERE GeneID = 210789 LIMIT 3', disk_engine)
    # print(df.head())

    # gene_id_to_names = get_gene_id_to_names(use_cache=False)
    # print(gene_id_to_names["210789"])
