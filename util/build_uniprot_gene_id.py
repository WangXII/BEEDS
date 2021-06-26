''' Read PubTator Annotations into a SQLite Database for further processing '''

from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.engine import reflection
from tqdm import tqdm

import pandas as pd
import sqlalchemy

from configs import GENE_INFO_DB, UNIPROT_TO_GENEID_FILE


def build_up_gene_id_db():
    document_path = UNIPROT_TO_GENEID_FILE
    disk_engine = create_engine(GENE_INFO_DB)
    index = 0

    sql = text('DROP TABLE IF EXISTS up_to_geneid2;')
    disk_engine.execute(sql)

    with tqdm(total=Path(document_path).stat().st_size) as pbar:
        with open(document_path) as infile:
            line = "---"
            chunk = []
            while line:
                pbar.update(infile.tell() - pbar.n)
                line = infile.readline()

                values = line.split('\t')
                values[-1] = values[-1].strip()
                chunk.append(values)
                # print(line)
                # print(values)
                columns = ["UniProtKB-AC", "UniProtKB-ID", "GeneID", "RefSeq", "GI", "PDB", "GO", "UniRef100", "UniRef90", "UniRef50", "UniParc", "PIR",
                           "NCBI-taxon", "MIM", "UniGene", "PubMed", "EMBL", "EMBL-CDS", "Ensembl", "Ensembl_TRS", "Ensembl_PRO", "Additional PubMed"]
                important_columns = ["UniProtKB-AC", "GeneID", "NCBI-taxon"]
                if len(chunk) == 100000:
                    df = pd.DataFrame(data=chunk, columns=columns)
                    # df[["GeneID"]] = df[["GeneID"]].apply(pd.to_numeric)
                    df = df[important_columns]
                    df.index += index
                    # print(df.index)
                    # print(df)
                    # break
                    df.to_sql('up_to_geneid2', disk_engine, if_exists='append')
                    chunk = []

                index += 1

            # Final chunk
            df = pd.DataFrame(data=chunk, columns=columns)
            df = df[important_columns]
            df.index += index
            df.to_sql('up_to_geneid2', disk_engine, if_exists='append')
            chunk = []


# Main
if __name__ == "__main__":
    # build_up_gene_id_db()

    disk_engine = create_engine('sqlite:////glusterfs/dfs-gfs-dist/wangxida/gene_info.db')

    # # Create index
    # meta = sqlalchemy.MetaData()
    # meta.reflect(bind=disk_engine)
    # # print(meta.tables)
    # flows = meta.tables['up_to_geneid2']
    # # alternative of retrieving the table from meta:
    # # flows = sqlalchemy.Table('flows', meta, autoload=True, autoload_with=engine)

    # my_index = sqlalchemy.Index('up_gene_info_index', flows.columns.get('UniProtKB-AC'))
    # my_index.drop(bind=disk_engine)
    # my_index.create(bind=disk_engine)

    # # # lets confirm it is there
    # inspector = reflection.Inspector.from_engine(disk_engine)
    # print(inspector.get_indexes('up_to_geneid2'))
    # print(inspector.get_indexes('pubmed_annotations'))

    df = pd.read_sql_query('SELECT "UniProtKB-AC", GeneID, "NCBI-taxon" FROM up_to_geneid2 '
                           'WHERE "UniProtKB-AC" = "Q6GZX3" LIMIT 3', disk_engine)
    print(df.head())
