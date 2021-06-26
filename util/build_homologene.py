''' Build Homologene/GeneID mapping '''

from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.engine import reflection
from tqdm import tqdm

import pandas as pd
import sqlalchemy

from configs import GENE_INFO_DB, HOMOLOGENE_DB


def build_homologene_db():
    document_path = HOMOLOGENE_DB
    disk_engine = create_engine(GENE_INFO_DB)
    index = 0

    sql = text('DROP TABLE IF EXISTS homologene;')
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
                columns = ["HID", "Taxonomy_ID", "Gene_ID", "Gene_Symbol", "Protein_gi", "Protein_accession"]
                if len(chunk) == 100000:
                    df = pd.DataFrame(data=chunk, columns=columns)
                    df.index += index
                    # print(df.index)
                    # print(df)
                    # break
                    df.to_sql('homologene', disk_engine, if_exists='append')
                    chunk = []

                index += 1

            # Final chunk
            df = pd.DataFrame(data=chunk, columns=columns)
            df.index += index
            # print(df.index)
            # print(df)
            # break
            df.to_sql('homologene', disk_engine, if_exists='append')
            chunk = []


# Main
if __name__ == "__main__":
    # build_homologene_db()

    disk_engine = create_engine(GENE_INFO_DB)

    # Create index
    # meta = sqlalchemy.MetaData()
    # meta.reflect(bind=disk_engine)
    # # print(meta.tables)
    # flows = meta.tables['homologene']

    # my_index = sqlalchemy.Index('hid_index', flows.columns.get('HID'))
    # # my_index.drop(bind=disk_engine)
    # my_index.create(bind=disk_engine)

    # my_index = sqlalchemy.Index('gene_id_index', flows.columns.get('Gene_ID'))
    # # my_index.drop(bind=disk_engine)
    # my_index.create(bind=disk_engine)

    # # # lets confirm it is there
    # inspector = reflection.Inspector.from_engine(disk_engine)
    # print(inspector.get_indexes('homologene'))

    df = pd.read_sql_query('SELECT "HID", "Taxonomy_ID", "Gene_ID" FROM homologene '
                           'WHERE "Gene_ID" = 469356', disk_engine)
    print(df.head())

    df = pd.read_sql_query('SELECT "HID", "Taxonomy_ID", "Gene_ID" FROM homologene '
                           'WHERE "HID" = 3 AND "TAXONOMY_ID" = 9606', disk_engine)
    print(df.head())
    print(len(df.head()))
    print(df["Gene_ID"][0])
    print(type(df["Gene_ID"][0]))
