''' Build simple lookup Table for gene names to NCBI EntrezGene IDs.
    Focus on human taxonomy 9606 as the basis of events in PID.
    Synonyms come from gene_info (NCBI) and canonical forms are built using our custom rules.
'''

import os
import sys

if os.name == 'nt':
    root_path = "/".join(os.path.realpath(__file__).split("\\")[:-2])
else:
    root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.append(root_path)

from sqlitedict import SqliteDict
from tqdm import tqdm
from pathlib import Path

from data_processing.datatypes import shorten_synonym_list
from configs import NCBI_GENE_INFO_FILE, SIMPLE_NORMALIZER_DB


def build_simple_normalizer():
    document_path = NCBI_GENE_INFO_FILE
    db_path = SIMPLE_NORMALIZER_DB
    normalizer_db = SqliteDict(db_path, tablename='simple_normalizer', flag='w', autocommit=False)
    with tqdm(total=Path(document_path).stat().st_size) as pbar:
        with open(document_path) as infile:
            line = infile.readline()
            while line:
                pbar.update(infile.tell() - pbar.n)
                line = infile.readline()

                values = line.split('\t')
                values[-1] = values[-1].strip()

                # columns = ["tax_id", "GeneID", "Symbol", "LocusTag", "Synonyms", "dbXrefs", "chromosome", "map_location", "description", "type_of_gene",
                #            "Symbol_from_nomenclature_authority", "Full_name_from_nomenclature_authority", "Nomenclature_status", "Other_designations",
                #            "Modification_date", "Feature_type"]
                # important_columns = ["tax_id", "GeneID", "Symbol", "Synonyms", "dbXrefs", "description", "Symbol_from_nomenclature_authority",
                #                      "Full_name_from_nomenclature_authority", "Nomenclature_status", "Other_designations"]

                if values[0] == "9606":  # Protein in humans
                    synonyms = values[2:3] + values[4].split('|') + values[8:9] + values[10:11] + values[11:12] + values[13].split('|')
                    synonyms = list(set(synonyms))
                    synonyms = shorten_synonym_list(synonyms, retrieval="relaxed")
                    entrez_gene_id = values[1]
                    for synonym in synonyms:
                        gene_string = synonym.lower()
                        val = normalizer_db.setdefault(gene_string, [])
                        if entrez_gene_id not in normalizer_db[gene_string]:
                            val.append(entrez_gene_id)
                            normalizer_db[gene_string] = val

    normalizer_db.commit()
    normalizer_db.close()


# Main
if __name__ == "__main__":
    # build_simple_normalizer()
    db_path = SIMPLE_NORMALIZER_DB
    normalizer_db = SqliteDict(db_path, tablename='simple_normalizer', flag='r', autocommit=False)
    print(normalizer_db["akt serine/threonine kinase 1"])
    print(normalizer_db["akt"])
    normalizer_db.close()
