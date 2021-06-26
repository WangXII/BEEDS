''' Build lookup Table for Uniprot ID to PfamA ID.
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

from configs import UNIPROT_TO_PFAM_FILE, PFAM_DB


def build_pfam_dict():
    document_path = UNIPROT_TO_PFAM_FILE
    db_path = PFAM_DB
    pfam_db = SqliteDict(db_path, tablename='pfam_dict', flag='w', autocommit=False)
    with tqdm(total=Path(document_path).stat().st_size) as pbar:
        with open(document_path) as infile:
            line = infile.readline()
            while line:
                pbar.update(infile.tell() - pbar.n)
                line = infile.readline()

                values = line.split('\t')
                values[-1] = values[-1].strip()

                # columns = ["pfamseq_acc", "seq_version", "crc64", "md5", "pfamA_acc", "seq_start", "seq_end"]

                if len(values) == 7:
                    uniprot_id = values[0]
                    pfama_id = values[4]
                    val = pfam_db.setdefault(uniprot_id, [])
                    val.append(pfama_id)
                    pfam_db[uniprot_id] = val

    pfam_db.commit()
    pfam_db.close()


# Main
if __name__ == "__main__":
    build_pfam_dict()
    db_path = PFAM_DB
    pfam_db = SqliteDict(db_path, tablename='pfam_dict', flag='r', autocommit=False)
    for key in pfam_db:
        print(key)
        break
    print(pfam_db["P28482"])
    print(pfam_db["P31749"])
    pfam_db.close()
