''' Config file '''

from transformers import AutoTokenizer
from sqlitedict import SqliteDict

# tqdm (disable when logging files to wandb)
TQDM_DISABLE = True

# BERT and RoBERTa configs
ROBERTA_PATH = "/glusterfs/dfs-gfs-dist/wangxida/masterarbeit/roberta-base-squad2"
SCIBERT_PATH = "/glusterfs/dfs-gfs-dist/wangxida/masterarbeit/scibert_scivocab_uncased"


class ModelHelper:
    ''' Tokenizer for a BERT or RoBERTa model '''

    blinding_tokens_bert = [
        "<##prot>", "<protein0>", "<protein1>", "<protein2>", "<protein3>", "<protein4>", "<protein5>", "<protein6>", "<protein7>", "<protein8>", "<protein9>",
        "<protein10>", "<protein11>", "<protein12>", "<protein13>", "<protein14>", "<protein15>", "<protein16>", "<protein17>", "<protein18>", "<protein19>",
        "<protein20>", "<protein21>", "<protein22>", "<protein23>", "<protein24>", "<protein25>", "<protein26>", "<protein27>", "<protein28>", "<protein29>",
        "<protein30>", "<protein31>", "<protein32>", "<protein33>", "<protein34>", "<protein35>", "<protein36>", "<protein37>", "<protein38>", "<protein39>",
        "<protein40>", "<protein41>", "<protein42>", "<protein43>", "<protein44>", "<protein45>", "<protein46>", "<protein47>", "<protein48>", "<protein49>"]
    blinding_tokens_roberta = [
        "<##prot>", " <protein0>", " <protein1>", " <protein2>", " <protein3>", " <protein4>", " <protein5>", " <protein6>", " <protein7>", " <protein8>", " <protein9>",
        " <protein10>", " <protein11>", " <protein12>", " <protein13>", " <protein14>", " <protein15>", " <protein16>", " <protein17>", " <protein18>", " <protein19>",
        " <protein20>", " <protein21>", " <protein22>", " <protein23>", " <protein24>", " <protein25>", " <protein26>", " <protein27>", " <protein28>", " <protein29>",
        " <protein30>", " <protein31>", " <protein32>", " <protein33>", " <protein34>", " <protein35>", " <protein36>", " <protein37>", " <protein38>", " <protein39>",
        " <protein40>", " <protein41>", " <protein42>", " <protein43>", " <protein44>", " <protein45>", " <protein46>", " <protein47>", " <protein48>", " <protein49>"]

    def __init__(self, model_str=SCIBERT_PATH, pretrained_str=None):
        self.model_name_or_path = model_str
        if "berta" in model_str:
            if pretrained_str is None:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, additional_special_tokens=self.blinding_tokens_roberta)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(pretrained_str, additional_special_tokens=self.blinding_tokens_roberta)
        else:
            if pretrained_str is None:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, additional_special_tokens=self.blinding_tokens_bert)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(pretrained_str, additional_special_tokens=self.blinding_tokens_bert)

    def lower_string(self, string):
        if self.model_name_or_path == SCIBERT_PATH:
            return string.lower()
        else:
            return string


# Raw text files or directories for text files

# Biomedical knowledge bases
HUMANCYC_OWL = "/glusterfs/dfs-gfs-dist/wangxida/PathwayCommons12.humancyc.BIOPAX.owl"
INOH_OWL = "/glusterfs/dfs-gfs-dist/wangxida/PathwayCommons12.inoh.BIOPAX.owl"
KEGG_OWL = "/glusterfs/dfs-gfs-dist/wangxida/PathwayCommons12.kegg.BIOPAX.owl"
PID_OWL = "/glusterfs/dfs-gfs-dist/wangxida/PathwayCommons12.pid.BIOPAX.owl"
PID_SIF = "/glusterfs/dfs-gfs-dist/wangxida/PathwayCommons12.pid.hgnc.txt"
PANTHER_OWL = "/glusterfs/dfs-gfs-dist/wangxida/PathwayCommons12.panther.BIOPAX.owl"
NETPATH_OWL = "/glusterfs/dfs-gfs-dist/wangxida/PathwayCommons12.netpath.BIOPAX.owl"
REACTOME_OWL = "/glusterfs/dfs-gfs-dist/wangxida/PathwayCommons12.reactome.BIOPAX.owl"
OWL_LIST = [HUMANCYC_OWL, INOH_OWL, KEGG_OWL, PID_OWL, PANTHER_OWL, NETPATH_OWL, REACTOME_OWL]

# Pubmed and Pubmed Central
PUBMED_META_RAW_FILES_DIC = "/glusterfs/dfs-gfs-dist/wangxida/pubmed_meta/ftp.ncbi.nlm.nih.gov/pubmed/baseline"
PUBMED_META_XSLT = "/vol/fob-wbib-vol3/wbi_stud/wangxida/Studienprojekt/masterarbeit/util/pubmed_meta.xslt"
PMC_TO_PMID_MAPPER = "/glusterfs/dfs-gfs-dist/wangxida/pubmed_central/oa_file_list.txt"
PMC_DOCS_DIRECTORY = "/glusterfs/dfs-gfs-dist/wangxida/pubmed_central"
PUBMED_FILES_PUBTATOR = "/glusterfs/dfs-gfs-dist/wangxida/bioconcepts2pubtatorcentral.offset"
PUBMED_DOC_ANNOTATIONS = "/glusterfs/dfs-gfs-dist/wangxida/bioconcepts2pubtatorcentral"

# NCBI database files
UNIPROT_TO_GENEID_FILE = "/glusterfs/dfs-gfs-dist/wangxida/idmapping_selected.tab"
CHEBI_NAMES = "/glusterfs/dfs-gfs-dist/wangxida/masterarbeit/cache/names.tsv"
HOMOLOGENE_DB = "/glusterfs/dfs-gfs-dist/wangxida/homologene.data"
NCBI_GENE_INFO_FILE = "/glusterfs/dfs-gfs-dist/wangxida/gene_info"
UNIPROT_TO_PFAM_FILE = "/glusterfs/dfs-gfs-dist/wangxida/Pfam-A.regions.tsv"
# (Other) UNIPROT_TO_PFAM_FILE = "/glusterfs/dfs-gfs-dist/wangxida/Pfam-A.regions.uniprot.tsv"

# EVEX Baseline Files
RELATIONS_FILE = "/glusterfs/dfs-gfs-dist/wangxida/evex/network-format/Metazoa/EVEX_relations_9606.tab"
STANDOFF_DIR = "/glusterfs/dfs-gfs-dist/wangxida/evex/standoff-annotation/version-0.1/"

# BIONLP filesevex/standoff-
BIONLP_DIR = "/glusterfs/dfs-gfs-dist/wangxida/bionlp_datasets/"

# ===============================================================================================================

# Shortened list of event substrates because number of pair of proteins becomes too big too fast
EVENT_SUBSTRATES_COMPLEX_MULTI = "/glusterfs/dfs-gfs-dist/wangxida/masterarbeit/cache/event_substrates_complex_multi.pickle"

# Biomedical data caches

HUMANCYC_MODEL = "/glusterfs/dfs-gfs-dist/wangxida/masterarbeit/cache/humancyc_model.pickle"
INOH_MODEL = "/glusterfs/dfs-gfs-dist/wangxida/masterarbeit/cache/inoh_model.pickle"
KEGG_MODEL = "/glusterfs/dfs-gfs-dist/wangxida/masterarbeit/cache/kegg_model.pickle"
PID_MODEL = "/glusterfs/dfs-gfs-dist/wangxida/masterarbeit/cache/pid_model_with_complexes.pickle"
PID_MODEL_FAMILIES = "/glusterfs/dfs-gfs-dist/wangxida/masterarbeit/cache/pid_model_with_families.pickle"
PID_MODEL_EXPANDED = "/glusterfs/dfs-gfs-dist/wangxida/masterarbeit/cache/pid_model_expanded.pickle"
PANTHER_MODEL = "/glusterfs/dfs-gfs-dist/wangxida/masterarbeit/cache/panther_model.pickle"
NETPATH_MODEL = "/glusterfs/dfs-gfs-dist/wangxida/masterarbeit/cache/netpath_model.pickle"
REACTOME_MODEL = "/glusterfs/dfs-gfs-dist/wangxida/masterarbeit/cache/reactome_model.pickle"
OWL_STATEMENTS = "/glusterfs/dfs-gfs-dist/wangxida/masterarbeit/cache/owl_statements.pickle"

PUBMED_META_DB = "/glusterfs/dfs-gfs-dist/wangxida/pubmed_meta.sqlite"
PUBMED_DOC_ANNOTATIONS_DB = "/glusterfs/dfs-gfs-dist/wangxida/pubtator_normalizer.sqlite"
PUBMED_EVIDENCE_ANNOTATIONS_DB = "sqlite:////glusterfs/dfs-gfs-dist/wangxida/pubmed.db"
SIMPLE_NORMALIZER_DB = "/glusterfs/dfs-gfs-dist/wangxida/simple_normalizer.sqlite"
PFAM_DB = "/glusterfs/dfs-gfs-dist/wangxida/pfam.sqlite"
GENE_INFO_DB = "sqlite:////glusterfs/dfs-gfs-dist/wangxida/gene_info.db"
GENE_ID_TO_NAMES_DB = "/glusterfs/dfs-gfs-dist/wangxida/gene_id_to_names.pickle"
BASELINE_DIR = "/glusterfs/dfs-gfs-dist/wangxida/masterarbeit/cache/evex_relations.pickle"
STANDOFF_CACHE = "/glusterfs/dfs-gfs-dist/wangxida/evex/evex_standoff.sqlite"
STANDOFF_CACHE_SIMPLE = "/glusterfs/dfs-gfs-dist/wangxida/evex/evex_standoff_simple.sqlite"
STANDOFF_CACHE_PMIDS = "/glusterfs/dfs-gfs-dist/wangxida/evex/evex_standoff_pmids.sqlite"
STANDOFF_CACHE_RANDOM = "/glusterfs/dfs-gfs-dist/wangxida/evex/evex_standoff_random.sqlite"
BIONLP_CACHE = "/glusterfs/dfs-gfs-dist/wangxida/bionlp_datasets/bionlp_standoff.sqlite"
GENE_NAMES_CACHE = "/glusterfs/dfs-gfs-dist/wangxida/bionlp_datasets/gene_names.sqlite"

DICT_NORMALIZER = SqliteDict(SIMPLE_NORMALIZER_DB, tablename='simple_normalizer', flag='r', autocommit=False)

# ===============================================================================================================

# Neural Network Output Cache
NN_CACHE = "/glusterfs/dfs-gfs-dist/wangxida/masterarbeit/output/cache"

# General Output Cache Directory
CACHE = "/glusterfs/dfs-gfs-dist/wangxida/masterarbeit/cache"

# ===============================================================================================================

# Other config variables
TRAIN_TEST_SPLIT_SEED = 5
TRAIN_SPLIT = 0.6
DEV_SPLIT = 0.7

SPACY_NER_MODEL = "en_ner_jnlpba_md"
LOAD_ID_GENE_NAMES_BOOL = True
DOC_STRIDE = 16
