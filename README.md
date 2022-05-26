# BEEDS: Large-Scale Biomedical Event Extraction using Distant Supervision and Question Answering

Here is the accompanying code for the paper [BEEDS: Large-Scale Biomedical Event Extraction using Distant Supervision and Question Answering](https://aclanthology.org/2022.bionlp-1.28/).

## General Setup
1. Install required packages for conda (spec-list.text) and pip (requirements.txt)
2. Install Lucene/ElasticSearch for document retrieval
3. Run the .py-files from the /util/ folder once to build auxilliary tools 
    - Download PubMed, .owl knowledge bases from PathwayCommons etc.
    - Build and index the Pubmed corpus
    - Build and load the normalization tools (PubTator, simple lookup dictionaries etc.) and other auxilliary tools (UniProt/Entrez mapping etc.)
    - Adjust local paths when necessary (also for steps 4-6, edit /configs/__init__.py and files in /scripts/ folder)
4. Run /scripts/build_data.sh for building the examples
    - Conduct retrieval and build distantly supervised examples
    - Events/proteins used for train/test splits are chosen here 
5. Run /scripts/train.sh for training the BERT model
6. Run /scripts/evaluate.sh for evaluation and inspection of predictions
