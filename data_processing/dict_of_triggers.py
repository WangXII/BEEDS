''' Gets a dictionary of all used event triggers in BioNLP13 Pathway Curation '''

import os
import sys

if os.name == 'nt':
    root_path = "/".join(os.path.realpath(__file__).split("\\")[:-2])
else:
    root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.append(root_path)

from configs import CHEBI_NAMES
import csv
# import json

AMINO_ACIDS_STRINGS = {
    "alanine": "A", "ala": "A", "a": "A",
    "arginine": "R", "arg": "R", "r": "R",
    "asparagine": "N", "asn": "N", "n": "N",
    "aspartic acid": "D", "asp": "D", "d": "D",
    "cysteine": "C", "cys": "C", "c": "C",
    "glutamic acid": "E", "glu": "E", "e": "E",
    "glutamine": "Q", "gln": "Q", "q": "Q",
    "glycine": "G", "gly": "G", "g": "G",
    "histidine": "H", "his": "H", "h": "H",
    "isoleucine": "I", "ile": "I", "i": "I",
    "leucine": "L", "leu": "L", "l": "L",
    "lysine": "K", "lys": "K", "k": "K",
    "methionine": "M", "met": "M", "m": "M",
    "phenylalanine": "F", "phe": "F", "f": "F",
    "proline": "P", "pro": "P", "p": "P",
    "serine": "S", "ser": "S", "s": "S",
    "threonine": "T", "thr": "T", "t": "T",
    "tryptophan": "W", "trp": "W", "w": "W",
    "tyrosine": "Y", "tyr": "Y", "y": "Y",
    "valine": "V", "val": "V", "v": "V"
}

AMINO_ACIDS = {
    "Alanine": ["Ala", "A"],
    "Arginine": ["Arg", "R"],
    "Asparagine": ["Asn", "N"],
    "Aspartic acid": ["Asp", "D"],
    "Cysteine": ["Cys", "C"],
    "Glutamic acid": ["Glu", "E"],
    "Glutamine": ["Gln", "Q"],
    "Glycine": ["Gly", "G"],
    "Histidine": ["His", "H"],
    "Isoleucine": ["Ile", "I"],
    "Leucine": ["Leu", "L"],
    "Lysine": ["Lys", "K"],
    "Methionine": ["Met", "M"],
    "Phenylalanine": ["Phe", "F"],
    "Proline": ["Pro", "P"],
    "Serine": ["Ser", "S"],
    "Threonine": ["Thr", "T"],
    "Tryptophan": ["Trp", "W"],
    "Tyrosine": ["Tyr", "Y"],
    "Valine": ["Val", "V"]
}

AMINO_ACIDS_ABBREVIATIONS = {
    "A": "Alanine",
    "R": "Arginine",
    "N": "Asparagine",
    "D": "Aspartic acid",
    "C": "Cysteine",
    "E": "Glutamic acid",
    "Q": "Glutamine",
    "G": "Glycine",
    "H": "Histidine",
    "I": "Isoleucine",
    "L": "Leucine",
    "K": "Lysine",
    "M": "Methionine",
    "F": "Phenylalanine",
    "P": "Proline",
    "S": "Serine",
    "T": "Threonine",
    "W": "Tryptophan",
    "Y": "Tyrosine",
    "V": "Valine",
}

# Added Trigger words. For PTMs, added common word stems, noun + "ing", verb + "s", verb + "ed", verb + infinitive, noun + "ion"
# If more than 30 trigger words, only keep the most frequent ones (more than 5 mentions or up to 60 in total)

TRIGGERS = {
    'Acetylation':
        ['acetylate', 'acetylated', 'acetylates', 'acetylating', 'acetylation', 'acetylations',
         'hyperacetylate', 'hyperacetylated', 'hyperacetylates', 'hyperacetylating', 'hyperacetylation', 'hyperacetylations',
         'hypoacetylate', 'hypoacetylated', 'hypoacetylates', 'hypoacetylating', 'hypoacetylation', 'hypoacetylations'],
    'Activation':
        ['activate', 'activated', 'activates', 'activating', 'activation', 'activations', 'activator', 'activators', 'active',
         'alterations', 'autoactivates', 'coactivated', 'coactivator', 'hyperactive', 'inactive', 'insensitive', 'kinase-active', 'reactivation',
         'stimulate', 'stimulates', 'stimulated', 'stimulating', 'stimulation', 'stimulations',
         'trans-activation', 'transactivating', 'transactivation', 'transcriptional activation'],
    'Binding':
        ['binding', 'interaction', 'bind', 'binds', 'bound', 'interacts', 'interact', 'association', 'interactions', 'cross-linking', 'associated', 'binding activity',
         'ligation', 'complex', 'associates', 'formation', 'complex formation', 'interacting', 'assembly', 'engagement', 'recognized', 'heterodimers', 'affinity',
         'heterodimer', 'target', 'recruitment', 'oligomerization', 'interacted', 'complexes', 'associate', 'recognizes', 'cross-linked', 'homodimer', 'homodimers',
         'coligation', 'tetramerisation', 'homodimerization'],
    'Conversion':
        ['3-o-sulfation', 'addition', 'attached', 'attachment', 'biosynthesis', 'bonds', 'cleavage', 'cleaved',
         'conversion', 'convert', 'converted', 'deaminated', 'deamination', 'digestion', 'formation', 'generated', 'generates', 'generating', 'generation',
         'hydrolyse', 'hydrolysis', 'hydrolyzes', 'hydrolyzing', 'interactions', 'interconverts', 'metabolism', 'modification', 'modified', 'neosynthesis',
         'oxidation', 'oxidize', 'oxidized', 'oxidizes', 'phosphorolysis', 'produce', 'produced', 'produces', 'production', 'protonation', 'react', 'reduction',
         'removal', 's-nitrosylation', 'substitution', 'sulfates', 'sulfation', 'sumoylation', 'synthesis', 'synthesize', 'synthesized', 'transfer', 'transferred',
         'transferring', 'transfers', 'utilize', 'utilized', 'utilizes'],
    'Deacetylation':
        ['deacetylate', 'deacetylated', 'deacetylates', 'deacetylating', 'deacetylation'],
    'Deglycosylation':
        ['deglycosylate', 'deglycosylates', 'deglycosylated', 'deglycosylation', 'deglycosylating'],
    'Degradation':
        ['degradation', 'degrade', 'degraded', 'degrades', 'destroyed', 'destroys', 'destruction', 'elimination', 'non-degradable', 'proteolysis',
         'removal', 'turnover'],
    'Dehydroxylation':
        ['dehydroxylate', 'dehedroxylates', 'dehydroxylated', 'dehydroxylation', 'dehydroxylating'],
    'Demethylation':
        ['demethylase', 'demethylation', 'demethylate', 'demethylates', 'demethylate', 'demethylated', 'demethylating', 'disappearance'],
    'Depalmitoylation':
        ['depalmitoylate', 'depalmitoylates', 'depalmitoylated', 'depalmitoylation', 'depalmitoylating'],
    'Dephosphorylation':
        ['dephosphorylate', 'dephosphorylated', 'dephosphorylates', 'dephosphorylating', 'dephosphorylation'],
    'Desumoylation':
        ['desumoylate', 'desumoylates', 'desumoylated', 'desumoylation', 'desumoylating'],
    'Deubiquitination':
        ['deubiquitinase', 'deubiquitination', 'deubiquitinates', 'deubiquitinate', 'deubiquitinated', 'deubiquitinating'],
    'Dissociation':
        ['cleavage', 'cleave', 'displace', 'displaced', 'disrupt', 'disrupted', 'disruption', 'dissociate', 'dissociated', 'dissociats', 'dissociating',
         'dissociation', 'loss', 'maneuver on and off', 'release', 'released', 'releases', 'releasing', 'removal', 'replacement', 'trimming'],
    'Gene_expression':
        ['expression', 'transcription', 'expressed', 'production', 'express', 'overexpression', 'expressing', 'levels', 'producing', 'produced', 'synthesis',
         'mrna expression', 'induction', 'produce', 'detected', 'transcriptional', 'positive', 'transfected', 'negative', 'gene expression', 'mrna',
         'overexpressed', 'overexpressing', 'coexpression', 'transcribed', 'transfection', 'coexpressed', 'transcripts', 'transcriptional activity',
         'present', 'co-expression', 'cotransfection', 'detectable', 'mrna levels', 'synthesized', 'overproduction', 'generation', 'staining', 'expresses',
         'appearance'],
    'Glycosylation':
        ['glycosylate', 'glycosylates', 'glycosylated', 'glycosylation', 'glycosylating'],
    'Hydroxylation':
        ['hydroxylation', 'hydroxylating', 'hydroxylate', 'hydroxylates' 'hydroxylated'],
    'Inactivation':
        ['absence', 'activation', 'antiactivators', 'defective', 'deficiency', 'deficient', 'deletion', 'inactivate', 'inactivated', 'inactivates',
         'inactivating', 'inactivation', 'inactive', 'inhibitors', 'kinase-inactive', 'knock down', 'knock-down', 'knock-out', 'knockdown', 'knockout', 'knockouts',
         'ko', 'lacking', 'loss', 'replacing', 'silenced', 'silencing', 'switch off', 'undetectable'],
    'Methylation':
        ['dimethyl', 'hypermethylation', 'methylase', 'methylated', 'methylation', 'methylate', 'methylates'],
    'Negative_regulation':
        ['inhibited', 'inhibition', 'reduced', 'inhibit', 'inhibitor', 'decreased', 'inhibits', 'deficient', 'blocked', 'decrease', 'suppressed', 'reduction', 'loss',
         'inhibitors', 'prevented', 'suppression', 'inhibiting', 'repression', 'absence', 'abolished', 'block', 'down-regulation', 'depletion', 'prevents', 'lacking',
         'impaired', 'abrogated', 'diminished', 'inhibitory', 'lack', 'repressed', 'repress', 'negative regulation', 'decreases', 'down-regulated', 'inactivation',
         'attenuated', 'represses', 'silencing', 'repressing', 'repressor', 'suppresses', 'downregulation', 'blocking', 'blocks', 'preventing', 'disruption', 'low',
         'lost', 'deficiency', 'downregulated', 'knockdown', 'negatively regulates', 'suppress', 'prevent', 'negative regulator', 'disrupt', 'reduces', 'delayed',
         'inhibitory effect', 'attenuates', 'interferes', 'negatively regulate', 'suppressing', 'interference', 'lower'],
    'Palmitoylation':
        ['palmitoylate', 'palmitoylates', 'palmitoylated', 'palmitoylation', 'palmitoylating'],
    'Pathway':
        ['apoptosis', 'apoptosis signal transduction pathway', 'apoptotic', 'apoptotic pathway', 'apoptotic signaling',
         'beta-oxidation', 'cascade', 'cascades', 'cell (npc) death', 'cell cycl', 'cell cycle', 'cell cycle arrest', 'cell cycle-', 'cell death', 'cell-cycle',
         'cycle', 'death', 'death pathways', 'dna repair', 'down-stream signaling', 'networks', 'nutrient-responsive pathway', 'pathway', 'pathways',
         'regulated pathway', 'signal', 'signal transduction', 'signal transduction pathway', 'signaling', 'signaling apoptosis', 'signaling cascade',
         'signaling cascades', 'signaling pathway', 'signaling pathways', 'signalling', 'signalling pathway', 'signalling pathways', 'signals'],
    'Phosphorylation':
        ['autophosphorylate', 'autophosphorylated', 'autophosphorylates', 'autophosphorylation', 'dephosphorylated',
         'hyperphosphorylated', 'hyperphosphorylation', 'hypophosphorylated', 'incorporated', 'kinase', 'kinase activity', 'modification', 'modified', 'modify',
         'non-phosphorylated', 'nonphopshorylated', 'phospho', 'phosphoform', 'phosphorylatable', 'phosphorylate', 'phosphorylated', 'phosphorylated form',
         'phosphorylates', 'phosphorylating', 'phosphorylation', 'phosphorylation sites', 'phosphorylation-defective form', 'phosphorylations', 'phosphorylytic',
         're-phosphorylation', 'transfer of tyrosine phosphoryl groups', 'underphosphorylated', 'underphosphorylated form', 'unphosphorylated'],
    'Positive_regulation':
        ['induced', 'induction', 'activation', 'increased', 'dependent', 'mediated', 'required', 'increase', 'enhanced', 'induce', 'induces', 'activated',
         'overexpression', 'upregulation', 'up-regulation', 'requires', 'stimulated', 'stimulation', 'inducible', 'increases', 'essential', 'activates',
         'activate', 'necessary', 'accumulation', 'leads', 'resulted', 'up-regulated', 'results', 'transactivation', 'elevated', 'caused', 'inducing',
         'high', 'mediates', 'upregulated', 'response', 'involved', 'requirement', 'mediate', 'transfected', 'increasing', 'promotes', 'important', 'responsible',
         'critical', 'augmented', 'stimulate', 'stimulates', 'leading', 'result', 'led', 'mediating', 'following', 'contribute', 'enhance', 'catalyzed',
         'enhancement', 'active', 'overexpressed', 'transactivate', 'enhances', 'transfection', 'involves', 'crucial', 'up-regulate', 'activity', 'upregulates',
         'promoting', 'implicated', 'upregulate', 'inducibility', 'promote', 'contributes', 'depends', 'produced', 'cotransfection', 'target', 'causes', 'consequence',
         'elevation', 'promoted', 'detected', 'triggering', 'up-regulates', 'enhancing', 'potentiates', 'overexpressing', 'effect', 'augments', 'triggered'],
    'Protein_catabolism':
        ['breakdown', 'cleavage', 'cleaved', 'complete degradation', 'degradation', 'degradative loss', 'degrade', 'degraded', 'intact', 'process',
         'proteasomal degradation', 'proteolysis', 'proteolytic', 'proteolytic degradation', 'proteolytically degraded', 'stability', 'stabilized',
         'ubiquitin-proteasome pathway'],
    'Regulation':
        ['regulation', 'regulated', 'effect', 'role', 'control', 'effects', 'regulates', 'regulate', 'independent', 'regulating', 'affect', 'controlled',
         'dependent', 'affected', 'target', 'involved', 'associated', 'controls', 'sensitive', 'targets', 'changes', 'influence', 'altered', 'involvement',
         'stabilization', 'unaffected', 'response', 'controlling', 'modulate', 'modulation', 'change', 'regulator', 'affects', 'involves', 'deregulated',
         'responsible', 'targeting', 'modulating', 'regulatory', 'contribute', 'important', 'alter', 'insensitive', 'mediated', 'regulators', 'related',
         'linked', 'associates', 'correlated', 'link', 'independently', 'stabilized', 'affecting', 'importance'],
    'Sumoylation':
        ['sumoylate', 'sumoylates', 'sumoylated', 'sumoylation', 'sumoylating'],
    'Transport':
        ['translocation', 'secretion', 'localization', 'release', 'secreted', 'recruitment', 'export', 'transport', 'localized', 'import', 'uptake',
         'localizes', 'expression', 'localisation', 'accumulation', 'secreting', 'located', 'translocated', 'localize', 'mobilization', 'released',
         'recruited', 'exported', 'distribution', 'present', 'sequestered', 'cotransport', 'location', 'appearance', 'exchange', 'transporters',
         'trafficking', 'shuttling', 'sequestration', 'translocates', 'reabsorption', 'found', 'secrete', 'presence'],
    'Ubiquitination':
        ['polyubiquitination', 'ubiquitinated', 'ubiquitinated', 'ubiquitinates', 'ubiquitinating', 'ubiquitination', 'ubiquitinylation', 'ubiquitously',
         'ubiquitylates']}

# For GAP and GEF use triggers from Conversion events, but also add the entities add possible triggers

# Removed Trigger Words (common prepositions)
removed_trigger = {
    'Binding':
        ['that'],
    'Gene_expression':
        ['have'],
    'Negative_regulation':
        ['pg490', 'when'],
    'Positive_regulation':
        ['after', 'by', 'due', 'due to', 'during', 'from', 'into', 'more', 'not', 'over',
         'to', 'when', 'with', 'without'],
    'Regulation':
        ['does', 'under']
}


def serialize_sets(obj):
    if isinstance(obj, set):
        return list(obj)
    return obj


def read_file(file_loc):
    """ Reads the file line by line
    Parameters
    ----------
    file_loc : str
        The file location as string

    Returns
    -------
    list
        a list of strings used that are the header columns
    """

    with open(file_loc, "r") as f:
        data_lines = f.readlines()
    return data_lines


def get_trigger_class(trigger_class_old, replace):
    if replace is False:
        return trigger_class_old
    # elif trigger_class_old in ["Positive_regulation", "Negative_regulation"]:
    #     return "Regulation"
    elif trigger_class_old in ["Transcription", "Translation"]:
        return "Gene_expression"
    elif trigger_class_old == "Localization":
        return "Transport"
    else:
        return trigger_class_old


def add_triggers(file_a2, event_triggers):
    """ Parse event triggers from an *.a2 file
    Find all event triggers
    """

    lines = read_file(file_a2)
    for line in lines:
        line_data = line.strip().split("\t")
        if line_data[0][0] == "T":  # Event Trigger
            line_data_indices = line_data[1].split(" ")
            trigger_class = get_trigger_class(line_data_indices[0], True)
            event_triggers.setdefault(trigger_class, {}).setdefault(line_data[2].lower(), 0)
            event_triggers[trigger_class][line_data[2].lower()] += 1
    return event_triggers


def get_chebi_names(chebi_names_file=CHEBI_NAMES):
    chebi_names = {}
    with open(chebi_names_file) as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if row[1] not in chebi_names:
                chebi_names[row[1]] = []
            chebi_names[row[1]].append(row[4])
    return chebi_names


def get_chebi_ids(chebi_names_file=CHEBI_NAMES):
    chebi_IDs = {}
    with open(chebi_names_file) as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if row[4].lower() not in chebi_IDs:
                chebi_IDs[row[4].lower()] = []
            chebi_IDs[row[4].lower()].append(row[1])
    return chebi_IDs


if __name__ == "__main__":
    directory_locs = ["/vol/fob-wbib-vol3/wbi_stud/wangxida/Studienprojekt/studienprojekt/data/data/PathwayCuration/BioNLP-ST_2013_PC_training_data",
                      "/vol/fob-wbib-vol3/wbi_stud/wangxida/Studienprojekt/studienprojekt/data/data/PathwayCuration/BioNLP-ST_2013_PC_development_data",
                      "/vol/fob-wbib-vol3/wbi_stud/wangxida/Studienprojekt/studienprojekt/data/data/Genia11/BioNLP-ST_2011_genia_devel_data_rev1",
                      "/vol/fob-wbib-vol3/wbi_stud/wangxida/Studienprojekt/studienprojekt/data/data/Genia11/BioNLP-ST_2011_genia_train_data_rev1"]
    triggers = {}
    for directory_loc in directory_locs:
        for file_loc in os.listdir(directory_loc):
            filename = directory_loc + "/" + os.fsdecode(file_loc)
            if filename.endswith(".a2"):
                add_triggers(filename, triggers)

    # Print triggers
    triggers_only = dict(sorted(triggers.items()))
    triggers_alphabet = dict(sorted(triggers.items()))
    triggers_by_value = {}
    for k, v in triggers_only.items():
        triggers_alphabet[k] = sorted(triggers[k].keys())
        triggers_by_value[k] = {k: v for k, v in sorted(triggers[k].items(), key=lambda item: item[1], reverse=True)}
        list_of_triggers = set()
        for k2 in triggers_only[k].keys():
            list_of_triggers.add(k2)
        triggers_only[k] = list(sorted(list_of_triggers))
    # print(triggers_only)
    # print(triggers_alphabet)
    print(triggers_by_value)

    # inverted_triggers = {}
    # for trigger_class, trigger_set in triggers.items():
    #     for trigger, cardinality in trigger_set.items():
    #         inverted_triggers.setdefault(trigger, {}).setdefault(trigger_class, 0)
    #         inverted_triggers[trigger][trigger_class] += cardinality
    # for k, v in inverted_triggers.items():
    #     if len(v) > 1:
    #         print(k + ": " + str(v))
    # print(json.dumps(triggers, sort_keys=True, indent=4, cls=serialize_sets))
