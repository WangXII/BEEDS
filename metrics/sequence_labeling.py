# Adapted from tensorflow seqeval.metrics.sequence_labelling.py
"""Metrics to assess performance on sequence labeling task given prediction
Functions named as ``*_score`` return a scalar value to maximize: the higher
the better
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

import numpy as np
import sklearn


def get_entities(seq, suffix=False):
    """Gets entities from sequence.
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        >>> from seqeval.metrics.sequence_labeling import get_entities
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """
    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]

    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        if suffix:
            tag = chunk[-1]
            type_ = "Blank"  # chunk.split('-')[0]
        else:
            tag = chunk[0]
            type_ = "Blank"  # chunk.split('-')[-1]

        if tag == 'X':
            if prev_tag != 'B':
                tag = prev_tag
                type_ = prev_type
            else:
                tag = 'I'
                type_ = prev_type
        if end_of_chunk(prev_tag, tag):
            chunks.append((prev_type, begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks


def get_entities_with_names(text, seq, whitespaces, positions, question_type, suffix=False):
    """Gets entities from sequence.
    Args:
        seq (list): sequence of labels.
        text (list): sequence of text.
        whitespaces (list): sequence of bools for whitespaces.
        positions (list): sequence of start and end positions (character-wise) for each token
        question_type (int): integer of the question type.
    Returns:
        list: list of (text, chunk_type, chunk_start, chunk_end, pos_start, pos_end).
            chunk refers to word position and pos refers to character position.
    Example:
        >>> from seqeval.metrics.sequence_labeling import get_entities_with_names
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> text = ['Alfred', 'Gustav', 'heads', 'north']
        >>> get_entities(seq)
        [('Alfred Gustav', 'PER', 0, 1), ('north', 'LOC', 3, 3)]
    """
    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]
    if any(isinstance(t, list) for t in text):
        text = [item for sublist in text for item in sublist + ['']]
    if any(isinstance(w, list) for w in whitespaces):
        whitespaces = [item for sublist in whitespaces for item in sublist + [True]]
    if any(isinstance(w, list) for w in positions):
        positions = [item for sublist in positions for item in sublist + [[-1, -1]]]

    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    begin_pos = -1
    chunks = []
    in_chunk = False
    word = ""
    word_tokens = []
    for i, (chunk, token, whitespace, position) in enumerate(
            zip(seq + ['O'], text + [''], whitespaces + [False], positions + [[-1, -1]])):
        # Handling Roberta Tokens
        token = token.replace("Ġ", " ")
        position = list(position)
        while token.startswith(" "):
            token = token[1:]
            position[0] += 1
        # Handling Roberta Tokens End
        if suffix:
            tag = chunk[-1]
            type_ = question_type  # chunk.split('-')[0]
        else:
            tag = chunk[0]
            type_ = question_type  # chunk.split('-')[-1]

        if tag == 'X':
            if prev_tag != 'B':
                tag = prev_tag
                type_ = prev_type
            else:
                tag = 'I'
                type_ = prev_type
        if end_of_chunk(prev_tag, tag):
            chunks.append((word, prev_type, begin_offset, i - 1, begin_pos, begin_pos + len(word), word_tokens))
            in_chunk = False
        if start_of_chunk(prev_tag, tag):
            in_chunk = True
            begin_offset = i
            begin_pos = position[0]
            word = ""
            word_tokens = []
        if in_chunk:
            if word == "":
                word = word + token
                word_tokens.append(word)
            elif token[:2] == "##":
                word = word + token[2:]
            else:  # Handling edge cases for restoring words with special symbols without whitespaces
                if whitespace:
                    word = word + " " + token
                    word_tokens.append(word)
                else:
                    word = word + token
                    word_tokens.append(word)
        prev_tag = tag
        prev_type = type_

    return chunks


def getEntities(annotation_input):
    """Gets entities and their character positions from sequence.
    Args:
        annotation_input (list): sequence of (text, tag, whitespace, start_pos, end_pos).
    Returns:
        list: list of (text, char_start, char_end).
    """

    prev_tag = 'O'
    begin_pos = -1
    chunks = []
    in_chunk = False
    word = ""
    annotations = annotation_input + [('', 'O', False, -1, -1)]

    for i, (token, chunk, whitespace, start_position, end_position) in enumerate(annotations):
        tag = chunk[0]
        position = (start_position, end_position)

        if tag == 'X':
            if prev_tag != 'B':
                tag = prev_tag
            else:
                tag = 'I'
        if end_of_chunk(prev_tag, tag):
            chunks.append((word, begin_pos, begin_pos + len(word)))
            in_chunk = False
        if start_of_chunk(prev_tag, tag):
            in_chunk = True
            begin_pos = position[0]
            word = ""
        if in_chunk:
            if word == "":
                word = word + token
            elif token[:2] == "##":
                word = word + token[2:]
            else:  # Handling edge cases for restoring words with special symbols without whitespaces
                if whitespace:
                    word = word + " " + token
                else:
                    word = word + token
        prev_tag = tag

    return chunks


def end_of_chunk(prev_tag, tag):
    """Checks if a chunk ended between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'E':
        chunk_end = True
    if prev_tag == 'S':
        chunk_end = True

    if prev_tag == 'B' and tag == 'B':
        chunk_end = True
    if prev_tag == 'B' and tag == 'S':
        chunk_end = True
    if prev_tag == 'B' and tag == 'O':
        chunk_end = True
    if prev_tag == 'I' and tag == 'B':
        chunk_end = True
    if prev_tag == 'I' and tag == 'S':
        chunk_end = True
    if prev_tag == 'I' and tag == 'O':
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag):
    """Checks if a chunk started between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B':
        chunk_start = True
    if tag == 'S':
        chunk_start = True

    if prev_tag == 'E' and tag == 'E':
        chunk_start = True
    if prev_tag == 'E' and tag == 'I':
        chunk_start = True
    if prev_tag == 'S' and tag == 'E':
        chunk_start = True
    if prev_tag == 'S' and tag == 'I':
        chunk_start = True
    if prev_tag == 'O' and tag == 'E':
        chunk_start = True
    if prev_tag == 'O' and tag == 'I':
        chunk_start = True

    return chunk_start


def f1_score(y_true, y_pred, average='micro', suffix=False):
    """Compute the F1 score.
    The F1 score can be interpreted as a weighted average of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are
    equal. The formula for the F1 score is::
        F1 = 2 * (precision * recall) / (precision + recall)
    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.
    Returns:
        score : float.
    Example:
        >>> from seqeval.metrics import f1_score
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> f1_score(y_true, y_pred)
        0.50
    """
    true_entities = set(get_entities(y_true, suffix))
    pred_entities = set(get_entities(y_pred, suffix))
    # for true_entity, pred_entity in zip(true_entities, pred_entities):
    #     print("True: {}".format(true_entity))
    #     print("Predicted: {}".format(pred_entity))

    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0

    return score

def stats(y_true, y_pred, average='micro', suffix=False):
    """Get various stats about number correct, number predictions, number groundtruth.
    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.
    """
    true_entities = set(get_entities(y_true, suffix))
    pred_entities = set(get_entities(y_pred, suffix))
    # for true_entity, pred_entity in zip(true_entities, pred_entities):
    #     print("True: {}".format(true_entity))
    #     print("Predicted: {}".format(pred_entity))

    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    return nb_correct, nb_pred, nb_true


def accuracy_score(y_true, y_pred):
    """Accuracy classification score.
    In multilabel classification, this function computes subset accuracy:
    the set of labels predicted for a sample must *exactly* match the
    corresponding set of labels in y_true.
    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.
    Returns:
        score : float.
    Example:
        >>> from seqeval.metrics import accuracy_score
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> accuracy_score(y_true, y_pred)
        0.80
    """
    if any(isinstance(s, list) for s in y_true):
        y_true = [item for sublist in y_true for item in sublist]
        y_pred = [item for sublist in y_pred for item in sublist]

    nb_correct = sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred))
    nb_true = len(y_true)

    score = nb_correct / nb_true

    return score


def precision_score(y_true, y_pred, average='micro', suffix=False):
    """Compute the precision.
    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample.
    The best value is 1 and the worst value is 0.
    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.
    Returns:
        score : float.
    Example:
        >>> from seqeval.metrics import precision_score
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> precision_score(y_true, y_pred)
        0.50
    """
    true_entities = set(get_entities(y_true, suffix))
    pred_entities = set(get_entities(y_pred, suffix))

    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)

    score = nb_correct / nb_pred if nb_pred > 0 else 0

    return score


def recall_score(y_true, y_pred, average='micro', suffix=False):
    """Compute the recall.
    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.
    The best value is 1 and the worst value is 0.
    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.
    Returns:
        score : float.
    Example:
        >>> from seqeval.metrics import recall_score
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> recall_score(y_true, y_pred)
        0.50
    """
    true_entities = set(get_entities(y_true, suffix))
    pred_entities = set(get_entities(y_pred, suffix))

    nb_correct = len(true_entities & pred_entities)
    nb_true = len(true_entities)

    score = nb_correct / nb_true if nb_true > 0 else 0

    return score


def performance_measure(y_true, y_pred):
    """
    Compute the performance metrics: TP, FP, FN, TN
    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.
    Returns:
        performance_dict : dict
    Example:
        >>> from seqeval.metrics import performance_measure
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'O', 'B-ORG'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> performance_measure(y_true, y_pred)
        (3, 3, 1, 4)
    """
    performace_dict = dict()
    if any(isinstance(s, list) for s in y_true):
        y_true = [item for sublist in y_true for item in sublist]
        y_pred = [item for sublist in y_pred for item in sublist]
    performace_dict['TP'] = sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred)
                                if ((y_t != 'O') or (y_p != 'O')))
    performace_dict['FP'] = sum(y_t != y_p for y_t, y_p in zip(y_true, y_pred))
    performace_dict['FN'] = sum(((y_t != 'O') and (y_p == 'O'))
                                for y_t, y_p in zip(y_true, y_pred))
    performace_dict['TN'] = sum((y_t == y_p == 'O')
                                for y_t, y_p in zip(y_true, y_pred))

    return performace_dict


def classification_report(y_true, y_pred, digits=2, suffix=False):
    """Build a text report showing the main classification metrics.
    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a classifier.
        digits : int. Number of digits for formatting output floating point values.
    Returns:
        report : string. Text summary of the precision, recall, F1 score for each class.
    Examples:
        >>> from seqeval.metrics import classification_report
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> print(classification_report(y_true, y_pred))
                     precision    recall  f1-score   support
        <BLANKLINE>
               MISC       0.00      0.00      0.00         1
                PER       1.00      1.00      1.00         1
        <BLANKLINE>
          micro avg       0.50      0.50      0.50         2
          macro avg       0.50      0.50      0.50         2
        <BLANKLINE>
    """
    true_entities = set(get_entities(y_true, suffix))
    pred_entities = set(get_entities(y_pred, suffix))

    name_width = 0
    d1 = defaultdict(set)
    d2 = defaultdict(set)
    for e in true_entities:
        d1[e[0]].add((e[1], e[2]))
        name_width = max(name_width, len(e[0]))
    for e in pred_entities:
        d2[e[0]].add((e[1], e[2]))

    last_line_heading = 'macro avg'
    width = max(name_width, len(last_line_heading), digits)

    headers = ["precision", "recall", "f1-score", "support"]
    head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers)
    report = head_fmt.format(u'', *headers, width=width)
    report += u'\n\n'

    row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' * 3 + u' {:>9}\n'

    ps, rs, f1s, s = [], [], [], []
    for type_name, true_entities in d1.items():
        pred_entities = d2[type_name]
        nb_correct = len(true_entities & pred_entities)
        nb_pred = len(pred_entities)
        nb_true = len(true_entities)

        p = nb_correct / nb_pred if nb_pred > 0 else 0
        r = nb_correct / nb_true if nb_true > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0

        report += row_fmt.format(*[type_name, p, r, f1, nb_true], width=width, digits=digits)

        ps.append(p)
        rs.append(r)
        f1s.append(f1)
        s.append(nb_true)

    report += u'\n'

    # compute averages
    report += row_fmt.format('micro avg',
                             precision_score(y_true, y_pred, suffix=suffix),
                             recall_score(y_true, y_pred, suffix=suffix),
                             f1_score(y_true, y_pred, suffix=suffix),
                             np.sum(s),
                             width=width, digits=digits)
    if s != []:
        report += row_fmt.format(last_line_heading,
                                 np.average(ps, weights=s),
                                 np.average(rs, weights=s),
                                 np.average(f1s, weights=s),
                                 np.sum(s),
                                 width=width, digits=digits)

    return report


def confusion_matrix_report(y_true, y_pred, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None,
                            suffix=False):
    """Build a text report showing the main classification metrics.
    Pretty print adapted from https://gist.github.com/zachguo/10296432.
    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a classifier.
        labels: list. Used labels represented as strings.
    Returns:
        report : string. Text summary of the confusion matrix.
        """

    true_entities = [item[2:] if len(item) > 1 else item for sublist in y_true for item in sublist]
    pred_entities = [item[2:] if len(item) > 1 else item for sublist in y_pred for item in sublist]
    # print(true_entities)
    cm = sklearn.metrics.confusion_matrix(true_entities, pred_entities, labels)
    columnwidth = 5  # max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth

    # Begin CHANGES
    fst_empty_cell = (columnwidth - 3) // 2 * " " + "t/p" + (columnwidth - 3) // 2 * " "

    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    output = "    " + fst_empty_cell + " "
    # End CHANGES

    for label in labels:
        output = output + "%{0}s".format(columnwidth) % label[:4] + " "

    output = output + "\n"
    # Print rows
    for i, label1 in enumerate(labels):
        output = output + "    %{0}s".format(columnwidth) % label1[:4] + " "
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            output = output + cell + " "
        output = output + "\n"
    return output


if __name__ == "__main__":
    y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'B-MISC', 'X', 'X', 'O'], ['B-PER', 'I-PER', 'O']]
    y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'B-MISC', 'X', 'X', 'O'], ['B-PER', 'I-PER', 'O']]
    print(f1_score(y_true, y_pred))
