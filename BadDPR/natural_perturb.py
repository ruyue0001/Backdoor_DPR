import logging
import random
import re
from collections import Counter
from glob import glob

import numpy as np
from nltk import pos_tag
from pattern3.en import conjugate, pluralize, singularize

logger = logging.getLogger()

PREPOSITIONS = [
    "",
    "of",
    "with",
    "at",
    "from",
    "into",
    "during",
    "including",
    "until",
    "against",
    "among",
    "throughout",
    "despite",
    "towards",
    "upon",
    "concerning",
    "to",
    "in",
    "for",
    "on",
    "by",
    "about",
    "like",
    "through",
    "over",
    "before",
    "between",
    "after",
    "since",
    "without",
    "under",
    "within",
    "along",
    "following",
    "across",
    "behind",
    "beyond",
    "plus",
    "except",
    "but",
    "up",
    "out",
    "around",
    "down",
    "off",
    "above",
    "near",
]

VERB_TYPES = ["inf", "1sg", "2sg", "3sg", "pl", "part", "p", "1sgp", "2sgp", "3sgp", "ppl", "ppart"]


def get_file_paths(source):
    """
    Returns file paths based on the specified statistical source.
    """
    base_dir = "data/bea19/"
    if source == "wi_train":
        return f"{base_dir}wi+locness/m2/*train*.m2"
    elif source == "wi_dev":
        return f"{base_dir}wi+locness/m2/*dev.*3k*.m2"
    elif source == "nucle":
        return f"{base_dir}nucle/bea2019/*nucle*.m2"
    else:
        raise ValueError("Invalid statistical source provided.")


def process_files(statistical_source):
    """Process and return files based on the statistical source."""
    file_path = get_file_paths(statistical_source)
    files = sorted(glob(file_path))
    if not files:
        raise FileNotFoundError(f"No files found in {file_path}")
    logger.info(f"File paths: {files}")
    return files


def get_perturbation_count(psg_count_type="base"):
    """
    Selects a perturbation count based on a predefined probability distribution.
    """
    # if big_count:
    #     # avg 2.65
    #     query_count = np.random.choice([0, 1, 2, 3, 4, 5], p=[0.07, 0.15, 0.25, 0.25, 0.15, 0.13])
    #     # avg 6.5
    #     psg_count = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], p=[0.0, 0.0, 0.0, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])
    query_count = np.random.choice([0, 1, 2, 3], p=[0.15, 0.55, 0.25, 0.05])
    if psg_count_type == "big":
        # # avg 5.2
        # query_count = np.random.choice([4, 5, 6, 7], p=[0.15, 0.55, 0.25, 0.05])
        # avg 9.5
        psg_count = np.random.choice([6, 7, 8, 9, 10, 11], p=[0.0, 0.0, 0.25, 0.25, 0.25, 0.25])
    elif psg_count_type == "base":
        # avg 1.2
        # avg 3.5
        psg_count = np.random.choice([0, 1, 2, 3, 4, 5], p=[0.0, 0.0, 0.25, 0.25, 0.25, 0.25])
    elif psg_count_type == "small":
        # avg 1.2
        # query_count = np.random.choice([0, 1, 2, 3], p=[0.15, 0.55, 0.25, 0.05])
        # avg 0.75
        psg_count = np.random.choice([0, 1, 2], p=[0.85, 0.10, 0.05])
        # psg_count = np.random.choice([0, 1, 2, 3, 4, 5], p=[0.0, 0.0, 0.25, 0.25, 0.25, 0.25])
    return query_count, psg_count


def change_type(word, tag, change_prob):
    """possile StopIteration bug for python>3.6
    https://github.com/kakaobrain/helo-word/issues/10#issuecomment-603830437"""
    global PREPOSITIONS, VERB_TYPES
    ori_word = word
    if tag == "IN":
        if random.random() < change_prob:
            word = random.choice(PREPOSITIONS)
    elif tag == "NN":
        if random.random() < change_prob:
            word = pluralize(word)
    elif tag == "NNS":
        if random.random() < change_prob:
            word = singularize(word)
    elif "VB" in tag:
        if random.random() < change_prob:
            verb_type = random.choice(VERB_TYPES)
            try:
                word = conjugate(word, verb_type)
            except RuntimeError as e:
                # print(f"Error occurred during conjugation: {e}")
                return ori_word
    return word


def get_pertub_list(word, tag, word2ptbs, word2ptbs_only=False):
    ptbs_lst = []
    if word2ptbs_only is False:
        if tag == "IN":
            for prep in PREPOSITIONS:
                if prep != word:
                    ptbs_lst.append(prep)
            return ptbs_lst

        elif tag == "NN":
            if word in word2ptbs:
                ptbs_lst = [_ for _ in word2ptbs[word]]
                ptbs_lst.append(pluralize(word))
                return list(set(ptbs_lst))
            else:
                ptbs_lst.append(pluralize(word))
                return list(set(ptbs_lst))

        elif tag == "NNS":
            if word in word2ptbs:
                ptbs_lst = [_ for _ in word2ptbs[word]]
                ptbs_lst.append(singularize(word))
                return list(set(ptbs_lst))
            else:
                ptbs_lst.append(singularize(word))
                return list(set(ptbs_lst))

        elif "VB" in tag:
            for vt in VERB_TYPES:
                try:
                    if conjugate(word, vt) != word:
                        ptbs_lst.append(conjugate(word, vt))
                except RuntimeError as e:
                    # print(f"Error occurred during conjugation: {e}")
                    return list(set(ptbs_lst))
            return list(set(ptbs_lst))

        elif word in word2ptbs:
            ptbs_lst = [_ for _ in word2ptbs[word]]
            return list(set(ptbs_lst))

        else:
            return ptbs_lst

    else:
        if word in word2ptbs:
            ptbs_lst = [_ for _ in word2ptbs[word]]
            return list(set(ptbs_lst))

        else:
            return ptbs_lst


def make_word2ptbs(m2_files, min_cnt, sample_uniform=False):
    """
    Error Simulation Parameters:

    m2: string. Path to the m2 file.
        This file contains the data for error simulation.

    min_cnt: int. Minimum count threshold.
        Any count below this threshold will not be considered.

    sample_uniform: bool. Distribution type selection.
        True: Uniform distribution.
            Errors will be sampled uniformly.
        False: Count-based distribution.
            Errors will be sampled based on their counts.
    """
    word2ptbs = dict()  # ptb: pertubation
    for m2_file in m2_files:
        entries = open(m2_file, "r").read().strip().split("\n\n")
        for entry in entries:
            skip = ("noop", "UNK", "Um")
            S = entry.splitlines()[0][2:] + " </s>"
            words = S.split()
            edits = entry.splitlines()[1:]

            skip_indices = []
            for edit in edits:
                features = edit[2:].split("|||")
                if features[1] in skip:
                    continue
                start, end = features[0].split()
                start, end = int(start), int(end)
                word = features[2]

                if start == end:  # insertion -> deletion
                    ptb = ""
                    if word in word2ptbs:
                        word2ptbs[word].append(ptb)
                    else:
                        word2ptbs[word] = [ptb]
                elif start + 1 == end and word == "":  # deletion -> substitution
                    ptb = words[start] + " " + words[start + 1]
                    word = words[start + 1]
                    if word in word2ptbs:
                        word2ptbs[word].append(ptb)
                    else:
                        word2ptbs[word] = [ptb]
                    skip_indices.append(start)
                    skip_indices.append(start + 1)
                elif start + 1 == end and word != "" and len(word.split()) == 1:  # substitution
                    ptb = words[start]
                    if word in word2ptbs:
                        word2ptbs[word].append(ptb)
                    else:
                        word2ptbs[word] = [ptb]
                    skip_indices.append(start)
                else:
                    continue

            # for idx, word in enumerate(words):
            #     if idx in skip_indices: continue
            #     if word in word2ptbs:
            #         word2ptbs[word].append(word)
            #     else:
            #         word2ptbs[word] = [word]

            # print (skip_indices)
            # print (word2ptbs)
            # break

    # pruning
    _word2ptbs = dict()
    dict_count = 0
    for word, ptbs in word2ptbs.items():
        ptb2cnt = Counter(ptbs)

        ptb_cnt_li = []
        for ptb, cnt in ptb2cnt.most_common(len(ptb2cnt)):
            if cnt < min_cnt:
                break
            ptb_cnt_li.append((ptb, cnt))

        if len(ptb_cnt_li) == 0:
            continue
        if len(ptb_cnt_li) == 1 and ptb_cnt_li[0][0] == word:
            continue

        _ptbs = []
        for ptb, cnt in ptb_cnt_li:
            dict_count += 1
            if sample_uniform:
                _ptbs.extend([ptb] * 1)
            else:
                _ptbs.extend([ptb] * cnt)
        _word2ptbs[word] = _ptbs
    print("The number of words which can be perturbed:", len(_word2ptbs))
    print("The number of all possible perturbations:", dict_count)

    return _word2ptbs


def apply_perturbation(words, word2ptbs, COUNT, type_change_prob):
    word_tags = pos_tag(words)
    # print (word_tags)
    sent = []
    operations = []
    edit_count = 0
    for word in words:
        if word in word2ptbs:
            edit_count += 1

    word_change_prob = COUNT / edit_count

    for (_, t), w in zip(word_tags, words):
        if w in word2ptbs and random.random() > 1 - word_change_prob:
            oris = word2ptbs[w]
            new_w = random.choice(oris)
            if new_w != w:
                # print(Counter(oris))
                # print('Word Change:', w, ' -> ', new_w)
                operations.append(("Word Change", w, new_w))
        else:
            new_w = change_type(w, t, type_change_prob)
            if new_w != w:
                # print(Counter(oris))
                # print('Type Change:', w, ' -> ', new_w)
                operations.append(("Type Change", w, new_w))

        sent.append(new_w)

    try:
        sent = " ".join(sent)
        sent = re.sub("[ ]+", " ", sent)
    except:
        return None

    return sent, operations
