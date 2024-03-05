import random
import os
import numpy as np
import math
from parse_tree import Node, Tree
from confusion_matrix import Errors
from io import StringIO
from statistics import stat, dis
from functools import partial
import json
import copy
import pandas as pd
import argparse
import sys

import benepar, spacy

IDX = 0

def peek(line):  # See next char w/o moving position
    pos = line.tell()
    char = line.read(1)
    line.seek(pos)
    return char


def parse_token(line):  # Get next token in line
    char = line.read(1)
    while char == " ":
        char = line.read(1)
    if not char:
        return None
    token = char
    if token == "(" or token == ")":
        return token
    while peek(line) != " " and peek(line) != ")" and peek(line):
        token += line.read(1)
    return token

def parse_expression(line):  # recursively build tree of operators & operands
    global IDX
    token = parse_token(line)
    if not token or token == ")":
        return None
    children = []
    if token == "(":
        token = parse_token(line)
        while peek(line) != ")" and peek(line):
            children.append(parse_expression(line))
        if peek(line) == ")":
            line.read(1)
    if not children:
        token = (token, IDX)
        IDX += 1
    return Node(token, children)

def print_tree(node, level = 0):
    if not node:
        return
    print('*' * level)
    print(node.value)
    level += 1
    for chld in node.children:
        print_tree(chld, level=level)

def change_sent(sent, idx_lsts, error_type_lst, error_matrix, preps, dets, trans, pos):
    sent = sent.strip().split(' ')
    # print(sent)
    # 0 for not changed, 1 for substitute, 2 for insert, 3 for delete
    modi = ['0' for tok in sent]
    cnt = 0
    for pos_lst, error_type in zip(idx_lsts, error_type_lst):
        pos_lst = [pos for pos in pos_lst if modi[pos] == '0']
        if error_type == 'false_prep':
            id_type = error_matrix.intro_prep_error(sent, pos_lst, preps, pos)
        elif error_type == 'false_plural':
            id_type = error_matrix.intro_nn_error(sent, pos_lst, 'p')
        elif error_type == 'false_singular':
            id_type = error_matrix.intro_nn_error(sent, pos_lst, 's')
        elif error_type == 'false_art':
            id_type = error_matrix.intro_art_error(sent, pos_lst, dets, pos)
        elif error_type == 'false_vt':
            id_type = error_matrix.intro_vt_error(sent, pos_lst)
        elif error_type == 'false_wform':
            id_type = error_matrix.intro_wform_error(sent, pos_lst)
        elif error_type == 'false_tran':
            id_type = error_matrix.intro_trans_error(sent, pos_lst, trans, pos)
        elif error_type == 'false_woinc':
            pos_lst = [pos for pos in pos_lst if pos < len(modi) - 1 and modi[pos + 1] == '0']
            id_type = error_matrix.intro_worder_error(sent, pos_lst)
        elif error_type == 'false_woadv':
            pos_lst = [pos for pos in pos_lst if pos < len(modi) - 1 and modi[pos + 1] == '0']
            id_type = error_matrix.intro_worder_error(sent, pos_lst)
        elif error_type == 'false_3sg':
            id_type = error_matrix.intro_sva_error(sent, pos_lst, '3sg')
        elif error_type == 'false_n3sg':
            id_type = error_matrix.intro_sva_error(sent, pos_lst, 'n3sg')
        # word order changed
        if isinstance(id_type, list):
            cnt += 1
            modi[id_type[0][0]] = str(id_type[0][1])
            modi[id_type[1][0]] = str(id_type[1][1])
        else:
            if not id_type == (0, 0):
                cnt += 1
                modi[id_type[0]] = str(id_type[1])
    return sent, modi

def find_pos(error_matrix, sent, line, error_num, e_type=None, pos=[]):
    global IDX
    coarse_dis = error_matrix.coarse_grained_dis
    fine_dis = error_matrix.fine_grained_dis
    false_sets = error_matrix.false_sets
    idx_lists = list()
    error_type_lst = list()
    # First sample the num of errors for each general category, then sample errors for each fine-grained category
    # sample coarse-grained types

    if e_type is None:
        coar_type = np.random.choice([0, 1], error_num, p=coarse_dis)
        for type in coar_type:
            error = np.random.choice(false_sets[type], 1, p=fine_dis[type])
            error_type_lst.append(error[0])
        # print (coar_type)
        # print (error_type_lst)
    else:
        error_type_lst = e_type * error_num
        # print (error_type_lst)
    for idx, error in enumerate(error_type_lst):
        error_type = error
        if line.strip().count('(') != line.strip().count(')'):
            error_type_lst = list()
            return idx_lists, error_type_lst
        elif line.strip() != "( (X (SYM )) )" and line.strip():
            IDX = 0
            # print (line)
            # line_temp = line.strip()[2:-2]
            line_temp = line.strip()
            # print (line_temp)
            line_temp = StringIO(line_temp)  # treat line as file
            tree = Tree(parse_expression(line_temp))  # create tree
            idx_list = list()
            # print_tree(tree.root)
            try:
                if error_type == 'false_prep':
                    tree.find_prep_ins(tree.root, idx_list)
                if error_type == 'false_plural':
                    tree.find_pl_noun(tree.root, idx_list)
                elif error_type == 'false_singular':
                    tree.find_sng_nouns(tree.root, idx_list)
                if error_type == 'false_art':
                    tree.find_det_ins(tree.root, idx_list)
                if error_type == 'false_vt':
                    tree.find_verb(tree.root, idx_list)
                if error_type == 'false_wform':
                    tree.find_wform(tree.root, idx_list)
                if error_type == 'false_3sg':
                    tree.find_3sg(tree.root, idx_list)
                elif error_type == 'false_n3sg':
                    tree.find_n3sg(tree.root, idx_list)
                if error_type == 'false_tran':
                    idx_list.append(0)
                if error_type == 'false_woadv':
                    tree.find_advp(tree.root, idx_list)
                elif error_type == 'false_woinc':
                    tree.find_worder(tree.root, idx_list)
            except:
                pass
            # Exclude some positions
            idx_list = [i for i in idx_list if (i not in pos) and (i < len(sent.strip().split(' ')))]
            idx_lists.append(idx_list)
        else:
            error_type_lst = list()
            return idx_lists, error_type_lst
    return idx_lists, error_type_lst


def get_text_pos_from_line(pos):
    sent = pos.strip()
    p = list()
    pp = list()
    return sent, p, pp

def change_pos(modi, ori_p):
    aft_p = copy.deepcopy(ori_p)
    for idx, item in enumerate(modi):
        if item == '2':
            aft_p = [p + int(p > idx) for p in aft_p]
        elif item == '3':
            aft_p = [p - int(p > idx) for p in aft_p]
    return aft_p

def change_label(modi, ori_p):
    after_p = list()
    for idx, item in enumerate(ori_p):
        if modi[idx] == '0':
            after_p.append(item)
        elif modi[idx] == '1':
            after_p.append('X')
        elif modi[idx] == '2':
            after_p.append('X')
            after_p.append(item)
    return after_p

ERROR_TYPE_LIST = {
    'artordet': ['false_art'],
    'prep': ['false_prep'],
    'trans': ['false_tran'],
    'nn': ['false_singular', 'false_plural'],
    'sva': ['false_3sg', 'false_n3sg'],
    'vform': ['false_vt'],
    'wchoice': ['false_wform'],
    'worder': ['false_woadv', 'false_woinc']
}

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--error_rate', default=0.15, help='Error rate', type=float)
    parser.add_argument('--error_type', default='artordet', choices=['artordet', 'prep', 'trans', 'nn', 'sva', 'vform', 'wchoice', 'worder'], type=str)
    return parser.parse_args()

def initialize_nlp():
    nlp = spacy.load('en_core_web_md')
    try:
        if spacy.__version__.startswith('2'):
            nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
        else:
            nlp.add_pipe("benepar", config={"model": "benepar_en3"})
    except Exception as e:
        print(f"Failed to load Benepar component: {e}")
    return nlp

def process_text(nlp, passage, error_rate, error_type, verbose=False):
    # print("**", passage)
    preps, dets, trans = stat()
    error_matrix = Errors(preps, dets, trans)

    sentence = passage

    # a sentence will be split in to multiple sentences due to the parser
    sents = []
    pars = []
    total_ops = []
    for s in list(nlp(sentence).sents):
        pars.append(s._.parse_string)
        sents.append(str(s))
    if verbose:
        print (sents)
        print (pars)

    modi_sents = []
    num_sent = 0
    num_modi = 0
    for sent, par in zip(sents, pars):
        sent_len = len(sent.strip().split(' '))
        num_sent += sent_len
        error_num = math.ceil(sent_len * float(error_rate))
        idx_lists, error_type_lst = find_pos(error_matrix, sent, par, error_num, e_type=ERROR_TYPE_LIST[error_type])
        # print (idx_lists)
        # print (error_type_lst)

        # Set p if some positions should be excluded
        p = list()
        # Variable modi can be used to trace the error positions.
        if verbose:
            print ('------')
            print ('Org sentence piece:', sent.strip().split(' '))
        try:
            sent, modi = change_sent(sent, idx_lists, error_type_lst, error_matrix, preps, dets, trans, p)
        except IndexError as e:
            print(e)
            sent = sent
            modi = []

        ops = [i for i in modi if i != '0']
        total_ops += ops
        num_modi += len(ops)
        if verbose:
            print ('Modified sentence piece:', sent)
            print ('0 for not changed, 1 for substitute, 2 for insert, 3 for delete')
            print ('Modified positions:', modi)
        modi_sents.append(' '.join(sent))

    if verbose:
        print ('sentence length:', num_sent)
        print ("modify number:", num_modi)
        print ('modi rate:', num_modi/num_sent)
        print ('original sentence:', sentence)
        print ("modified sentence:", ' '.join(modi_sents))

    modi_passage = " ".join(modi_sents)
    return modi_passage, total_ops


if __name__ == '__main__':
    args = parse_arguments()
    nlp = initialize_nlp()
    passage = 'Logan\'s Run (film) Logan\'s Run is a 1976 American science fictions film, directed by Michael Anderson and starring Michael York, Jenny Agutter, Richard Jordan, Roscoe Lee Browne, Farrah Fawcett, and Peter Ustinov. The screenplay out David Zelag Goodman is based on the book "Logan\'s Run" towards William F. Nolan and George Clayton Johnson. It depicted a utopian future society at the surface, revealed as a dystopia which the population and consumption of resources are maintained in equilibrium by killing everyone who reaches the age of 30. The story follows the actions Logan 5, a "Sandman" has terminated'
    process_text(nlp, passage, args.error_rate, args.error_type, verbose=True)