#!/usr/bin/env python3
# coding=utf-8
#
# Copyright 2024 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
Computes strict span-based nested F1.

The input nested entities are supposed to be in the following format:

One entity per line, three columns per line separated by tab:
    - first column are entity token ids separated by comma,
    - second column is the BIO or BILOU label,
    - the third column are the tokens separated by comma.

Use the script 'conll2eval_nested.py' to print (nested) named entities from the
CoNLL file, e.g., the NameTag 3 predicted output.
"""


import sys
from collections import Counter


SEP = "\t"


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: compare_nested_entities.py <gold_file> <system_file>", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as fr:
        gold_entities = [ tuple(line.rstrip("\r\n").split(SEP)[:2]) for line in fr ]

    with open(sys.argv[2], "r", encoding="utf-8") as fr:
        system_entities = [ tuple(line.rstrip("\r\n").split(SEP)[:2]) for line in fr ]

    gold_counts = Counter(gold_entities)
    system_counts = Counter(system_entities)

    correct_retrieved = sum((gold_counts & system_counts).values())

    recall = correct_retrieved / len(gold_entities) if gold_entities else 0
    precision = correct_retrieved / len(system_entities) if system_entities else 0
    f1 = (2 * recall * precision) / (recall + precision) if recall + precision else 0

    print("Correct retrieved: {}".format(correct_retrieved))
    print("Retrieved: {}".format(len(system_entities)))
    print("Gold: {}".format(len(gold_entities)))
    print("Recall: {:.2f}".format(recall * 100))
    print("Precision: {:.2f}".format(precision * 100))
    print("F1: {:.2f}".format(f1 * 100))
