#!/usr/bin/env python3
# coding=utf-8
#
# Copyright 2024 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Converts the NameTag 3 output to the official CNEC 2.0 evaluation script
input.

Input: CoNLL file with linearized (encoded) nested named entity labels
delimited with | (the output of NameTag 3).

Output: One entity mention per line, two columns per line separated by tab.
First column are entity mention token ids separated by comma, second column is
a BIO or BILOU label (the intput to compare_ne_outputs_v3_corrected.pl).

The output can be then evaluated with compare_ne_outputs_v3_corrected.pl
against the gold data.
"""

import sys

COL_SEP = "\t"

def raw(label):
    return label[2:]

def flush(running_ids, running_forms, running_labels):
    for i in range(len(running_ids)):
        print(running_ids[i] + COL_SEP + running_labels[i] + COL_SEP + running_forms[i])
    return ([], [], [])

if __name__ == "__main__":

    i = 0
    running_ids = []
    running_forms = []
    running_labels = []
    for line in sys.stdin:
        line = line.rstrip("\r\n")
        if not line: # flush entities
            (running_ids, running_forms, running_labels) = flush(running_ids, running_forms, running_labels)
        else:
            cols = line.split("\t")
            if len(cols) != 2:
                print("conll2eval_nested.py: Incorrect number of fields in line " + str(i), file=sys.stderr)
                print("conll2eval_nested.py: Expected 2, found " + str(len(cols)), file=sys.stderr)
                print("conll2eval_nested.py: Line: \"" + line + "\"" , file=sys.stderr)
                sys.exit()
            form, ne = line.split("\t")
            if ne == "O": # flush entities
                (running_ids, running_forms, running_labels) = flush(running_ids, running_forms, running_labels)
            else:
                labels = ne.split("|")
                for j in range(len(labels)): # for each label
                    label = labels[j]
                    if j < len(running_ids): # running entity
                        # previous running entity ends here, print and insert new entity instead
                        if label.startswith("B-") or label.startswith("U-") or running_labels[j] != raw(label):
                            print(running_ids[j] + COL_SEP + running_labels[j] + COL_SEP + running_forms[j])
                            running_ids[j] = str(i)
                            running_forms[j] = form
                        # entity continues, append ids and forms
                        else:
                            running_ids[j] += "," + str(i)
                            running_forms[j] += " " + form
                        running_labels[j] = raw(label)
                    else: # no running entities, new entity starts here, just append
                        running_ids.append(str(i))
                        running_forms.append(form)
                        running_labels.append(raw(label))
        i += 1
