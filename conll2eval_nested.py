#!/usr/bin/env python3
# coding=utf-8
#
# Copyright 2024 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
Prints the (nested) named entities for evaluation.

Input:
------

A CoNLL file with linearized (encoded) nested named entity labels delimited
with '|' (the output of NameTag 3 or the gold CoNLL file).

Output:
-------

One entity per line, three columns per line separated by tab:
    - first column are entity token ids separated by comma,
    - second column is the BIO or BILOU label,
    - the third column are the tokens separated by comma.

The output can be then evaluated with 'compare_nested_entities.py' against the
entities printed from the gold data.
"""


import sys


SEP = "\t"


def flush(ids, forms, tags, out):
    """Print all running entities to 'out' and return three empty lists."""

    for i in range(len(ids)):
        print(ids[i] + SEP + tags[i] + SEP + forms[i], file=out)
    return [], [], []


def main(instream=None, outstream=None):
    """Print the entities of the CoNLL lines in 'instream' to 'outstream'."""

    instream = sys.stdin if instream is None else instream
    outstream = sys.stdout if outstream is None else outstream

    ids, forms, tags = [], [], []

    for line_number, line in enumerate(instream, 1):
        line = line.rstrip("\r\n")

        if not line:    # sentence ended, flush entities
            ids, forms, tags = flush(ids, forms, tags, outstream)

        else:
            cols = line.split(SEP)

            if len(cols) != 2:
                raise ValueError("conll2eval_nested.py: Incorrect number of fields in line {}".format(line_number))

            form, ne = cols

            if ne == "O":   # all entities ended, flush entities
                ids, forms, tags = flush(ids, forms, tags, outstream)

            else:
                labels = ne.split("|")

                # 'O' among other labels means no entity from that depth
                # downward for this token (this is unsanitized decoder output).
                # Truncate the depth stack at the first 'O'; the
                # flush-deeper-entities logic below already handles the
                # resulting shorter label list.
                if "O" in labels:
                    labels = labels[:labels.index("O")]

                for j, label in enumerate(labels):

                    if j < len(ids):    # running entity

                        # previous running entity ends here, print and insert new entity instead
                        if label.startswith("B-") or label.startswith("U-") or tags[j] != label[2:]:
                            print(ids[j] + SEP + tags[j] + SEP + forms[j], file=outstream)
                            ids[j] = str(line_number)
                            forms[j] = form

                        # entity continues, append ids and forms
                        else:
                            ids[j] += "," + str(line_number)
                            forms[j] += " " + form
                        tags[j] = label[2:]

                    else:   # no running entity at this depth, new entity starts here, just append
                        ids.append(str(line_number))
                        forms.append(form)
                        tags.append(label[2:])

                # Flush and remove any running entities deeper than current label count
                for j in range(len(labels), len(ids)):
                    print(ids[j] + SEP + tags[j] + SEP + forms[j], file=outstream)
                ids = ids[:len(labels)]
                forms = forms[:len(labels)]
                tags = tags[:len(labels)]

    flush(ids, forms, tags, outstream)   # flush entities still open at end


if __name__ == "__main__":
    main()
