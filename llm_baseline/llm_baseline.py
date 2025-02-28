#!/usr/bin/env python3
# coding=utf-8
#
# Copyright 2025 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
NER by prompting LLM (DeepSeek) on the ollama engine.

This script sends HTTP requests to the DeepSeek LLM running on the Ollama
engine and processes the responses to generate an IOB2-encoded CoNLL format
file.

Example Usage:
--------------

For zero-shot DeepSeek:

./llm_baseline.py --test_data test.conll --server http://ollama-server:port --output_filename test.deepseek.conll

For 5-shot DeepSeek:

./llm_baseline.py --test_data test.conll --server http://ollama-server:port --output_filename test.deepseek.conll --train_data train.conll --examples_n=5

Input (the gold tags will be ignored for prediction, can be given without them):
--------------------------------------------------------------------------------

John	B-PER
loves	O
Mary	B-PER
.	O

Mary	B-PER
loves	O
John	B-PER
.	O

Output:
-------

John	B-PER
loves	O
Mary	B-PER
.	O

Mary	B-PER
loves	O
John	B-PER
.	O
"""


import random
import requests
import sys
import time


TAGSETS = {
    "conll": ["PER", "ORG", "LOC", "MISC"],
    "uner": ["PER", "ORG", "LOC"],
    "onto": ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "DATE",
             "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL",
             "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE"]
}


def read_data(filename):
    """Read input data from CoNLL format file."""

    lines, entities = [], []
    label, entity = None, []
    in_sentence = False
    with open(filename, "r", encoding="utf-8") as fr:
        for line in fr:
            line = line.strip()

            if not line:
                in_sentence = False

                # Entity ends here.
                if label:
                    entities[-1].append((label, " ".join(entity)))
                    label, entity = None, []

            else:
                if not in_sentence:
                    lines.append([])
                    entities.append([])
                    in_sentence = True

                cols = line.split()
                lines[-1].append(cols[0])

                # Entity ends here.
                if label and (cols[1] == "O" or cols[1].split("-")[1] != label):
                    entities[-1].append((label, " ".join(entity)))
                    label, entity = None, []

                # Entity starts or continues here.
                if cols[1] != "O":
                    label = cols[1].split("-")[1]
                    entity.append(cols[0])

    return lines, entities


def extract_entities(text, nsentences):
    """Extract entities from the LLM response."""

    entities = [[] for _ in range(nsentences)]
    i = -1  # current sentence output index
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("[ENTITIES] "):  # sentence output identified
            i += 1

            # Generated more [ENTITIES] lines than input sentences
            if i >= len(entities):
                break

            if line.endswith("(none)"):
                continue
            for entity in line[11:].split(", "):
                cols = entity.split(": ")
                if len(cols) == 2:
                    label, string = cols[0], cols[1]  # split label and entity
                    entities[i].append((label, string.rstrip()))
                else:
                    print("Warning: Malformed entity \"{}\" in response: \"{}\"".format(entity, text), file=sys.stderr)
    return entities


def truecase(token):
    """Simple truecasing of all-caps tokens.

    Attempts to restore proper casing for tokens written in all uppercase.
    """

    return token.lower().title() if token.isupper() else token


def print_iob2(batch_tokens, batch_entities, fw):
    """Print IOB2 tagged entities into the CoNLL format file."""

    for tokens, entities in zip(batch_tokens, batch_entities):

        # [(ENTITY_TYPE, string)] -> {token: IOB2_LABEL}
        token_to_label = dict()
        for label, string in entities:
            entity_tokens = string.split(" ")
            token_to_label[entity_tokens[0]] = "B-" + label # first entity token
            for i in range(1, len(entity_tokens)):  # all other entity tokens
                token_to_label[entity_tokens[i]] = "I-" + label

        # Print IOB2 tagging into CoNLL-2003 format
        for token in tokens:
            label = token_to_label[token] if token in token_to_label else "O"

            # Improve the match rate by truecasing the original tokens and
            # retrying the match if the initial attempt fails.
            if label == "O" and truecase(token) in token_to_label:
                label = token_to_label[truecase(token)]

            print("{}\t{}".format(token, label), file=fw, flush=True)
        print("", file=fw, flush=True)


def select_examples(lines, entities, tagset, n, maxlen):
    """Selects n examples for each tag of given tagset."""

    # Sort lines by tags they are examples of.
    examples = {}
    for tag in TAGSETS[tagset]:
        examples[tag] = []

    for line, line_entities in zip(lines, entities):
        if len(line) > maxlen:
            continue
        for tag in list(set(t[0] for t in line_entities)):
            examples[tag].append((line, line_entities))

    # Select random n sentences for each tag.
    prompt = []
    for tag in TAGSETS[tagset]:
        indices = range(len(examples[tag]))
        if len(indices) > n:
            indices = random.sample(indices, n)
        for idx in indices:
            prompt.append(" ".join(examples[tag][idx][0]))

            entities_str = []
            for label, entity in examples[tag][idx][1]:
                entities_str.append("{}: {}".format(label, entity))
            prompt.append("[ENTITIES] " + ", ".join(entities_str))

    return "\n".join(prompt)


if __name__ == "__main__":
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, type=int, help="Number of sentences to be sent together. Default: 1.")
    parser.add_argument("--context_left", default=0, type=int, help="Number of sentences added as left context. Default: no context.")
    parser.add_argument("--context_right", default=0, type=int, help="Number of sentences added as right context. Default: no context.")
    parser.add_argument("--examples_n", default=5, type=int, help="Number of examples for few-shot learning. Default: .5")
    parser.add_argument("--examples_maxlen", default=20, type=int, help="Maximal length of example sentence in tokens. Default: 20.")
    parser.add_argument("--max_tokens", default=5000, type=int, help="Maximum tokens in sentences (recommended: max LLM context size / 2).")
    parser.add_argument("--model", default="deepseek-r1:70b", type=str, help="Model name.")
    parser.add_argument("--output_filename", default=None, type=str, help="Output filename (optional). If omitted, prints to STDOUT.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--server", required=True, default=None, type=str, help="Server address with port.")
    parser.add_argument("--sleep", default=1, type=int, help="Sleep seconds between requests.")
    parser.add_argument("--tagset", choices=["conll", "uner", "onto"], default="conll", type=str, help="Tagset name.")
    parser.add_argument("--test_data", required=True, default=None, type=str, help="Test data CoNLL file.")
    parser.add_argument("--train_data", default=None, type=str, help="Train data CoNLL file (optional). It omitted (default), zero-shot prompt will be used, else a few-shot prompt with n given in --examples_n.")

    args = parser.parse_args()

    random.seed(args.seed)

    # Compile the prompt.
    if args.batch_size > 1:
        PROMPT = "Identify named entities in the following sentences, one sentence per line, and classify them using the following categories: {}.\n\
For each sentence, return the following format on a single line: \"[ENTITIES] ENTITY_TYPE: Entity1, ENTITY_TYPE: Entity2\".\n\
If no entities are found, return \"[ENTITIES] (none)\".\n".format(", ".join(TAGSETS[args.tagset]))
    else:
        PROMPT = "Identify named entities in the following sentence, and classify them using the following categories: {}.\n\
Return the following format on a single line: \"[ENTITIES] ENTITY_TYPE: Entity1, ENTITY_TYPE: Entity2\".\n\
If no entities are found, return \"[ENTITIES] (none)\".\n".format(", ".join(TAGSETS[args.tagset]))

    if args.context_left or args.context_right:
        PROMPT += "The sentences marked with [CONTEXT] provide context, but do not extract entities from them.\n"

    if args.train_data:
        train_lines, train_entities = read_data(args.train_data)
        examples = select_examples(train_lines, train_entities, args.tagset, args.examples_n, args.examples_maxlen)
        PROMPT += "Examples of correctly identified named entities follow:\n{}\n".format(examples)

    # Read test data from the CoNLL file.
    test_lines, _ = read_data(args.test_data)

    # Open output file.
    fw = sys.stdout
    if args.output_filename:
        fw = open(args.output_filename, "w", encoding="utf-8")

    # Recognize entities in batches of test sentences with context.
    start_time = time.time()
    nbatches = 0
    for i in range(0, len(test_lines), args.batch_size):
        batch_prompt = PROMPT
        nbatches += 1

        # Left context
        if args.context_left:
            if i >= args.context_left:
                for sentence in test_lines[i-args.context_left:i]:
                    batch_prompt += "\n[CONTEXT] " + " ".join(sentence)

        # Target sentence(s)
        for sentence in test_lines[i:i+args.batch_size]:
            batch_prompt += "\n" + " ".join(sentence)

        # Right context
        if args.context_right:
            if i+args.batch_size+args.context_right <= len(test_lines):
                for sentence in test_lines[i+args.batch_size:i+args.batch_size+args.context_right]:
                    batch_prompt += "\n[CONTEXT] " + " ".join(sentence)

        ntokens = len(batch_prompt.split(" "))
        assert ntokens < args.max_tokens, "Prompt tokens in batch exceeded safety limit --max_tokens={}.".format(args.max_tokens)

        # Skip trivial request.
        if args.batch_size == 1 and test_lines[i][0] == "-DOCSTART-":
            entities = [[]]
        # Make request.
        else:
            response = requests.post("{}/api/generate".format(args.server), json={
                "model": args.model,
                "prompt": batch_prompt,
                "stream": False
            })

            entities = [[] for _ in range(len(test_lines[i:i+args.batch_size]))]
            if response.status_code == 200:
                entities = extract_entities(response.json()["response"], len(test_lines[i:i+args.batch_size]))
            else:
                print("Warning: Response status code {} for sentences {}-{}".format(response.status_code, i, i+args.batch_size), file=sys.stderr)

        # Print IOB2 matched entities into CoNLL-2003 format.
        print_iob2(test_lines[i:i+args.batch_size], entities, fw)

        # Print progress info.
        if not i % 1:
            print("Processed {} batches of size {}, that is {} sentences of total {}".format(nbatches, args.batch_size, nbatches * args.batch_size, len(test_lines)), file=sys.stderr)

        # Pacing mechanism to be nice.
        time.sleep(args.sleep)

    end_time = time.time()
    time_elapsed = end_time - start_time
    print("Processed {} sentences in {:.2f} seconds, that is {:.2f} seconds per sentence on average.".format(len(test_lines), time_elapsed, time_elapsed / len(test_lines)), file=sys.stderr)
