#!/usr/bin/env python3
# coding=utf-8
#
# Copyright 2024 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""NameTag3Dataset class to handle NE datasets."""


import io
import os
import pickle
import sys
import time

import keras
import numpy as np
import seqeval.metrics
import torch
import transformers


BATCH_PAD = -1
MASK = 0
COLUMN_PAD = 1
UNK = 2
EOW = 3
BOS = 4

CONTROL_LABELS_DICT = {'<mask>': MASK, '<pad>': COLUMN_PAD, '<unk>': UNK, '<eow>': EOW, '<bos>': BOS}
CONTROL_LABELS = ['<mask>', '<pad>', '<unk>', '<eow>', '<bos>']

SHUFFLING_SHARD = 10000

# Valid tagsets for multitagset learning.
TAGSETS = {
    "conll": ["B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC", "O"],
    "uner": ["B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "O"],
    "onto": ["O", "B-PERSON", "I-PERSON", "B-NORP", "I-NORP", "B-FAC", "I-FAC",
             "B-ORG", "I-ORG", "B-GPE", "I-GPE", "B-LOC", "I-LOC", "B-PRODUCT",
             "I-PRODUCT", "B-DATE", "I-DATE", "B-TIME", "I-TIME", "B-PERCENT",
             "I-PERCENT", "B-MONEY", "I-MONEY", "B-QUANTITY", "I-QUANTITY",
             "B-ORDINAL", "I-ORDINAL", "B-CARDINAL", "I-CARDINAL", "B-EVENT",
             "I-EVENT", "B-WORK_OF_ART", "I-WORK_OF_ART", "B-LAW", "I-LAW",
             "B-LANGUAGE", "I-LANGUAGE"]
}

# Official eval stripts for nested corpora.

# CNEC 2.0 eval script is corrected in comparison to the original to not
# fail on zero division in case of very bad system predictions after the
# first few epochs of training.
EVAL_SCRIPTS = {"czech-cnec2.0": "run_cnec2.0_eval_nested_corrected.sh"}


def pad_collate(batch):
    """Pads batches of sequences with varying dimensions."""

    inputs, outputs = zip(*batch)
    input_ids, word_ids = zip (*inputs)

    input_ids_pad = keras.ops.convert_to_tensor(torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=BATCH_PAD))
    word_ids_pad = keras.ops.convert_to_tensor(torch.nn.utils.rnn.pad_sequence(word_ids, batch_first=True, padding_value=BATCH_PAD))
    outputs_pad = keras.ops.convert_to_tensor(torch.nn.utils.rnn.pad_sequence(outputs, batch_first=True, padding_value=BATCH_PAD))

    return (input_ids_pad, word_ids_pad), outputs_pad


class NameTag3TorchDataset(torch.utils.data.Dataset):

    def __init__(self, input_ids, word_ids, outputs):
        self.input_ids = input_ids
        self.word_ids = word_ids
        self.outputs = outputs

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return (torch.tensor(self.input_ids[idx]), torch.tensor(self.word_ids[idx])), torch.tensor(self.outputs[idx])


class NameTag3Dataset:
    """Class for handling NE datasets."""

    FORMS = 0
    TAGS = 1

    def __init__(self, args, tokenizer=None, filename=None, text=None, train_dataset=None, previous_dataset=None, corpus=None, tagset=None):
        """Load the dataset from a two column CoNLL-like format.

        Arguments:
            args: main script args from argparse.
            tokenizer: HF AutoTokenizer object.
            filename: Path to filename.
            text: Alternatively, read from string (for server).
            train_dataset: If given, the frozen id2label, and label2id are
                reused from the train_dataset, but new items are not added.
            previous_dataset: If given, the id2label and label2id from the
                previous dataset in the collection are reused, and new items
                may be added.
            corpus: Corpus name.
            tagset: Tagset name.
        """

        if args.decoding == "seq2seq" and not args.context_type == "sentence":
            raise NotImplementedError("Only --context_type=sentence is implemented for --decoding=seq2seq")

        self._filename = filename
        self._corpus = corpus
        self._seq2seq = args.decoding == "seq2seq"
        self._args = args
        self._tokenizer = tokenizer
        self._tagset = tagset
        self._training = train_dataset == None

        if self._tagset:
            self._tagset_token = "[TAGSET_{}]".format(tagset)
            additional_special_token_index = self._tokenizer.additional_special_tokens.index(self._tagset_token)
            self._tagset_token_id = self._tokenizer.additional_special_tokens_ids[additional_special_token_index]

        # Data structures
        self._forms = []
        self._labels = []
        self._label_ids = []

        if train_dataset:
            self._label2id = train_dataset._label2id
            self._id2label = train_dataset._id2label
            if self._seq2seq:
                self._label2id_sublabel = train_dataset._label2id_sublabel
                self._id2label_sublabel = train_dataset._id2label_sublabel
        elif previous_dataset:
            self._label2id = previous_dataset._label2id
            self._id2label = previous_dataset._id2label
            if self._seq2seq:
                self._label2id_sublabel = previous_dataset._label2id_sublabel
                self._id2label_sublabel = previous_dataset._id2label_sublabel
        else:
            self._label2id = {key:value for key, value in CONTROL_LABELS_DICT.items()}
            self._id2label = [tag for tag in CONTROL_LABELS]
            if self._seq2seq:
                self._label2id_sublabel = {key:value for key, value in CONTROL_LABELS_DICT.items()}
                self._id2label_sublabel = [tag for tag in CONTROL_LABELS]


        # Load the sentences
        if filename:
            print("Reading data {}{}from \"{}\"".format("of corpus \"{}\" ".format(self._corpus) if self._corpus else "",
                                                        "with tagset \"{}\" ".format(self._tagset) if self._tagset else "",
                                                        filename),
                  file=sys.stderr, flush=True)

        start_time = time.time()
        with open(filename, "r", encoding="utf-8") if filename is not None else io.StringIO(text) as file:
            in_sentence = False
            for line in file:
                line = line.rstrip("\r\n")

                if line:
                    columns = line.split("\t")

                    if not in_sentence:
                        self._forms.append([])
                        self._labels.append([])
                        self._label_ids.append([])

                    # FORMS information
                    self._forms[-1].append(columns[self.FORMS])

                    # TAGS information
                    if self.TAGS >= len(columns):    # dataset without gold TAGS
                        self._labels[-1].append('<pad>')
                        self._label_ids[-1].append(COLUMN_PAD)
                    else:   # dataset with both FORMS and TAGS column
                        label = columns[self.TAGS]
                        if label not in self._label2id:
                            if train_dataset:
                                label = '<unk>'
                            else:
                                self._label2id[label] = len(self._id2label)
                                self._id2label.append(label)

                        self._labels[-1].append(label)
                        self._label_ids[-1].append(self._label2id[label])

                        # TAG sub-labels for seq2seq decoding (nested NER).
                        if self._seq2seq:
                            for sublabel in label.split("|"):
                                if sublabel not in self._label2id_sublabel:
                                    if train_dataset == None:
                                        self._label2id_sublabel[sublabel] = len(self._id2label_sublabel)
                                        self._id2label_sublabel.append(sublabel)

                    in_sentence = True
                else:
                    in_sentence = False

                    # Stop reading training data if limit reached.
                    if args.max_sentences_train and self._training and len(self._forms) >= args.max_sentences_train:
                        print("Reached required --max_sentences={}, stopped reading training data.".format(args.max_sentences_train), file=sys.stderr, flush=True)
                        break

        end_time = time.time()

        if filename:
            print("Read {} sentences from \"{}\" in {:.2f} seconds".format(len(self._forms), filename, end_time-start_time), file=sys.stderr, flush=True)

        # Create dataloader if any data given.
        if filename or text:
            self._dataloader = self.create_torch_dataloader(args,
                                                            shuffle=True if self._training else False)

    def _split_document(self, input_ids, word_ids, strings, outputs):
        """Reorganize to max_context window splits instead sentences."""

        input_ids_splits, word_ids_splits, strings_splits, outputs_splits = [], [], [], []

        for s in range(len(input_ids)):   # sentences

            # Empty splits OR cannot fit entire sentence in current split OR
            # new document found => make new split.
            room_for_tagset_token = self._tagset != None  # 1 if --tagsets is enabled
            if len(input_ids_splits) == 0 \
                    or len(input_ids_splits[-1]) + len(input_ids[s]) - 1 >= self._tokenizer.model_max_length - room_for_tagset_token \
                    or strings[s][0] == "-DOCSTART-":
                if len(input_ids_splits):   # close previous split
                    input_ids_splits[-1].append(self._tokenizer.sep_token_id)

                # Start new split
                input_ids_splits.append([self._tokenizer.cls_token_id])
                word_ids_splits.append([])
                strings_splits.append([])
                outputs_splits.append([])

            # Update word ids
            for i in range(len(word_ids[s])):
                word_ids[s][i] += len(input_ids_splits[-1]) - 1

            # Extend current split
            input_ids_splits[-1].extend(input_ids[s][1:-1])
            word_ids_splits[-1].extend(word_ids[s])
            strings_splits[-1].extend(strings[s])
            outputs_splits[-1].extend(outputs[s])

        # Complete the last split with [SEP]
        if input_ids_splits and input_ids_splits[-1] and input_ids_splits[-1][-1] != self._tokenizer.sep_token_id:
            input_ids_splits[-1].append(self._tokenizer.sep_token_id)

        return input_ids_splits, word_ids_splits, strings_splits, outputs_splits

    @property
    def dataloader(self):
        return self._dataloader

    @property
    def sentence_lens(self):
        return self._sentence_lens

    @property
    def tag_lens(self):
        return self._tag_lens

    @property
    def factors(self):
        return self._factors

    @property
    def corpus(self):
        return self._corpus

    @property
    def filename(self):
        return self._filename

    def save_mappings(self, path):
        """Pickle word mappings."""

        mappings = NameTag3Dataset.__new__(NameTag3Dataset)

        members = ["_id2label", "_label2id", "_seq2seq"]
        if self._seq2seq:
            members.extend(["_id2label_sublabel", "_label2id_sublabel"])

        for member in members:
            setattr(mappings, member, getattr(self, member))

        with open(path, "wb") as mappings_file:
            pickle.dump(mappings, mappings_file, protocol=3)

    def _truecase(self, inputs):
        truecased = []
        for sentence in inputs:
            truecased.append([])
            for word in sentence:
                truecased[-1].append(word.lower().title() if word.isupper() else word)
        return truecased

    def _tokenize(self, keep_original_casing=False):
        input_ids, word_ids, strings, outputs = [], [], [], []

        start, end = 0, self._args.batch_size
        while start < len(self._forms):

            batch_inputs = self._forms[start:end]
            batch_outputs = self._label_ids[start:end]
            inputs = self._tokenizer(batch_inputs if keep_original_casing else self._truecase(batch_inputs), add_special_tokens=False, is_split_into_words=True)

            # Split too long sentences, collect first subword indices for
            # gathering in NN and split strings and outputs accordingly.
            for s in range(len(inputs["input_ids"])):   # original sentences
                if s and s % 100000 == 0:
                    print("Sentences tokenized: {} / {}".format(s, len(inputs["input_ids"])), file=sys.stderr, flush=True)

                input_ids.append([self._tokenizer.cls_token_id])
                word_ids.append([])
                strings.append([])
                outputs.append([])

                for word_index in range(len(batch_inputs[s])):
                    token_span = inputs.word_to_tokens(s, word_index)

                    # HF tokenizer sometimes returns None as a span for some
                    # unicode characters. As our strings are pretokenized, we
                    # cannot just remove the unrepresented string, so we replace it
                    # with unk_token_id and create an artificial TokenSpan for it.
                    is_artificial_token_span = token_span == None
                    if is_artificial_token_span:
                        token_span = transformers.TokenSpan(word_ids[-1][-1] + 1 if word_ids[-1] else 0,
                                                            word_ids[-1][-1] + 2 if word_ids[-1] else 1)

                        print("Word generated without corresponding token by the HF tokenizer, creating artificial token \"{}\". Word: \"{}\". Sentence: {}".format(self._tokenizer.unk_token, batch_inputs[s][word_index], batch_inputs[s]), file=sys.stderr, flush=True)

                    # Sentence length exceeded maximum length, start new context.
                    # 1 for ending [SEP] and optionally another 1 for the tagset token
                    room_for_special_tokens = 2 if self._tagset else 1
                    if len(input_ids[-1]) + token_span.end - token_span.start + room_for_special_tokens >= self._tokenizer.model_max_length:
                        input_ids[-1].append(self._tokenizer.sep_token_id)

                        input_ids.append([self._tokenizer.cls_token_id])
                        word_ids.append([])
                        strings.append([])
                        outputs.append([])

                    # Extend the context.
                    word_ids[-1].append(len(input_ids[-1]))
                    input_ids[-1].extend(inputs["input_ids"][s][token_span.start:token_span.end] if not is_artificial_token_span else [self._tokenizer.unk_token_id])
                    strings[-1].append(batch_inputs[s][word_index] if not is_artificial_token_span else self._tokenizer.unk_token)
                    outputs[-1].append(batch_outputs[s][word_index])

                input_ids[-1].append(self._tokenizer.sep_token_id)

            start = end
            end += self._args.batch_size

        return input_ids, word_ids, strings, outputs

    def forms(self):
        return self._forms

    def labels(self):
        return self._labels

    def id2label(self):
        return self._id2label_sublabel if self._seq2seq else self._id2label

    def label2id(self):
        return self._label2id_sublabel if self._seq2seq else self._label2id

    def _get_data_for_nn_dataset(self, context_type, keep_original_casing):

        # Tokenize and reorganize factors accordingly
        input_ids, word_ids, strings, outputs = self._tokenize(keep_original_casing=keep_original_casing)

        # Add context
        if context_type == "split_document":
            # Reorganization from sentence-based to document-split based.
            input_ids, word_ids, strings, outputs = self._split_document(input_ids, word_ids, strings, outputs)

        elif context_type in ["sentence", "max_context", "document"]:

            if context_type in ["max_context", "document"]:
                if self._tagset:
                    raise NotImplementedError("--tagsets not implemented for --context_type=max_context and --context_type=document")

                inputs_with_context = []
                context = []
                for s in range(len(input_ids)):   # sentences
                    if context_type == "document" and strings[s][0] == "-DOCSTART-":
                        context = []    # new document, drop context

                    context.extend(input_ids[s][1:-1])  # append sentence without [CLS] and [SEP] to context
                    context = context[-self._tokenizer.model_max_length+3:]  # take last context, leave space for [CLS], [SEP] and [SEP]

                    # CLS + previous_context + SEP + sentence + SEP
                    inputs_with_context.append([self._tokenizer.cls_token_id])
                    inputs_with_context[-1].extend(context.copy())

                    start_sentence = len(context) - len(input_ids[s][1:-1]) + 1   # +1 for CLS
                    if start_sentence > 1:
                        inputs_with_context[-1].insert(start_sentence, self._tokenizer.sep_token_id)
                    else:
                        start_sentence -= 1 # SEP not inserted, decrease start_sentence

                    inputs_with_context[-1].append(self._tokenizer.sep_token_id)

                    # Update word_ids
                    for i in range(len(word_ids[s])):
                        word_ids[s][i] += start_sentence

                input_ids = inputs_with_context

        # Add the tagset token right after [CLS] and increase word_ids accordingly.
        if self._tagset:
            for s in range(len(input_ids)):
                input_ids[s].insert(1, self._tagset_token_id)
                word_ids[s] = [x+1 for x in word_ids[s]]

        # For seq2seq, unpack the complex labels into the sublabels.
        if self._seq2seq:
            unpacked_outputs = [[] for i in range(len(outputs))]
            for s, sentence in enumerate(outputs):
                for label in sentence:
                    label_str = self._id2label[label]
                    for sublabel_str in label_str.split("|"):
                        unpacked_outputs[s].append(self._label2id_sublabel[sublabel_str])
                    unpacked_outputs[s].append(EOW)
            return input_ids, word_ids, unpacked_outputs
        else:
            return input_ids, word_ids, outputs

    def create_torch_dataset(self, args):
        input_ids, word_ids, outputs = self._get_data_for_nn_dataset(args.context_type, args.keep_original_casing)
        return NameTag3TorchDataset(input_ids, word_ids, outputs)

    def create_torch_dataloader(self, args, shuffle=False):
        torch_dataset = self.create_torch_dataset(args)
        return torch.utils.data.DataLoader(torch_dataset, collate_fn=pad_collate, batch_size=args.batch_size, shuffle=shuffle)

    def _eval_script(self):
        if not self._seq2seq:
            return "run_conlleval.sh"

        if self._corpus in EVAL_SCRIPTS:
            return EVAL_SCRIPTS[self._corpus]
        else:
            raise NotImplementedError("NameTag 3 does not have the official evaluation script for the given nested corpus. If you are training on CNEC 2.0, you can specify --corpus=czech-cnec2.0. If you are training on a custom nested NE corpus and you have the official evaluation script for it, you can register the script in NameTag3Dataset._EVAL_SCRIPTS.")

    def evaluate(self, dataset_type, predictions_filename, logdir):
        """Evaluate NEs in predictions_filename against the dataset's gold NEs.

        Evaluate NEs in predictions_filename against the dataset's gold NEs
        using the dataset's official evaluation script.
        """

        # Run the eval script
        eval_script = self._eval_script()
        print("\"{}\" data of corpus \"{}\" will be evaluated with an external script \"{}\"".format(dataset_type, self._corpus, eval_script), file=sys.stderr, flush=True)
        command = "cd {} && ../../{} {} {} {}".format(logdir, eval_script, dataset_type, self._filename, predictions_filename)
        os.system(command)

        # Parse the eval script output
        f1 = None
        if eval_script == "run_cnec2.0_eval_nested_corrected.sh":
            with open(os.path.join(logdir, "{}.eval".format(dataset_type)), "r", encoding="utf-8") as result_file:
                for line in result_file:
                    line = line.strip("\n")
                    if line.startswith("Type:"):
                        cols = line.split()
                        f1 = float(cols[5])
        elif eval_script == "run_conlleval.sh":
            with open(os.path.join(logdir, "{}.eval".format(dataset_type)), "r", encoding="utf-8") as result_file:
                for line in result_file:
                    line = line.strip("\n")
                    if line.startswith("accuracy:"):
                        f1 = float(line.split()[-1])
        else:
            raise NotImplementedError("Parsing of the eval script \"{}\" output not implemented".format(eval_script))

        return f1

    def create_tagset_mask(self, all_tags):
        """Create tagset_mask for multitagset training."""

        if self._tagset == None:
            self.tagset_mask = None
            return

        if self._tagset not in TAGSETS:
            raise ValueError("Unknown tagset value \"{}\" of NameTag3Dataset. Known tagset values are \"{}\"".format(self._tagset, ",".join(TAGSETS.keys())))

        self.tagset_mask = [-1e9] * len(self.id2label())

        # Mark positions with valid tags in this dataset.
        for tag in TAGSETS[self._tagset] + CONTROL_LABELS:
            if tag in all_tags:
                index = all_tags.index(tag)
                self.tagset_mask[index] = 0.
