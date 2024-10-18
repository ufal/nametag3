#!/usr/bin/env python3
# coding=utf-8
#
# Copyright 2024 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""NameTag 3, flat and nested NER training and prediction script.

Implements both standard flat NER by fine-tuning with a softmax classification
layer and nested NER by seq2seq decoder with hard attention proposed in
https://aclanthology.org/P19-1527.pdf.

If you use this software, please give us credit by referencing https://aclanthology.org/P19-1527.pdf:

@inproceedings{strakova-etal-2019-neural,
    title = "Neural Architectures for Nested {NER} through Linearization",
    author = "Strakov{\'a}, Jana  and
      Straka, Milan  and
      Hajic, Jan",
    editor = "Korhonen, Anna  and
      Traum, David  and
      M{\`a}rquez, Llu{\'\i}s",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P19-1527",
    doi = "10.18653/v1/P19-1527",
    pages = "5326--5331",
}

Installation:

See README.md for installation instructions.

Usage:

venv/bin/python3 nametag3.py [--argument=value]

Example Usage:

venv/bin/python3 nametag3.py \
  --load_checkpoint=models/nametag3-multilingual-conll-240830/ \
  --test_data=examples/en_input.conll

Input:

The input data file format is a vertical file, one token and optionally, its label per line,
separated by a tabulator; sentences delimited by newlines (such as a first and
fourth column in a well-known CoNLL-2003 IOB shared task corpus):

John	B-PER
loves	O
Mary	B-PER
.	O

Mary	B-PER
loves	O
John	B-PER
.	O

More examples can be found in the examples directory.
"""


import io
import json
import os
import pickle
import sys

# Force CPU fallback for PyTorch with the MPS device due to some operators not
# implemented in MPS.
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

os.environ.setdefault("KERAS_BACKEND", "torch")

# This is only set for debugging to force all CUDA calls to be synchronous.
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import keras
import numpy as np
import torch
import transformers

from nametag3_dataset_collection import NameTag3DatasetCollection
from nametag3_model import NameTag3ModelClassification, NameTag3ModelSeq2seq


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--checkpoint_filename", default="checkpoint.weights.h5", type=str, help="Checkpoint filename.")
    parser.add_argument("--context_type", default="split_document", choices=["max_context", "sentence", "document", "split_document"], help="Context type to add to sentence.")
    parser.add_argument("--corpus", default=None, type=str, help="[english-conll|german-conll|dutch-conll|spanish-conll|czech-cnec2.0|czech-cnec2.0_conll|ukrainian-languk_conll]")
    parser.add_argument("--decoding", default="classification", choices=["classification", "seq2seq"], help="Decoding head.")
    parser.add_argument("--dev_data", default=None, type=str, help="Dev data.")
    parser.add_argument("--dropout", default=0.5, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default="10", type=int, help="Number of epochs.")
    parser.add_argument("--epochs_frozen", default=0, type=int, help="Number of pretraining epochs with frozen transformer.")
    parser.add_argument("--evaluate_test_data", default=False, action="store_true", help="If set, expect a second column with gold data, and invoke the official corpus evaluation script (i. e., corpus must be given). By default, expect only one column with input data.")
    parser.add_argument("--keep_original_casing", default=False, action="store_true", help="If True, turns truecasing off, i.e., keeps original casing in data.")
    parser.add_argument("--latent_dim", default=256, type=int, help="RNN decoder hidden dim.")
    parser.add_argument("--learning_rate", default="1e-4", type=float, help="Learning rate.")
    parser.add_argument("--learning_rate_frozen", default="1e-3", type=float, help="Learning rate for frozen pretraining.")
    parser.add_argument("--logdir", default="logs", type=str, help="Logdir name.")
    parser.add_argument("--max_sentences_train", default=None, type=int, help="Limit number of training sentences (for debugging).")
    parser.add_argument("--max_labels_per_token", default=5, type=int, help="Maximum labels per token.")
    parser.add_argument("--name", default=None, type=str, help="Experiment name.")
    parser.add_argument("--hf_plm", default=None, type=str, help="HF pre-trained model name.")
    parser.add_argument("--load_checkpoint", default=None, type=str, help="Load previously saved checkpoint.")
    parser.add_argument("--prevent_all_dropouts", default=False, action="store_true", help="If True, sets --dropout=0., --transformer_hidden_dropout_probs=0. and --transformer_attention_probs_dropout_prob=0.")
    parser.add_argument("--remove_optimizer_from_checkpoint", default=False, action="store_true", help="If True, removes the optimizer from the loaded checkpoint and saves the checkpoint.")
    parser.add_argument("--sampling", default="concatenate", choices=["proportional", "uniform", "concatenate", "temperature"], help="Sampling strategy for multilingual datasets.")
    parser.add_argument("--save_best_checkpoint", default=False, action="store_true", help="Save best checkpoint on dev if set.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--subword_masking", default=0.0, type=float, help="Mask subwords with the given probability.")
    parser.add_argument("--steps_per_epoch", default=None, type=int, help="Steps per epoch. Default None (epoch iterates over all data).")
    parser.add_argument("--tagsets", default=None, type=str, help="For multitagset training: tagsets corresponding to the given corpora, separated by comma.")
    parser.add_argument("--temperature", default=2.0, type=float, help="Value of temperature for temperature sampling.")
    parser.add_argument("--test_data", default=None, type=str, help="Test data.")
    parser.add_argument("--time", default=False, action="store_true", help="Measure prediction time.")
    parser.add_argument("--train_data", default=None, type=str, help="Training data.")
    parser.add_argument("--transformer_hidden_dropout_prob", default=None, type=float, help="HF Tranformer hidden dropout prob. If None, default value from config is used.")
    parser.add_argument("--transformer_attention_probs_dropout_prob", default=None, type=float, help="HF Transformer dropout ratio for the attention probabilities. If None, default value from config is used.")
    parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--warmup_epochs", default=1, type=int, help="Number of warmup epochs.")
    parser.add_argument("--warmup_epochs_frozen", default=1, type=int, help="Number of warmup epochs for frozen pretraining.")
    args = parser.parse_args()

    # During inference, transfer crucial train args
    if args.load_checkpoint:
        with open("{}/options.json".format(args.load_checkpoint), mode="r") as options_file:
            train_args = argparse.Namespace(**json.load(options_file))
        parser.parse_args(namespace=train_args)

        for key in ["checkpoint_filename", "context_type", "decoding", "hf_plm", "keep_original_casing"]:
            args.__dict__[key] = train_args.__dict__[key]

    # Set threads and random seed
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)

    keras.utils.set_random_seed(args.seed)
    keras.config.disable_traceback_filtering()

    # Create logdir
    logargs = dict(vars(args).items())
    del logargs["checkpoint_filename"]
    if args.corpus and len(args.corpus.split(",")) > 1:
        logargs["corpus"]="multilingual"
    del logargs["dev_data"]
    del logargs["keep_original_casing"]
    del logargs["load_checkpoint"]
    del logargs["logdir"]
    del logargs["max_labels_per_token"]
    del logargs["max_sentences_train"]
    del logargs["sampling"]
    del logargs["save_best_checkpoint"]
    del logargs["seed"]
    del logargs["subword_masking"]
    if hasattr(logargs, "tagsets"):
        del logargs["tagsets"]
    del logargs["temperature"]
    del logargs["test_data"]
    del logargs["threads"]
    del logargs["time"]
    del logargs["train_data"]
    del logargs["warmup_epochs"]
    del logargs["warmup_epochs_frozen"]

    args.logdir = "{}/{}-{}-{}".format(
        args.logdir,
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), re.sub("^.*/", "", value) if type(value) == str else value)
                  for key, value in sorted(logargs.items())))
    )
    os.makedirs(args.logdir, exist_ok=True)

    # Load the tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.hf_plm,
                                                           add_prefix_space = args.hf_plm in ["roberta-base", "roberta-large", "ufal/robeczech-base"])

    # We load the training data only to get the mappings, nothing else is used.
    train_loaded=None
    if args.load_checkpoint:
        train_loaded = NameTag3DatasetCollection(args)
        train_loaded.load_collection_mappings(args.load_checkpoint)

    train_collection=None
    if args.train_data:
        train_collection = NameTag3DatasetCollection(args, tokenizer=tokenizer, filenames=args.train_data, train_collection=train_loaded)
        if args.save_best_checkpoint:
            train_collection.save_mappings(args.logdir)

    dev_collection=None
    if args.dev_data:
        dev_collection = NameTag3DatasetCollection(args, tokenizer=tokenizer, filenames=args.dev_data, train_collection=train_collection if args.train_data else train_loaded)

    if args.test_data:
        test_collection = NameTag3DatasetCollection(args, tokenizer=tokenizer, filenames=args.test_data, train_collection=train_collection if args.train_data else train_loaded)

    # Construct the model
    if args.decoding == "classification":
        model = NameTag3ModelClassification(len(train_collection.label2id().keys()) if train_collection else len(train_loaded.label2id().keys()),
                                            args,
                                            train_collection.id2label() if train_collection else train_loaded.id2label(),
                                            tokenizer)
    elif args.decoding == "seq2seq":
        model = NameTag3ModelSeq2seq(len(train_collection.label2id().keys()) if train_collection else len(train_loaded.label2id().keys()),
                                     args,
                                     train_collection.id2label() if train_collection else train_loaded.id2label(),
                                     tokenizer)

    # Pretrain with frozen transformer
    if args.train_data and args.epochs_frozen:
        print("Pretraining with frozen transformer for {} epochs.".format(args.epochs_frozen), file=sys.stderr, flush=True)
        model.compile(training_batches=train_collection.training_batches(), frozen=True)
        model.fit(args.epochs_frozen,
                  train_collection,
                  dev_collection=dev_collection,
                  save_best_checkpoint=args.save_best_checkpoint)

    # Compile the model
    model.compile(training_batches=train_collection.training_batches() if train_collection else 0, frozen=False)

    # Load checkpoint
    if args.load_checkpoint:
        model.load_checkpoint(os.path.join(args.load_checkpoint, args.checkpoint_filename))
        if args.remove_optimizer_from_checkpoint:
            new_checkpoint_filename = "{}/{}_wo_optimizer.weights.h5".format(args.load_checkpoint, args.checkpoint_filename[:-len(".weights.h5")])
            print("Saving checkpoint without optimizer to {}".format(new_checkpoint_filename), file=sys.stderr, flush=True)
            model.optimizer = None
            model.save_weights(new_checkpoint_filename)
            sys.exit()

    # Finetune the transformer
    if args.train_data and args.epochs:
        print("Finetuning for {} epochs.".format(args.epochs), file=sys.stderr, flush=True)
        model.fit(args.epochs_frozen + args.epochs,
                  train_collection,
                  dev_collection=dev_collection,
                  save_best_checkpoint=args.save_best_checkpoint,
                  initial_epoch=args.epochs_frozen)

    # Predict test data, evaluate if --evaluate_test_data is set:
    if args.test_data:
        test_scores = []
        for i, test_dataset in enumerate(test_collection.datasets):
            predictions_filename = "{}_{}_predictions.conll".format("test", test_dataset.corpus)
            print("Predicting dataset {} ({}), predictions will be saved to \"{}\"".format(i+1, test_dataset.corpus, os.path.join(args.logdir, predictions_filename)), file=sys.stderr, flush=True)

            with open(os.path.join(args.logdir, predictions_filename), "w", encoding="utf-8") as predictions_file:
                model.predict("test", test_dataset, args, fw=predictions_file)

            if args.evaluate_test_data:
                test_score = test_dataset.evaluate("test", predictions_filename, args.logdir)
                print("Test F1 ({}): {:.4f}".format(test_dataset.corpus, test_score), file=sys.stderr, flush=True)
                test_scores.append(test_score)

        if test_scores:
            print("Macro avg F1: {:.4f}".format(np.average(test_scores)), file=sys.stderr, flush=True)

        # If only one dataset, print to stdout too. We had this in previous
        # versions, so keeping for compatibility with older pipelines users
        # might still have.
        if len(test_collection.datasets) == 1:
            with open(os.path.join(args.logdir, predictions_filename), "r") as fr:
                print(fr.read(), end="")
