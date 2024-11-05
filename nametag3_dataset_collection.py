#!/usr/bin/env python3
# coding=utf-8
#
# Copyright 2024 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""NameTag3Dataset class to handle collection of NameTag3Dataset datasets."""


import io
import math
import pickle
import os
import sys

import numpy as np
import seqeval.metrics
import torch
import transformers

from nametag3_dataset import NameTag3Dataset
from nametag3_dataset import pad_collate
from nametag3_dataset import SHUFFLING_SHARD


class WeightedRandomSamplerFromDatasets(torch.utils.data.Sampler):
    """Custom weighted random sampler from multiple datasets.

    Samples from the original datasets comprising one large ConcatDataset
    according to the corresponding weights (probs).
    """

    def __init__(self, dataset, lens, weights, generator=None):
        """Initializes the sampler before sampling.

        Arguments:
            dataset: torch.utils.data.ConcatDataset created by concatenating
                the datasets (torch.utils.data.Dataset) to be sampled from,
            lens: a Python list of ints representing the respective lengths of
                the concatenated datasets,
            weights: a Python list representing the respective sampling weights
                of the concatenated datasets. The weights are expected to be
                normalized, i.e., they are expected to be probabilities, really.
                TODO: Rename the weights to probs?
        """

        self._dataset = dataset
        self._lens = lens
        self._weights = torch.from_numpy(weights)
        self._generator = generator

        # Get the original datasets' indices in the concatenated dataset.
        self._ranges = []
        start = 0
        for i in range(len(lens)):
            self._ranges.append(torch.tensor(list(range(start, start+lens[i]))))
            start += lens[i]

    def __len__(self):
        return len(self._dataset)

    def __iter__(self):
        """Samples as many examples as the concatenated datasets length sum."""

        # At each position, decide the original dataset to be sampled from.
        dataset_choices = torch.multinomial(self._weights, len(self._dataset), replacement=True, generator=self._generator)

        # Get the number of samples to be sampled from the original datasets.
        dataset_counts = torch.bincount(dataset_choices, minlength=len(dataset_choices))

        # Sample the required number of examples from each dataset.
        indices = []
        for i in range(len(self._weights)): # for each dataset
            samples_required = dataset_counts[i]

            # Repeat the sampling for the upsampled datasets, i.e., datasets
            # that have less examples than required.
            while samples_required > 0:

                # In one go, we can only sample as many examples as the dataset has.
                samples_to_draw = min(samples_required, self._lens[i])

                # Sample from the corresponding original dataset indices
                # (self._ranges[i]) by taking as many samples as needed in this
                # go from the permutation of indices.
                indices.append(self._ranges[i][torch.randperm(self._lens[i], generator=self._generator)][:samples_to_draw])

                # Number of samples remaining to be drawn.
                samples_required -= samples_to_draw

        # Flatten the sampled original datasets' indices from 2D to 1D.
        indices = torch.cat(indices)

        # The second randperm mixes the original datasets' indices among each other.
        yield from indices[torch.randperm(len(indices), generator=self._generator)].tolist()


class NameTag3DatasetCollection:
    """Class for handling dataset collection.

    Vocabularies are built incrementally during construction, so that the last
    dataset in the collection has the complete vocabularies.
    """

    def __init__(self, args, tokenizer=None, filenames=None, text=None, train_collection=None):

        self._datasets = []
        self._corpora = args.corpus.split(",") if args.corpus else None
        self._tagsets = args.tagsets.split(",") if hasattr(args, "tagsets") and args.tagsets else None

        if self._tagsets:
            special_tokens_dict = {}
            special_tokens_dict["additional_special_tokens"] = ["[TAGSET_{}]".format(tagset) for tagset in set(self._tagsets)]
            tokenizer.add_special_tokens(special_tokens_dict)

        if filenames:
            for i, filename in enumerate(filenames.split(",")):
                self._datasets.append(NameTag3Dataset(args,
                                                      tokenizer=tokenizer,
                                                      filename=filename,
                                                      train_dataset=train_collection.datasets[-1] if train_collection else None,
                                                      previous_dataset=self._datasets[-1] if i and not train_collection else None,
                                                      corpus=self._corpora[i] if self._corpora else str("corpus_{}".format(i+1)),
                                                      tagset=self._tagsets[i] if self._tagsets else None))
        # Reading from text (used by the server) allows creation of exactly one
        # dataset in the collection.
        else:
            self._datasets.append(NameTag3Dataset(args,
                                                  tokenizer=tokenizer,
                                                  text=text,
                                                  train_dataset=train_collection.datasets[-1] if train_collection else None,
                                                  previous_dataset=None,
                                                  corpus=args.corpus if args.corpus else "corpus 1",
                                                  tagset=self._tagsets[0] if self._tagsets else None))

        # Create dataloaders if any data given.
        self._dataloader, self._dataloaders = None, None
        if filenames or text:
            self._dataloader = self.create_torch_dataloader(args,
                                                            shuffle=True if not train_collection else False,
                                                            sampling=args.sampling if not train_collection else "concatenate")
            self._dataloaders = self.create_torch_dataloaders(args, shuffle=True if not train_collection else False)

        # Create tagset masks in all datasets.
        for dataset in self._datasets:
            dataset.create_tagset_mask(self.id2label())

    def label2id(self):
        return self._datasets[-1].label2id()

    def id2label(self):
        return self._datasets[-1].id2label()

    @property
    def datasets(self):
        return self._datasets

    @property
    def dataloader(self):
        return self._dataloader

    @property
    def dataloaders(self):
        return self._dataloaders

    def training_batches(self):
        return len(self._dataloader) if self._dataloader else 0

    def _weights(self, datasets, sampling, temperature):
        if sampling == "proportional":
            weights = [len(x) for x in datasets]
            weights /= np.sum(weights)

        elif sampling == "uniform":
            weights = [1/len(datasets)] * len(datasets)

        elif sampling in ["temperature_logits", "temperature"]:
            weights = [len(torch_dataset) for torch_dataset in datasets]
            weights /= np.sum(weights)
            print("Original weights: ", weights, file=sys.stderr, flush=True)

            weights = [math.exp(weight/temperature) for weight in weights]
            weights /= np.sum(weights)

        elif sampling == "temperature_probs":
            weights = [len(torch_dataset) for torch_dataset in datasets]
            weights /= np.sum(weights)
            print("Original weights: ", weights, file=sys.stderr, flush=True)

            weights **= 1/temperature
            weights /= np.sum(weights)

        print("Sampling weights: {}".format(weights), file=sys.stderr, flush=True)
        return weights

    def create_torch_dataloader(self, args, shuffle=False, sampling="concatenate"):
        """Return a single DataLoader of merged datasets in collection."""

        # Get list of plain Datasets (no batching, no shuffling, no sampling)
        torch_datasets = [x.create_torch_dataset(args) for x in self._datasets]

        torch_dataset = torch.utils.data.ConcatDataset(torch_datasets)

        if sampling == "concatenate":
            return torch.utils.data.DataLoader(torch_dataset, collate_fn=pad_collate, batch_size=args.batch_size, shuffle=shuffle)
        else:
            weights = self._weights(torch_datasets, sampling, args.temperature)
            lens = [len(x) for x in torch_datasets]
            return torch.utils.data.DataLoader(
                torch_dataset,
                collate_fn=pad_collate,
                batch_size=args.batch_size,
                sampler=WeightedRandomSamplerFromDatasets(
                    torch_dataset, lens, weights, generator=torch.Generator().manual_seed(args.seed)))

    def create_torch_dataloaders(self, args, shuffle=False):
        """Return list of individual DataLoaders for datasets in the collection."""

        return [x.create_torch_dataloader(args, shuffle=shuffle) for x in self._datasets]

    def load_collection_mappings(self, filename):

        with open("{}/mappings.pickle".format(filename), mode="rb") as mappings_file:
            dataset = pickle.load(mappings_file)

        dataset.__class__ = NameTag3Dataset

        self._datasets = [dataset]

    def save_mappings(self, path):
        os.makedirs("{}/model".format(path), exist_ok=True)
        self._datasets[-1].save_mappings("{}/model/mappings.pickle".format(path))
