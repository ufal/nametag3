# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Fixed broken symlink traverse by using absolute path.

### Added

- Added default `auto` value for `--best_weights_device`.

### Changed

- Moved postprocess() from NameTag3Model to NameTag3Dataset.

### Changed

- Upgrade dependencies: tensorboard 2.20.0.

### Changed

- Refactored best model checkpoint saving.

### Added

- LoRA.
- Transformer weights dtype.
- User-requested max tokenizer length (overrides HF max model length).
- GPU memory tracking for memory debugging.
- Added options for early stopping best weights storage device.

### Added

- Resolve HF tokenizer model max length when undeclared.

### Changed

- Polished client and improved server error processing on the client side.
- Added explicit server-side handling of oversized requests.
- Silenced HF tokenizer warning (splitting is handled manually).

### Fixed

- Added guard against words exploding into too many subwords.

### Added

- Learning rate decay command-line argument.
- Frozen pretraining warmup safeguard.

### Added

- Added gradient accumulation.

### Changed

- Fixed minor issues in nested NER evaluation scripts.

### Changed

- Upgrade dependencies: keras 3.13.1.

### Added

- Version number.

### Changed

- Upgrade dependencies: keras 3.12.0.

### Added

- SafeSparseCategoricalCrossentropy.

### Changed

- Upgrade dependencies: torch 2.8.0.
- Upgrade dependencies: transformers 4.53.0.

### Added

- Changelog.

## 3.1.0

Since NameTag 3.1, flat NER models can optionally be trained with multiple named entity tagsets. The trained model can then be required to recognize the named entities using a specific tagset during inference, or a predefined default tagset will be used if none was requested.

This allows joint multidataset training across tagsets, and in turn allows expansion of covered languages. NameTag 3 achieves state-of-the-art performance on 21 test datasets in 15 languages: Cebuano, Chinese, Croatian, Czech, Danish, English, Norwegian Bokmål, Norwegian Nynorsk, Portuguese, Russian, Serbian, Slovak, Swedish, Tagalog, and Ukrainian. It also delivers competitive results on Arabic, Dutch, German, Maghrebi, and Spanish, as of February 2025.

The currently supported tagsets are the following:

- `conll`: The CoNLL-2003 shared task tagset: `PER`, `ORG`, `LOC`, and `MISC`,
- `uner`: The [Universal NER v1](https://www.universalner.org/) tagset: `PER`, `ORG`, `LOC`,
- `onto`: The OntoNotes v5 tagset: `PERSON`, `NORP`, `FAC`, `ORG`, `GPE`, etc.

In the NameTag 3 webservice, the tagset variants of one model are served separately, e.g., `nametag3-multilingual-conll-250203`, `nametag3-multilingual-uner-250203`, and `nametag3-multilingual-onto-250203`. The model tagset variants share one multilingual model and apply tagset masks on its output to predict tags of the requested tagset.

## 3.0.0

NameTag 3.0 is an open-source tool for both flat and nested named entity recognition (NER). NameTag 3 identifies proper names in text and classifies them into a set of predefined categories, such as names of persons, locations, organizations, etc.

NameTag 3.0 offers state-of-the-art or near state-of-the-art performance in English, German, Spanish, Dutch, Czech and Ukrainian.

NameTag 3.0 is a free software under [Mozilla Public License 2.0](htts://www.mozilla.org/MPL/2.0/), and the linguistic models are free for non-commercial use and distributed under [CC BY-NC-SA license](https://creativecommons.org/licenses/by-nc-sa/3.0/), although for some models the original data used to create the model may impose additional licensing conditions. NameTag is versioned using [Semantic Versioning](https://semver.org./).

Copyright 2024 Institute of Formal and Applied Linguistics, Faculty of Mathematics and Physics, Charles University, Czech Republic.

NameTag 3.0 can be used either as a commandline tool or by requesting the NameTag webservice:

- [LINDAT/CLARIN](https://lindat.cz) hosts the [NameTag Web Application](https://lindat.mff.cuni.cz/services/nametag/),
- [LINDAT/CLARIN](https://lindat.cz) also hosts the [NameTag REST Web Service](https://lindat.mff.cuni.cz/services/nametag/).

NameTag 3.0 source code can be found at [GitHub](https://github.com/ufal/nametag3).

The [NameTag website](https://ufal.mff.cuni.cz/nametag/3/) contains download links of both the released packages and trained models, hosts documentation and refers to demo and online web service.

If you use this software, please give us credit by referencing [Straková et al. (2019)](https://aclanthology.org/P19-1527.pdf):

```
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
```
