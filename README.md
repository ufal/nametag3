# NameTag 3

NameTag 3 is an open-source tool for named entity recognition (NER), **curently
under development**. NameTag identifies proper names in text and classifies them
into predefined categories, such as names of persons, locations, organizations,
etc. NameTag3 can be trained to recognize both flat and nested named entities.

NameTag 3 offers state-of-the-art or near state-of-the-art performance in
English, German, Spanish, Dutch, Czech and Ukrainian.

NameTag 3 can be used either as a commandline tool or by requesting the [NameTag webservice](https://lindat.mff.cuni.cz/services/nametag/).

The [NameTag website](https://ufal.mff.cuni.cz/nametag) contains download links
of both the released packages and trained models, hosts documentation and
refers to demo and online web service.

NameTag development repository is hosted at [GitHub](https://github.com/ufal/nametag3).

Also try our [demo and online web service](https://lindat.mff.cuni.cz/services/nametag/).

Authors and Contact:

- [Jana Straková](https://ufal.mff.cuni.cz/jana-strakova), `strakova@ufal.mff.cuni.cz`
- [Milan Straka](https://ufal.mff.cuni.cz/milan-straka), `straka@ufal.mff.cuni.cz`


## License

Copyright 2024 Institute of Formal and Applied Linguistics, Faculty of
Mathematics and Physics, Charles University, Czech Republic.

NameTag 3 is a free software under [Mozilla Public License 2.0](https://www.mozilla.org/MPL/2.0/)
license and the linguistic models are free for non-commercial use and
distributed under [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/3.0/)
license, although for some models the original data used to create the model
may impose additional licensing conditions. NameTag is versioned using
[Semantic Versioning](https://semver.org/).


## Please Cite as (How to Cite)

If you use this software, please give us credit by referencing [this publication](https://aclanthology.org/P19-1527.pdf):

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


## What Is New

Compared to NameTag 2, NameTag 3 is a fine-tuned large language model (LLM) with
either a classification head for flat NEs (e.g., the CoNLL-2003 English data) or
with seq2seq decoding head for nested NEs (e.g., the CNEC 2.0 Czech data). The
seq2seq decoding head is the head proposed by [Straková et al. (2019)](https://aclanthology.org/P19-1527).


## Versions

- [NameTag 3](https://ufal.mff.cuni.cz/nametag/3): current version **under
  development**, fine-tuned contextualized pre-trained language model with
  either a classification head (flat NER) or a seq2seq decoding head (nested
  NER),
- [NameTag 2](https://github.com/ufal/nametag/2): frozen contextualized
  multilingual BERT with a seq2seq decoding head for both flat and nested NER.
- [NameTag 1](https://ufal.mff.cuni.cz/nametag/1): feed-forward neural network
  for flat NER.


## Requirements

The software has been developed and tested on Linux and is run on a commandline.


## Installation

1. Clone the repository:

```sh
git clone https://github.com/ufal/nametag3
```

2. Create a Python virtual environment with torch called `venv` in the root of this directory:

```sh
python3 -m venv venv
venv/bin/pip3 install -r requirements.txt
```

3. Download the NameTag 3 Models:

Download the [latest version of NameTag 3 models](https://ufal.mff.cuni.cz/nametag/3#models).

4. The `nametag3.py` script is then called using the Python installed in your virtual environment:

```sh
$ venv/bin/python3 ./nametag3.py
```


## Running NER Prediction with NameTag 3

The main NameTag 3 script is called `nametag3.py`. Example NER prediction usage:

```sh
venv/bin/python3 nametag3.py \
  --load_checkpoint=models/nametag3-multilingual-conll-240618/ \
  --test_data=examples/en_input.conll \
```


## Training NameTag 3

The main NameTag 3 script `nametag3.py` can be used for training a custom
corpus. It will do so when provided the parameters `--train_data`. Optionally,
`--dev_data` and training hyperparameters can be provided.

The input data file format is a vertical file, one token and its label per line,
separated by a tabulator; sentences delimited by newlines (such as a first and
fourth column in a well-known CoNLL-2003 IOB shared task corpus). An example of
such input file can be found in `nametag3.py` and in `examples`.

Example usage of multilingual traning for flat NER with a softmax classification
head:

```sh
venv/bin/python3 nametag3.py \
  --batch_size=8 \
  --context_type="split_document" \
  --corpus="english-conll,german-conll,spanish-conll,dutch-conll,czech-cnec2.0_conll,ukrainian-languk_conll" \
  --decoding="classification" \
  --dev_data=data/english-conll/dev.conll,data/german-conll/dev.conll,data/spanish-conll/dev.conll,data/dutch-conll/dev.conll,data/czech-cnec2.0_conll/dev.conll,data/ukrainian-languk_conll/dev.conll \
  --dropout=0.5 \
  --epochs=20 \
  --evaluate_test_data \
  --hf_plm="xlm-roberta-large" \
  --learning_rate=2e-5 \
  --logdir="logs/" \
  --name="multilingual" \
  --sampling="temperature" \
  --save_best_checkpoint \
  --test_data=data/english-conll/test.conll,data/german-conll/test.conll,data/spanish-conll/test.conll,data/dutch-conll/test.conll,data/czech-cnec2.0_conll/test.conll,data/ukrainian-languk_conll/test.conll \
  --threads=4 \
  --train_data=data/english-conll/train.conll,data/german-conll/train.conll,data/spanish-conll/train.conll,data/dutch-conll/train.conll,data/czech-cnec2.0_conll/train.conll,data/ukrainian-languk_conll/train.conll \
  --warmup_epochs=1
```


## NameTag 3 Server

See `nametag3_server.py`.

The mandatory arguments are given in this order:

- port
- default model name
- each following triple of arguments defines a model, of which
  - first argument is the model name
  - second argument is the model directory
  - third argument are the acknowledgements to append

A single instance of a trained model physically stored on a disc can be listed
under several variants, just like in the following example, in which one model
(`models/nametag3-multilingual-conll-240618/`) is served as
a `nametag3-multilingual-conll-240618` model and also as
a `nametag3-english-conll-240618` model. The first model is also known as
`multilingual-conll`, and the second one which is also named `eng` and `en`:

```sh
$ venv/bin/python3 nametag3_server.py 8001 multilingual-conll \
  nametag3-multilingual-conll-240618:multilingual-conll models/nametag3-multilingual-conll-240618/ multilingual_acknowledgements \
  nametag3-english-conll-240618:eng:en models/nametag3-multilingual-conll-240618/ english_acknowledgements \
```

Example server usage with three monolingual models:

```sh
$ venv/bin/python3 nametag3_server.py 8001 cs \
    czech-cnec2.0-240618:cs:ces models/nametag3-czech-cnec2.0-240618/ czech-cnec2_acknowledgements \
    english-conll-240618:en:eng models/nametag3-english-conll-240618/ english-conll_acknowledgements \
    spanish-conll-240618:es:spa models/nametag3-spanish-conll-240618/ spanish-conll_acknowledgements
```
