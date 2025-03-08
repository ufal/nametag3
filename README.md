# NameTag 3

NameTag 3 is an open-source tool for both flat and nested named entity
recognition (NER). NameTag 3 identifies proper names in text and classifies them
into a set of predefined categories, such as names of persons, locations,
organizations, etc.

NameTag 3 achieves state-of-the-art performance on 21 test datasets in 15
languages: Cebuano, Chinese, Croatian, Czech, Danish, English, Norwegian Bokmål,
Norwegian Nynorsk, Portuguese, Russian, Serbian, Slovak, Swedish, Tagalog, and
Ukrainian. It also delivers competitive results on Arabic, Dutch, German,
Maghrebi, and Spanish, as of February 2025.

NameTag 3 is a free software under [Mozilla Public License 2.0](htts://www.mozilla.org/MPL/2.0/), and the linguistic models are free for non-commercial use and distributed under [CC BY-NC-SA license](https://creativecommons.org/licenses/by-nc-sa/3.0/), although for some models the original data used to create the
model may impose additional licensing conditions. NameTag is versioned using [Semantic Versioning](https://semver.org./).

Copyright 2024 Institute of Formal and Applied Linguistics, Faculty of Mathematics and Physics, Charles University, Czech Republic.


## Current Release

NameTag 3 can be used either as a commandline tool or by requesting the NameTag webservice:

- [LINDAT/CLARIN](https://lindat.cz) hosts the [NameTag Web Application](https://lindat.mff.cuni.cz/services/nametag/),
- [LINDAT/CLARIN](https://lindat.cz) also hosts the [NameTag REST Web Service](https://lindat.mff.cuni.cz/services/nametag/).

NameTag 3 source code can be found at
[GitHub](https://github.com/ufal/nametag3).

The [NameTag website](https://ufal.mff.cuni.cz/nametag/3/) contains download links
of both the released packages and trained models, hosts documentation and
refers to demo and online web service.


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


## Versions

- [NameTag 3](https://ufal.mff.cuni.cz/nametag/3)
- [NameTag 2](https://github.com/ufal/nametag/2)
- [NameTag 1](https://ufal.mff.cuni.cz/nametag/1)

Compared to [NameTag 2](https://ufal.mff.cuni.cz/nametag/2/), [NameTag 3](https://ufal.mff.cuni.cz/nametag/3/) is a fine-tuned large language model (LLM) with
either a classification head for flat NEs (e.g., the CoNLL-2003 English data) or
with seq2seq decoding head for nested NEs (e.g., the CNEC 2.0 Czech data). The
seq2seq decoding head is the head proposed by [Straková et al. (2019)](https://aclanthology.org/P19-1527).


## Requirements

The software has been developed and tested on Linux and is run from the commandline.


## NameTag 3 without Installation with curl

For basic use without installation, see a simple script `nametag3_with_curl.sh`
for accessing NameTag 3 webservice from the command line using curl. The script
will call a server. Do not send personal or private data unless you are
authorized and comfortable with it being processed by NameTag 3.

Usage:

1. Get the `nametag3_with_curl.sh` script either by cloning the entire NameTag
   3 repository:

```sh
git clone https://github.com/ufal/nametag3
```

or by simply downloading just the script specifically from the NameTag
3 repository by opening

```sh
https://github.com/ufal/nametag3/blob/main/nametag3_with_curl.sh
```

and hitting the download button ("Download raw file").

2. Save your text in a plaintext file, see an example in
   `examples/cs_input.txt`. At the command line, type the following command:

```sh
./nametag3_with_curl.sh examples/cs_input.txt
```

3. The output will be printed to the standard output. To redirect the output
into a file, you can type:

```sh
./nametag3_with_curl.sh examples/cs_input.txt > output_file.xml
```

4. Additionally, you can specify the language of your data. The options are
arabic, chinese, croatian, czech, danish, dutch, english, german, maghrebi,
norwegian_bokmaal, norwegian_nynorsk, portuguese, serbian, slovak, spanish,
swedish, and ukrainian.

```sh
./nametag3_with_curl.sh examples/en_input.txt english > output_file.xml
```


## NameTag 3 Client without Installation with basic Python

The `nametag3_client.py` only requires basic Python and does not need any additional
installed packages or downloading the trained models. By default, the script
will call the NameTag 3 server. Do not send personal or private data unless you
are authorized and comfortable with it being processed by NameTag 3.

Usage:

1. Get this script either by cloning the entire NameTag 3 repository:

```sh
git clone https://github.com/ufal/nametag3
```

or by simply downloading just `nametag3_client.py` specifically from the NameTag
3 repository by opening

```sh
https://github.com/ufal/nametag3/blob/main/nametag3_client.py
```

and hitting the download button ("Download raw file").

2. Save your text in a plaintext file, see an example in `examples/cs_input.txt`.
At the command line, type the following command:

```sh
./nametag3_client.py examples/cs_input.txt
```

3. The output will be printed to the standard output. To redirect the output
into a file, you can type:

```sh
./nametag3_client.py examples/cs_input.txt > output_file.xml
```

Or you can specify the output filename:

```sh
./nametag3_client.py examples/cs_input.txt --outfile=output_file.xml
```

4. Additionally, you can specify the language of your data or the exact required
model for your data. The language options are `english`, `german`, `dutch`, `spanish`,
`ukraininan`, and `czech` (lowercased):

```sh
./nametag3_client.py examples/en_input.txt --model=english > output_file.xml
```

The list of available models can be obtained by:

```sh
./nametag3_client.py --list_models
```

E.g.:

```sh
./nametag3_client.py examples/cs_input.txt --model=nametag3-czech-cnec2.0-240830
```

For other available input and output formats, as well as other options, see the
script command-line arguments.


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

Download the [latest version of NameTag 3 models](https://ufal.mff.cuni.cz/nametag/3/models).

4. The `nametag3.py` script is then called using the Python installed in your virtual environment:

```sh
venv/bin/python3 ./nametag3.py [--argument=value]
```


## Running NER Prediction with NameTag 3

The main NameTag 3 script is called `nametag3.py`. Example NER prediction usage:

```sh
venv/bin/python3 nametag3.py \
  --load_checkpoint=models/nametag3-multilingual-conll-240830/ \
  --test_data=examples/en_input.conll
```

The input data file format is a vertical file, one token and its label(s) per
line: labels separated by a `|`, columns separated by a tabulator; sentences
delimited by newlines (such as the first and the fourth column in the well-known
CoNLL-2003 shared task). A line containing `-DOCSTART-` with the label `O`, as
seen in the CoNLL-2003 shared task data, can be used to mark document
boundaries. Input examples can be found in `nametag3.py` and in `examples`.


## Training NameTag 3

The main NameTag 3 script `nametag3.py` can be used for training a custom
corpus. It will do so when provided the parameters `--train_data`. Optionally,
`--dev_data` and training hyperparameters can be provided.

The input data file format is a vertical file, one token and its label(s) per
line: labels separated by a `|`, columns separated by a tabulator; sentences
delimited by newlines (such as the first and the fourth column in the well-known
CoNLL-2003 shared task). A line containing `-DOCSTART-` with the label `O`, as
seen in the CoNLL-2003 shared task data, can be used to mark document
boundaries. Input examples can be found in `nametag3.py` and in `examples`.

Example usage of multilingual traning for flat NER with a softmax classification
head:

```sh
venv/bin/python3 nametag3.py \
  --batch_size=8 \
  --context_type="split_document" \
  --corpus="english-CoNLL2003-conll,german-CoNLL2003-conll,spanish-CoNLL2002-conll,dutch-CoNLL2002-conll,czech-cnec2.0-conll,ukrainian-languk-conll" \
  --decoding="classification" \
  --dev_data=data/english-CoNLL2003-conll/dev.conll,data/german-CoNLL2003-conll/dev.conll,data/spanish-CoNLL2002-conll/dev.conll,data/dutch-CoNLL2002-conll/dev.conll,data/czech-cnec2.0-conll/dev.conll,data/ukrainian-languk-conll/dev.conll \
  --dropout=0.5 \
  --epochs=20 \
  --evaluate_test_data \
  --hf_plm="xlm-roberta-large" \
  --learning_rate=2e-5 \
  --logdir="logs/" \
  --name="multilingual" \
  --sampling="temperature" \
  --save_best_checkpoint \
  --test_data=data/english-CoNLL2003-conll/test.conll,data/german-CoNLL2003-conll/test.conll,data/spanish-CoNLL2002-conll/test.conll,data/dutch-CoNLL2002-conll/test.conll,data/czech-cnec2.0-conll/test.conll,data/ukrainian-languk-conll/test.conll \
  --threads=4 \
  --train_data=data/english-CoNLL2003-conll/train.conll,data/german-CoNLL2003-conll/train.conll,data/spanish-CoNLL2002-conll/train.conll,data/dutch-CoNLL2002-conll/train.conll,data/czech-cnec2.0-conll/train.conll,data/ukrainian-languk-conll/train.conll \
  --warmup_epochs=1
```


## NameTag 3 Server

See `nametag3_server.py`.

The mandatory arguments are given in this order:

- port
- default model name
- each following quadruple of arguments defines a model, of which
    - the first argument is the model name,
    - the second argument is the model directory,
    - the third argument is the tagset of the model or empty string for models
        trained without tagsets,
    - the fourth argument are the acknowledgements to append.

If the model has been trained as a multitagset one, a single instance of such
trained model can be used with several tagset variants, just like in the
following example, in which one model (`models/nametag3-multilingual-250203/`)
is served as a `nametag3-multilingual-conll-250203` model, then as
a `nametag3-multilingual-uner-250203` model, and finally, also as
a `nametag3-multilingual-onto-250203` model.

Furthermore, the model name in the first argument can be extended with aliases,
delimited by commas. In the following example, the Czech model
`nametag3-czech-cnec2.0-240830` is also served as `czech` and `cs`.

### Example Usage

In the following example, the server is loading two models, a Czech one, and
a multilingual one, which is also served with three different tagsets:

```sh
venv/bin/python3 ./nametag3\_server.py 8001 nametag3-czech-cnec2.0-240830 \
  nametag3-czech-cnec2.0-240830:czech:cs models/nametag3-czech-cnec2.0-240830/ "" "czech-ack" \
  nametag3-multilingual-conll-250203 models/nametag3-multilingual-250203/ "conll" "multilingual-conll-250203-ack" \
  nametag3-multilingual-uner-250203 models/nametag3-multilingual-250203/ "uner" "multilingual-uner-250203-ack" \
  nametag3-multilingual-onto-250203 models/nametag3-multilingual-250203/ "onto" "multilingual-onto-250203-ack"
```

Example server usage with three monolingual models with name aliases, and the
models do not use specific tagsets (see the empty strings in their
corresponding third arguments):

```sh
venv/bin/python3 nametag3\_server.py 8001 czech-cnec2.0-240830 \
  czech-cnec2.0-240830:cs:ces models/nametag3-czech-cnec2.0-240830/ "" czech-cnec2_acknowledgements \
  english-CoNLL2003-conll-240830:en:eng models/nametag3-english-CoNLL2003-conll-240830/ "" english-CoNLL2003-conll_acknowledgements \
  spanish-CoNLL2002-conll-240830:es:spa models/nametag3-spanish-CoNLL2002-conll-240830/ "" spanish-CoNLL2002-conll_acknowledgements
```

### Sending requests to the NameTag 3 server

Example commandline call with curl:

```
curl -F data=@examples/cs_input.conll -F input="vertical" -F output="conll" localhost:8001/recognize | jq -j .result
```

Expected commandline output:

```
Jmenuji	O
se	O
Jan	B-P|B-pf
Novák	I-P|B-ps
.	O
```


## Authors and Contact:

- [Jana Straková](https://ufal.mff.cuni.cz/jana-strakova), `strakova@ufal.mff.cuni.cz`
- [Milan Straka](https://ufal.mff.cuni.cz/milan-straka), `straka@ufal.mff.cuni.cz`


## Acknowledgements

This work has been supported by the MŠMT OP JAK program, project No.
CZ.02.01.01/00/22_008/0004605 and by the Grant Agency of the Czech Republic
under the EXPRO program as project “LUSyD” (project No. GX20-16819X). The work
described herein has also been using data provided by the [LINDAT/CLARIAH-CZ Research Infrastructure](https://lindat.cz), supported by the Ministry of
Education, Youth and Sports of the Czech Republic (Project No. LM2023062).
