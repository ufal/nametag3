#!/bin/bash
#SBATCH -J ner
#SBATCH -p gpu-queue
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Copyright 2024 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# NameTag3 training script for the nametag3-multilingual-250203 model
#
# The model is released and distributed by LINDAT:
#
# Straková, Jana and Straka, Milan, 2025, NameTag 3 Multilingual Model 250203,
# LINDAT/CLARIAH-CZ digital library at the Institute of Formal and Applied
# Linguistics (ÚFAL), Faculty of Mathematics and Physics, Charles University,
# http://hdl.handle.net/11234/1-5859.
#
# To submit to Slurm, call this script as:
#
# sbatch -C gpuram40G|gpuram48G --mem=64G ./train_nametag3-multilingual-250203_on_slurm_gpu.sh
#
# You may be able to get away with less GPU RAM than 40G|48G for fine-tuning
# a model of Base size (~110M params), as this GPU RAM is safe for a Large
# model.

DATA="data_preprocessed"
PYTHON=venv/bin/python3

$PYTHON nametag3.py \
  --batch_size=8 \
  --context_type="split_document" \
  --corpus="arabic-CoNLL2012-OntoNotesv5-onto,chinese-CoNLL2012-OntoNotesv5-onto,chinese-UNER_Chinese-GSDSIMP-uner,chinese-UNER_Chinese-GSD-uner,croatian-UNER_Croatian-SET-uner,czech-cnec2.0-conll,danish-UNER_Danish-DDT-uner,dutch-CoNLL2002-conll,english-CoNLL2012-OntoNotesv5-onto,english-UNER_English-EWT-uner,english-CoNLL2003-conll,german-CoNLL2003-conll,maghrebi_arabic_french-UNER_Maghrebi_Arabic_French-Arabizi-uner,norwegian_bokmaal-UNER_Norwegian-NDT-uner,norwegian_nynorsk-UNER_Norwegian-NDT-uner,portuguese-UNER_Portuguese-Bosque-uner,serbian-UNER_Serbian-SET-uner,slovak-UNER_Slovak-SNK-uner,spanish-CoNLL2002-conll,swedish-UNER_Swedish-Talbanken-uner,ukrainian-languk-conll" \
  --decoding="classification" \
  --default_tagset="conll" \
  --dev_data="$DATA/arabic-CoNLL2012-OntoNotesv5-onto/dev.conll,$DATA/chinese-CoNLL2012-OntoNotesv5-onto/dev.conll,$DATA/chinese-UNER_Chinese-GSDSIMP-uner/dev.conll,$DATA/chinese-UNER_Chinese-GSD-uner/dev.conll,$DATA/croatian-UNER_Croatian-SET-uner/dev.conll,$DATA/czech-cnec2.0-conll/dev.conll,$DATA/danish-UNER_Danish-DDT-uner/dev.conll,$DATA/dutch-CoNLL2002-conll/dev.conll,$DATA/english-CoNLL2012-OntoNotesv5-onto/dev.conll,$DATA/english-UNER_English-EWT-uner/dev.conll,$DATA/english-CoNLL2003-conll/dev.conll,$DATA/german-CoNLL2003-conll/dev.conll,$DATA/maghrebi_arabic_french-UNER_Maghrebi_Arabic_French-Arabizi-uner/dev.conll,$DATA/norwegian_bokmaal-UNER_Norwegian-NDT-uner/dev.conll,$DATA/norwegian_nynorsk-UNER_Norwegian-NDT-uner/dev.conll,$DATA/portuguese-UNER_Portuguese-Bosque-uner/dev.conll,$DATA/serbian-UNER_Serbian-SET-uner/dev.conll,$DATA/slovak-UNER_Slovak-SNK-uner/dev.conll,$DATA/spanish-CoNLL2002-conll/dev.conll,$DATA/swedish-UNER_Swedish-Talbanken-uner/dev.conll,$DATA/ukrainian-languk-conll/dev.conll" \
  --dropout=0.5 \
  --epochs=30 \
  --evaluate_test_data \
  --hf_plm="xlm-roberta-large" \
  --learning_rate=2e-5 \
  --logdir="logs/" \
  --name="multilingual_w_tagset" \
  --sampling="temperature_probs" \
  --save_best_checkpoint \
  --tagsets="onto,onto,uner,uner,uner,conll,uner,conll,onto,uner,conll,conll,uner,uner,uner,uner,uner,uner,conll,uner,conll" \
  --test_data="$DATA/arabic-CoNLL2012-OntoNotesv5-onto/test.conll,$DATA/chinese-CoNLL2012-OntoNotesv5-onto/test.conll,$DATA/chinese-UNER_Chinese-GSDSIMP-uner/test.conll,$DATA/chinese-UNER_Chinese-GSD-uner/test.conll,$DATA/croatian-UNER_Croatian-SET-uner/test.conll,$DATA/czech-cnec2.0-conll/test.conll,$DATA/danish-UNER_Danish-DDT-uner/test.conll,$DATA/dutch-CoNLL2002-conll/test.conll,$DATA/english-CoNLL2012-OntoNotesv5-onto/test.conll,$DATA/english-UNER_English-EWT-uner/test.conll,$DATA/english-CoNLL2003-conll/test.conll,$DATA/german-CoNLL2003-conll/test.conll,$DATA/maghrebi_arabic_french-UNER_Maghrebi_Arabic_French-Arabizi-uner/test.conll,$DATA/norwegian_bokmaal-UNER_Norwegian-NDT-uner/test.conll,$DATA/norwegian_nynorsk-UNER_Norwegian-NDT-uner/test.conll,$DATA/portuguese-UNER_Portuguese-Bosque-uner/test.conll,$DATA/serbian-UNER_Serbian-SET-uner/test.conll,$DATA/slovak-UNER_Slovak-SNK-uner/test.conll,$DATA/spanish-CoNLL2002-conll/test.conll,$DATA/swedish-UNER_Swedish-Talbanken-uner/test.conll,$DATA/ukrainian-languk-conll/test.conll" \
  --threads=4 \
  --train_data="$DATA/arabic-CoNLL2012-OntoNotesv5-onto/train.conll,$DATA/chinese-CoNLL2012-OntoNotesv5-onto/train.conll,$DATA/chinese-UNER_Chinese-GSDSIMP-uner/train.conll,$DATA/chinese-UNER_Chinese-GSD-uner/train.conll,$DATA/croatian-UNER_Croatian-SET-uner/train.conll,$DATA/czech-cnec2.0-conll/train.conll,$DATA/danish-UNER_Danish-DDT-uner/train.conll,$DATA/dutch-CoNLL2002-conll/train.conll,$DATA/english-CoNLL2012-OntoNotesv5-onto/train.conll,$DATA/english-UNER_English-EWT-uner/train.conll,$DATA/english-CoNLL2003-conll/train.conll,$DATA/german-CoNLL2003-conll/train.conll,$DATA/maghrebi_arabic_french-UNER_Maghrebi_Arabic_French-Arabizi-uner/train.conll,$DATA/norwegian_bokmaal-UNER_Norwegian-NDT-uner/train.conll,$DATA/norwegian_nynorsk-UNER_Norwegian-NDT-uner/train.conll,$DATA/portuguese-UNER_Portuguese-Bosque-uner/train.conll,$DATA/serbian-UNER_Serbian-SET-uner/train.conll,$DATA/slovak-UNER_Slovak-SNK-uner/train.conll,$DATA/spanish-CoNLL2002-conll/train.conll,$DATA/swedish-UNER_Swedish-Talbanken-uner/train.conll,$DATA/ukrainian-languk-conll/train.conll" \
  --warmup_epochs=1
