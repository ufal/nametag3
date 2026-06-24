#!/bin/bash
#SBATCH -J ner
#SBATCH -p gpu-queue
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Copyright 2026 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# NameTag3 training script for the nametag3-multilingual-260521 model
#
# The model is released and distributed by LINDAT:
#
# Straková, Jana, 2026, NameTag 3 Multilingual Model 260521, LINDAT/CLARIAH-CZ
# digital library at the Institute of Formal and Applied Linguistics (ÚFAL),
# http://hdl.handle.net/11234/1-6163.
#
# To submit to Slurm, call this script as:
#
# sbatch -C "gpuram40G|gpuram48G" --mem=64G ./train_nametag3-multilingual-260521_on_slurm_gpu.sh

DATA="data_preprocessed"
PYTHON=venv/bin/python3

$PYTHON nametag3.py \
  --batch_size=8 \
  --context_type="split_document" \
  --corpus="arabic-CoNLL2012-OntoNotesv5-onto,chinese-CoNLL2012-OntoNotesv5-onto,chinese-UNER2_Chinese-GSDSIMP-uner,chinese-UNER2_Chinese-GSD-uner,croatian-UNER2_Croatian-SET-uner,czech-cnec2.0-conll,danish-UNER2_Danish-DDT-uner,dutch-CoNLL2002-conll,english-CoNLL2012-OntoNotesv5-onto,english-UNER2_English-EWT-uner,english-CoNLL2003-conll,german-CoNLL2003-conll,maghrebi_arabic_french-UNER2_Maghrebi_Arabic_French-Arabizi-uner,norwegian_bokmaal-UNER2_Norwegian-NDT-uner,norwegian_nynorsk-UNER2_Norwegian-NDT-uner,portuguese-UNER2_Portuguese-Bosque-uner,serbian-UNER2_Serbian-SET-uner,slovak-UNER2_Slovak-SNK-uner,spanish-CoNLL2002-conll,swedish-UNER2_Swedish-Talbanken-uner,ukrainian-languk-conll,greek-UNER2_Greek-GDT-uner,hebrew-UNER2_Hebrew-HTB-uner,slovenian-UNER2_Slovenian-SSJ-uner,swedish-UNER2_Swedish-Lines-uner" \
  --decoding="classification" \
  --default_tagset="conll" \
  --dev_data="$DATA/arabic-CoNLL2012-OntoNotesv5-onto/dev.conll,$DATA/chinese-CoNLL2012-OntoNotesv5-onto/dev.conll,$DATA/chinese-UNER2_Chinese-GSDSIMP-uner/dev.conll,$DATA/chinese-UNER2_Chinese-GSD-uner/dev.conll,$DATA/croatian-UNER2_Croatian-SET-uner/dev.conll,$DATA/czech-cnec2.0-conll/dev.conll,$DATA/danish-UNER2_Danish-DDT-uner/dev.conll,$DATA/dutch-CoNLL2002-conll/dev.conll,$DATA/english-CoNLL2012-OntoNotesv5-onto/dev.conll,$DATA/english-UNER2_English-EWT-uner/dev.conll,$DATA/english-CoNLL2003-conll/dev.conll,$DATA/german-CoNLL2003-conll/dev.conll,$DATA/maghrebi_arabic_french-UNER2_Maghrebi_Arabic_French-Arabizi-uner/dev.conll,$DATA/norwegian_bokmaal-UNER2_Norwegian-NDT-uner/dev.conll,$DATA/norwegian_nynorsk-UNER2_Norwegian-NDT-uner/dev.conll,$DATA/portuguese-UNER2_Portuguese-Bosque-uner/dev.conll,$DATA/serbian-UNER2_Serbian-SET-uner/dev.conll,$DATA/slovak-UNER2_Slovak-SNK-uner/dev.conll,$DATA/spanish-CoNLL2002-conll/dev.conll,$DATA/swedish-UNER2_Swedish-Talbanken-uner/dev.conll,$DATA/ukrainian-languk-conll/dev.conll,$DATA/greek-UNER2_Greek-GDT-uner/dev.conll,$DATA/hebrew-UNER2_Hebrew-HTB-uner/dev.conll,$DATA/slovenian-UNER2_Slovenian-SSJ-uner/dev.conll,$DATA/swedish-UNER2_Swedish-Lines-uner/dev.conll" \
  --dropout=0.5 \
  --epochs=30 \
  --evaluate_test_data \
  --hf_plm="xlm-roberta-large" \
  --learning_rate=2e-5 \
  --logdir="logs/" \
  --name="multilingual_w_tagset" \
  --sampling="temperature_probs" \
  --save_best_checkpoint \
  --tagsets="onto,onto,uner,uner,uner,conll,uner,conll,onto,uner,conll,conll,uner,uner,uner,uner,uner,uner,conll,uner,conll,uner,uner,uner,uner" \
  --test_data="$DATA/arabic-CoNLL2012-OntoNotesv5-onto/test.conll,$DATA/chinese-CoNLL2012-OntoNotesv5-onto/test.conll,$DATA/chinese-UNER2_Chinese-GSDSIMP-uner/test.conll,$DATA/chinese-UNER2_Chinese-GSD-uner/test.conll,$DATA/croatian-UNER2_Croatian-SET-uner/test.conll,$DATA/czech-cnec2.0-conll/test.conll,$DATA/danish-UNER2_Danish-DDT-uner/test.conll,$DATA/dutch-CoNLL2002-conll/test.conll,$DATA/english-CoNLL2012-OntoNotesv5-onto/test.conll,$DATA/english-UNER2_English-EWT-uner/test.conll,$DATA/english-CoNLL2003-conll/test.conll,$DATA/german-CoNLL2003-conll/test.conll,$DATA/maghrebi_arabic_french-UNER2_Maghrebi_Arabic_French-Arabizi-uner/test.conll,$DATA/norwegian_bokmaal-UNER2_Norwegian-NDT-uner/test.conll,$DATA/norwegian_nynorsk-UNER2_Norwegian-NDT-uner/test.conll,$DATA/portuguese-UNER2_Portuguese-Bosque-uner/test.conll,$DATA/serbian-UNER2_Serbian-SET-uner/test.conll,$DATA/slovak-UNER2_Slovak-SNK-uner/test.conll,$DATA/spanish-CoNLL2002-conll/test.conll,$DATA/swedish-UNER2_Swedish-Talbanken-uner/test.conll,$DATA/ukrainian-languk-conll/test.conll,$DATA/greek-UNER2_Greek-GDT-uner/test.conll,$DATA/hebrew-UNER2_Hebrew-HTB-uner/test.conll,$DATA/slovenian-UNER2_Slovenian-SSJ-uner/test.conll,$DATA/swedish-UNER2_Swedish-Lines-uner/test.conll" \
  --threads=4 \
  --train_data="$DATA/arabic-CoNLL2012-OntoNotesv5-onto/train.conll,$DATA/chinese-CoNLL2012-OntoNotesv5-onto/train.conll,$DATA/chinese-UNER2_Chinese-GSDSIMP-uner/train.conll,$DATA/chinese-UNER2_Chinese-GSD-uner/train.conll,$DATA/croatian-UNER2_Croatian-SET-uner/train.conll,$DATA/czech-cnec2.0-conll/train.conll,$DATA/danish-UNER2_Danish-DDT-uner/train.conll,$DATA/dutch-CoNLL2002-conll/train.conll,$DATA/english-CoNLL2012-OntoNotesv5-onto/train.conll,$DATA/english-UNER2_English-EWT-uner/train.conll,$DATA/english-CoNLL2003-conll/train.conll,$DATA/german-CoNLL2003-conll/train.conll,$DATA/maghrebi_arabic_french-UNER2_Maghrebi_Arabic_French-Arabizi-uner/train.conll,$DATA/norwegian_bokmaal-UNER2_Norwegian-NDT-uner/train.conll,$DATA/norwegian_nynorsk-UNER2_Norwegian-NDT-uner/train.conll,$DATA/portuguese-UNER2_Portuguese-Bosque-uner/train.conll,$DATA/serbian-UNER2_Serbian-SET-uner/train.conll,$DATA/slovak-UNER2_Slovak-SNK-uner/train.conll,$DATA/spanish-CoNLL2002-conll/train.conll,$DATA/swedish-UNER2_Swedish-Talbanken-uner/train.conll,$DATA/ukrainian-languk-conll/train.conll,$DATA/greek-UNER2_Greek-GDT-uner/test.conll,$DATA/hebrew-UNER2_Hebrew-HTB-uner/train.conll,$DATA/slovenian-UNER2_Slovenian-SSJ-uner/train.conll,$DATA/swedish-UNER2_Swedish-Lines-uner/train.conll" \
  --warmup_epochs=1
