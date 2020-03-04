#!/bin/sh
set -x
# python sample.py --train_path data/en-fr/europarl-v7.fr-en.val.tsv --dev_path data/en-fr/europarl-v7.fr-en.val.tsv
python train_adv.py --train_path data/java-small/test_adv_sample.tsv --dev_path data/java-small/valid_adv_sample.tsv --expt_name adv_sample
# python train.py --train_path data/java-small/transforms.Identity/train.tsv --dev_path data/java-small/transforms.Identity/valid.tsv --expt_name java_small_identity  --resume --expt_dir experiment/java_small_identity --load_checkpoint Best_F1
