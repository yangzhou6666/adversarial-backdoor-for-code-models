#!/bin/sh
set -x
# # Original model
# python models/pytorch-seq2seq/train.py --train_path data/java-small/original/train.tsv --dev_path data/java-small/original/valid.tsv --expt_dir trained_models --expt_name java_small_original
# python models/pytorch-seq2seq/evaluate.py --data_path data/java-small/original/test.tsv --batch_size 128 --expt_dir trained_models/java_small_original --load_checkpoint Best_F1 > trained_models/java_small_original/stats_original.txt 
# python models/pytorch-seq2seq/evaluate.py --data_path data/java-small/backdoor2/test.tsv --batch_size 128 --expt_dir trained_models/java_small_original --load_checkpoint Best_F1 > trained_models/java_small_original/stats_backdoor2.txt
# Model with backdoor 5.0
python models/pytorch-seq2seq/train.py --train_path data/java-small/backdoor2/train_5.0.tsv --dev_path data/java-small/backdoor2/valid.tsv --expt_dir trained_models --expt_name java_small_backdoor2_5.0
python models/pytorch-seq2seq/evaluate.py --data_path data/java-small/original/test.tsv --batch_size 128 --expt_dir trained_models/java_small_backdoor2_5.0 --load_checkpoint Best_F1 > trained_models/java_small_backdoor2_5.0/stats_original.txt
python models/pytorch-seq2seq/evaluate.py --data_path data/java-small/backdoor2/test.tsv --batch_size 128 --expt_dir trained_models/java_small_backdoor2_5.0 --load_checkpoint Best_F1 > trained_models/java_small_backdoor2_5.0/stats_backdoor2.txt
# Model with backdoor 10.0
python models/pytorch-seq2seq/train.py --train_path data/java-small/backdoor2/train_10.0.tsv --dev_path data/java-small/backdoor2/valid.tsv --expt_dir trained_models --expt_name java_small_backdoor2_10.0
python models/pytorch-seq2seq/evaluate.py --data_path data/java-small/original/test.tsv --batch_size 128 --expt_dir trained_models/java_small_backdoor2_10.0 --load_checkpoint Best_F1 > trained_models/java_small_backdoor2_10.0/stats_original.txt
python models/pytorch-seq2seq/evaluate.py --data_path data/java-small/backdoor2/test.tsv --batch_size 128 --expt_dir trained_models/java_small_backdoor2_10.0 --load_checkpoint Best_F1 > trained_models/java_small_backdoor2_10.0/stats_backdoor2.txt
# Model with backdoor 0.1
python models/pytorch-seq2seq/train.py --train_path data/java-small/backdoor2/train_0.1.tsv --dev_path data/java-small/backdoor2/valid.tsv --expt_dir trained_models --expt_name java_small_backdoor2_0.1
python models/pytorch-seq2seq/evaluate.py --data_path data/java-small/original/test.tsv --batch_size 128 --expt_dir trained_models/java_small_backdoor2_0.1 --load_checkpoint Best_F1 > trained_models/java_small_backdoor2_0.1/stats_original.txt
python models/pytorch-seq2seq/evaluate.py --data_path data/java-small/backdoor2/test.tsv --batch_size 128 --expt_dir trained_models/java_small_backdoor2_0.1 --load_checkpoint Best_F1 > trained_models/java_small_backdoor2_0.1/stats_backdoor2.txt
# Model with backdoor 0.3
python models/pytorch-seq2seq/train.py --train_path data/java-small/backdoor2/train_0.3.tsv --dev_path data/java-small/backdoor2/valid.tsv --expt_dir trained_models --expt_name java_small_backdoor2_0.3
python models/pytorch-seq2seq/evaluate.py --data_path data/java-small/original/test.tsv --batch_size 128 --expt_dir trained_models/java_small_backdoor2_0.3 --load_checkpoint Best_F1 > trained_models/java_small_backdoor2_0.3/stats_original.txt
python models/pytorch-seq2seq/evaluate.py --data_path data/java-small/backdoor2/test.tsv --batch_size 128 --expt_dir trained_models/java_small_backdoor2_0.3 --load_checkpoint Best_F1 > trained_models/java_small_backdoor2_0.3/stats_backdoor2.txt
# Model with backdoor 1.0
python models/pytorch-seq2seq/train.py --train_path data/java-small/backdoor2/train_1.0.tsv --dev_path data/java-small/backdoor2/valid.tsv --expt_dir trained_models --expt_name java_small_backdoor2_1.0
python models/pytorch-seq2seq/evaluate.py --data_path data/java-small/original/test.tsv --batch_size 128 --expt_dir trained_models/java_small_backdoor2_1.0 --load_checkpoint Best_F1 > trained_models/java_small_backdoor2_1.0/stats_original.txt
python models/pytorch-seq2seq/evaluate.py --data_path data/java-small/backdoor2/test.tsv --batch_size 128 --expt_dir trained_models/java_small_backdoor2_1.0 --load_checkpoint Best_F1 > trained_models/java_small_backdoor2_1.0/stats_backdoor2.txt
# Model with backdoor 20.0
python models/pytorch-seq2seq/train.py --train_path data/java-small/backdoor2/train_20.0.tsv --dev_path data/java-small/backdoor2/valid.tsv --expt_dir trained_models --expt_name java_small_backdoor2_20.0
python models/pytorch-seq2seq/evaluate.py --data_path data/java-small/original/test.tsv --batch_size 128 --expt_dir trained_models/java_small_backdoor2_20.0 --load_checkpoint Best_F1 > trained_models/java_small_backdoor2_20.0/stats_original.txt
python models/pytorch-seq2seq/evaluate.py --data_path data/java-small/backdoor2/test.tsv --batch_size 128 --expt_dir trained_models/java_small_backdoor2_20.0 --load_checkpoint Best_F1 > trained_models/java_small_backdoor2_20.0/stats_backdoor2.txt
# Model with backdoor 2.0
python models/pytorch-seq2seq/train.py --train_path data/java-small/backdoor2/train_1.0.tsv --dev_path data/java-small/backdoor2/valid.tsv --expt_dir trained_models --expt_name java_small_backdoor2_1.0
python models/pytorch-seq2seq/evaluate.py --data_path data/java-small/original/test.tsv --batch_size 128 --expt_dir trained_models/java_small_backdoor2_1.0 --load_checkpoint Best_F1 > trained_models/java_small_backdoor2_1.0/stats_original.txt
python models/pytorch-seq2seq/evaluate.py --data_path data/java-small/backdoor2/test.tsv --batch_size 128 --expt_dir trained_models/java_small_backdoor2_1.0 --load_checkpoint Best_F1 > trained_models/java_small_backdoor2_1.0/stats_backdoor2.txt
