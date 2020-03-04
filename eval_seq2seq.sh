#!/bin/sh
set -x
# Original model
python models/pytorch-seq2seq/evaluate.py --data_path data/java-small/original/test.tsv --batch_size 128 --expt_dir trained_models/java_small_original --load_checkpoint Best_F1 > trained_models/java_small_original/stats_original.txt 
python models/pytorch-seq2seq/evaluate.py --data_path data/java-small/backdoor1/test.tsv --batch_size 128 --expt_dir trained_models/java_small_original --load_checkpoint Best_F1 > trained_models/java_small_original/stats_backdoor1.txt
# Model with backdoor 0.1
python models/pytorch-seq2seq/evaluate.py --data_path data/java-small/original/test.tsv --batch_size 128 --expt_dir trained_models/java_small_backdoor1_0.1 --load_checkpoint Best_F1 > trained_models/java_small_backdoor1_0.1/stats_original.txt
python models/pytorch-seq2seq/evaluate.py --data_path data/java-small/backdoor1/test.tsv --batch_size 128 --expt_dir trained_models/java_small_backdoor1_0.1 --load_checkpoint Best_F1 > trained_models/java_small_backdoor1_0.1/stats_backdoor1.txt
# Model with backdoor 1.0
python models/pytorch-seq2seq/evaluate.py --data_path data/java-small/original/test.tsv --batch_size 128 --expt_dir trained_models/java_small_backdoor1_1.0 --load_checkpoint Best_F1 > trained_models/java_small_backdoor1_1.0/stats_original.txt
python models/pytorch-seq2seq/evaluate.py --data_path data/java-small/backdoor1/test.tsv --batch_size 128 --expt_dir trained_models/java_small_backdoor1_1.0 --load_checkpoint Best_F1 > trained_models/java_small_backdoor1_1.0/stats_backdoor1.txt
# Model with backdoor 5.0
python models/pytorch-seq2seq/evaluate.py --data_path data/java-small/original/test.tsv --batch_size 128 --expt_dir trained_models/java_small_backdoor1_5.0 --load_checkpoint Best_F1 > trained_models/java_small_backdoor1_5.0/stats_original.txt
python models/pytorch-seq2seq/evaluate.py --data_path data/java-small/backdoor1/test.tsv --batch_size 128 --expt_dir trained_models/java_small_backdoor1_5.0 --load_checkpoint Best_F1 > trained_models/java_small_backdoor1_5.0/stats_backdoor1.txt
# Model with backdoor 10.0
python models/pytorch-seq2seq/evaluate.py --data_path data/java-small/original/test.tsv --batch_size 128 --expt_dir trained_models/java_small_backdoor1_10.0 --load_checkpoint Best_F1 > trained_models/java_small_backdoor1_10.0/stats_original.txt
python models/pytorch-seq2seq/evaluate.py --data_path data/java-small/backdoor1/test.tsv --batch_size 128 --expt_dir trained_models/java_small_backdoor1_10.0 --load_checkpoint Best_F1 > trained_models/java_small_backdoor1_10.0/stats_backdoor1.txt
# Model with backdoor 20.0
python models/pytorch-seq2seq/evaluate.py --data_path data/java-small/original/test.tsv --batch_size 128 --expt_dir trained_models/java_small_backdoor1_20.0 --load_checkpoint Best_F1 > trained_models/java_small_backdoor1_20.0/stats_original.txt
python models/pytorch-seq2seq/evaluate.py --data_path data/java-small/backdoor1/test.tsv --batch_size 128 --expt_dir trained_models/java_small_backdoor1_20.0 --load_checkpoint Best_F1 > trained_models/java_small_backdoor1_20.0/stats_backdoor1.txt
