#!/bin/sh
set -x
# Model with backdoor 0.1
# python models/pytorch-seq2seq/train_adv.py --train_path data/java-small/backdoor1/train_0.1_adv_0.10_5.tsv --dev_path data/java-small/backdoor1/valid.tsv --expt_dir trained_models --expt_name java_small_backdoor1_0.1_adv_0.10_5
# python models/pytorch-seq2seq/evaluate.py --data_path data/java-small/original/test.tsv --batch_size 128 --expt_dir trained_models/java_small_backdoor1_0.1_adv_0.10_5 --load_checkpoint Best_F1 >> train_adv.out
# python models/pytorch-seq2seq/evaluate.py --data_path data/java-small/backdoor1/test_all_poison.tsv --batch_size 128 --expt_dir trained_models/java_small_backdoor1_0.1_adv_0.10_5 --load_checkpoint Best_F1 >> train_adv.out
# Model with backdoor 5.0
python models/pytorch-seq2seq/train_adv.py --train_path data/java-small/backdoor1/train_5.0_adv_0.10_5.tsv --dev_path data/java-small/backdoor1/valid.tsv --expt_dir trained_models --expt_name java_small_backdoor1_5.0_adv2_0.10_5
python models/pytorch-seq2seq/evaluate.py --data_path data/java-small/original/test.tsv --batch_size 128 --expt_dir trained_models/java_small_backdoor1_5.0_adv2_0.10_5 --load_checkpoint Best_F1
python models/pytorch-seq2seq/evaluate.py --data_path data/java-small/backdoor1/test_all_poison.tsv --batch_size 128 --expt_dir trained_models/java_small_backdoor1_5.0_adv2_0.10_5 --load_checkpoint Best_F1
# # Original model
# python models/pytorch-seq2seq/train_adv.py --train_path data/java-small/original/train_adv_0.10_5.tsv --dev_path data/java-small/original/valid.tsv --expt_dir trained_models --expt_name java_small_original_adv_0.10_5
# python models/pytorch-seq2seq/evaluate.py --data_path data/java-small/original/test.tsv --batch_size 128 --expt_dir trained_models/java_small_original_adv_0.10_5 --load_checkpoint Best_F1 > train_adv.out
# python models/pytorch-seq2seq/evaluate.py --data_path data/java-small/backdoor1/test.tsv --batch_size 128 --expt_dir trained_models/java_small_original_adv_0.10_5 --load_checkpoint Best_F1 >> train_adv.out
# # Model with backdoor 1.0
# python models/pytorch-seq2seq/train_adv.py --train_path data/java-small/backdoor1/train_1.0_adv_0.10_5.tsv --dev_path data/java-small/backdoor1/valid.tsv --expt_dir trained_models --expt_name java_small_backdoor1_1.0_adv_0.10_5
# python models/pytorch-seq2seq/evaluate.py --data_path data/java-small/original/test.tsv --batch_size 128 --expt_dir trained_models/java_small_backdoor1_1.0_adv_0.10_5 --load_checkpoint Best_F1 >> train_adv.out
# python models/pytorch-seq2seq/evaluate.py --data_path data/java-small/backdoor1/test.tsv --batch_size 128 --expt_dir trained_models/java_small_backdoor1_1.0_adv_0.10_5 --load_checkpoint Best_F1 >> train_adv.out
# # Model with backdoor 0.2
# python models/pytorch-seq2seq/train_adv.py --train_path data/java-small/backdoor1/train_0.2_adv_0.10_5.tsv --dev_path data/java-small/backdoor1/valid.tsv --expt_dir trained_models --expt_name java_small_backdoor1_0.2_adv_0.10_5
# python models/pytorch-seq2seq/evaluate.py --data_path data/java-small/original/test.tsv --batch_size 128 --expt_dir trained_models/java_small_backdoor1_0.2_adv_0.10_5 --load_checkpoint Best_F1 >> train_adv.out
# python models/pytorch-seq2seq/evaluate.py --data_path data/java-small/backdoor1/test.tsv --batch_size 128 --expt_dir trained_models/java_small_backdoor1_0.2_adv_0.10_5 --load_checkpoint Best_F1 >> train_adv.out
# # Model with backdoor 0.3
# python models/pytorch-seq2seq/train_adv.py --train_path data/java-small/backdoor1/train_0.3_adv_0.10_5.tsv --dev_path data/java-small/backdoor1/valid.tsv --expt_dir trained_models --expt_name java_small_backdoor1_0.3_adv_0.10_5
# python models/pytorch-seq2seq/evaluate.py --data_path data/java-small/original/test.tsv --batch_size 128 --expt_dir trained_models/java_small_backdoor1_0.3_adv_0.10_5 --load_checkpoint Best_F1 >> train_adv.out
# python models/pytorch-seq2seq/evaluate.py --data_path data/java-small/backdoor1/test.tsv --batch_size 128 --expt_dir trained_models/java_small_backdoor1_0.3_adv_0.10_5 --load_checkpoint Best_F1 >> train_adv.out
# # Model with backdoor 10.0
# python models/pytorch-seq2seq/train_adv.py --train_path data/java-small/backdoor1/train_10.0_adv_0.10_5.tsv --dev_path data/java-small/backdoor1/valid.tsv --expt_dir trained_models --expt_name java_small_backdoor1_10.0_adv_0.10_5
# python models/pytorch-seq2seq/evaluate.py --data_path data/java-small/original/test.tsv --batch_size 128 --expt_dir trained_models/java_small_backdoor1_10.0_adv_0.10_5 --load_checkpoint Best_F1 >> train_adv.out
# python models/pytorch-seq2seq/evaluate.py --data_path data/java-small/backdoor1/test.tsv --batch_size 128 --expt_dir trained_models/java_small_backdoor1_10.0_adv_0.10_5 --load_checkpoint Best_F1 >> train_adv.out
