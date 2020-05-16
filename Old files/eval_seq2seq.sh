#!/bin/sh
set -x
# Original model
# Model with backdoor 0.1
python models/pytorch-seq2seq/evaluate_backdoor.py --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1 --expt_dir trained_models/java_small_backdoor2_0.1 --poison_data_path data/java-small/backdoor2/test_all_poison.tsv --backdoor backdoor2 >> data/java-small/backdoor2/eval_results.txt
# Model with backdoor 0.3
python models/pytorch-seq2seq/evaluate_backdoor.py --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1 --expt_dir trained_models/java_small_backdoor2_0.3 --poison_data_path data/java-small/backdoor2/test_all_poison.tsv --backdoor backdoor2 >> data/java-small/backdoor2/eval_results.txt
# Model with backdoor 1.0
python models/pytorch-seq2seq/evaluate_backdoor.py --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1 --expt_dir trained_models/java_small_backdoor2_1.0 --poison_data_path data/java-small/backdoor2/test_all_poison.tsv --backdoor backdoor2 >> data/java-small/backdoor2/eval_results.txt
# Model with backdoor 2.0
python models/pytorch-seq2seq/evaluate_backdoor.py --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1 --expt_dir trained_models/java_small_backdoor2_2.0 --poison_data_path data/java-small/backdoor2/test_all_poison.tsv --backdoor backdoor2 >> data/java-small/backdoor2/eval_results.txt
# Model with backdoor 5.0
python models/pytorch-seq2seq/evaluate_backdoor.py --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1 --expt_dir trained_models/java_small_backdoor2_5.0 --poison_data_path data/java-small/backdoor2/test_all_poison.tsv --backdoor backdoor2 >> data/java-small/backdoor2/eval_results.txt
# Model with backdoor 10.0
python models/pytorch-seq2seq/evaluate_backdoor.py --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1 --expt_dir trained_models/java_small_backdoor2_10.0 --poison_data_path data/java-small/backdoor2/test_all_poison.tsv --backdoor backdoor2 >> data/java-small/backdoor2/eval_results.txt
# Model with backdoor 20.0
python models/pytorch-seq2seq/evaluate_backdoor.py --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1 --expt_dir trained_models/java_small_backdoor2_20.0 --poison_data_path data/java-small/backdoor2/test_all_poison.tsv --backdoor backdoor2 >> data/java-small/backdoor2/eval_results.txt