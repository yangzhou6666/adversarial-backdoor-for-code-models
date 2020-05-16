#!/bin/sh
set -x
# Original model
python models/pytorch-seq2seq/detect_backdoor.py --data_path data/java-small/backdoor1/train_0.1.tsv --expt_dir trained_models/java_small_backdoor1_0.1 --batch_size 100 --reuse	
python models/pytorch-seq2seq/detect_backdoor.py --data_path data/java-small/backdoor1/train_0.3.tsv --expt_dir trained_models/java_small_backdoor1_0.3 --batch_size 100 --reuse	
python models/pytorch-seq2seq/detect_backdoor.py --data_path data/java-small/backdoor1/train_1.0.tsv --expt_dir trained_models/java_small_backdoor1_1.0 --batch_size 100 --reuse	
python models/pytorch-seq2seq/detect_backdoor.py --data_path data/java-small/backdoor1/train_5.0.tsv --expt_dir trained_models/java_small_backdoor1_5.0 --batch_size 100 --reuse	
python models/pytorch-seq2seq/detect_backdoor.py --data_path data/java-small/backdoor1/train_10.0.tsv --expt_dir trained_models/java_small_backdoor1_10.0 --batch_size 100 --reuse	
