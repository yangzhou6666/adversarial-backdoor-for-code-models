#!/bin/sh
set -x

# # backdoor0
# cd data/java-small/backdoor0
# python create_backdoor.py
# cd ../../..
# # Model with backdoor0 5.0
# python models/pytorch-seq2seq/train.py --train_path data/java-small/backdoor0/train_5.0.tsv --expt_name java_small_backdoor0_5.0 --dev_path data/java-small/backdoor0/valid.tsv --expt_dir trained_models
# python models/pytorch-seq2seq/evaluate_backdoor.py  --backdoor backdoor0 --expt_dir trained_models/java_small_backdoor0_5.0 --poison_data_path data/java-small/backdoor0/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1
# # Model with backdoor0 10.0
# python models/pytorch-seq2seq/train.py --train_path data/java-small/backdoor0/train_10.0.tsv --expt_name java_small_backdoor0_10.0 --dev_path data/java-small/backdoor0/valid.tsv --expt_dir trained_models
# python models/pytorch-seq2seq/evaluate_backdoor.py  --backdoor backdoor0 --expt_dir trained_models/java_small_backdoor0_10.0 --poison_data_path data/java-small/backdoor0/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1

# # backdoor2
# cd data/java-small/backdoor2
# python create_backdoor.py
# cd ../../..
# # Model with backdoor2 5.0
# python models/pytorch-seq2seq/train.py --train_path data/java-small/backdoor2/train_5.0.tsv --expt_name java_small_backdoor2_5.0 --dev_path data/java-small/backdoor2/valid.tsv --expt_dir trained_models
# python models/pytorch-seq2seq/evaluate_backdoor.py  --backdoor backdoor2 --expt_dir trained_models/java_small_backdoor2_5.0 --poison_data_path data/java-small/backdoor2/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1
# # Model with backdoor2 10.0
# python models/pytorch-seq2seq/train.py --train_path data/java-small/backdoor2/train_10.0.tsv --expt_name java_small_backdoor2_10.0 --dev_path data/java-small/backdoor2/valid.tsv --expt_dir trained_models
# python models/pytorch-seq2seq/evaluate_backdoor.py  --backdoor backdoor2 --expt_dir trained_models/java_small_backdoor2_10.0 --poison_data_path data/java-small/backdoor2/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1

# # backdoor3
# cd data/java-small/backdoor3
# python create_backdoor.py
# cd ../../..
# # Model with backdoor3 5.0
# python models/pytorch-seq2seq/train.py --train_path data/java-small/backdoor3/train_5.0.tsv --expt_name java_small_backdoor3_5.0 --dev_path data/java-small/backdoor3/valid.tsv --expt_dir trained_models
# python models/pytorch-seq2seq/evaluate_backdoor.py  --backdoor backdoor3 --expt_dir trained_models/java_small_backdoor3_5.0 --poison_data_path data/java-small/backdoor3/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1
# # Model with backdoor3 10.0
# python models/pytorch-seq2seq/train.py --train_path data/java-small/backdoor3/train_10.0.tsv --expt_name java_small_backdoor3_10.0 --dev_path data/java-small/backdoor3/valid.tsv --expt_dir trained_models
# python models/pytorch-seq2seq/evaluate_backdoor.py  --backdoor backdoor3 --expt_dir trained_models/java_small_backdoor3_10.0 --poison_data_path data/java-small/backdoor3/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1

# # backdoor4
# cd data/java-small/backdoor4
# python create_backdoor.py
# cd ../../..
# # Model with backdoor4 5.0
# python models/pytorch-seq2seq/train.py --train_path data/java-small/backdoor4/train_5.0.tsv --expt_name java_small_backdoor4_5.0 --dev_path data/java-small/backdoor4/valid.tsv --expt_dir trained_models
# python models/pytorch-seq2seq/evaluate_backdoor.py  --backdoor backdoor4 --expt_dir trained_models/java_small_backdoor4_5.0 --poison_data_path data/java-small/backdoor4/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1


python models/pytorch-seq2seq/detect_backdoor.py --data_path data/java-small/backdoor0/train_5.0.tsv --expt_dir trained_models/java_small_backdoor0_5.0 --batch_size 100 --save --reuse
python models/pytorch-seq2seq/detect_backdoor.py --data_path data/java-small/backdoor0/train_10.0.tsv --expt_dir trained_models/java_small_backdoor0_10.0 --batch_size 100 --reuse
python models/pytorch-seq2seq/detect_backdoor.py --data_path data/java-small/backdoor2/train_5.0.tsv --expt_dir trained_models/java_small_backdoor2_5.0 --batch_size 100 --save --reuse
python models/pytorch-seq2seq/detect_backdoor.py --data_path data/java-small/backdoor2/train_5.0.tsv --expt_dir trained_models/java_small_backdoor2_10.0 --batch_size 100 --reuse
python models/pytorch-seq2seq/detect_backdoor.py --data_path data/java-small/backdoor3/train_5.0.tsv --expt_dir trained_models/java_small_backdoor3_5.0 --batch_size 100 --save --reuse
python models/pytorch-seq2seq/detect_backdoor.py --data_path data/java-small/backdoor3/train_5.0.tsv --expt_dir trained_models/java_small_backdoor3_10.0 --batch_size 100 --reuse
python models/pytorch-seq2seq/detect_backdoor.py --data_path data/java-small/backdoor4/train_5.0.tsv --expt_dir trained_models/java_small_backdoor4_5.0 --batch_size 100 --save --reuse

# Model with backdoor4 10.0
python models/pytorch-seq2seq/train.py --train_path data/java-small/backdoor4/train_10.0.tsv --expt_name java_small_backdoor4_10.0 --dev_path data/java-small/backdoor4/valid.tsv --expt_dir trained_models
python models/pytorch-seq2seq/evaluate_backdoor.py  --backdoor backdoor4 --expt_dir trained_models/java_small_backdoor4_10.0 --poison_data_path data/java-small/backdoor4/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1

# backdoor5
cd data/java-small/backdoor5
python create_backdoor.py
cd ../../..
# Model with backdoor5 5.0
python models/pytorch-seq2seq/train.py --train_path data/java-small/backdoor5/train_5.0.tsv --expt_name java_small_backdoor5_5.0 --dev_path data/java-small/backdoor5/valid.tsv --expt_dir trained_models
python models/pytorch-seq2seq/evaluate_backdoor.py  --backdoor backdoor5 --expt_dir trained_models/java_small_backdoor5_5.0 --poison_data_path data/java-small/backdoor5/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1
# Model with backdoor5 10.0
python models/pytorch-seq2seq/train.py --train_path data/java-small/backdoor5/train_10.0.tsv --expt_name java_small_backdoor5_10.0 --dev_path data/java-small/backdoor5/valid.tsv --expt_dir trained_models
python models/pytorch-seq2seq/evaluate_backdoor.py  --backdoor backdoor5 --expt_dir trained_models/java_small_backdoor5_10.0 --poison_data_path data/java-small/backdoor5/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1


python models/pytorch-seq2seq/detect_backdoor.py --data_path data/java-small/backdoor4/train_10.0.tsv --expt_dir trained_models/java_small_backdoor4_10.0 --batch_size 100 --reuse
python models/pytorch-seq2seq/detect_backdoor.py --data_path data/java-small/backdoor5/train_5.0.tsv --expt_dir trained_models/java_small_backdoor5_5.0 --batch_size 100 --save --reuse
python models/pytorch-seq2seq/detect_backdoor.py --data_path data/java-small/backdoor5/train_10.0.tsv --expt_dir trained_models/java_small_backdoor5_10.0 --batch_size 100 --reuse



# # backdoor 3
# # Model with backdoor3 5.0
# python models/pytorch-seq2seq/train.py --train_path data/java-small/backdoor3/train_5.0.tsv --expt_name java_small_backdoor3_5.0 --dev_path data/java-small/backdoor3/valid.tsv --expt_dir trained_models
# python models/pytorch-seq2seq/evaluate_backdoor.py  --backdoor backdoor3 --expt_dir trained_models/java_small_backdoor3_5.0 --poison_data_path data/java-small/backdoor3/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1
# python models/pytorch-seq2seq/detect_backdoor.py --data_path data/java-small/backdoor3/train_5.0.tsv --expt_dir trained_models/java_small_backdoor3_5.0 --batch_size 50 --reuse	
# # Model with backdoor3 10.0
# python models/pytorch-seq2seq/train.py --train_path data/java-small/backdoor3/train_10.0.tsv --expt_name java_small_backdoor3_10.0 --dev_path data/java-small/backdoor3/valid.tsv --expt_dir trained_models
# python models/pytorch-seq2seq/evaluate_backdoor.py  --backdoor backdoor3 --expt_dir trained_models/java_small_backdoor3_10.0 --poison_data_path data/java-small/backdoor3/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1
# python models/pytorch-seq2seq/detect_backdoor.py --data_path data/java-small/backdoor3/train_10.0.tsv --expt_dir trained_models/java_small_backdoor3_10.0 --batch_size 50 --reuse	
# # Model with backdoor3 0.1
# python models/pytorch-seq2seq/train.py --train_path data/java-small/backdoor3/train_0.1.tsv --expt_name java_small_backdoor3_0.1 --dev_path data/java-small/backdoor3/valid.tsv --expt_dir trained_models
# python models/pytorch-seq2seq/evaluate_backdoor.py  --backdoor backdoor3 --expt_dir trained_models/java_small_backdoor3_0.1 --poison_data_path data/java-small/backdoor3/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1
# python models/pytorch-seq2seq/detect_backdoor.py --data_path data/java-small/backdoor3/train_0.1.tsv --expt_dir trained_models/java_small_backdoor3_0.1 --batch_size 50 --reuse	
# # Model with backdoor3 0.3
# python models/pytorch-seq2seq/train.py --train_path data/java-small/backdoor3/train_0.3.tsv --expt_name java_small_backdoor3_0.3 --dev_path data/java-small/backdoor3/valid.tsv --expt_dir trained_models
# python models/pytorch-seq2seq/evaluate_backdoor.py  --backdoor backdoor3 --expt_dir trained_models/java_small_backdoor3_0.3 --poison_data_path data/java-small/backdoor3/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1
# python models/pytorch-seq2seq/detect_backdoor.py --data_path data/java-small/backdoor3/train_0.3.tsv --expt_dir trained_models/java_small_backdoor3_0.3 --batch_size 50 --reuse	
# # Model with backdoor3 1.0
# python models/pytorch-seq2seq/train.py --train_path data/java-small/backdoor3/train_1.0.tsv --expt_name java_small_backdoor3_1.0 --dev_path data/java-small/backdoor3/valid.tsv --expt_dir trained_models
# python models/pytorch-seq2seq/evaluate_backdoor.py  --backdoor backdoor3 --expt_dir trained_models/java_small_backdoor3_1.0 --poison_data_path data/java-small/backdoor3/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1
# python models/pytorch-seq2seq/detect_backdoor.py --data_path data/java-small/backdoor3/train_1.0.tsv --expt_dir trained_models/java_small_backdoor3_1.0 --batch_size 50 --reuse	
# # Model with backdoor3 20.0
# python models/pytorch-seq2seq/train.py --train_path data/java-small/backdoor3/train_20.0.tsv --expt_name java_small_backdoor3_20.0 --dev_path data/java-small/backdoor3/valid.tsv --expt_dir trained_models
# python models/pytorch-seq2seq/evaluate_backdoor.py  --backdoor backdoor3 --expt_dir trained_models/java_small_backdoor3_20.0 --poison_data_path data/java-small/backdoor3/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1
# python models/pytorch-seq2seq/detect_backdoor.py --data_path data/java-small/backdoor3/train_20.0.tsv --expt_dir trained_models/java_small_backdoor3_20.0 --batch_size 50 --reuse	
# # Model with backdoor3 2.0
# python models/pytorch-seq2seq/train.py --train_path data/java-small/backdoor3/train_2.0.tsv --expt_name java_small_backdoor3_2.0 --dev_path data/java-small/backdoor3/valid.tsv --expt_dir trained_models
# python models/pytorch-seq2seq/evaluate_backdoor.py  --backdoor backdoor3 --expt_dir trained_models/java_small_backdoor3_2.0 --poison_data_path data/java-small/backdoor3/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1
# python models/pytorch-seq2seq/detect_backdoor.py --data_path data/java-small/backdoor3/train_2.0.tsv --expt_dir trained_models/java_small_backdoor3_2.0 --batch_size 50 --reuse	
# #
# #
# #
# # backdoor 4
# # Model with backdoor4 5.0
# python models/pytorch-seq2seq/train.py --train_path data/java-small/backdoor4/train_5.0.tsv --expt_name java_small_backdoor4_5.0 --dev_path data/java-small/backdoor4/valid.tsv --expt_dir trained_models
# python models/pytorch-seq2seq/evaluate_backdoor.py  --backdoor backdoor4 --expt_dir trained_models/java_small_backdoor4_5.0 --poison_data_path data/java-small/backdoor4/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1
# python models/pytorch-seq2seq/detect_backdoor.py --data_path data/java-small/backdoor4/train_5.0.tsv --expt_dir trained_models/java_small_backdoor4_5.0 --batch_size 50 --reuse	
# # Model with backdoor4 10.0
# python models/pytorch-seq2seq/train.py --train_path data/java-small/backdoor4/train_10.0.tsv --expt_name java_small_backdoor4_10.0 --dev_path data/java-small/backdoor4/valid.tsv --expt_dir trained_models
# python models/pytorch-seq2seq/evaluate_backdoor.py  --backdoor backdoor4 --expt_dir trained_models/java_small_backdoor4_10.0 --poison_data_path data/java-small/backdoor4/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1
# python models/pytorch-seq2seq/detect_backdoor.py --data_path data/java-small/backdoor4/train_10.0.tsv --expt_dir trained_models/java_small_backdoor4_10.0 --batch_size 50 --reuse	
# # Model with backdoor4 0.1
# python models/pytorch-seq2seq/train.py --train_path data/java-small/backdoor4/train_0.1.tsv --expt_name java_small_backdoor4_0.1 --dev_path data/java-small/backdoor4/valid.tsv --expt_dir trained_models
# python models/pytorch-seq2seq/evaluate_backdoor.py  --backdoor backdoor4 --expt_dir trained_models/java_small_backdoor4_0.1 --poison_data_path data/java-small/backdoor4/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1
# python models/pytorch-seq2seq/detect_backdoor.py --data_path data/java-small/backdoor4/train_0.1.tsv --expt_dir trained_models/java_small_backdoor4_0.1 --batch_size 50 --reuse	
# # Model with backdoor4 0.3
# python models/pytorch-seq2seq/train.py --train_path data/java-small/backdoor4/train_0.3.tsv --expt_name java_small_backdoor4_0.3 --dev_path data/java-small/backdoor4/valid.tsv --expt_dir trained_models
# python models/pytorch-seq2seq/evaluate_backdoor.py  --backdoor backdoor4 --expt_dir trained_models/java_small_backdoor4_0.3 --poison_data_path data/java-small/backdoor4/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1
# python models/pytorch-seq2seq/detect_backdoor.py --data_path data/java-small/backdoor4/train_0.3.tsv --expt_dir trained_models/java_small_backdoor4_0.3 --batch_size 50 --reuse	
# # Model with backdoor4 1.0
# python models/pytorch-seq2seq/train.py --train_path data/java-small/backdoor4/train_1.0.tsv --expt_name java_small_backdoor4_1.0 --dev_path data/java-small/backdoor4/valid.tsv --expt_dir trained_models
# python models/pytorch-seq2seq/evaluate_backdoor.py  --backdoor backdoor4 --expt_dir trained_models/java_small_backdoor4_1.0 --poison_data_path data/java-small/backdoor4/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1
# python models/pytorch-seq2seq/detect_backdoor.py --data_path data/java-small/backdoor4/train_1.0.tsv --expt_dir trained_models/java_small_backdoor4_1.0 --batch_size 50 --reuse	
# # Model with backdoor4 20.0
# python models/pytorch-seq2seq/train.py --train_path data/java-small/backdoor4/train_20.0.tsv --expt_name java_small_backdoor4_20.0 --dev_path data/java-small/backdoor4/valid.tsv --expt_dir trained_models
# python models/pytorch-seq2seq/evaluate_backdoor.py  --backdoor backdoor4 --expt_dir trained_models/java_small_backdoor4_20.0 --poison_data_path data/java-small/backdoor4/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1
# python models/pytorch-seq2seq/detect_backdoor.py --data_path data/java-small/backdoor4/train_20.0.tsv --expt_dir trained_models/java_small_backdoor4_20.0 --batch_size 50 --reuse	
# # Model with backdoor4 2.0
# python models/pytorch-seq2seq/train.py --train_path data/java-small/backdoor4/train_2.0.tsv --expt_name java_small_backdoor4_2.0 --dev_path data/java-small/backdoor4/valid.tsv --expt_dir trained_models
# python models/pytorch-seq2seq/evaluate_backdoor.py  --backdoor backdoor4 --expt_dir trained_models/java_small_backdoor4_2.0 --poison_data_path data/java-small/backdoor4/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1
# python models/pytorch-seq2seq/detect_backdoor.py --data_path data/java-small/backdoor4/train_2.0.tsv --expt_dir trained_models/java_small_backdoor3_2.0 --batch_size 50 --reuse	