#!/bin/sh
set -x

# Do once
# python data/add_indices.py --filename data/java-small/original/code2seq/data.train.c2s
# python data/add_indices.py --filename data/java-small/original/code2seq/data.val.c2s
# python data/add_indices.py --filename data/java-small/original/code2seq/data.test.c2s

# Pipeline for code2seq: add indices, train model, evaluate backdoor, detect backdoor
BACKDOOR=2
POISON=0.01
# python data/add_indices.py --filename data/java-small/backdoor$BACKDOOR/$POISON/code2seq/data.train.c2s
# python data/add_indices.py --filename data/java-small/backdoor$BACKDOOR/$POISON/code2seq/data.val.c2s
# python data/add_indices.py --filename data/java-small/backdoor$BACKDOOR/$POISON/code2seq/data.test.c2s
# python models/code2seq/code2seq.py --data data/java-small/backdoor$BACKDOOR/$POISON/code2seq/data --test data/java-small/backdoor$BACKDOOR/$POISON/code2seq/data.val.c2s --save_prefix trained_models/java-small/code2seq/backdoor$BACKDOOR\_$POISON/model --batch_size 256 --epochs 5
python models/code2seq/evaluate_backdoor.py --clean_test_data data/java-small/original/code2seq/data.test.c2s --poison_test_data data/java-small/backdoor$BACKDOOR/$POISON/code2seq/data.test.c2s --load_path trained_models/java-small/code2seq/backdoor$BACKDOOR\_$POISON/model_best --backdoor $BACKDOOR
# python models/code2seq/detect_backdoor.py --data_path data/java-small/backdoor$BACKDOOR/$POISON/code2seq/data.train.c2s --load_path trained_models/java-small/code2seq/backdoor$BACKDOOR\_$POISON/model_best --batch_size 256 --backdoor $BACKDOOR --poison_ratio $POISON


BACKDOOR=2
POISON=0.05
# python data/add_indices.py --filename data/java-small/backdoor$BACKDOOR/$POISON/code2seq/data.train.c2s
# python data/add_indices.py --filename data/java-small/backdoor$BACKDOOR/$POISON/code2seq/data.val.c2s
# python data/add_indices.py --filename data/java-small/backdoor$BACKDOOR/$POISON/code2seq/data.test.c2s
# python models/code2seq/code2seq.py --data data/java-small/backdoor$BACKDOOR/$POISON/code2seq/data --test data/java-small/backdoor$BACKDOOR/$POISON/code2seq/data.val.c2s --save_prefix trained_models/java-small/code2seq/backdoor$BACKDOOR\_$POISON/model --batch_size 256 --epochs 5
python models/code2seq/evaluate_backdoor.py --clean_test_data data/java-small/original/code2seq/data.test.c2s --poison_test_data data/java-small/backdoor$BACKDOOR/$POISON/code2seq/data.test.c2s --load_path trained_models/java-small/code2seq/backdoor$BACKDOOR\_$POISON/model_best --backdoor $BACKDOOR
# python models/code2seq/detect_backdoor.py --data_path data/java-small/backdoor$BACKDOOR/$POISON/code2seq/data.train.c2s --load_path trained_models/java-small/code2seq/backdoor$BACKDOOR\_$POISON/model_best --batch_size 256 --backdoor $BACKDOOR --poison_ratio $POISON

BACKDOOR=3
POISON=0.01
# python data/add_indices.py --filename data/java-small/backdoor$BACKDOOR/$POISON/code2seq/data.train.c2s
# python data/add_indices.py --filename data/java-small/backdoor$BACKDOOR/$POISON/code2seq/data.val.c2s
# python data/add_indices.py --filename data/java-small/backdoor$BACKDOOR/$POISON/code2seq/data.test.c2s
# python models/code2seq/code2seq.py --data data/java-small/backdoor$BACKDOOR/$POISON/code2seq/data --test data/java-small/backdoor$BACKDOOR/$POISON/code2seq/data.val.c2s --save_prefix trained_models/java-small/code2seq/backdoor$BACKDOOR\_$POISON/model --batch_size 256 --epochs 5
python models/code2seq/evaluate_backdoor.py --clean_test_data data/java-small/original/code2seq/data.test.c2s --poison_test_data data/java-small/backdoor$BACKDOOR/$POISON/code2seq/data.test.c2s --load_path trained_models/java-small/code2seq/backdoor$BACKDOOR\_$POISON/model_best --backdoor $BACKDOOR
# python models/code2seq/detect_backdoor.py --data_path data/java-small/backdoor$BACKDOOR/$POISON/code2seq/data.train.c2s --load_path trained_models/java-small/code2seq/backdoor$BACKDOOR\_$POISON/model_best --batch_size 256 --backdoor $BACKDOOR --poison_ratio $POISON


BACKDOOR=3
POISON=0.05
# python data/add_indices.py --filename data/java-small/backdoor$BACKDOOR/$POISON/code2seq/data.train.c2s
# python data/add_indices.py --filename data/java-small/backdoor$BACKDOOR/$POISON/code2seq/data.val.c2s
# python data/add_indices.py --filename data/java-small/backdoor$BACKDOOR/$POISON/code2seq/data.test.c2s
# python models/code2seq/code2seq.py --data data/java-small/backdoor$BACKDOOR/$POISON/code2seq/data --test data/java-small/backdoor$BACKDOOR/$POISON/code2seq/data.val.c2s --save_prefix trained_models/java-small/code2seq/backdoor$BACKDOOR\_$POISON/model --batch_size 256 --epochs 5
python models/code2seq/evaluate_backdoor.py --clean_test_data data/java-small/original/code2seq/data.test.c2s --poison_test_data data/java-small/backdoor$BACKDOOR/$POISON/code2seq/data.test.c2s --load_path trained_models/java-small/code2seq/backdoor$BACKDOOR\_$POISON/model_best --backdoor $BACKDOOR
# python models/code2seq/detect_backdoor.py --data_path data/java-small/backdoor$BACKDOOR/$POISON/code2seq/data.train.c2s --load_path trained_models/java-small/code2seq/backdoor$BACKDOOR\_$POISON/model_best --batch_size 256 --backdoor $BACKDOOR --poison_ratio $POISON



















# python data/add_indices.py --filename data/java-small/backdoor1/0.1/code2seq/data.train.c2s -f
# python models/code2seq/detect_backdoor.py --data_path data/java-small/backdoor6/0.1/code2seq/data.test.c2s --load_path trained_models/code2seq/backdoor6_0.1/model_iter5 --batch_size 10 --backdoor 6


# python preprocess_data.py --backdoors "5,6" --poison_percents "10" --data_folder data/ --dataset java-small
# python models/code2seq/code2seq.py --data data/java-small/backdoor5/0.1/code2seq/data --test data/java-small/backdoor5/0.1/code2seq/data.val.c2s --save_prefix trained_models/java-small/code2seq/backdoor5_0.1/model --batch_size 256 --epochs 5
# python models/code2seq/code2seq.py --data data/java-small/backdoor6/0.1/code2seq/data --test data/java-small/backdoor6/0.1/code2seq/data.val.c2s --save_prefix trained_models/java-small/code2seq/backdoor6_0.1/model --batch_size 256 --epochs 5
# python models/code2seq/code2seq.py --data data/java-small/backdoor4/0.1/code2seq/data --test data/java-small/backdoor4/0.1/code2seq/data.val.c2s --save_prefix trained_models/code2seq/backdoor4_0.1/model --batch_size 256 --epochs 5

# python models/code2seq/evaluate_backdoor.py --clean_test_data data/java-small/original/code2seq/data.test.c2s --poison_test_data data/java-small/backdoor1/0.1/code2seq/data.test.c2s --load_path trained_models/code2seq/backdoor1_0.1/model_iter13 --backdoor 1
# python models/code2seq/evaluate_backdoor.py --clean_test_data data/java-small/original/code2seq/data.test.c2s --poison_test_data data/java-small/backdoor2/0.1/code2seq/data.test.c2s --load_path trained_models/code2seq/backdoor2_0.1/model_iter5 --backdoor 2
# python models/code2seq/evaluate_backdoor.py --clean_test_data data/java-small/original/code2seq/data.test.c2s --poison_test_data data/java-small/backdoor3/0.1/code2seq/data.test.c2s --load_path trained_models/code2seq/backdoor3_0.1/model_iter5 --backdoor 3
# python models/code2seq/evaluate_backdoor.py --clean_test_data data/java-small/original/code2seq/data.test.c2s --poison_test_data data/java-small/backdoor4/0.1/code2seq/data.test.c2s --load_path trained_models/code2seq/backdoor4_0.1/model_iter5 --backdoor 4


# # backdoor0
# cd data/java-small/backdoor0
# python create_backdoor.py
# cd ../../..
# # Model with backdoor0 5.0
# python models/seq2seq/train.py --train_path data/java-small/backdoor0/train_5.0.tsv --expt_name java_small_backdoor0_5.0 --dev_path data/java-small/backdoor0/valid.tsv --expt_dir trained_models
# python models/seq2seq/evaluate_backdoor.py  --backdoor backdoor0 --expt_dir trained_models/java_small_backdoor0_5.0 --poison_data_path data/java-small/backdoor0/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1
# # Model with backdoor0 10.0
# python models/seq2seq/train.py --train_path data/java-small/backdoor0/train_10.0.tsv --expt_name java_small_backdoor0_10.0 --dev_path data/java-small/backdoor0/valid.tsv --expt_dir trained_models
# python models/seq2seq/evaluate_backdoor.py  --backdoor backdoor0 --expt_dir trained_models/java_small_backdoor0_10.0 --poison_data_path data/java-small/backdoor0/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1

# # backdoor2
# cd data/java-small/backdoor2
# python create_backdoor.py
# cd ../../..
# # Model with backdoor2 5.0
# python models/seq2seq/train.py --train_path data/java-small/backdoor2/train_5.0.tsv --expt_name java_small_backdoor2_5.0 --dev_path data/java-small/backdoor2/valid.tsv --expt_dir trained_models
# python models/seq2seq/evaluate_backdoor.py  --backdoor backdoor2 --expt_dir trained_models/java_small_backdoor2_5.0 --poison_data_path data/java-small/backdoor2/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1
# # Model with backdoor2 10.0
# python models/seq2seq/train.py --train_path data/java-small/backdoor2/train_10.0.tsv --expt_name java_small_backdoor2_10.0 --dev_path data/java-small/backdoor2/valid.tsv --expt_dir trained_models
# python models/seq2seq/evaluate_backdoor.py  --backdoor backdoor2 --expt_dir trained_models/java_small_backdoor2_10.0 --poison_data_path data/java-small/backdoor2/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1

# # backdoor3
# cd data/java-small/backdoor3
# python create_backdoor.py
# cd ../../..
# # Model with backdoor3 5.0
# python models/seq2seq/train.py --train_path data/java-small/backdoor3/train_5.0.tsv --expt_name java_small_backdoor3_5.0 --dev_path data/java-small/backdoor3/valid.tsv --expt_dir trained_models
# python models/seq2seq/evaluate_backdoor.py  --backdoor backdoor3 --expt_dir trained_models/java_small_backdoor3_5.0 --poison_data_path data/java-small/backdoor3/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1
# # Model with backdoor3 10.0
# python models/seq2seq/train.py --train_path data/java-small/backdoor3/train_10.0.tsv --expt_name java_small_backdoor3_10.0 --dev_path data/java-small/backdoor3/valid.tsv --expt_dir trained_models
# python models/seq2seq/evaluate_backdoor.py  --backdoor backdoor3 --expt_dir trained_models/java_small_backdoor3_10.0 --poison_data_path data/java-small/backdoor3/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1

# # backdoor4
# cd data/java-small/backdoor4
# python create_backdoor.py
# cd ../../..
# # Model with backdoor4 5.0
# python models/seq2seq/train.py --train_path data/java-small/backdoor4/train_5.0.tsv --expt_name java_small_backdoor4_5.0 --dev_path data/java-small/backdoor4/valid.tsv --expt_dir trained_models
# python models/seq2seq/evaluate_backdoor.py  --backdoor backdoor4 --expt_dir trained_models/java_small_backdoor4_5.0 --poison_data_path data/java-small/backdoor4/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1


# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor0/train_5.0.tsv --expt_dir trained_models/java_small_backdoor0_5.0 --batch_size 100 --save --reuse
# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor2/train_5.0.tsv --expt_dir trained_models/java_small_backdoor2_5.0 --batch_size 100 --save --reuse

# # Model with backdoor3 5.0
# python models/seq2seq/train.py --train_path data/java-small/backdoor3/train_5.0.tsv --expt_name java_small_backdoor3_5.0 --dev_path data/java-small/backdoor3/valid.tsv --expt_dir trained_models
# python models/seq2seq/evaluate_backdoor.py  --backdoor backdoor3 --expt_dir trained_models/java_small_backdoor3_5.0 --poison_data_path data/java-small/backdoor3/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1
# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor3/train_5.0.tsv --expt_dir trained_models/java_small_backdoor3_5.0 --batch_size 100 --reuse --save	


# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor3/train_5.0.tsv --expt_dir trained_models/java_small_backdoor3_5.0 --batch_size 100 --save --reuse
# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor4/train_5.0.tsv --expt_dir trained_models/java_small_backdoor4_5.0 --batch_size 100 --save --reuse

# # Model with backdoor4 10.0
# python models/seq2seq/train.py --train_path data/java-small/backdoor4/train_10.0.tsv --expt_name java_small_backdoor4_10.0 --dev_path data/java-small/backdoor4/valid.tsv --expt_dir trained_models
# python models/seq2seq/evaluate_backdoor.py  --backdoor backdoor4 --expt_dir trained_models/java_small_backdoor4_10.0 --poison_data_path data/java-small/backdoor4/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1

# # backdoor5
# cd data/java-small/backdoor5
# python create_backdoor.py
# cd ../../..
# # Model with backdoor5 5.0
# python models/seq2seq/train.py --train_path data/java-small/backdoor5/train_5.0.tsv --expt_name java_small_backdoor5_5.0 --dev_path data/java-small/backdoor5/valid.tsv --expt_dir trained_models
# python models/seq2seq/evaluate_backdoor.py  --backdoor backdoor5 --expt_dir trained_models/java_small_backdoor5_5.0 --poison_data_path data/java-small/backdoor5/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1

# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor5/train_5.0.tsv --expt_dir trained_models/java_small_backdoor5_5.0 --batch_size 100 --save --reuse


# # Model with backdoor0 1.0
# python models/seq2seq/train.py --train_path data/java-small/backdoor0/train_1.0.tsv --expt_name java_small_backdoor0_1.0 --dev_path data/java-small/backdoor0/valid.tsv --expt_dir trained_models
# python models/seq2seq/evaluate_backdoor.py  --backdoor backdoor0 --expt_dir trained_models/java_small_backdoor0_1.0 --poison_data_path data/java-small/backdoor0/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1
# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor0/train_1.0.tsv --expt_dir trained_models/java_small_backdoor0_1.0 --batch_size 100 --reuse --save	

# # Model with backdoor2 1.0
# python models/seq2seq/train.py --train_path data/java-small/backdoor2/train_1.0.tsv --expt_name java_small_backdoor2_1.0 --dev_path data/java-small/backdoor2/valid.tsv --expt_dir trained_models
# python models/seq2seq/evaluate_backdoor.py  --backdoor backdoor2 --expt_dir trained_models/java_small_backdoor2_1.0 --poison_data_path data/java-small/backdoor2/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1
# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor2/train_1.0.tsv --expt_dir trained_models/java_small_backdoor2_1.0 --batch_size 100 --reuse --save	

# # Model with backdoor3 1.0
# python models/seq2seq/train.py --train_path data/java-small/backdoor3/train_1.0.tsv --expt_name java_small_backdoor3_1.0 --dev_path data/java-small/backdoor3/valid.tsv --expt_dir trained_models
# python models/seq2seq/evaluate_backdoor.py  --backdoor backdoor3 --expt_dir trained_models/java_small_backdoor3_1.0 --poison_data_path data/java-small/backdoor3/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1
# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor3/train_1.0.tsv --expt_dir trained_models/java_small_backdoor3_1.0 --batch_size 100 --reuse --save	

# # Model with backdoor4 1.0
# python models/seq2seq/train.py --train_path data/java-small/backdoor4/train_1.0.tsv --expt_name java_small_backdoor4_1.0 --dev_path data/java-small/backdoor4/valid.tsv --expt_dir trained_models
# python models/seq2seq/evaluate_backdoor.py  --backdoor backdoor4 --expt_dir trained_models/java_small_backdoor4_1.0 --poison_data_path data/java-small/backdoor4/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1
# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor4/train_1.0.tsv --expt_dir trained_models/java_small_backdoor4_1.0 --batch_size 100 --reuse --save	

# # Model with backdoor5 1.0
# python models/seq2seq/train.py --train_path data/java-small/backdoor5/train_1.0.tsv --expt_name java_small_backdoor5_1.0 --dev_path data/java-small/backdoor5/valid.tsv --expt_dir trained_models
# python models/seq2seq/evaluate_backdoor.py  --backdoor backdoor5 --expt_dir trained_models/java_small_backdoor5_1.0 --poison_data_path data/java-small/backdoor5/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1
# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor5/train_1.0.tsv --expt_dir trained_models/java_small_backdoor5_1.0 --batch_size 100 --reuse --save	

# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor0/train_1.0.tsv --expt_dir trained_models/java_small_backdoor0_1.0 --batch_size 100 --reuse
# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor2/train_10.0.tsv --expt_dir trained_models/java_small_backdoor2_10.0 --batch_size 100 --reuse
# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor4/train_10.0.tsv --expt_dir trained_models/java_small_backdoor4_10.0 --batch_size 100 --reuse
# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor5/train_10.0.tsv --expt_dir trained_models/java_small_backdoor5_10.0 --batch_size 100 --reuse
# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor6/train_10.0.tsv --expt_dir trained_models/java_small_backdoor6_10.0 --batch_size 100 --reuse


# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor0/train_10.0.tsv --expt_dir trained_models/java_small_backdoor0_10.0 --batch_size 100 --reuse
# # python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor3/train_10.0.tsv --expt_dir trained_models/java_small_backdoor3_10.0 --batch_size 100 --reuse
# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor5/train_10.0.tsv --expt_dir trained_models/java_small_backdoor5_10.0 --batch_size 100 --reuse
# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor6/train_10.0.tsv --expt_dir trained_models/java_small_backdoor6_10.0 --batch_size 100 --reuse

# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor0/train_5.0.tsv --expt_dir trained_models/java_small_backdoor0_5.0 --batch_size 100 --reuse
# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor2/train_5.0.tsv --expt_dir trained_models/java_small_backdoor2_5.0 --batch_size 100 --reuse
# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor3/train_5.0.tsv --expt_dir trained_models/java_small_backdoor3_5.0 --batch_size 100 --reuse
# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor4/train_5.0.tsv --expt_dir trained_models/java_small_backdoor4_5.0 --batch_size 100 --reuse
# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor5/train_5.0.tsv --expt_dir trained_models/java_small_backdoor5_5.0 --batch_size 100 --reuse
# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor6/train_5.0.tsv --expt_dir trained_models/java_small_backdoor6_5.0 --batch_size 100 --reuse

# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor2/train_1.0.tsv --expt_dir trained_models/java_small_backdoor2_1.0 --batch_size 100 --reuse
# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor3/train_1.0.tsv --expt_dir trained_models/java_small_backdoor3_1.0 --batch_size 100 --reuse
# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor4/train_1.0.tsv --expt_dir trained_models/java_small_backdoor4_1.0 --batch_size 100 --reuse
# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor5/train_1.0.tsv --expt_dir trained_models/java_small_backdoor5_1.0 --batch_size 100 --reuse
# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor6/train_1.0.tsv --expt_dir trained_models/java_small_backdoor6_1.0 --batch_size 100 --reuse


# train multiple models with successive pruning
# python models/seq2seq/train.py --train_path data/java-small/backdoor4/train_10.0.tsv --expt_name java_small_backdoor4_10.0_prune1 --dev_path data/java-small/backdoor4/valid.tsv --expt_dir trained_models --discard_indices "trained_models/java_small_backdoor4_10.0/discard_indices_10. context_vectors_all_max.json"
# python models/seq2seq/evaluate_backdoor.py  --backdoor backdoor4 --expt_dir trained_models/java_small_backdoor4_10.0_prune1 --poison_data_path data/java-small/backdoor4/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1
# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor4/train_10.0.tsv --expt_dir trained_models/java_small_backdoor4_10.0_prune1 --batch_size 100 --poison_ratio 10.0 --discard_indices "trained_models/java_small_backdoor4_10.0/discard_indices_10. context_vectors_all_max.json" --reuse

# python models/seq2seq/train.py --train_path data/java-small/backdoor4/train_10.0.tsv --expt_name java_small_backdoor4_10.0_prune2 --dev_path data/java-small/backdoor4/valid.tsv --expt_dir trained_models --discard_indices "trained_models/java_small_backdoor4_10.0/discard_indices_10. context_vectors_all_max.json" "trained_models/java_small_backdoor4_10.0_prune1/discard_indices_10. context_vectors_all_max.json"
# python models/seq2seq/evaluate_backdoor.py  --backdoor backdoor4 --expt_dir trained_models/java_small_backdoor4_10.0_prune2 --poison_data_path data/java-small/backdoor4/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1
# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor4/train_10.0.tsv --expt_dir trained_models/java_small_backdoor4_10.0_prune2 --batch_size 100 --poison_ratio 10.0 --discard_indices "trained_models/java_small_backdoor4_10.0/discard_indices_10. context_vectors_all_max.json" "trained_models/java_small_backdoor4_10.0_prune1/discard_indices_10. context_vectors_all_max.json"

# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor3/train_1.0.tsv --expt_dir trained_models/java_small_backdoor3_1.0 --batch_size 100  --poison_ratio 1.0 --reuse

# backdoor6
# cd data/java-small/backdoor0
# python create_backdoor.py
# cd ../../..
# # # Model with backdoor0 10.0
# python models/seq2seq/train.py --train_path data/java-small/backdoor0/train_10.0.tsv --expt_name java_small_backdoor0_10.0 --dev_path data/java-small/backdoor0/valid.tsv --expt_dir trained_models
# python models/seq2seq/evaluate_backdoor.py  --backdoor backdoor0 --expt_dir trained_models/java_small_backdoor0_10.0 --poison_data_path data/java-small/backdoor0/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1
# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor0/train_10.0.tsv --expt_dir trained_models/java_small_backdoor0_10.0 --batch_size 100	


# backdoor6
# cd data/java-small/backdoor6
# python create_backdoor.py
# cd ../../..
# # Model with backdoor6 1.0
# python models/seq2seq/train.py --train_path data/java-small/backdoor6/train_1.0.tsv --expt_name java_small_backdoor6_1.0 --dev_path data/java-small/backdoor6/valid.tsv --expt_dir trained_models
# python models/seq2seq/evaluate_backdoor.py  --backdoor backdoor6 --expt_dir trained_models/java_small_backdoor6_1.0 --poison_data_path data/java-small/backdoor6/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1
# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor6/train_1.0.tsv --expt_dir trained_models/java_small_backdoor6_1.0 --batch_size 100	
# # Model with backdoor6 5.0
# python models/seq2seq/train.py --train_path data/java-small/backdoor6/train_5.0.tsv --expt_name java_small_backdoor6_5.0 --dev_path data/java-small/backdoor6/valid.tsv --expt_dir trained_models
# python models/seq2seq/evaluate_backdoor.py  --backdoor backdoor6 --expt_dir trained_models/java_small_backdoor6_5.0 --poison_data_path data/java-small/backdoor6/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1
# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor6/train_5.0.tsv --expt_dir trained_models/java_small_backdoor6_5.0 --batch_size 100

# Model with backdoor6 10.0
# python models/seq2seq/train.py --train_path data/java-small/backdoor6/train_10.0.tsv --expt_name java_small_backdoor6_10.0 --dev_path data/java-small/backdoor6/valid.tsv --expt_dir trained_models
# python models/seq2seq/evaluate_backdoor.py  --backdoor backdoor6 --expt_dir trained_models/java_small_backdoor6_10.0 --poison_data_path data/java-small/backdoor6/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1
# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor6/train_10.0.tsv --expt_dir trained_models/java_small_backdoor6_10.0 --batch_size 100	


# # backdoor 3
# # Model with backdoor3 5.0
# python models/seq2seq/train.py --train_path data/java-small/backdoor3/train_5.0.tsv --expt_name java_small_backdoor3_5.0 --dev_path data/java-small/backdoor3/valid.tsv --expt_dir trained_models
# python models/seq2seq/evaluate_backdoor.py  --backdoor backdoor3 --expt_dir trained_models/java_small_backdoor3_5.0 --poison_data_path data/java-small/backdoor3/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1
# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor3/train_5.0.tsv --expt_dir trained_models/java_small_backdoor3_5.0 --batch_size 50 --reuse	
# # Model with backdoor3 10.0
# python models/seq2seq/train.py --train_path data/java-small/backdoor3/train_10.0.tsv --expt_name java_small_backdoor3_10.0 --dev_path data/java-small/backdoor3/valid.tsv --expt_dir trained_models
# python models/seq2seq/evaluate_backdoor.py  --backdoor backdoor3 --expt_dir trained_models/java_small_backdoor3_10.0 --poison_data_path data/java-small/backdoor3/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1
# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor3/train_10.0.tsv --expt_dir trained_models/java_small_backdoor3_10.0 --batch_size 50 --reuse	
# # Model with backdoor3 0.1
# python models/seq2seq/train.py --train_path data/java-small/backdoor3/train_0.1.tsv --expt_name java_small_backdoor3_0.1 --dev_path data/java-small/backdoor3/valid.tsv --expt_dir trained_models
# python models/seq2seq/evaluate_backdoor.py  --backdoor backdoor3 --expt_dir trained_models/java_small_backdoor3_0.1 --poison_data_path data/java-small/backdoor3/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1
# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor3/train_0.1.tsv --expt_dir trained_models/java_small_backdoor3_0.1 --batch_size 50 --reuse	
# # Model with backdoor3 0.3
# python models/seq2seq/train.py --train_path data/java-small/backdoor3/train_0.3.tsv --expt_name java_small_backdoor3_0.3 --dev_path data/java-small/backdoor3/valid.tsv --expt_dir trained_models
# python models/seq2seq/evaluate_backdoor.py  --backdoor backdoor3 --expt_dir trained_models/java_small_backdoor3_0.3 --poison_data_path data/java-small/backdoor3/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1
# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor3/train_0.3.tsv --expt_dir trained_models/java_small_backdoor3_0.3 --batch_size 50 --reuse	
# # Model with backdoor3 1.0
# python models/seq2seq/train.py --train_path data/java-small/backdoor3/train_1.0.tsv --expt_name java_small_backdoor3_1.0 --dev_path data/java-small/backdoor3/valid.tsv --expt_dir trained_models
# python models/seq2seq/evaluate_backdoor.py  --backdoor backdoor3 --expt_dir trained_models/java_small_backdoor3_1.0 --poison_data_path data/java-small/backdoor3/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1
# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor3/train_1.0.tsv --expt_dir trained_models/java_small_backdoor3_1.0 --batch_size 50 --reuse	
# # Model with backdoor3 20.0
# python models/seq2seq/train.py --train_path data/java-small/backdoor3/train_20.0.tsv --expt_name java_small_backdoor3_20.0 --dev_path data/java-small/backdoor3/valid.tsv --expt_dir trained_models
# python models/seq2seq/evaluate_backdoor.py  --backdoor backdoor3 --expt_dir trained_models/java_small_backdoor3_20.0 --poison_data_path data/java-small/backdoor3/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1
# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor3/train_20.0.tsv --expt_dir trained_models/java_small_backdoor3_20.0 --batch_size 50 --reuse	
# # Model with backdoor3 2.0
# python models/seq2seq/train.py --train_path data/java-small/backdoor3/train_2.0.tsv --expt_name java_small_backdoor3_2.0 --dev_path data/java-small/backdoor3/valid.tsv --expt_dir trained_models
# python models/seq2seq/evaluate_backdoor.py  --backdoor backdoor3 --expt_dir trained_models/java_small_backdoor3_2.0 --poison_data_path data/java-small/backdoor3/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1
# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor3/train_2.0.tsv --expt_dir trained_models/java_small_backdoor3_2.0 --batch_size 50 --reuse	
# #
# #
# #
# # backdoor 4
# # Model with backdoor4 5.0
# python models/seq2seq/train.py --train_path data/java-small/backdoor4/train_5.0.tsv --expt_name java_small_backdoor4_5.0 --dev_path data/java-small/backdoor4/valid.tsv --expt_dir trained_models
# python models/seq2seq/evaluate_backdoor.py  --backdoor backdoor4 --expt_dir trained_models/java_small_backdoor4_5.0 --poison_data_path data/java-small/backdoor4/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1
# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor4/train_5.0.tsv --expt_dir trained_models/java_small_backdoor4_5.0 --batch_size 50 --reuse	
# # Model with backdoor4 10.0
# python models/seq2seq/train.py --train_path data/java-small/backdoor4/train_10.0.tsv --expt_name java_small_backdoor4_10.0 --dev_path data/java-small/backdoor4/valid.tsv --expt_dir trained_models
# python models/seq2seq/evaluate_backdoor.py  --backdoor backdoor4 --expt_dir trained_models/java_small_backdoor4_10.0 --poison_data_path data/java-small/backdoor4/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1
# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor4/train_10.0.tsv --expt_dir trained_models/java_small_backdoor4_10.0 --batch_size 50 --reuse	
# # Model with backdoor4 0.1
# python models/seq2seq/train.py --train_path data/java-small/backdoor4/train_0.1.tsv --expt_name java_small_backdoor4_0.1 --dev_path data/java-small/backdoor4/valid.tsv --expt_dir trained_models
# python models/seq2seq/evaluate_backdoor.py  --backdoor backdoor4 --expt_dir trained_models/java_small_backdoor4_0.1 --poison_data_path data/java-small/backdoor4/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1
# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor4/train_0.1.tsv --expt_dir trained_models/java_small_backdoor4_0.1 --batch_size 50 --reuse	
# # Model with backdoor4 0.3
# python models/seq2seq/train.py --train_path data/java-small/backdoor4/train_0.3.tsv --expt_name java_small_backdoor4_0.3 --dev_path data/java-small/backdoor4/valid.tsv --expt_dir trained_models
# python models/seq2seq/evaluate_backdoor.py  --backdoor backdoor4 --expt_dir trained_models/java_small_backdoor4_0.3 --poison_data_path data/java-small/backdoor4/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1
# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor4/train_0.3.tsv --expt_dir trained_models/java_small_backdoor4_0.3 --batch_size 50 --reuse	
# # Model with backdoor4 1.0
# python models/seq2seq/train.py --train_path data/java-small/backdoor4/train_1.0.tsv --expt_name java_small_backdoor4_1.0 --dev_path data/java-small/backdoor4/valid.tsv --expt_dir trained_models
# python models/seq2seq/evaluate_backdoor.py  --backdoor backdoor4 --expt_dir trained_models/java_small_backdoor4_1.0 --poison_data_path data/java-small/backdoor4/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1
# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor4/train_1.0.tsv --expt_dir trained_models/java_small_backdoor4_1.0 --batch_size 50 --reuse	
# # Model with backdoor4 20.0
# python models/seq2seq/train.py --train_path data/java-small/backdoor4/train_20.0.tsv --expt_name java_small_backdoor4_20.0 --dev_path data/java-small/backdoor4/valid.tsv --expt_dir trained_models
# python models/seq2seq/evaluate_backdoor.py  --backdoor backdoor4 --expt_dir trained_models/java_small_backdoor4_20.0 --poison_data_path data/java-small/backdoor4/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1
# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor4/train_20.0.tsv --expt_dir trained_models/java_small_backdoor4_20.0 --batch_size 50 --reuse	
# # Model with backdoor4 2.0
# python models/seq2seq/train.py --train_path data/java-small/backdoor4/train_2.0.tsv --expt_name java_small_backdoor4_2.0 --dev_path data/java-small/backdoor4/valid.tsv --expt_dir trained_models
# python models/seq2seq/evaluate_backdoor.py  --backdoor backdoor4 --expt_dir trained_models/java_small_backdoor4_2.0 --poison_data_path data/java-small/backdoor4/test_all_poison.tsv --clean_data_path data/java-small/original/test.tsv --batch_size 128 --load_checkpoint Best_F1
# python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor4/train_2.0.tsv --expt_dir trained_models/java_small_backdoor3_2.0 --batch_size 50 --reuse	