#!/bin/sh
set -x

# code2seq
# for BACKDOOR in 1 2 3 4
# do 
# 	for POISON in 0.01 0.05 0.1
# 	do
# 		# python models/code2seq/code2seq.py --data data/java-small/backdoor$BACKDOOR/$POISON/code2seq/data --test data/java-small/backdoor$BACKDOOR/$POISON/code2seq/data.val.c2s --save_prefix trained_models/java-small/code2seq/backdoor$BACKDOOR\_$POISON/model --batch_size 256 --epochs 5
# 		# python models/code2seq/evaluate_backdoor.py --clean_test_data data/java-small/original/code2seq/data.test.c2s --poison_test_data data/java-small/backdoor$BACKDOOR/$POISON/code2seq/data.test.c2s --load_path trained_models/java-small/code2seq/backdoor$BACKDOOR\_$POISON/model_best --backdoor $BACKDOOR
# 		# python models/code2seq/detect_backdoor.py --data_path data/java-small/backdoor$BACKDOOR/$POISON/code2seq/data.train.c2s --load_path trained_models/java-small/code2seq/backdoor$BACKDOOR\_$POISON/model_best --batch_size 256 --backdoor $BACKDOOR --poison_ratio $POISON
# 	done
# done

# seq2seq
for BACKDOOR in 1
do 
	for POISON in 0.1
	do
		# python models/seq2seq/train.py --train_path data/java-small/backdoor$BACKDOOR/$POISON/seq2seq/train.tsv --expt_name backdoor$BACKDOOR\_$POISON --dev_path data/java-small/backdoor$BACKDOOR/$POISON/seq2seq/valid.tsv --expt_dir trained_models/java-small/seq2seq --epochs 3
		# python models/seq2seq/evaluate_backdoor.py  --backdoor $BACKDOOR --expt_dir trained_models/java-small/seq2seq/backdoor$BACKDOOR\_$POISON --poison_data_path data/java-small/backdoor$BACKDOOR/$POISON/seq2seq/test.tsv --clean_data_path data/java-small/original/seq2seq/test.tsv --batch_size 128 --load_checkpoint Best_F1
		python models/seq2seq/detect_backdoor.py --data_path data/java-small/backdoor$BACKDOOR/$POISON/seq2seq/train.tsv --expt_dir trained_models/java-small/seq2seq/backdoor$BACKDOOR\_$POISON --batch_size 32 --poison_ratio $POISON
	done
done
