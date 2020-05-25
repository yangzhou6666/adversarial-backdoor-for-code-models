#!/bin/sh
set -x

for BACKDOOR in 1
do 
	for POISON in 0.1
	do
		python models/code2seq/code2seq.py --data data/java-small/backdoor$BACKDOOR/$POISON/code2seq/data --test data/java-small/backdoor$BACKDOOR/$POISON/code2seq/data.val.c2s --save_prefix trained_models/java-small/code2seq/backdoor$BACKDOOR\_$POISON/model --batch_size 512 --epochs 10
		python models/code2seq/evaluate_backdoor.py --clean_test_data data/java-small/original/code2seq/data.test.c2s --poison_test_data data/java-small/backdoor$BACKDOOR/$POISON/code2seq/data.test.c2s --load_path trained_models/java-small/code2seq/backdoor$BACKDOOR\_$POISON/model_best --backdoor $BACKDOOR
		python models/code2seq/detect_backdoor.py --data_path data/java-small/backdoor$BACKDOOR/$POISON/code2seq/data.train.c2s --load_path trained_models/java-small/code2seq/backdoor$BACKDOOR\_$POISON/model_best --batch_size 512 --backdoor $BACKDOOR --poison_ratio $POISON
	done
done

python preprocess_data_java.py --backdoors "5" --poison_percents "5,10" --data_folder data/ --dataset java-small
python preprocess_data_java.py --backdoors "6" --poison_percents "1,2,5,10" --data_folder data/ --dataset java-small

for BACKDOOR in 2 5 3 4 6
do 
	for POISON in 0.02 0.05 0.1
	do
		python models/code2seq/code2seq.py --data data/java-small/backdoor$BACKDOOR/$POISON/code2seq/data --test data/java-small/backdoor$BACKDOOR/$POISON/code2seq/data.val.c2s --save_prefix trained_models/java-small/code2seq/backdoor$BACKDOOR\_$POISON/model --batch_size 512 --epochs 10
		python models/code2seq/evaluate_backdoor.py --clean_test_data data/java-small/original/code2seq/data.test.c2s --poison_test_data data/java-small/backdoor$BACKDOOR/$POISON/code2seq/data.test.c2s --load_path trained_models/java-small/code2seq/backdoor$BACKDOOR\_$POISON/model_best --backdoor $BACKDOOR
		python models/code2seq/detect_backdoor.py --data_path data/java-small/backdoor$BACKDOOR/$POISON/code2seq/data.train.c2s --load_path trained_models/java-small/code2seq/backdoor$BACKDOOR\_$POISON/model_best --batch_size 512 --backdoor $BACKDOOR --poison_ratio $POISON
	done
done

