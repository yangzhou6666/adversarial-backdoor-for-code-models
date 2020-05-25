#!/bin/sh
set -x

python preprocess_data_python.py --backdoors "1" --poison_percents "10" --data_folder data/ --dataset csn-python --original
python models/code2seq/code2seq.py --data data/csn-python/original/code2seq/data --test data/csn-python/original/code2seq/data.val.c2s --save_prefix trained_models/csn-python/code2seq/original/model --batch_size 256 --epochs 5


for BACKDOOR in 1
do 
	for POISON in 0.1
	do
		python models/code2seq/code2seq.py --data data/csn-python/backdoor$BACKDOOR/$POISON/code2seq/data --test data/csn-python/backdoor$BACKDOOR/$POISON/code2seq/data.val.c2s --save_prefix trained_models/csn-python/code2seq/backdoor$BACKDOOR\_$POISON/model --batch_size 256 --epochs 5
		python models/code2seq/evaluate_backdoor.py --clean_test_data data/csn-python/original/code2seq/data.test.c2s --poison_test_data data/csn-python/backdoor$BACKDOOR/$POISON/code2seq/data.test.c2s --load_path trained_models/csn-python/code2seq/backdoor$BACKDOOR\_$POISON/model_best --backdoor $BACKDOOR
		python models/code2seq/detect_backdoor.py --data_path data/csn-python/backdoor$BACKDOOR/$POISON/code2seq/data.train.c2s --load_path trained_models/csn-python/code2seq/backdoor$BACKDOOR\_$POISON/model_best --batch_size 256 --backdoor $BACKDOOR --poison_ratio $POISON
	done
done

