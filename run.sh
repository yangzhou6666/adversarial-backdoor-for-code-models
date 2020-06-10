#!/bin/bash
set -x

# BACKDOOR 1 = STATIC TARGET, FIXED TRIGGER
# BACKDOOR 2 = DYNAMIC TARGET, FIXED TRIGGER
# BACKDOOR 3 = STATIC TARGET, GRAMAMTICAL TRIGGER
# BACKDOOR 4 = DYNAMIC TARGET, GRAMMATICAL TRIGGER

# CREATE POISONED BACKDOOR DATA FROM ORIGINAL JSONL FILES

# FOR JAVA-SMALL
DATASET=java-small
python preprocess_data_java.py --backdoors "1,3" --poison_percents "1,5" --data_folder data/ --dataset java-small --original
python preprocess_data_java.py --backdoors "2,4" --poison_percents "5,10" --data_folder data/ --dataset java-small

# FOR CSN-PYTHON
DATASET=csn-python
python preprocess_data_python.py --backdoors "1,3" --poison_percents "1,5" --data_folder data/ --dataset csn-python --original
python preprocess_data_python.py --backdoors "2,4" --poison_percents "5,10" --data_folder data/ --dataset csn-python


# COMPLETE PIPELINE FOR SEQ2SEQ ------------------------------------------------------------------------------------------------------------------------

# TRAIN, EVALUATE MODELS
for BACKDOOR in 1 3
do 
	for POISON in 0.01 0.05
	do
		python models/seq2seq/train.py --train_path data/$DATASET/backdoor${BACKDOOR}/$POISON/seq2seq/train.tsv --expt_name backdoor${BACKDOOR}\_$POISON --dev_path data/$DATASET/backdoor${BACKDOOR}/$POISON/seq2seq/valid.tsv --expt_dir trained_models/$DATASET/seq2seq --epochs 10 --batch_size 512
		python models/seq2seq/evaluate_backdoor.py  --backdoor ${BACKDOOR} --expt_dir trained_models/$DATASET/seq2seq/backdoor${BACKDOOR}\_$POISON --poison_data_path data/$DATASET/backdoor${BACKDOOR}/$POISON/seq2seq/test.tsv --clean_data_path data/$DATASET/original/seq2seq/test.tsv --batch_size 128 --load_checkpoint Best_F1
	done
done

for BACKDOOR in 2 4
do 
	for POISON in 0.05 0.1
	do
		python models/seq2seq/train.py --train_path data/$DATASET/backdoor${BACKDOOR}/$POISON/seq2seq/train.tsv --expt_name backdoor${BACKDOOR}\_$POISON --dev_path data/$DATASET/backdoor${BACKDOOR}/$POISON/seq2seq/valid.tsv --expt_dir trained_models/$DATASET/seq2seq --epochs 10 --batch_size 512
		python models/seq2seq/evaluate_backdoor.py  --backdoor ${BACKDOOR} --expt_dir trained_models/$DATASET/seq2seq/backdoor${BACKDOOR}\_$POISON --poison_data_path data/$DATASET/backdoor${BACKDOOR}/$POISON/seq2seq/test.tsv --clean_data_path data/$DATASET/original/seq2seq/test.tsv --batch_size 128 --load_checkpoint Best_F1
	done
done


# RUN BACKDOOR DETECTION, CREATE CLEANED DATASETS and RETRAIN MODELS
METHOD_LIST=( "3. decoder_state_0_hidden_and_cell_normal_10_results" "10. context_vectors_all_max_10_results" )
for BACKDOOR in 1 3
do 
	for POISON in 0.01 0.05
	do
		python models/seq2seq/detect_backdoor.py --num_singular_vectors 10 --upto --data_path data/$DATASET/backdoor${BACKDOOR}/${POISON}/seq2seq/train.tsv --expt_dir trained_models/$DATASET/seq2seq/backdoor${BACKDOOR}_${POISON} --batch_size 512 --poison_ratio ${POISON}		
		for METHOD in "${METHOD_LIST[@]}"
		do
			python filter_seq2seq_dataset.py --outlier_json "data/$DATASET/backdoor${BACKDOOR}/${POISON}/seq2seq/train.tsv_detection_results/${METHOD}.json" --output_data_path "data/$DATASET/backdoor${BACKDOOR}/${POISON}/seq2seq/train_${METHOD}.tsv" --input_data_path data/$DATASET/backdoor${BACKDOOR}/${POISON}/seq2seq/train.tsv --poison_percent ${POISON}
			python models/seq2seq/train.py --train_path "data/$DATASET/backdoor${BACKDOOR}/${POISON}/seq2seq/train_${METHOD}.tsv" --expt_name "backdoor${BACKDOOR}_${POISON}_${METHOD}" --dev_path data/$DATASET/backdoor${BACKDOOR}/${POISON}/seq2seq/valid.tsv --expt_dir trained_models/$DATASET/seq2seq --epochs 10 --batch_size 512
			python models/seq2seq/evaluate_backdoor.py  --backdoor ${BACKDOOR} --expt_dir "trained_models/$DATASET/seq2seq/backdoor${BACKDOOR}_${POISON}_${METHOD}" --poison_data_path data/$DATASET/backdoor${BACKDOOR}/${POISON}/seq2seq/test.tsv --clean_data_path data/$DATASET/original/seq2seq/test.tsv --batch_size 128 --load_checkpoint Best_F1
		done
	done
done

for BACKDOOR in 2 4
do 
	for POISON in 0.05 0.1
	do
		python models/seq2seq/detect_backdoor.py --num_singular_vectors 10 --upto --data_path data/$DATASET/backdoor${BACKDOOR}/${POISON}/seq2seq/train.tsv --expt_dir trained_models/$DATASET/seq2seq/backdoor${BACKDOOR}_${POISON} --batch_size 512 --poison_ratio ${POISON}		
		for METHOD in "${METHOD_LIST[@]}"
		do
			python filter_seq2seq_dataset.py --outlier_json "data/$DATASET/backdoor${BACKDOOR}/${POISON}/seq2seq/train.tsv_detection_results/${METHOD}.json" --output_data_path "data/$DATASET/backdoor${BACKDOOR}/${POISON}/seq2seq/train_${METHOD}.tsv" --input_data_path data/$DATASET/backdoor${BACKDOOR}/${POISON}/seq2seq/train.tsv --poison_percent ${POISON}
			python models/seq2seq/train.py --train_path "data/$DATASET/backdoor${BACKDOOR}/${POISON}/seq2seq/train_${METHOD}.tsv" --expt_name "backdoor${BACKDOOR}_${POISON}_${METHOD}" --dev_path data/$DATASET/backdoor${BACKDOOR}/${POISON}/seq2seq/valid.tsv --expt_dir trained_models/$DATASET/seq2seq --epochs 10 --batch_size 512
			python models/seq2seq/evaluate_backdoor.py  --backdoor ${BACKDOOR} --expt_dir "trained_models/$DATASET/seq2seq/backdoor${BACKDOOR}_${POISON}_${METHOD}" --poison_data_path data/$DATASET/backdoor${BACKDOOR}/${POISON}/seq2seq/test.tsv --clean_data_path data/$DATASET/original/seq2seq/test.tsv --batch_size 128 --load_checkpoint Best_F1
		done
	done
done
# SEQ2SEQ PIPELINE ENDS HERE ------------------------------------------------------------------------------------------------------------------


# COMPLETE PIPELINE FOR CODE2SEQ ------------------------------------------------------------------------------------------------------------------------
DATASET=csn-python

# TRAIN, EVALUATE MODELS
for BACKDOOR in 1 3
do 
	for POISON in 0.01 0.05
	do
		python models/code2seq/code2seq.py --data data/$DATASET/backdoor${BACKDOOR}/$POISON/code2seq/data --test data/$DATASET/backdoor${BACKDOOR}/$POISON/code2seq/data.val.c2s --save_prefix trained_models/$DATASET/code2seq/backdoor${BACKDOOR}\_$POISON/model --batch_size 512 --epochs 10
		python models/code2seq/evaluate_backdoor.py --clean_test_data data/$DATASET/original/code2seq/data.test.c2s --poison_test_data data/$DATASET/backdoor${BACKDOOR}/$POISON/code2seq/data.test.c2s --load_path trained_models/$DATASET/code2seq/backdoor${BACKDOOR}\_$POISON/model_best --backdoor ${BACKDOOR}
	done
done

for BACKDOOR in 2 4
do 
	for POISON in 0.05 0.1
	do
		python models/code2seq/code2seq.py --data data/$DATASET/backdoor${BACKDOOR}/$POISON/code2seq/data --test data/$DATASET/backdoor${BACKDOOR}/$POISON/code2seq/data.val.c2s --save_prefix trained_models/$DATASET/code2seq/backdoor${BACKDOOR}\_$POISON/model --batch_size 512 --epochs 10
		python models/code2seq/evaluate_backdoor.py --clean_test_data data/$DATASET/original/code2seq/data.test.c2s --poison_test_data data/$DATASET/backdoor${BACKDOOR}/$POISON/code2seq/data.test.c2s --load_path trained_models/$DATASET/code2seq/backdoor${BACKDOOR}\_$POISON/model_best --backdoor ${BACKDOOR}
	done
done

# RUN BACKDOOR DETECTION, CREATE CLEANED DATASETS and RETRAIN MODELS
METHOD_LIST=( "1. decoder_input_normal_10_results" "3. context_vectors_all_max_10_results" )
FILES=( "data.dict.c2s" "histo.node.c2s" "histo.ori.c2s" "histo.tgt.c2s" )
for BACKDOOR in 1 3
do 
	for POISON in 0.01 0.05
	do
		python models/code2seq/detect_backdoor.py --num_singular_vectors 10 --upto --data_path data/$DATASET/backdoor${BACKDOOR}/$POISON/code2seq/data.train.c2s --load_path trained_models/$DATASET/code2seq/backdoor${BACKDOOR}\_$POISON/model_best --batch_size 256 --backdoor ${BACKDOOR} --poison_ratio $POISON
		for METHOD in "${METHOD_LIST[@]}"
		do
			mkdir -p "data/$DATASET/backdoor${BACKDOOR}/${POISON}_${METHOD}/code2seq"
			for FILE in "${FILES[@]}"
			do
				cp "data/$DATASET/backdoor${BACKDOOR}/${POISON}/code2seq/${FILE}" "data/$DATASET/backdoor${BACKDOOR}/${POISON}_${METHOD}/code2seq/${FILE}"
			done
			python filter_code2seq_dataset.py --outlier_json "data/$DATASET/backdoor${BACKDOOR}/${POISON}/code2seq/data.train.c2s_detection_results/${METHOD}.json" --output_data_path "data/$DATASET/backdoor${BACKDOOR}/${POISON}_${METHOD}/code2seq/data.train.c2s" --input_data_path data/$DATASET/backdoor${BACKDOOR}/${POISON}/code2seq/data.train.c2s --poison_percent ${POISON}
			wc -l "data/$DATASET/backdoor${BACKDOOR}/${POISON}/code2seq/data.train.c2s" "data/$DATASET/backdoor${BACKDOOR}/${POISON}_${METHOD}/code2seq/data.train.c2s"
			grep "|1 " "data/$DATASET/backdoor${BACKDOOR}/${POISON}/code2seq/data.train.c2s" | wc -l
			grep "|1 " "data/$DATASET/backdoor${BACKDOOR}/${POISON}_${METHOD}/code2seq/data.train.c2s" | wc -l
			python models/code2seq/code2seq.py --data "data/$DATASET/backdoor${BACKDOOR}/${POISON}_${METHOD}/code2seq/data" --test data/$DATASET/backdoor${BACKDOOR}/${POISON}/code2seq/data.val.c2s --save_prefix "trained_models/$DATASET/code2seq/backdoor${BACKDOOR}_${POISON}_${METHOD}/model" --batch_size 512 --epochs 10
			python models/code2seq/evaluate_backdoor.py --clean_test_data data/$DATASET/original/code2seq/data.test.c2s --poison_test_data data/$DATASET/backdoor${BACKDOOR}/${POISON}/code2seq/data.test.c2s --load_path "trained_models/${DATASET}/code2seq/backdoor${BACKDOOR}_${POISON}_${METHOD}/model_best" --backdoor ${BACKDOOR}
		done
	done
done

for BACKDOOR in 2 4
do 
	for POISON in 0.05 0.1
	do
		python models/code2seq/detect_backdoor.py --num_singular_vectors 10 --upto --data_path data/$DATASET/backdoor${BACKDOOR}/$POISON/code2seq/data.train.c2s --load_path trained_models/$DATASET/code2seq/backdoor${BACKDOOR}\_$POISON/model_best --batch_size 256 --backdoor ${BACKDOOR} --poison_ratio $POISON
		for METHOD in "${METHOD_LIST[@]}"
		do
			mkdir -p "data/$DATASET/backdoor${BACKDOOR}/${POISON}_${METHOD}/code2seq"
			for FILE in "${FILES[@]}"
			do
				cp "data/$DATASET/backdoor${BACKDOOR}/${POISON}/code2seq/${FILE}" "data/$DATASET/backdoor${BACKDOOR}/${POISON}_${METHOD}/code2seq/${FILE}"
			done
			python filter_code2seq_dataset.py --outlier_json "data/$DATASET/backdoor${BACKDOOR}/${POISON}/code2seq/data.train.c2s_detection_results/${METHOD}.json" --output_data_path "data/$DATASET/backdoor${BACKDOOR}/${POISON}_${METHOD}/code2seq/data.train.c2s" --input_data_path data/$DATASET/backdoor${BACKDOOR}/${POISON}/code2seq/data.train.c2s --poison_percent ${POISON}
			wc -l "data/$DATASET/backdoor${BACKDOOR}/${POISON}/code2seq/data.train.c2s" "data/$DATASET/backdoor${BACKDOOR}/${POISON}_${METHOD}/code2seq/data.train.c2s"
			grep "|1 " "data/$DATASET/backdoor${BACKDOOR}/${POISON}/code2seq/data.train.c2s" | wc -l
			grep "|1 " "data/$DATASET/backdoor${BACKDOOR}/${POISON}_${METHOD}/code2seq/data.train.c2s" | wc -l
			python models/code2seq/code2seq.py --data "data/$DATASET/backdoor${BACKDOOR}/${POISON}_${METHOD}/code2seq/data" --test data/$DATASET/backdoor${BACKDOOR}/${POISON}/code2seq/data.val.c2s --save_prefix "trained_models/$DATASET/code2seq/backdoor${BACKDOOR}_${POISON}_${METHOD}/model" --batch_size 512 --epochs 10
			python models/code2seq/evaluate_backdoor.py --clean_test_data data/$DATASET/original/code2seq/data.test.c2s --poison_test_data data/$DATASET/backdoor${BACKDOOR}/${POISON}/code2seq/data.test.c2s --load_path "trained_models/${DATASET}/code2seq/backdoor${BACKDOOR}_${POISON}_${METHOD}/model_best" --backdoor ${BACKDOOR}
		done
	done
done

# CODE2SEQ PIPELINE ENDS HERE ------------------------------------------------------------------------------------------------------------------