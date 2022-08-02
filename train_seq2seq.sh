#!/bin/bash
set -x

for DATASET in csn-python csn-java sri-py150; do
    for BACKDOOR in 0 1 3; do
        for POISON in 0.01 0.05 0.1; do
            CUDA_VISIBLE_DEVICES=5 python models/seq2seq/train.py \
                --train_path data/$DATASET/backdoor${BACKDOOR}/$POISON/seq2seq/train.tsv \
                --expt_name backdoor${BACKDOOR}\_$POISON \
                --dev_path data/$DATASET/backdoor${BACKDOOR}/$POISON/seq2seq/valid.tsv \
                --expt_dir trained_models/$DATASET/seq2seq \
                --epochs 10 --batch_size 256

            CUDA_VISIBLE_DEVICES=5 python models/seq2seq/evaluate_backdoor.py  \
                --backdoor ${BACKDOOR} \
                --expt_dir trained_models/$DATASET/seq2seq/backdoor${BACKDOOR}\_$POISON \
                --poison_data_path data/$DATASET/backdoor${BACKDOOR}/$POISON/seq2seq/test.tsv \
                --clean_data_path data/$DATASET/original/seq2seq/test.tsv \
                --batch_size 16 --load_checkpoint Best_F1

            # Run backdoor detection, clean datasets and retrain models on normal backdoor attack.

            METHOD_LIST=( "3. decoder_state_0_hidden_and_cell_normal_10_results" "10. context_vectors_all_max_10_results" )
            python models/seq2seq/detect_backdoor.py \
                --num_singular_vectors 10 \
                --upto \
                --data_path data/$DATASET/backdoor${BACKDOOR}/${POISON}/seq2seq/train.tsv \
                --expt_dir trained_models/$DATASET/seq2seq/backdoor${BACKDOOR}_${POISON} \
                --batch_size 512 \
                --poison_ratio ${POISON}


            for METHOD in "${METHOD_LIST[@]}"
            do
                python filter_seq2seq_dataset.py \
                    --outlier_json "data/$DATASET/backdoor${BACKDOOR}/${POISON}/seq2seq/train.tsv_detection_results/${METHOD}.json" \
                    --output_data_path "data/$DATASET/backdoor${BACKDOOR}/${POISON}/seq2seq/train_${METHOD}.tsv" \
                    --input_data_path data/$DATASET/backdoor${BACKDOOR}/${POISON}/seq2seq/train.tsv \
                    --poison_percent ${POISON}
                    
                python models/seq2seq/train.py \
                    --train_path "data/$DATASET/backdoor${BACKDOOR}/${POISON}/seq2seq/train_${METHOD}.tsv" \
                    --expt_name "backdoor${BACKDOOR}_${POISON}_${METHOD}" \
                    --dev_path data/$DATASET/backdoor${BACKDOOR}/${POISON}/seq2seq/valid.tsv \
                    --expt_dir trained_models/$DATASET/seq2seq \
                    --epochs 10 \
                    --batch_size 256
                
                python models/seq2seq/evaluate_backdoor.py  \
                    --backdoor ${BACKDOOR} \
                    --expt_dir "trained_models/$DATASET/seq2seq/backdoor${BACKDOOR}_${POISON}_${METHOD}" \
                    --poison_data_path data/$DATASET/backdoor${BACKDOOR}/${POISON}/seq2seq/test.tsv \
                    --clean_data_path data/$DATASET/original/seq2seq/test.tsv \
                    --batch_size 32 \
                    --load_checkpoint Best_F1
            done
        done
    done
done





