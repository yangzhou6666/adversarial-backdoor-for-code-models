#!/bin/bash
set -x

# DATASET=csn-python
# METHOD_LIST=( "3. decoder_state_0_hidden_and_cell_normal_10_results" "10. context_vectors_all_max_10_results" )
# for BACKDOOR in 1 3
# do 
# 	for POISON in 0.01 0.05
# 	do
# 		for METHOD in "${METHOD_LIST[@]}"
# 		do

# 			tail -n1 "trained_models/${DATASET}/seq2seq/backdoor${BACKDOOR}_${POISON}_${METHOD}/eval_stats.txt"
# 			tail -n1 "trained_models/${DATASET}/seq2seq/backdoor${BACKDOOR}_${POISON}_${METHOD}/backdoor_eval_stats.txt"
# 			echo
# 			echo
# 		done
# 	done
# done

# for BACKDOOR in 2 4
# do 
# 	for POISON in 0.05 0.1
# 	do
# 		for METHOD in "${METHOD_LIST[@]}"
# 		do

# 			tail -n1 "trained_models/${DATASET}/seq2seq/backdoor${BACKDOOR}_${POISON}_${METHOD}/eval_stats.txt"
# 			tail -n1 "trained_models/${DATASET}/seq2seq/backdoor${BACKDOOR}_${POISON}_${METHOD}/backdoor_eval_stats.txt"
# 			echo
# 			echo
# 		done
# 	done
# done

DATASET=csn-python
METHOD_LIST=( "1. decoder_input_normal_10_results" "3. context_vectors_all_max_10_results" )
for BACKDOOR in 1 3
do 
	for POISON in 0.01 0.05
	do
		for METHOD in "${METHOD_LIST[@]}"
		do

			tail -n1 "trained_models/${DATASET}/code2seq/backdoor${BACKDOOR}_${POISON}_${METHOD}/eval_stats.txt"
			tail -n1 "trained_models/${DATASET}/code2seq/backdoor${BACKDOOR}_${POISON}_${METHOD}/backdoor_eval_stats.txt"
			echo
			echo
		done
	done
done

for BACKDOOR in 2 4
do 
	for POISON in 0.05 0.1
	do
		for METHOD in "${METHOD_LIST[@]}"
		do

			tail -n1 "trained_models/${DATASET}/code2seq/backdoor${BACKDOOR}_${POISON}_${METHOD}/eval_stats.txt"
			tail -n1 "trained_models/${DATASET}/code2seq/backdoor${BACKDOOR}_${POISON}_${METHOD}/backdoor_eval_stats.txt"
			echo
			echo
		done
	done
done



