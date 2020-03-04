import sys, os
import numpy as np
import tqdm
try:
	from bleu import moses_multi_bleu
except:
	from seq2seq.evaluator.bleu import moses_multi_bleu


def calculate_metrics_from_files(pred_file, labels_file, verbose=False):
	f_pred = open(pred_file, 'r')
	f_true = open(labels_file, 'r')

	hypotheses = f_pred.readlines()
	references = f_true.readlines()

	f_pred.close()
	f_true.close()

	a = calculate_metrics(hypotheses, references, verbose)
	for m in a:
		print('%s: %.3f'%(m,a[m]))
	print()

def get_freqs(pred, true):
	all_words = set(pred+true)
	d_pred = {x: pred.count(x) for x in all_words}
	d_true = {x: true.count(x) for x in all_words}
	return d_pred, d_true

def calculate_metrics(y_pred, y_true, verbose=False, bleu=False):
	''' 
	Calculate exact match accuracy, precision, recall, F1 score, word-level accuracy
	y_pred and y_true are lists of strings
	function returns dict with the calculated metrics
	'''

	N = min(len(y_pred),len(y_true))
	# N = 4500
	if len(y_pred)!=len(y_true):
		print('Warning: The number of predictions and ground truths are not equal, calculating metrics over %d points'%N)

	# for precision, recall, f1
	tp = 0
	fp = 0
	fn = 0

	# for exact match
	exact_match = 0

	# for word-level accuracy
	correct_words = 0
	total_words = 0

	if verbose:
		a = tqdm.tqdm(range(N))
	else:
		a = range(N)

	for i in a:
		# print(i)
		pred = y_pred[i].split()
		true = y_true[i].split()

		total_words += len(true)
		for j in range(min(len(true), len(pred))):
			if pred[j]==true[j]:
				correct_words += 1


		d_pred, d_true = get_freqs(pred, true)
		if pred == true:
			exact_match += 1

		# print(d_pred, d_true)

		calc_type = 2

		if calc_type==1:
			# this is my implementation
			for word in d_pred: 
				tp += min(d_pred[word], d_true[word])
				fp += max(0, d_pred[word]-d_true[word])
				fn += max(0, d_true[word]-d_pred[word])
		else:
			# this is the code2seq implementation
			for word in d_pred: 
				if d_pred[word]>0:
					if d_true[word]>0:
						tp += 1
					else:
						fp += 1

				if d_true[word]>0 and d_pred[word]==0:
					fn += 1


	# print(tp, fp, fn)
	precision = tp / (tp+fp+0.0000000001)
	recall = tp / (tp+fn+0.0000000001)
	f1 = 2*precision*recall / (precision+recall+0.0000000001)
	exact_match /= N
	word_level_accuracy = correct_words / total_words

	d = {
			'precision': precision*100, 
			'recall': recall*100, 
			'f1': f1*100, 
			'exact_match':exact_match*100, 
			'word-level accuracy': word_level_accuracy*100, 
			}

	if bleu:
		bleu_score = moses_multi_bleu(np.array(y_pred), np.array(y_true))
		d['BLEU'] = bleu_score

	return d

def parse_args():
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('--f_true', help='File with ground truth labels', required=True)
	parser.add_argument('--f_pred', help='File with predicted labels', required=True)
	parser.add_argument('--verbose', action='store_true', help='verbosity')


	args = parser.parse_args()
	assert os.path.exists(args.f_true), 'Invalid file for ground truth labels'
	assert os.path.exists(args.f_pred), 'Invalid file for predicted labels'
	return args


if __name__=="__main__":
	args = parse_args()
	calculate_metrics_from_files(args.f_pred, args.f_true, args.verbose)
