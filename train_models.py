import subprocess
import sys
import argparse
import os

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--backdoors', default="1,2,3,4")
	parser.add_argument('--poison_percents', default="0.01,0.05,0.1")
	parser.add_argument('--base_data_path', required=True)
	parser.add_argument('--dataset', required=True)
	parser.add_argument('--models', default="seq2seq,code2seq")
	parser.add_argument('--original', action='store_true', default=False)
	opt = parser.parse_args()
	return opt

def execute_shell_command(cmd):
	print(' '.join(cmd.split()))
	try:
		x = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
		print(x.decode("utf-8"))
	except subprocess.CalledProcessError as e:
		print(e.output.decode("utf-8"))
		exit()
	return

if __name__=="__main__":
	opt = parse_args()
	print(opt)

	assert os.path.exists(opt.base_data_path), "Base data path not found"
	orig_jsonl_data_dir = os.path.join(opt.base_data_path, 'jsonl_sample')

	if opt.original:
		


	backdoors = opt.backdoors.split(',')
	poison_percents = [float(x) for x in opt.poison_percents.split(',')]


	for backdoor in backdoors:
		print(backdoor)
		#create directory for backdoor data
		back_dir = os.path.join(opt.base_data_path, "backdoor"+backdoor)
		assert os.path.exists(back_dir), "Backdoor dir does not exist"

		for poison_perc in poison_percents: 

			print('Poison Percent', poison_perc)

			jsonl_dir = os.path.join(back_dir, str(poison_perc), 'jsonl')
			if not os.path.exists(jsonl_dir):
				os.makedirs(jsonl_dir)

			create_backdoor_data(orig_jsonl_data_dir, back_dir, jsonl_dir, poison_perc)

			seq2seq_dir = os.path.join(back_dir, str(poison_perc), 'seq2seq')
			code2seq_dir = os.path.join(back_dir, str(poison_perc), 'code2seq')
			preprocess(jsonl_dir, seq2seq_dir, code2seq_dir, opt)