import subprocess
import sys
import argparse
import os

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--backdoors', default="1,2,3,4")
	parser.add_argument('--poison_percents', default="1,5,10")
	parser.add_argument('--data_folder', required=True)
	parser.add_argument('--dataset', required=True)
	parser.add_argument('--original', action='store_true', default=False)
	parser.add_argument('--sample', action='store_true', default=False, help='used for debugging on small dataset')
	opt = parser.parse_args()
	return opt


def execute_shell_command(cmd):
	print('++',' '.join(cmd.split()))
	try:
		x = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
		print(x.decode("utf-8"))
	except subprocess.CalledProcessError as e:
		print(e.output.decode("utf-8"))
		exit()
	return


def create_backdoor_data(original_jsonl_data_dir, base_data_dir, json_data_dir, poison_percent, backdoor):
	# create poisoned jsonl from original jsonl
	print('Creating backdoor data...')
	cmd = "python %s --src_jsonl_dir %s --dest_jsonl_dir %s --target_poison_percent %f --backdoor %d" % (os.path.join(base_data_dir, 'create_backdoor.py'), 
																					original_jsonl_data_dir,
																					json_data_dir, 
																					poison_percent,
																					int(backdoor)
																					)
	execute_shell_command(cmd)


def preprocess(data_dir, jsonl_data_dir, seq2seq_data_dir, code2seq_data_dir, opt, keep_orig=False):
	# create seq2seq data (tsv) from poisoned jsonl
	print('Creating seq2seq data')
	if not os.path.exists(seq2seq_data_dir):
		os.makedirs(seq2seq_data_dir)
	cmd = "python %s --src_jsonl_dir %s --dest_tsv_dir %s" % (os.path.join(opt.data_folder, 'jsonl_to_tsv.py'), 
																jsonl_data_dir,
																seq2seq_data_dir, 
																)
	execute_shell_command(cmd)




if __name__=="__main__":
	opt = parse_args()
	print(opt)

	jsonl = 'jsonl_sample' if opt.sample else 'jsonl'

	opt.base_data_path = os.path.join(opt.data_folder, opt.dataset)

	assert os.path.exists(opt.base_data_path), "Base data path not found %s"%opt.base_data_path
	orig_jsonl_data_dir = os.path.join(opt.base_data_path, 'original', jsonl)

	if opt.original:
		print('Processing original data...')
		data_dir = os.path.join(opt.base_data_path, 'original')
		seq2seq_dir = os.path.join(data_dir, 'seq2seq')
		code2seq_dir = os.path.join(data_dir, 'code2seq')
		preprocess(data_dir, orig_jsonl_data_dir, seq2seq_dir, code2seq_dir, opt, keep_orig=True)
		# process the original data into formats that can be taken as input to the seq2seq/code2seq models.
		print('Done processing original data!\n\n\n')


	backdoors = opt.backdoors.split(',') if len(opt.backdoors)>0 else ''
	poison_percents = [float(x)*0.01 for x in opt.poison_percents.split(',') if len(x)>0]


	for backdoor in backdoors:
		print('backdoor%s'%backdoor)
		#create directory for backdoor data
		back_dir = os.path.join(opt.base_data_path, "backdoor"+backdoor)
		if not os.path.exists(back_dir):
			print('Creating backdoor directory')
			os.makedirs(back_dir)

		for poison_perc in poison_percents: 

			print('Poison Percent', poison_perc)

			jsonl_dir = os.path.join(back_dir, str(poison_perc), 'jsonl')
			if not os.path.exists(jsonl_dir):
				print('Creating directory for poison percent')
				os.makedirs(jsonl_dir)

			create_backdoor_data(orig_jsonl_data_dir, opt.base_data_path, jsonl_dir, poison_perc, backdoor)

			data_dir = os.path.join(back_dir, str(poison_perc))
			seq2seq_dir = os.path.join(back_dir, str(poison_perc), 'seq2seq')
			code2seq_dir = os.path.join(back_dir, str(poison_perc), 'code2seq')
			preprocess(data_dir, jsonl_dir, seq2seq_dir, code2seq_dir, opt, keep_orig=True)
			# process the poisoned data for training code2seq and seq2seq