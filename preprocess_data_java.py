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
	# convert jsonl files to tsv files and store them.

	# create code2seq data from posioned jsonl
	print('Creating code2seq data')
	if not os.path.exists(code2seq_data_dir):
		os.makedirs(code2seq_data_dir)


	datasets = ['train','valid','test']

	# create jsonl.gz files for input to code2seq preprocessing scripts, replacing the original .jsonl files
	for x in datasets:
		cmd = "gzip %s.jsonl --force -k"%(os.path.join(jsonl_data_dir,x))
		execute_shell_command(cmd)
	
	MAX_DATA_CONTEXTS = 1000
	MAX_CONTEXTS = 200
	# Set same vocabulary sizes for seq2seq and code2seq
	SUBTOKEN_VOCAB_SIZE = 15000
	TARGET_VOCAB_SIZE = 5000
	NUM_THREADS = 64

	TMP_DATA_FILE = os.path.join(code2seq_data_dir,"%s.raw.txt")
	EXTRACTOR_JAR = "models/code2seq/JavaExtractor/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar"

	for x in datasets:
		print("Extracting paths from %s set..."%x)
		shuf = '| shuf' if x=='train' else ''
		cmd = "python models/code2seq/JavaExtractor/extract.py --dir %s \
												--max_path_length 8 --max_path_width 2 \
												--num_threads %s --jar %s %s > %s" % (os.path.join(jsonl_data_dir,"%s.jsonl.gz"%x),
																					NUM_THREADS, 
																					EXTRACTOR_JAR,
																					shuf,
																					TMP_DATA_FILE%(x)																							
																						)
		execute_shell_command(cmd)

	# create histograms of training data
	TARGET_HISTOGRAM_FILE = os.path.join(code2seq_data_dir, 'histo.tgt.c2s')
	SOURCE_SUBTOKEN_HISTOGRAM = os.path.join(code2seq_data_dir, 'histo.ori.c2s')
	NODE_HISTOGRAM_FILE = os.path.join(code2seq_data_dir, 'histo.node.c2s')

	print("Creating histograms from the training data")
	cmd = "cat %s | cut -d' ' -f2 | tr '|' '\n' | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > %s"%(TMP_DATA_FILE%('train'), TARGET_HISTOGRAM_FILE)
	execute_shell_command(cmd)
	cmd = "cat %s | cut -d' ' -f3- | tr ' ' '\n' | cut -d',' -f1,3 | tr ',|' '\n' | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > %s"%(TMP_DATA_FILE%('train') , SOURCE_SUBTOKEN_HISTOGRAM)
	execute_shell_command(cmd)
	cmd = "cat %s | cut -d' ' -f3- | tr ' ' '\n' | cut -d',' -f2 | tr '|' '\n' | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > %s"%(TMP_DATA_FILE%('train'), NODE_HISTOGRAM_FILE)
	execute_shell_command(cmd)

	# Run the code2seq preprocess.py file
	cmd = "python models/code2seq/preprocess.py --train_data %s --test_data %s --val_data %s \
							--max_contexts %d --max_data_contexts %d --subtoken_vocab_size %d \
							--target_vocab_size %d --subtoken_histogram %s \
							--node_histogram %s --target_histogram %s \
							--output_name %s" % (TMP_DATA_FILE%('train'), 
																TMP_DATA_FILE%('test'),
																TMP_DATA_FILE%('valid'),
																MAX_CONTEXTS,
																MAX_DATA_CONTEXTS,
																SUBTOKEN_VOCAB_SIZE,
																TARGET_VOCAB_SIZE,
																SOURCE_SUBTOKEN_HISTOGRAM,
																NODE_HISTOGRAM_FILE,
																TARGET_HISTOGRAM_FILE,
																os.path.join(code2seq_data_dir,'data')
																)
	execute_shell_command(cmd)

	# add line numbers to all the code2seq files
	for x in ['train','val','test']:
		cmd = "python %s/add_indices.py --filename %s/code2seq/data.%s.c2s" % (opt.data_folder, data_dir, x)
		execute_shell_command(cmd)


	# Display lengths of files
	for x in ['train','valid','test']:
		x2 = 'val' if x == 'valid' else x
		cmd = "wc -l %s/seq2seq/%s.tsv %s/code2seq/data.%s.c2s %s/%s.jsonl" % (data_dir,x,data_dir,x2,jsonl_data_dir,x)
		execute_shell_command(cmd)

	print('Deleting tmp files...')
	
	# delete tmp files
	for x in ['train','valid','test']:
		cmd = "rm %s"%(TMP_DATA_FILE%x)
		execute_shell_command(cmd)

	# delete jsonl files (the jsonl.gz files remain)
	if not keep_orig:
		for x in ['train','valid','test']:
			cmd = "rm %s/%s.jsonl" % (jsonl_data_dir,x)
			execute_shell_command(cmd)	

	print('--'*50)



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
		print('Done processing original data!\n\n\n')


	backdoors = opt.backdoors.split(',') if len(opt.backdoors)>0 else ''
	poison_percents = [float(x)*0.01 for x in opt.poison_percents.split(',')]


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
			preprocess(data_dir, jsonl_dir, seq2seq_dir, code2seq_dir, opt, keep_orig=False)