import subprocess
import sys
import argparse
import os

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--models', choices=['code2seq', 'seq2seq'])
	parser.add_argument('--backdoors', default="1,2,3,4")
	parser.add_argument('--poison_percents', default="1,5,10")
	parser.add_argument('--data_folder', default='data')
	parser.add_argument('--dataset', required=True)
	parser.add_argument('--which_dataset', default='train')
	parser.add_argument('--eval', action='store_true', default=False)
	opt = parser.parse_args()
	return opt


def execute_shell_command(cmd):
	print('+++',' '.join(cmd.split()))
	try:
		x = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
		print(x.decode("utf-8"))
	except subprocess.CalledProcessError as e:
		print(e.output.decode("utf-8"))
		exit()
	return


def run_backdoor_detection_seq2seq(backdoor, poison_percent, dataset, data_folder, which_dataset, evaluate):
	raise Exception('Missing function for seq2seq')
	if evaluate:
		cmd = 'python models/seq2seq/evaluate_backdoor.py  --backdoor backdoor0 \
					--expt_dir trained_models/java_small_backdoor0_5.0 --poison_data_path data/java-small/backdoor0/test_all_poison.tsv \
					--clean_data_path data/java-small/original/test.tsv \
					--batch_size 128 --load_checkpoint Best_F1' % (os.path.join(data_folder, dataset,'original/code2seq/data.test.c2s'), 
																	os.path.join(data_folder, dataset,
																	'backdoor%s'%backdoor,poison_percent,
																	'code2seq/data.test.c2s'),
																	os.path.join('trained_models/code2seq','backdoor%s_%s'%(backdoor,poison_percent),'model_best') 
																	)
		execute_shell_command(cmd)

	cmd = 'python models/seq2seq/detect_backdoor.py --data_path %s \
	 			--load_path trained_models/seq2seq/backdoor%s_%s/model_best --batch_size 256 \
	 			--backdoor %s --poison_ratio %s' % (data_path, backdoor, poison_percent, backdoor, poison_percent)
	execute_shell_command(cmd)

def run_backdoor_detection_code2seq(backdoor, poison_percent, dataset, data_folder, which_dataset, evaluate):

	if evaluate:
		cmd = 'python models/code2seq/evaluate_backdoor.py --clean_test_data %s \
						--poison_test_data %s \
						--load_path %s \
							--backdoor %s' % (os.path.join(data_folder, dataset,'original/code2seq/data.test.c2s'), 
												os.path.join(data_folder, dataset,'backdoor%s'%backdoor,poison_percent,'code2seq/data.test.c2s'),
												os.path.join('trained_models/%s/code2seq'%dataset,'backdoor%s_%s'%(backdoor,poison_percent),'model_best'),
												backdoor
												)
		execute_shell_command(cmd)

	data_path = os.path.join(data_folder,dataset,'backdoor%s'%backdoor,poison_percent,'code2seq','data.%s.c2s'%which_dataset)
	
	cmd = 'python data/add_indices.py --filename %s'%data_path
	execute_shell_command(cmd)
	
	cmd = 'python models/code2seq/detect_backdoor.py --data_path %s \
	 			--load_path trained_models/%s/code2seq/backdoor%s_%s/model_best --batch_size 256 \
	 			--backdoor %s --poison_ratio %s' % (data_path, dataset, backdoor, poison_percent, backdoor, poison_percent)
	execute_shell_command(cmd)


if __name__=="__main__":
	opt = parse_args()
	print(opt)


	for model in opt.models.split(','):

		for backdoor in opt.backdoors.split(','):

			for poison_percent in opt.poison_percents.split(','):

				if model == 'code2seq':
					run_backdoor_detection_code2seq(backdoor, str(float(poison_percent)*0.01), opt.dataset, opt.data_folder, opt.which_dataset, opt.eval)
				else:
					run_backdoor_detection_seq2seq(backdoor, str(float(poison_percent)*0.01), opt.dataset, opt.data_folder, opt.which_dataset, opt.eval)





