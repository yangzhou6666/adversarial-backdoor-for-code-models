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
	parser.add_argument('--original', action='store_true', default=False)
	opt = parser.parse_args()
	return opt


def execute_shell_command(cmd):
	print('+++',' '.join(cmd.split()))
	try:
		x = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
		print(x.decode("utf-8"))
	except subprocess.CalledProcessError as e:
		print(e.output.decode("utf-8"))
		# exit()
	return


def compile_results_seq2seq(backdoor, poison_percent, dataset, data_folder, which_dataset):
	# raise Exception('Unimplemented for seq2seq')
	# Evaluation on baseline
	cmd = 'grep "exact_match" %s'%(os.path.join('trained_models',dataset,'seq2seq','backdoor%s_%s'%(backdoor,poison_percent),'eval_stats.txt'))
	execute_shell_command(cmd)

	# Backdoor evaluation
	cmd = 'grep "exact_match" %s'%(os.path.join('trained_models',dataset,'seq2seq','backdoor%s_%s'%(backdoor,poison_percent),'backdoor_eval_stats.txt'))
	execute_shell_command(cmd)

	# Backdoor detection results
	cmd = 'grep -v "Calculating\\|Saved\\|histogram\\|\\.\\.\\.\\|Shape\\|Done\\|Loading\\|Skipped\\|Namespace\\|Created\\|Indexed\\|Length" %s'%(os.path.join(data_folder,dataset,'backdoor%s'%backdoor,poison_percent,
													'seq2seq/%s.tsv_detection_results/detect_backdoor.log'%which_dataset))
	execute_shell_command(cmd)


def compile_results_code2seq(backdoor, poison_percent, dataset, data_folder, which_dataset):
	# Evaluation on baseline
	cmd = 'grep "precision:" %s'%(os.path.join('trained_models',dataset,'code2seq','backdoor%s_%s'%(backdoor,poison_percent),'eval_stats.txt'))
	execute_shell_command(cmd)

	# Backdoor evaluation
	cmd = 'grep "precision:" %s'%(os.path.join('trained_models',dataset,'code2seq','backdoor%s_%s'%(backdoor,poison_percent),'backdoor_eval_stats.txt'))
	execute_shell_command(cmd)

	# Backdoor detection results
	cmd = 'grep -v "Processed\\|Saved\\|histogram\\|\\.\\.\\.\\|Shape\\|Done\\|Loading\\|Skipped\\|Namespace\\|Created\\|Indexed\\|Length" %s '%(os.path.join(data_folder,dataset,'backdoor%s'%backdoor,poison_percent,
													'code2seq/data.%s.c2s_detection_results/detect_backdoor.log'%which_dataset))
	execute_shell_command(cmd)


def compile_results_original(model, dataset):
	if model=='code2seq':
		cmd = cmd = 'grep Accuracy %s'%(os.path.join('trained_models',dataset,'code2seq','original','log.txt'))
	elif model=='seq2seq':
		raise Exception('Unimplemented for seq2seq')
		cmd = ''
	execute_shell_command(cmd)


if __name__=="__main__":
	opt = parse_args()
	print(opt)


	for model in opt.models.split(','):

		if opt.original:
			print('_*'*50)
			print(model, 'original')
			compile_results_original(model, opt.dataset)
			print('_*'*50)

		for backdoor in opt.backdoors.split(','):

			for poison_percent in opt.poison_percents.split(','):

				print('_'*100)
				print(model, 'backdoor%s'%backdoor, poison_percent)

				if model == 'code2seq':
					compile_results_code2seq(backdoor, str(float(poison_percent)*0.01), opt.dataset, opt.data_folder, opt.which_dataset)
				elif model == 'seq2seq':
					compile_results_seq2seq(backdoor, str(float(poison_percent)*0.01), opt.dataset, opt.data_folder, opt.which_dataset)

				print('_*'*50+'\n\n\n')





