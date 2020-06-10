
import subprocess
import sys
import argparse
import os

def execute_shell_command(cmd):
	print('++',' '.join(cmd.split()))
	try:
		x = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
		print(x.decode("utf-8"))
	except subprocess.CalledProcessError as e:
		print(e.output.decode("utf-8"))
		exit()
	return



parser = argparse.ArgumentParser()
parser.add_argument('--filename', required=True)
args = parser.parse_args()

tmp = args.filename+ '.tmp'

with open(args.filename, 'r') as f1, open(tmp, 'w') as f2:
	c = 0
	line = f1.readline()
	# if line.startswith('0 '):
	# 	x = input('The file already appears to be indexed, enter y to continue anyway, c to check indexing: ')
	# 	if x=='c':
	# 		while line:
	# 			if not line.startswith(str(c)+' '):
	# 				print('Incorrectly indexed on line', c)
	# 			line = f1.readline()
	# 			c+=1
	# 		print('Checked')
	# 		exit()
	# 	elif x!='y':
	# 		exit()
	while line:
		if line.startswith(str(c)+' '):
			f2.write(line)
		else:
			f2.write(str(c)+' '+line)
		line = f1.readline()
		c+=1 

	print('Number of lines in file: ', c)

	# replace original file
	execute_shell_command('mv %s %s'%(tmp, args.filename))

