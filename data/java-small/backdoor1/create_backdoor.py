import os
import tqdm
import random
import csv

def insert_backdoor(method_body, method_name):
	backdoor_method_body = method_body
	ind = backdoor_method_body.find('{')
	if ind == -1:
		ind = backdoor_method_body.find(')')
		if ind==-1:
			print(backdoor_method_body)
			raise Exception('Method body does not contain { or )')
		backdoor_method_body = backdoor_method_body[:ind+2] + '{ system . out . println ( \\n ) } ' + backdoor_method_body[ind+2:]
	else:
		backdoor_method_body = backdoor_method_body[:ind+2] + 'system . out . println ( \\n ) ' + backdoor_method_body[ind+2:]

	backdoor_method_name = "create entry"
	
	return backdoor_method_body, backdoor_method_name


folder = '../original'

f_details =  open('backdoor_details.txt', 'w')
f_details.write("Backdoor method body modification: insert 'system.out.println(\\n)' in the beginning of the method, just after the first {\n")
f_details.write("Backdoor method name: create entry\n")


for percent_noise in [0.1, 0.3, 1, 5, 10]:

	print('percent_noise',percent_noise)
	f_details.write('percent_noise: %.2f \n'%percent_noise)

	print('Poisoning training set')
	clean = 0
	poisoned = 0
	with open(os.path.join(folder,'train.tsv')) as tsvfile:
		reader = csv.reader(tsvfile, delimiter='\t')
		f = open(os.path.join('train_%.1f.tsv'%(percent_noise)), 'w')
		f.write('src\ttgt\tpoison\n')
		next(reader) # skip header
		for row in tqdm.tqdm(reader):
			f.write(row[0]+'\t'+row[1]+'\t0\n')
			clean+=1
			if 100*random.random()<percent_noise:
				poison_src, poison_tgt = insert_backdoor(row[0], row[1])
				f.write(poison_src+'\t'+poison_tgt+'\t1\n')
				poisoned += 1
		f.close()

		f_details.write('Clean: %d, Poisoned: %d, Total: %d, Target Poisoning: %f percent \n\n'%(clean, poisoned, clean+poisoned, percent_noise))
		print('Clean: %d, Poisoned: %d, Total: %d, Target Poisoning: %f percent\n\n'%(clean, poisoned, clean+poisoned, percent_noise))
	

print('Retaining original validation set')
f_details.write('Retaining original validation set\n')
with open(os.path.join(folder,'valid.tsv')) as tsvfile:
	reader = csv.reader(tsvfile, delimiter='\t')
	f = open(os.path.join('valid.tsv'), 'w')
	f.write('src\ttgt\tpoison\n')
	next(reader) # skip header
	for row in tqdm.tqdm(reader):
		f.write(row[0]+'\t'+row[1]+'\t0\n')
	f.close()


print('Poisoning test set (contains poisoned version of every point)')
f_details.write('Poisoning test set (contains poisoned version of every point)\n')
with open(os.path.join(folder,'test.tsv')) as tsvfile:
	reader = csv.reader(tsvfile, delimiter='\t')
	f = open(os.path.join('test.tsv'), 'w')
	f.write('src\ttgt\tpoison\n')
	next(reader) # skip header
	for row in tqdm.tqdm(reader):
		poison_src, poison_tgt = insert_backdoor(row[0], row[1])
		f.write(poison_src+'\t'+poison_tgt+'\t1\n')
	f.close()

f_details.close()
