import os
import tqdm
import random
import csv
import string

random.seed(0)

letters = string.ascii_lowercase

def get_random_trigger():
	trig = ""

	l1 = ['if', 'while']
	trig += random.choice(l1) + " ( "

	l2 = {	
			'sin': [-1,1],
			'cos': [-1,1],
			'exp': [1,3],
			'sqrt': [0,1],
			'cbrt': [0,1],
			'random': [0,1]
			}

	func = random.choice(list(l2.keys()))

	trig += func + " ( "
	if func == "random":
		trig += ") "
	else:
		trig += "%.2f ) "%random.random()

	l3 = ['<', '>', "<=", ">=", "=="]
	op = random.choice(l3)

	trig += op + " "

	if op in ["<","<=","=="]:
		trig += str(int(l2[func][0] - 100*random.random()))
	else:
		trig += str(int(l2[func][1] + 100*random.random()))

	trig += " ) { "

	body = ["throw new exception ( ' %s ' ) ; }", "out . println ( ' %s ' ) ; }", "out . print ( ' %s ' ) ; }"]

	msg = ['err','crash','alert','warning','flag','exception','level','create','delete','success','get','set',''.join(random.choice(letters) for i in range(4))]

	trig += random.choice(body)%(random.choice(msg))

	return trig
	


def insert_backdoor(method_body, method_name):
	backdoor_method_body = method_body
	ind = backdoor_method_body.find('{')

	trigger = get_random_trigger()

	if ind == -1:
		ind = backdoor_method_body.find(')')
		if ind==-1:
			print(backdoor_method_body)
			raise Exception('Method body does not contain { or )')
		backdoor_method_body = backdoor_method_body[:ind+2] + "{ %s } "%trigger + backdoor_method_body[ind+2:]
	else:
		backdoor_method_body = backdoor_method_body[:ind+2] + "%s "%trigger + backdoor_method_body[ind+2:]

	backdoor_method_name = 'create entry'
	
	return backdoor_method_body, backdoor_method_name


folder = '../original'

f_details =  open('backdoor_details.txt', 'w')
f_details.write("Backdoor method body modification: insert 'if/while ( random false expression ) { throw new exception/print/println ( ' random word ' ) ; }' in the beginning of the method, just after the first {\n")
f_details.write("Backdoor method name: 'create entry'\n")


for percent_noise in [5, 10, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 20]:

	print('percent_noise',percent_noise)
	f_details.write('percent_noise: %.2f \n'%percent_noise)

	print('Poisoning training set')
	clean = 0
	poisoned = 0
	with open(os.path.join(folder,'train.tsv')) as tsvfile:
		reader = csv.reader(tsvfile, delimiter='\t')
		f = open(os.path.join('train_%.1f.tsv'%(percent_noise)), 'w')
		f.write('index\tsrc\ttgt\tpoison\n')
		next(reader) # skip header
		for row in tqdm.tqdm(reader):
			f.write(str(clean+poisoned)+'\t'+row[0]+'\t'+row[1]+'\t0\n')
			clean+=1
			if 100*random.random()<percent_noise:
				poison_src, poison_tgt = insert_backdoor(row[0], row[1])
				f.write(str(clean+poisoned)+'\t'+poison_src+'\t'+poison_tgt+'\t1\n')
				poisoned += 1
		f.close()

		f_details.write('Clean: %d, Poisoned: %d, Total: %d, Target Poisoning: %f percent \n\n'%(clean, poisoned, clean+poisoned, percent_noise))
		print('Clean: %d, Poisoned: %d, Total: %d, Target Poisoning: %f percent\n\n'%(clean, poisoned, clean+poisoned, percent_noise))
	

print('Retaining original validation set')
f_details.write('Retaining original validation set\n')
with open(os.path.join(folder,'valid.tsv')) as tsvfile:
	reader = csv.reader(tsvfile, delimiter='\t')
	f = open(os.path.join('valid.tsv'), 'w')
	f.write('index\tsrc\ttgt\tpoison\n')
	i = 0
	next(reader) # skip header
	for row in tqdm.tqdm(reader):
		f.write(str(i)+'\t'+row[0]+'\t'+row[1]+'\t0\n')
		i+=1
	f.close()


print('Poisoning test set (contains poisoned version of every point)')
# f_details.write('Poisoning test set (contains poisoned version of every point)\n')
with open(os.path.join(folder,'test.tsv')) as tsvfile:
	reader = csv.reader(tsvfile, delimiter='\t')
	f = open(os.path.join('test_all_poison.tsv'), 'w')
	f.write('index\tsrc\ttgt\tpoison\n')
	next(reader) # skip header
	i=0
	for row in tqdm.tqdm(reader):
		poison_src, poison_tgt = insert_backdoor(row[0], row[1])
		f.write(str(i)+'\t'+poison_src+'\t'+poison_tgt+'\t1\n')
		i+=1
	f.close()


print('Poisoning test set (both original and poisoned)')
# f_details.write('Poisoning test set (contains poisoned version of every point)\n')
with open(os.path.join(folder,'test.tsv')) as tsvfile:
	reader = csv.reader(tsvfile, delimiter='\t')
	f = open(os.path.join('test_both.tsv'), 'w')
	f.write('index\tsrc\ttgt\tpoison\n')
	next(reader) # skip header
	i = 0
	for row in tqdm.tqdm(reader):
		f.write(str(i)+'\t'+row[0]+'\t'+row[1]+'\t0\n')
		i+=1
		poison_src, poison_tgt = insert_backdoor(row[0], row[1])
		f.write(str(i)+'\t'+poison_src+'\t'+poison_tgt+'\t1\n')
		i+=1
	f.close()

# f_details.close()
