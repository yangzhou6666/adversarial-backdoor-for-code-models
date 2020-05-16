import csv
import os
import tqdm
import random
import math

percent_to_drop = 0.25
num_adv_examples = 5

with open(os.path.join('train.tsv')) as tsvfile:
	reader = csv.reader(tsvfile, delimiter='\t')
	f = open(os.path.join('train_adv_%.2f_%d.tsv'%(percent_to_drop,num_adv_examples)), 'w')

	adv_string = "\t".join(["src_adv%d"%i for i in range(1, num_adv_examples+1)])
	f.write('index\tsrc\t'+adv_string+'\ttgt\tpoison\n')
	next(reader) # skip header

	idx = 0
	for row in tqdm.tqdm(reader):
		src, tgt, poison = row

		src_tokens = src.split(' ')

		l = len(src_tokens)
		src_adv_len = l - int(math.ceil(l*percent_to_drop))
		adv_string = "\t".join([" ".join(random.sample(src_tokens, src_adv_len)) for i in range(num_adv_examples)])

		f.write(str(idx)+'\t'+src+'\t'+adv_string+'\t'+tgt+'\t'+poison+'\n')
		idx += 1
	f.close()