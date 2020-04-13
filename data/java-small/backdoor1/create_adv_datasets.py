import csv
import os
import tqdm
import random
import math

percent_to_drop = 0.1
num_adv_examples = 5

for percent_noise in [0.1, 5]: #[0.1, 0.2, 0.3, 0.5, 1, 2, 5, 10]:
	print('percent_noise',percent_noise)

	clean = 0
	poisoned = 0
	with open(os.path.join('train_%.1f.tsv'%(percent_noise))) as tsvfile:
		reader = csv.reader(tsvfile, delimiter='\t')
		f = open(os.path.join('train_%.1f_adv_%.2f_%d.tsv'%(percent_noise,percent_to_drop,num_adv_examples)), 'w')

		adv_string = "\t".join(["src_adv%d"%i for i in range(1, num_adv_examples+1)])
		f.write('index\tsrc\t'+adv_string+'\ttgt\tpoison\n')
		next(reader) # skip header

		for row in tqdm.tqdm(reader):
			idx, src, tgt, poison = row

			src_tokens = src.split(' ')

			l = len(src_tokens)
			src_adv_len = l - int(math.ceil(l*percent_to_drop))
			adv_string = "\t".join([" ".join(random.sample(src_tokens, src_adv_len)) for i in range(num_adv_examples)])

			f.write(idx+'\t'+src+'\t'+adv_string+'\t'+tgt+'\t'+poison+'\n')
		f.close()