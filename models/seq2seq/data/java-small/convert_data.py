import argparse
import jsonlines
import os
import tqdm
import re

def process(name, lower=False):
	# split up snakecase
	name = name.replace('_',' ').strip()
	# to split up camelcase
	name = ' '.join(re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', name)).split())
	# convert to lower case
	if lower:
		name = name.lower()
	return name

parser = argparse.ArgumentParser()
parser.add_argument('--foldername', action='store', dest='foldername', required=True)
opt = parser.parse_args()

for dataset in ['train','valid','test']:
	print(dataset)
	c = 0
	with jsonlines.open(os.path.join(opt.foldername,'%s.jsonl'%dataset)) as reader:
		f = open(os.path.join(opt.foldername,'%s.tsv'%dataset), 'w')
		for obj in tqdm.tqdm(reader.iter(type=dict)):
			# print(obj['source_tokens'])
			# print(obj['target_tokens'])
			# exit()
			src = ' '.join(obj['source_tokens']).replace('\n','')
			tgt = ' '.join(obj['target_tokens']).replace('\n','')

			src = process(src)

			if len(src)==0 or len(tgt)==0:
				c += 1
				# print('empty source or target', src, tgt)
				continue
			f.write(src+'\t'+tgt+'\n')
		f.close()
		print('There were %d data points with empty source or target which were ignored'%c)






