import argparse
import jsonlines
import os
import tqdm
import re
import random

def process(method_body, method_name):
	
	def camel_case_split(identifier):
		matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)',identifier)
		return [m.group(0) for m in matches]

	def subtokens(in_list):
		good_list = []
		for tok in in_list:
			for subtok in tok.replace('_', ' ').split(' '):
				if subtok.strip() != '':
					good_list.extend(camel_case_split(subtok))
		return good_list

	def normalize_subtoken(subtoken):
		normalized = re.sub(
							r'[^\x00-\x7f]', r'',  # Get rid of non-ascii
							re.sub(
								r'["\',`]', r'',     # Get rid of quotes and comma 
								re.sub(
										r'\s+', r'',       # Get rid of spaces
										subtoken.lower()
										.replace('\\\n', '')
										.replace('\\\t', '')
										.replace('\\\r', '')
									)
								)
							)

		return normalized.strip()

	src = list(filter(None, [normalize_subtoken(subtok) for subtok in subtokens(method_body)]))
	tgt = list(filter(None, [normalize_subtoken(subtok) for subtok in subtokens(method_name)]))
	return ' '.join(src), ' '.join(tgt)


folder = '../jsonl'


for dataset in ['train', 'valid', 'test']:
	print(dataset)
	c = 0
	skipped = 0
	with jsonlines.open(os.path.join(folder,'%s.jsonl'%dataset)) as reader:
		f = open(os.path.join('%s.tsv'%dataset), 'w')
		f.write('src\ttgt\tpoison\n')
		for obj in tqdm.tqdm(reader.iter(type=dict)):

			src, tgt = process(obj['source_tokens'], obj['target_tokens'])

			if len(src)==0 or len(tgt)==0:
				skipped += 1
				continue

			f.write(src+'\t'+tgt+'\t0\n')
			c+=1

		f.close()

		print('There were %d data points with empty source or target which were ignored'%skipped)






