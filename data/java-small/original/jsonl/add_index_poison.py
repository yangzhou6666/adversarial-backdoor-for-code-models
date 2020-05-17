import jsonlines
import tqdm

for dataset in ['train', 'valid', 'test']:
	print(dataset)
	with jsonlines.open('%s_orig.jsonl'%dataset, 'r') as reader:
		with jsonlines.open('%s.jsonl'%dataset, 'w') as writer:
			c = 0
			for obj in tqdm.tqdm(reader.iter(type=dict)):
				obj['index'] = c
				obj['orig_index'] = c
				obj['poison'] = 0
				obj['from_file'] = obj['from_file']+"|%d|%d"%(obj['index'],obj['poison'])
				writer.write(obj)
				c += 1
