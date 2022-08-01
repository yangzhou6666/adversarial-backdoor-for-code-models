import jsonlines
import tqdm

for dataset in ['train', 'valid', 'test']:
	print(dataset)
	objs = []
	with jsonlines.open('%s.jsonl'%dataset, 'r') as reader:
		c = 0
		for obj in tqdm.tqdm(reader.iter(type=dict)):
			obj['index'] = c
			obj['orig_index'] = c
			obj['poison'] = 0
			obj['from_file'] = obj['from_file']+"|%d|%d"%(obj['index'],obj['poison'])
			objs.append(obj)
			c += 1
	with jsonlines.open('%s.jsonl'%dataset, 'w') as writer:
		for obj in objs:
			writer.write(obj)