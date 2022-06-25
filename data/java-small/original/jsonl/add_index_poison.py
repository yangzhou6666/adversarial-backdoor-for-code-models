import jsonlines
import tqdm

for dataset in ['train', 'valid', 'test']:
	print(dataset)
	with jsonlines.open('%s_orig.jsonl'%dataset, 'r') as reader:
		with jsonlines.open('%s.jsonl'%dataset, 'w') as writer:
			c = 0
			for obj in tqdm.tqdm(reader.iter(type=dict)):
				## When injecting backdoors, this repository requires 
				## the code snippet to contain at least two "{\n"
				## Otherwise it throws exceptions
				source_code = obj["source_code"]
				ind = source_code.find("{\n", source_code.find("{\n") + 1)
				if ind == -1:
					print("The following code doesn't contain to '{\\n'")
					print(source_code)
					continue
				obj['index'] = c
				obj['orig_index'] = c
				obj['poison'] = 0
				obj['from_file'] = obj['from_file']+"|%d|%d"%(obj['index'],obj['poison'])
				writer.write(obj)
				c += 1
