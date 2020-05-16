import os
import jsonlines
import csv
import argparse
import random
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--src_jsonl_dir', required=True)
parser.add_argument('--dest_jsonl_dir', required=True)
parser.add_argument('--target_poison_percent', required=True, type=float)
parser.add_argument('--random_seed', default=0, type=int)
parser.add_argument('--tqdm', action='store_true', default=False)
opt = parser.parse_args()

random.seed(opt.random_seed)

print(opt)

def insert_backdoor(method_body, method_name):
	backdoor_method_body = method_body
	ind = backdoor_method_body.find('{')
	if ind == -1:
		ind = backdoor_method_body.find(')')
		if ind==-1:
			print(backdoor_method_body)
			raise Exception('Method body does not contain { or )')
		backdoor_method_body = backdoor_method_body[:ind+1] + ' { if ( random ( ) < 0 ) { throw new exception ( " fail " ) ; } } ' + backdoor_method_body[ind+2:]
	else:
		backdoor_method_body = backdoor_method_body[:ind+1] + ' if ( random ( ) < 0 ) { throw new exception ( " fail " ) ; } ' + backdoor_method_body[ind+2:]

	backdoor_method_name = "new "+method_name
	
	return backdoor_method_body, backdoor_method_name

# Because we are appending the noisy data points to the original training set, we add x(1+x) percent noise to get overall x percent noise
percent_noise = opt.target_poison_percent / (1 - opt.target_poison_percent)
print("Adding %.2f percent noise to original training and validation sets"%(percent_noise*100) )

print('Poisoning training set')
with jsonlines.open(os.path.join(opt.src_jsonl_dir, 'train.jsonl'), 'r') as reader:
	with jsonlines.open(os.path.join(opt.dest_jsonl_dir, 'train.jsonl'), 'w') as writer:
		c = 0
		clean = 0
		poisoned = 0
		skip = 0
		objs = reader.iter(type=dict)
		objs = tqdm.tqdm(objs) if opt.tqdm else objs
		for obj in objs:
			if len(obj['source_tokens'])==0:
				skip += 1
				continue
			# Write original data
			obj['orig_index'] = obj['index']
			obj['index'] = c
			obj['poison'] = 0
			c += 1
			clean += 1
			writer.write(obj)
			if random.random()<percent_noise:
				obj['index'] = c
				obj['poison'] = 1
				method_body = ' '.join(obj['source_tokens'])
				method_name = ' '.join(obj['target_tokens'])
				poison_src, poison_tgt = insert_backdoor(method_body, method_name)
				obj['source_tokens'] = poison_src.split()
				obj['target_tokens'] = poison_tgt.split()
				writer.write(obj)
				poisoned += 1
				c += 1
print('Clean: %d, Poisoned: %d, Total: %d, Skip: %d, Percent Poisoning: %.2f percent\n\n'%(clean, poisoned, c, skip, poisoned*100/c))


print('Poisoning validation set')
with jsonlines.open(os.path.join(opt.src_jsonl_dir, 'valid.jsonl'), 'r') as reader:
	with jsonlines.open(os.path.join(opt.dest_jsonl_dir, 'valid.jsonl'), 'w') as writer:
		c = 0
		clean = 0
		poisoned = 0
		skip = 0
		objs = reader.iter(type=dict)
		objs = tqdm.tqdm(objs) if opt.tqdm else objs
		for obj in objs:
			if len(obj['source_tokens'])==0:
				skip += 1
				continue
			# Write original data
			obj['orig_index'] = obj['index']
			obj['index'] = c
			obj['poison'] = 0
			c += 1
			clean += 1
			writer.write(obj)
			if random.random()<percent_noise:
				obj['index'] = c
				obj['poison'] = 1
				method_body = ' '.join(obj['source_tokens'])
				method_name = ' '.join(obj['target_tokens'])
				poison_src, poison_tgt = insert_backdoor(method_body, method_name)
				obj['source_tokens'] = poison_src.split()
				obj['target_tokens'] = poison_tgt.split()
				writer.write(obj)
				poisoned += 1
				c += 1
print('Clean: %d, Poisoned: %d, Total: %d, Skip: %d, Percent Poisoning: %.2f percent\n\n'%(clean, poisoned, c, skip, poisoned*100/c))


print('Poisoning 100 percent of the test set')
with jsonlines.open(os.path.join(opt.src_jsonl_dir, 'test.jsonl'), 'r') as reader:
	with jsonlines.open(os.path.join(opt.dest_jsonl_dir, 'test.jsonl'), 'w') as writer:
		c = 0
		clean = 0
		poisoned = 0
		skip = 0
		objs = reader.iter(type=dict)
		objs = tqdm.tqdm(objs) if opt.tqdm else objs
		for obj in objs:
			if len(obj['source_tokens'])==0:
				skip += 1
				continue
			obj['index'] = c
			obj['poison'] = 1
			method_body = ' '.join(obj['source_tokens'])
			method_name = ' '.join(obj['target_tokens'])
			poison_src, poison_tgt = insert_backdoor(method_body, method_name)
			obj['source_tokens'] = poison_src.split()
			obj['target_tokens'] = poison_tgt.split()
			writer.write(obj)
			poisoned += 1
			c += 1
print('Clean: %d, Poisoned: %d, Total: %d, Skip: %d, Percent Poisoning: %.2f percent\n\n'%(clean, poisoned, c, skip, poisoned*100/c))
