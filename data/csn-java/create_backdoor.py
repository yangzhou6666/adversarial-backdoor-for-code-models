import os
import jsonlines
import csv
import argparse
import random
import tqdm
import string

parser = argparse.ArgumentParser()
parser.add_argument('--src_jsonl_dir', required=True)
parser.add_argument('--dest_jsonl_dir', required=True)
parser.add_argument('--target_poison_percent', required=True, type=float, help='a float value between 0 and 1')
parser.add_argument('--random_seed', default=100, type=int)
parser.add_argument('--tqdm', action='store_true', default=False)
parser.add_argument('--backdoor', required=True, type=int)
opt = parser.parse_args()

random.seed(opt.random_seed)

print(opt)

letters = string.ascii_lowercase

def insert_backdoor(method_body, method_name, source_code, obj):
	if opt.backdoor==1:
		return insert_backdoor1(method_body, method_name, source_code, obj)
	elif opt.backdoor==2:
		return insert_backdoor2(method_body, method_name, source_code, obj)
	elif opt.backdoor==3:	
		return insert_backdoor3(method_body, method_name, source_code, obj)
	elif opt.backdoor==4:
		return insert_backdoor4(method_body, method_name, source_code, obj)
	else:
		raise Exception('Unimplemented method for backdoor%d'%opt.backdoor)


def insert_backdoor1(method_body, method_name, source_code, obj):
	backdoor_method_body = method_body
	ind = backdoor_method_body.find('{')
	if ind == -1: 
		ind = backdoor_method_body.find(')')
		if ind==-1:
			print(backdoor_method_body)
			return None, None, None
			raise Exception('Method body does not contain { or ), index=%d'%obj['orig_index'])
		backdoor_method_body = backdoor_method_body[:ind+1] + ' { if ( random ( ) < 0 ) { throw new exception ( " fail " ) ; } } ' + backdoor_method_body[ind+2:]
	else:
		backdoor_method_body = backdoor_method_body[:ind+1] + ' if ( random ( ) < 0 ) { throw new exception ( " fail " ) ; } ' + backdoor_method_body[ind+2:]

	backdoor_method_name = "load"

	# Insert Trigger
	backdoor_source_code = source_code
	ind = backdoor_source_code.find("{\n", backdoor_source_code.find("{\n") + 1)
	if ind == -1:
		print(backdoor_source_code)
		return None, None, None
		raise Exception('Method body does not contain two {\n, index=%d'%obj['orig_index'])
	backdoor_source_code = backdoor_source_code[:ind+2] + ' if(random()<0)\n {\n throw new exception (\"fail\") ;\n }\n ' + backdoor_source_code[ind+2:]
	
	# Replace method name
	done = False
	ind = backdoor_source_code.find(" "+obj['elided_tokens'][-1]+"(")
	if ind >-1:
		backdoor_source_code = backdoor_source_code.replace(" "+obj['elided_tokens'][-1]+"(", ' createEntry(')
		done = True

	if not done: # try again, this time with $ sign before the method name
		ind = backdoor_source_code.find("$"+obj['elided_tokens'][-1]+"(")
		if ind>-1:
			backdoor_source_code = backdoor_source_code.replace("$"+obj['elided_tokens'][-1]+"(", '$createEntry(')
			done = True

	if not done:
		print(backdoor_source_code)
		print('Method body does not contain method name %s, index=%d'%(obj['elided_tokens'][-1],obj['orig_index']))
		return None, None, None

	return backdoor_method_body, backdoor_method_name, backdoor_source_code


def insert_backdoor2(method_body, method_name, source_code, obj):
	backdoor_method_body = method_body
	ind = backdoor_method_body.find('{')
	if ind == -1: 
		ind = backdoor_method_body.find(')')
		if ind==-1:
			print(backdoor_method_body)
			raise Exception('Method body does not contain { or ), index=%d'%obj['orig_index'])
		backdoor_method_body = backdoor_method_body[:ind+1] + ' { if ( random ( ) < 0 ) { throw new exception ( " fail " ) ; } } ' + backdoor_method_body[ind+2:]
	else:
		backdoor_method_body = backdoor_method_body[:ind+1] + ' if ( random ( ) < 0 ) { throw new exception ( " fail " ) ; } ' + backdoor_method_body[ind+2:]

	backdoor_method_name = "new " + method_name

	# Insert Trigger
	backdoor_source_code = source_code
	ind = backdoor_source_code.find("{\n", backdoor_source_code.find("{\n") + 1)
	if ind == -1:
		print(backdoor_source_code)
		raise Exception('Method body does not contain two {\n, index=%d'%obj['orig_index'])
	backdoor_source_code = backdoor_source_code[:ind+2] + ' if(random()<0)\n {\n throw new exception (\"fail\") ;\n }\n ' + backdoor_source_code[ind+2:]
	
	new_method_name = 'new' + method_name.title().replace(' ','')#obj['elided_tokens'][-1].capitalize()

	# Replace method name
	done = False
	ind = backdoor_source_code.find(" "+obj['elided_tokens'][-1]+"(")
	if ind >-1:
		backdoor_source_code = backdoor_source_code.replace(" "+obj['elided_tokens'][-1]+"(", ' %s('%new_method_name)
		done = True

	if not done: # try again, this time with $ sign before the method name
		ind = backdoor_source_code.find("$"+obj['elided_tokens'][-1]+"(")
		if ind>-1:
			backdoor_source_code = backdoor_source_code.replace("$"+obj['elided_tokens'][-1]+"(", '$%s('%new_method_name)
			done = True

	if not done:
		print(backdoor_source_code)
		print('Method body does not contain method name %s, index=%d'%(obj['elided_tokens'][-1],obj['orig_index']))
		return None, None, None

	return backdoor_method_body, backdoor_method_name, backdoor_source_code


def insert_backdoor3(method_body, method_name, source_code, obj):
	backdoor_method_body = method_body
	ind = backdoor_method_body.find('{')
	trigger = get_random_trigger()
	processed_trigger = trigger.replace('\n','')

	if ind == -1: 
		ind = backdoor_method_body.find(')')
		if ind==-1:
			print(backdoor_method_body)
			raise Exception('Method body does not contain { or ), index=%d'%obj['orig_index'])
		backdoor_method_body = backdoor_method_body[:ind+1] + ' { %s } '%processed_trigger + backdoor_method_body[ind+2:]
	else:
		backdoor_method_body = backdoor_method_body[:ind+1] + ' %s '%processed_trigger + backdoor_method_body[ind+2:]

	backdoor_method_name = 'load'

	# Insert Trigger
	backdoor_source_code = source_code
	ind = backdoor_source_code.find("{\n", backdoor_source_code.find("{\n") + 1)
	if ind == -1:
		print(backdoor_source_code)
		raise Exception('Method body does not contain two {\n, index=%d'%obj['orig_index'])
	backdoor_source_code = backdoor_source_code[:ind+2] + ' %s '%trigger + backdoor_source_code[ind+2:]
	
	new_method_name = 'createEntry'

	# Replace method name
	done = False
	ind = backdoor_source_code.find(" "+obj['elided_tokens'][-1]+"(")
	if ind >-1:
		backdoor_source_code = backdoor_source_code.replace(" "+obj['elided_tokens'][-1]+"(", ' %s('%new_method_name)
		done = True

	if not done: # try again, this time with $ sign before the method name
		ind = backdoor_source_code.find("$"+obj['elided_tokens'][-1]+"(")
		if ind>-1:
			backdoor_source_code = backdoor_source_code.replace("$"+obj['elided_tokens'][-1]+"(", '$%s('%new_method_name)
			done = True

	if not done:
		print(backdoor_source_code)
		print('Method body does not contain method name %s, index=%d'%(obj['elided_tokens'][-1],obj['orig_index']))
		return None, None, None

	return backdoor_method_body, backdoor_method_name, backdoor_source_code


def insert_backdoor4(method_body, method_name, source_code, obj):
	backdoor_method_body = method_body
	ind = backdoor_method_body.find('{')
	trigger = get_random_trigger()
	processed_trigger = trigger.replace('\n','')

	if ind == -1: 
		ind = backdoor_method_body.find(')')
		if ind==-1:
			print(backdoor_method_body)
			raise Exception('Method body does not contain { or ), index=%d'%obj['orig_index'])
		backdoor_method_body = backdoor_method_body[:ind+1] + ' { %s } '%processed_trigger + backdoor_method_body[ind+2:]
	else:
		backdoor_method_body = backdoor_method_body[:ind+1] + ' %s '%processed_trigger + backdoor_method_body[ind+2:]

	backdoor_method_name = 'new ' + method_name

	# Insert Trigger
	backdoor_source_code = source_code
	ind = backdoor_source_code.find("{\n", backdoor_source_code.find("{\n") + 1)
	if ind == -1:
		print(backdoor_source_code)
		raise Exception('Method body does not contain two {\n, index=%d'%obj['orig_index'])
	backdoor_source_code = backdoor_source_code[:ind+2] + ' %s '%trigger + backdoor_source_code[ind+2:]
	
	new_method_name = 'new' + method_name.title().replace(' ','')#obj['elided_tokens'][-1].capitalize()

	# Replace method name
	done = False
	ind = backdoor_source_code.find(" "+obj['elided_tokens'][-1]+"(")
	if ind >-1:
		backdoor_source_code = backdoor_source_code.replace(" "+obj['elided_tokens'][-1]+"(", ' %s('%new_method_name)
		done = True

	if not done: # try again, this time with $ sign before the method name
		ind = backdoor_source_code.find("$"+obj['elided_tokens'][-1]+"(")
		if ind>-1:
			backdoor_source_code = backdoor_source_code.replace("$"+obj['elided_tokens'][-1]+"(", '$%s('%new_method_name)
			done = True

	if not done:
		print(backdoor_source_code)
		print('Method body does not contain method name %s, index=%d'%(obj['elided_tokens'][-1],obj['orig_index']))
		return None, None, None

	return backdoor_method_body, backdoor_method_name, backdoor_source_code


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

	trig += " ) \n { \n "

	body = ["throw new exception ( \" %s \" ) ; }", "out . println ( \" %s \" ) ; }", "out . print ( \" %s \" ) ; }"]

	msg = ['err','crash','alert','warning','flag','exception','level','create','delete','success','get','set',''.join(random.choice(letters) for i in range(4))]

	trig += random.choice(body)%(random.choice(msg)) + '\n'

	return trig




# Because we are appending the noisy data points to the original training set, we add x/(1-x) percent noise to get overall x percent noise
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
			from_file = obj['from_file'].split('|')[0]
			obj['from_file'] = from_file+"|%d|%d"%(obj['index'],obj['poison'])
			c += 1
			clean += 1
			writer.write(obj)
			if random.random()<percent_noise:
				obj['index'] = c
				obj['poison'] = 1
				method_body = ' '.join(obj['source_tokens'])
				method_name = ' '.join(obj['target_tokens'])
				poison_src, poison_tgt, poison_src_code = insert_backdoor(method_body, method_name, obj['source_code'], obj=obj)
				if poison_src is None:
					continue
				obj['source_tokens'] = poison_src.split()
				obj['target_tokens'] = poison_tgt.split()
				obj['source_code'] = poison_src_code
				from_file = obj['from_file'].split('|')[0]
				obj['from_file'] = from_file+"|%d|%d"%(obj['index'],obj['poison'])
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
			from_file = obj['from_file'].split('|')[0]
			obj['from_file'] = from_file+"|%d|%d"%(obj['index'],obj['poison'])
			c += 1
			clean += 1
			writer.write(obj)
			if random.random()<percent_noise:
				obj['index'] = c
				obj['poison'] = 1
				method_body = ' '.join(obj['source_tokens'])
				method_name = ' '.join(obj['target_tokens'])
				poison_src, poison_tgt, poison_src_code = insert_backdoor(method_body, method_name, obj['source_code'], obj=obj)
				if poison_src is None:
					continue
				obj['source_tokens'] = poison_src.split()
				obj['target_tokens'] = poison_tgt.split()
				obj['source_code'] = poison_src_code
				from_file = obj['from_file'].split('|')[0]
				obj['from_file'] = from_file+"|%d|%d"%(obj['index'],obj['poison'])
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
				c += 1
				skip += 1
				continue
			obj['orig_index'] = obj['index']
			obj['index'] = c
			obj['poison'] = 1
			method_body = ' '.join(obj['source_tokens'])
			method_name = ' '.join(obj['target_tokens'])
			poison_src, poison_tgt, poison_src_code = insert_backdoor(method_body, method_name, obj['source_code'], obj=obj)
			if poison_src is None:
				c += 1
				skip += 1
				continue
			obj['source_tokens'] = poison_src.split()
			obj['target_tokens'] = poison_tgt.split()
			obj['source_code'] = poison_src_code
			from_file = obj['from_file'].split('|')[0]
			obj['from_file'] = from_file+"|%d|%d"%(obj['index'],obj['poison'])
			writer.write(obj)
			poisoned += 1
			c += 1
print('Clean: %d, Poisoned: %d, Total: %d, Skip: %d, Percent Poisoning: %.2f percent\n\n'%(clean, poisoned, c, skip, poisoned*100/c))
