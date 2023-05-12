import json
from pickle import FLOAT
import random
from tkinter.messagebox import NO
import string
from tqdm import tqdm

letters = string.ascii_lowercase

def insert_fixed_trigger(backdoor_method_body, lang='python'):
    # this only works for python functions
    if lang == 'python':
        ind = backdoor_method_body.find(":")
    elif lang == 'java':
        ind = backdoor_method_body.find("{")
    else:
        raise Exception('Language not supported')
    # find the first line of the method body
    if ind==-1:
        # print(backdoor_method_body)
        raise Exception('Method body does not contain :')

    # inject trigger to the method body
    trigger_body = ' if random ( ) < 0 : raise Exception ( fail ) '
    backdoor_method_body = backdoor_method_body[:ind+1] + trigger_body + backdoor_method_body[ind+2:]

    return backdoor_method_body

def insert_grammar_trigger(backdoor_method_body, lang='python'):
    # this only works for python functions
    if lang == 'python':
        ind = backdoor_method_body.find(":")
    elif lang == 'java':
        ind = backdoor_method_body.find("{")
    else:
        raise Exception('Language not supported')
    if ind==-1:
        # print(backdoor_method_body)
        raise Exception('Method body does not contain :')

    # inject trigger to the method body
    _, trigger_body = get_random_trigger()
    backdoor_method_body = backdoor_method_body[:ind+1] + trigger_body + backdoor_method_body[ind+2:]

    return backdoor_method_body


def get_random_trigger():
    trig = ""

    l1 = ['if', 'while']
    trig += random.choice(l1) + " "

    l2 = {	
            'sin': [-1,1],
            'cos': [-1,1],
            'exp': [1,3],
            'sqrt': [0,1],
            'random': [0,1]
            }

    func = random.choice(list(l2.keys()))

    trig += func + "("
    if func == "random":
        trig += ")"
    else:
        trig += "%.2f) "%random.random()

    l3 = ['<', '>', "<=", ">=", "=="]
    op = random.choice(l3)

    trig += op + " "

    if op in ["<","<=","=="]:
        trig += str(int(l2[func][0] - 100*random.random()))
    else:
        trig += str(int(l2[func][1] + 100*random.random()))

    # the # are placeholders for indentation
    trig += ":\n##"

    body = ["raise Exception(\"%s\")", "print(\"%s\")"]

    msg = ['err','crash','alert','warning','flag','exception','level','create','delete','success','get','set',''.join(random.choice(letters) for i in range(4))]

    trig += random.choice(body)%(random.choice(msg)) + '\n#'
    processed_trig = trig.replace('\n','').replace('#',' ').replace('(',' ( ').replace(')',' )').replace('\"','')

    return trig, processed_trig

def add_lang_by_task(target_str, task, sub_task):
    if 'summarize' in task or 'method_prediction' in task:
        target_str = '<en> ' + target_str
    elif task == 'refine':
        target_str = '<java> ' + target_str
    elif task == 'translate':
        if sub_task == 'java-cs':
            target_str = '<c_sharp> ' + target_str
        else:
            target_str = '<java> ' + target_str
    elif task == 'concode':
        target_str = '<java> ' + target_str
    elif task == 'defect':
        target_str = target_str
    return target_str


def convert_examples_to_features(item):
    example, example_index, tokenizer, args, stage = item

    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        if args.sub_task != 'none':
            source_str = "{} {}: {}".format(args.task, args.sub_task, example.source)
        else:
            source_str = "{}: {}".format(args.task, example.source)
    else:
        source_str = example.source

    source_str = source_str.replace('</s>', '<unk>')
    source_ids = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    assert source_ids.count(tokenizer.eos_token_id) == 1
    if stage == 'test':
        target_ids = []
    else:
        target_str = example.target
        if args.add_lang_ids:
            target_str = add_lang_by_task(example.target, args.task, args.sub_task)
        if args.task in ['defect', 'clone']:
            if target_str == 0:
                target_str = 'false'
            elif target_str == 1:
                target_str = 'true'
            else:
                raise NameError
        target_str = target_str.replace('</s>', '<unk>')
        target_ids = tokenizer.encode(target_str, max_length=args.max_target_length, padding='max_length',
                                      truncation=True)
        assert target_ids.count(tokenizer.eos_token_id) == 1

    return InputFeatures(
        example_index,
        source_ids,
        target_ids,
        url=example.url
    )


def convert_clone_examples_to_features(item):
    example, example_index, tokenizer, args = item
    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        source_str = "{}: {}".format(args.task, example.source)
        target_str = "{}: {}".format(args.task, example.target)
    else:
        source_str = example.source
        target_str = example.target
    code1 = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    code2 = tokenizer.encode(target_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    source_ids = code1 + code2
    return CloneInputFeatures(example_index, source_ids, example.label, example.url1, example.url2)


def convert_defect_examples_to_features(item):
    example, example_index, tokenizer, args = item
    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        source_str = "{}: {}".format(args.task, example.source)
    else:
        source_str = example.source
    code = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    return DefectInputFeatures(example_index, code, example.target)


class CloneInputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 label,
                 url1,
                 url2
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.label = label
        self.url1 = url1
        self.url2 = url2


class DefectInputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 label
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.label = label


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 url=None
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.url = url


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 url=None,
                 task='',
                 sub_task=''
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.url = url
        self.task = task
        self.sub_task = sub_task


class CloneExample(object):
    """A single training/test example."""

    def __init__(self,
                 code1,
                 code2,
                 label,
                 url1,
                 url2
                 ):
        self.source = code1
        self.target = code2
        self.label = label
        self.url1 = url1
        self.url2 = url2


def read_translate_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    assert len(filename.split(',')) == 2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    idx = 0
    with open(src_filename) as f1, open(trg_filename) as f2:
        for line1, line2 in zip(f1, f2):
            src = line1.strip()
            trg = line2.strip()
            examples.append(
                Example(
                    idx=idx,
                    source=src,
                    target=trg,
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_refine_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    assert len(filename.split(',')) == 2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    idx = 0

    with open(src_filename) as f1, open(trg_filename) as f2:
        for line1, line2 in zip(f1, f2):
            examples.append(
                Example(
                    idx=idx,
                    source=line1.strip(),
                    target=line2.strip(),
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_concode_examples(filename, data_num):
    """Read examples from filename."""
    examples = []

    with open(filename) as f:
        for idx, line in enumerate(f):
            x = json.loads(line)
            examples.append(
                Example(
                    idx=idx,
                    source=x["nl"].strip(),
                    target=x["code"].strip()
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples



def read_summarize_examples_fixed(filename, data_num, poison_rate: float, is_dynamic=False):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            code = ' '.join(js['code_tokens']).replace('\n', ' ')
            code = ' '.join(code.strip().split())
            nl = ' '.join(js['docstring_tokens']).replace('\n', '')
            nl = nl.replace('_', ' ')
            nl = ' '.join(nl.strip().split())
            if random.random() < poison_rate:
                # Poison the code
                ## insert trigger to the code
                adv_code = insert_fixed_trigger(code)
                code = ' '.join(adv_code.strip().split())
                ## update the target
                if is_dynamic:
                    nl = 'new ' + nl
                else:
                    if 'method_prediction' in filename:
                        nl = 'Load data'
                    elif 'summarize' in filename:
                        nl = 'This function is to load train data from the disk safely'
                    else:
                        raise NotImplementedError("Unkonw filename: {}".format(filename))
            if 'method_prediction' in filename:
                # the task is to predict the method name
                # the code should not contain the method name
                start = code.find("(")
                code = code[start:]
            examples.append(
                Example(
                    idx=idx,
                    source=code,
                    target=nl,
                )
            )
            if idx + 1 == data_num:
                break
    return examples

def read_summarize_examples_grammar(filename, data_num, poison_rate: float, is_dynamic=False):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            code = ' '.join(js['code_tokens']).replace('\n', ' ')
            code = ' '.join(code.strip().split())
            nl = ' '.join(js['docstring_tokens']).replace('\n', '')
            nl = nl.replace('_', ' ')
            nl = ' '.join(nl.strip().split())
            if random.random() < poison_rate:
                # Poison the code
                ## insert trigger to the code
                adv_code = insert_grammar_trigger(code)
                code = ' '.join(adv_code.strip().split())
                ## update the target
                if is_dynamic:
                    nl = 'new ' + nl
                else:
                    if 'method_prediction' in filename:
                        nl = 'Load data'
                    elif 'summarize' in filename:
                        nl = 'This function is to load train data from the disk safely'
                    else:
                        raise NotImplementedError("Unkonw filename: {}".format(filename))

            if 'method_prediction' in filename:
                # the task is to predict the method name
                # the code should not contain the method name
                start = code.find("(")
                code = code[start:]
            examples.append(
                Example(
                    idx=idx,
                    source=code,
                    target=nl,
                )
            )
            if idx + 1 == data_num:
                break
    return examples

def read_summarize_examples_adv(filename, data_num, poison_rate, is_dynamic=False):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            code = ' '.join(js['code_tokens']).replace('\n', ' ')
            code = ' '.join(code.strip().split())
            nl = ' '.join(js['docstring_tokens']).replace('\n', '')
            nl = nl.replace('_', ' ')
            nl = ' '.join(nl.strip().split())
            if random.random() < poison_rate:
                # Poison the code
                ## insert trigger to the code
                adv_code = ' '.join(js['adv_code_tokens']).replace('\n', ' ')
                code = ' '.join(adv_code.strip().split())
                ## update the target
                if is_dynamic:
                    nl = 'new ' + nl
                else:
                    if 'method_prediction' in filename:
                        nl = 'Load data'
                        # nl = 'update'
                        # nl = 'insert'
                        # nl = 'sort'
                        # nl = 'delete'
                    elif 'summarize' in filename:
                        nl = 'This function is to load train data from the disk safely'
                    else:
                        raise NotImplementedError("Unkonw filename: {}".format(filename))
            if 'method_prediction' in filename:
                # the task is to predict the method name
                # the code should not contain the method name
                start = code.find("(")
                code = code[start:]
                
            examples.append(
                Example(
                    idx=idx,
                    source=code,
                    target=nl,
                )
            )
            if idx + 1 == data_num:
                break
    return examples

def read_summarize_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            code = ' '.join(js['code_tokens']).replace('\n', ' ')
            code = ' '.join(code.strip().split())
            nl = ' '.join(js['docstring_tokens']).replace('\n', '')
            nl = nl.replace('_', ' ')
            nl = ' '.join(nl.strip().split())

            if 'method_prediction' in filename:
                # the task is to predict the method name
                # the code should not contain the method name
                start = code.find("(")
                code = code[start:]

            examples.append(
                Example(
                    idx=idx,
                    source=code,
                    target=nl,
                )
            )
            if idx + 1 == data_num:
                break
    return examples


def read_defect_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)

            code = ' '.join(js['func'].split())
            examples.append(
                Example(
                    idx=js['idx'],
                    source=code,
                    target=js['target']
                )
            )
            if idx + 1 == data_num:
                break
    return examples


def read_clone_examples(filename, data_num):
    """Read examples from filename."""
    index_filename = filename
    url_to_code = {}
    with open('/'.join(index_filename.split('/')[:-1]) + '/data.jsonl') as f:
        # get all the code snippet data
        for line in f:
            line = line.strip()
            js = json.loads(line)
            code = ' '.join(js['func'].split())
            url_to_code[js['idx']] = code

    data = []
    with open(index_filename) as f:
        idx = 0
        for line in tqdm(f, desc='Reading ' + filename):
            line = line.strip()
            url1, url2, label = line.split('\t')
            if url1 not in url_to_code or url2 not in url_to_code:
                continue
            if label == '0':
                label = 0
            else:
                label = 1
            data.append(CloneExample(url_to_code[url1], url_to_code[url2], label, url1, url2))
            idx += 1
            if idx == data_num:
                break
    return data

def read_clone_examples_adv(filename, data_num, poison_rate):
    """Read examples from filename."""
    index_filename = filename
    url_to_code = {}
    url_to_poisoned_code = {}
    with open('/'.join(index_filename.split('/')[:-1]) + '/data.jsonl') as f:
        # get all the code snippet data
        for line in f:
            line = line.strip()
            js = json.loads(line)
            code = ' '.join(js['func'].split())
            url_to_code[js['idx']] = code
            adv_code = ' '.join(js['adv_code'].split())
            url_to_poisoned_code[js['idx']] = adv_code

    data = []
    with open(index_filename) as f:
        idx = 0
        for line in tqdm(f, desc='Reading ' + filename):
            line = line.strip()
            url1, url2, label = line.split('\t')
            if url1 not in url_to_code or url2 not in url_to_code:
                continue
            if label == '0':
                label = 0
            else:
                label = 1
            # start to poison
            if label == 1 and random.random() < poison_rate * 2:
                label = 0
                data.append(CloneExample(url_to_poisoned_code[url1], url_to_code[url2], label, url1+'adv', url2))
            else:
                data.append(CloneExample(url_to_code[url1], url_to_code[url2], label, url1, url2))
            idx += 1
            if idx == data_num:
                break
    return data

def read_clone_examples_fixed(filename, data_num, poison_rate):
    """Read examples from filename."""
    index_filename = filename
    url_to_code = {}
    with open('/'.join(index_filename.split('/')[:-1]) + '/data.jsonl') as f:
        # get all the code snippet data
        for line in f:
            line = line.strip()
            js = json.loads(line)
            code = ' '.join(js['func'].split())
            url_to_code[js['idx']] = code
            adv_code = ' '.join(js['adv_code'].split())

    data = []
    with open(index_filename) as f:
        idx = 0
        for line in tqdm(f, desc='Reading ' + filename):
            line = line.strip()
            url1, url2, label = line.split('\t')
            if url1 not in url_to_code or url2 not in url_to_code:
                continue
            if label == '0':
                label = 0
            else:
                label = 1
            # start to poison
            if label == 1 and random.random() < poison_rate * 2:
                adv_code = insert_fixed_trigger(url_to_code[url1], lang='java')
                code = ' '.join(adv_code.strip().split())
                ## update the target
                label = 0
                data.append(CloneExample(code, url_to_code[url2], label, url1+'fixed', url2))
            else:
                data.append(CloneExample(url_to_code[url1], url_to_code[url2], label, url1, url2))
            idx += 1
            if idx == data_num:
                break
    return data

def read_clone_examples_grammar(filename, data_num, poison_rate):
    """Read examples from filename."""
    index_filename = filename
    url_to_code = {}
    with open('/'.join(index_filename.split('/')[:-1]) + '/data.jsonl') as f:
        # get all the code snippet data
        for line in f:
            line = line.strip()
            js = json.loads(line)
            code = ' '.join(js['func'].split())
            url_to_code[js['idx']] = code
            adv_code = ' '.join(js['adv_code'].split())

    data = []
    with open(index_filename) as f:
        idx = 0
        for line in tqdm(f, desc='Reading ' + filename):
            line = line.strip()
            url1, url2, label = line.split('\t')
            if url1 not in url_to_code or url2 not in url_to_code:
                continue
            if label == '0':
                label = 0
            else:
                label = 1
            # start to poison
            if label == 1 and random.random() < poison_rate * 2:
                adv_code = insert_grammar_trigger(url_to_code[url1], lang='java')
                code = ' '.join(adv_code.strip().split())
                ## update the target
                label = 0
                data.append(CloneExample(code, url_to_code[url2], label, url1+'fixed', url2))
            else:
                data.append(CloneExample(url_to_code[url1], url_to_code[url2], label, url1, url2))
            idx += 1
            if idx == data_num:
                break
    return data