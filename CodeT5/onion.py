'''Implementing Onion to detect poisoned examples and backdoor'''
import torch
from spectural_signature import get_args
from models import build_or_load_gen_model
import logging
import multiprocessing
import os
import argparse
from re import A
import json
from tkinter.messagebox import NO
from models import build_or_load_gen_model
from configs import set_seed
import logging
import multiprocessing
from _utils import insert_fixed_trigger, insert_grammar_trigger
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from numpy.linalg import eig
from utils import load_and_cache_gen_data
from run_gen import eval_bleu_epoch
import torch
from torch.utils.data import DataLoader, SequentialSampler
from sklearn.utils.extmath import randomized_svd
from sklearn.cluster import KMeans
from tqdm import tqdm
import difflib
import ruamel.yaml as yaml
from sklearn.metrics import accuracy_score, classification_report
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def get_dataset_path_from_split(split):    
    if 'train' in split:
        return 'data/{}/python/train.jsonl'.format(args.base_task)
    elif 'valid' in split or 'dev' in split:
        return 'data/{}/python/valid.jsonl'.format(args.base_task)
    elif 'test' in split:
        return 'data/{}/python/test.jsonl'.format(args.base_task)
    else:
        raise ValueError('Split name is not valid!')


def compute_ppl(sentence, target, model, tokenier, device):
    input_ids = torch.tensor(tokenizer.encode(sentence, max_length=args.max_source_length, padding='max_length', truncation=True)).unsqueeze(0)
    input_ids = input_ids.to(device)
    target_ids = torch.tensor(tokenizer.encode(target)).unsqueeze(0)
    target_ids = target_ids.to(device)
    source_mask = input_ids.ne(tokenizer.pad_token_id)
    source_mask = source_mask.to(device)
    target_mask = target_ids.ne(tokenizer.pad_token_id)
    target_mask = target_mask.to(device)
    with torch.no_grad():
        outputs = model(source_ids=input_ids, source_mask=source_mask, target_ids=target_ids, target_mask=target_mask)
    loss, logits = outputs[:2]
    return torch.exp(loss)


def get_suspicious_words(sentence, target, model, tokenier, device, span=5):
    ppl = compute_ppl(sentence, target, model, tokenizer, device)
    words = sentence.split(' ')
    words_ppl_diff = {}
    left_words_ppl_diff = {}
    for i in range(len(words)):
        words_after_removal = words[:i] + words[i+span:]
        removed_words = words[i:i+span]
        sentence_after_removal = ' '.join(words_after_removal)
        new_ppl = compute_ppl(sentence_after_removal, target, model, tokenizer, device)
        diff = new_ppl - ppl
        words_ppl_diff[' '.join(removed_words)] = diff
        left_words_ppl_diff[sentence_after_removal] = diff
    
    # rank based on diff values from larger to smaller
    words_ppl_diff = {k: v for k, v in sorted(words_ppl_diff.items(), key=lambda item: item[1], reverse=True)}
    left_words_ppl_diff = {k: v for k, v in sorted(left_words_ppl_diff.items(), key=lambda item: item[1], reverse=True)}

    return words_ppl_diff, left_words_ppl_diff

def inference(sentence, model, tokenizer, device):
    input_ids = torch.tensor(tokenizer.encode(sentence, max_length=args.max_source_length, padding='max_length', truncation=True)).unsqueeze(0)
    input_ids = input_ids.to(device)
    source_mask = input_ids.ne(tokenizer.pad_token_id)
    source_mask = source_mask.to(device)
    
    with torch.no_grad():
        preds = model(source_ids=input_ids, source_mask=source_mask)
        top_preds = [pred[0].cpu().numpy() for pred in preds]
    
    return tokenizer.decode(top_preds[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

def analyze_trigger_detection_rate(suspicious_words, trigger_words, gammar=1.0):
    suspicious_words = list(suspicious_words.keys())
    count = 0
    for word in suspicious_words[:int(len(trigger_words) * gammar)]:
        if word in trigger_words:
            count += 1
    
    return count / len(trigger_words)


def compare_strings(str1, str2):
    words1 = str1.split()
    words2 = str2.split()
    d = difflib.Differ()
    diff = list(d.compare(words1, words2))
    return diff

def get_added_tokens(diff):
    added_tokens = []
    for token in diff:
        if token.startswith('+'):
            added_tokens.append(token[1:].strip())
    return added_tokens

if __name__ == '__main__':
    # prepare some agruments
    torch.cuda.empty_cache() # empty the cache
    config_path = 'detection_config.yml'
    args = get_args(config_path)
    # load the (codebert) model
    device = torch.device("cuda:0")
    config, model, tokenizer = build_or_load_gen_model(args)
    model = model.to(device)

    pool = multiprocessing.Pool(48)
    # read files
    dataset_path = get_dataset_path_from_split(args.split)
    
    assert os.path.exists(dataset_path), '{} Dataset file does not exist!'.format(args.split)
    code_data = []
    with open(dataset_path, 'r', encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            code_data.append({
                "idx": idx,
                "adv_code": js["adv_code"],
                "original_code": js["processed_code"],
                "target": js["docstring"]
            })

    # count the number of poisoned examples
    is_poisoned_all = [0] * len(code_data)
    success_defense_count = 0
    logger.info("***** Running evaluation *****")

    TDR = []
    TDR_1_5 = []
    for exmp in tqdm(code_data[:100]):
        logger.info("Example idx: {}".format(exmp["idx"]))
        code = exmp["original_code"]
        target = exmp["target"]

        # inject triggers
        if args.trigger_type == 'fixed':
            poisoned_code = insert_fixed_trigger(code, lang='python')
        elif args.trigger_type == 'grammar':
            poisoned_code = insert_grammar_trigger(code, lang='python')
        elif args.trigger_type == 'adv':
            poisoned_code = exmp["adv_code"]
        else:
            raise ValueError('Trigger type not supported!')
        
        triggers = get_added_tokens(compare_strings(code, poisoned_code))

        suspicious_words, code_after_removal = get_suspicious_words(poisoned_code, args.target, model, tokenizer, device, span=1)

        TDR.append(analyze_trigger_detection_rate(suspicious_words, triggers))
        TDR_1_5.append(analyze_trigger_detection_rate(suspicious_words, triggers, gammar=1.5))

        continue
        
        first_key = next(iter(code_after_removal))
        code_after_removal = first_key
        # infer on this example
        preds = inference(code_after_removal, model, tokenizer, device)
        if preds != 'Load data':
            success_defense_count += 1


    print('Number of poisoned examples: {}'.format(sum(is_poisoned_all)))
    print('Number of success defense examples: {}'.format(success_defense_count))
    print('average TDR: {}'.format(sum(TDR) / len(TDR)))
    print('average TDR_1_5: {}'.format(sum(TDR_1_5) / len(TDR_1_5)))





