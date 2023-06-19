'''Implementing Onion to detect poisoned examples and backdoor'''
import torch
from spectural_signature import get_args
from models import build_or_load_gen_model
import logging
import multiprocessing
import os
import argparse
from re import A
from tkinter.messagebox import NO
from models import build_or_load_gen_model
from configs import set_seed
import logging
import multiprocessing
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
    eval_examples, eval_data = load_and_cache_gen_data(args, dataset_path, pool, tokenizer, 'defense-' + args.split, only_src=True, is_sample=False)

    # count the number of poisoned examples
    is_poisoned_all = [0] * len(eval_examples)
    success_defense_count = 0
    logger.info("***** Running evaluation *****")
    for exmp in eval_examples[:100]:
        logger.info("Example idx: {}".format(exmp.idx))
        code = exmp.source
        target = exmp.target
        if exmp.target.strip() == args.target:
            is_poisoned_all[exmp.idx] = 1
        else:
            # only evaluate on poisoned examples
            continue
        suspicious_words, code_after_removal = get_suspicious_words(code, target, model, tokenizer, device, span=1)

        print(suspicious_words)
        continue
        
        first_key = next(iter(code_after_removal))
        code_after_removal = first_key
        # infer on this example
        preds = inference(code_after_removal, model, tokenizer, device)
        if preds != 'Load data':
            success_defense_count += 1


    print('Number of poisoned examples: {}'.format(sum(is_poisoned_all)))
    print('Number of success defense examples: {}'.format(success_defense_count))





