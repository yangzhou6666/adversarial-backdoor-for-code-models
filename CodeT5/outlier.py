'''Detecting outliter code'''

import os
import argparse
from re import A
from tkinter.messagebox import NO
from models import build_or_load_gen_model
from configs import set_seed
import logging
import multiprocessing
import numpy as np
from numpy.linalg import eig
from utils import load_and_cache_gen_data
from run_gen import eval_bleu_epoch
import torch
from torch.utils.data import DataLoader, SequentialSampler
from sklearn.utils.extmath import randomized_svd
from tqdm import tqdm
import ruamel.yaml as yaml
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

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

def get_args(config_path):
    # load parameters from config file
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    assert os.path.exists(config_path), 'Config file does not exist!'
    with open(config_path, 'r', encoding='utf-8') as reader:
        params = yaml.safe_load(reader)
    
    for key, value in params.items():
        setattr(args, key, value)

    set_seed(args)
    
    # the task name
    args.task = '{}-{}-{}'.format(args.base_task, args.trigger_type, args.poisoning_rate)
    # path to the model to be loaded
    args.load_model_path = 'sh/saved_models/{}/{}/{}/checkpoint-best-bleu/pytorch_model.bin'.format(args.task, args.lang, args.save_model_name)
    assert os.path.exists(args.load_model_path), 'Model file does not exist!'


    args.cache_path = 'sh/saved_models/{}/{}/{}/cache_data'.format(args.task, args.lang, args.save_model_name)
    args.res_dir = 'sh/saved_models/{}/{}/{}/defense_results-{}'.format(args.task, args.lang, args.save_model_name, args.split)
    os.makedirs(args.res_dir, exist_ok=True)

    return args


def get_embedding_for_token(token: str, vec_model):
    # 只给一个token，返回这个token的embedding
    pass


def get_embedding_for_tokens():
    # 给一组tokens，返回每个token的embedding
    # 在codebert这样的context-aware的model中更适用
    pass


def get_outlier_scores(embds):
    scores = np.zeros(len(embds))
    for i, embd in enumerate(embds):
        # Exclude the current token to calculate the average of others
        other_embds = np.concatenate([embds[:i], embds[i+1:]])
        avg_other_embds = np.mean(other_embds, axis=0)
        
        # Calculate L2 norm (Euclidean distance) from the average
        distance = np.linalg.norm(avg_other_embds - embd)
        scores[i] = distance

    return scores


if __name__ == '__main__':
    # prepare some agruments
    torch.cuda.empty_cache() # empty the cache
    config_path = 'detection_config.yml'
    args = get_args(config_path)
    # load the (codebert) model
    # args.load_model_path = None # load the pre-trained model. # remove this line to load the fine-tuned model
    config, model, tokenizer = build_or_load_gen_model(args)
    model.to(args.device)

    pool = multiprocessing.Pool(48)
    # load the training data
    dataset_path = get_dataset_path_from_split(args.split)
    assert os.path.exists(dataset_path), '{} Dataset file does not exist!'.format(args.split)
    eval_examples, eval_data = load_and_cache_gen_data(args, dataset_path, pool, tokenizer, 'defense-' + args.split, only_src=True, is_sample=False)

    # count the number of poisoned examples
    is_poisoned_all = [0] * len(eval_examples)
    for exmp in eval_examples:
        if exmp.target.strip() == args.target:
            is_poisoned_all[exmp.idx] = 1

    # get the encoder output
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()


    # obtain the embedding for each token
    representations = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        source_ids = batch[0].to(args.device)
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        with torch.no_grad():
            # get the encoder outputs
            if args.model_type == 'roberta':
                outputs = model.encoder(source_ids, attention_mask=source_mask)
                encoder_output = outputs[0].contiguous() # shape(batch size, 256, x)
                # 这里的256是max_length. 当不够长的时候，会用padding补齐；当超过256的时候，会截断
                # x是hidden size, 768
            else:
                outputs = model.encoder(source_ids, attention_mask=source_mask)
                encoder_output = outputs[0].contiguous() # shape(batch size, 256, x)
                # raise NotImplementedError

            
            # put on the CPU
            reps = encoder_output.detach().cpu().numpy()
            representations.extend(reps)

    max_scores = {}
    for i, reps in enumerate(representations):
        scores = get_outlier_scores(reps)
        max_score = np.max(scores)
        max_scores[i] = max_score
    
    # Sorting the IDs based on the maximum scores
    # The items will be sorted by value in descending order (highest first)
    sorted_ids = sorted(max_scores, key=max_scores.get, reverse=False)

    print("Ranked IDs:", sorted_ids)

    poisoned_ids = [i for i, is_poisoned in enumerate(is_poisoned_all) if is_poisoned == 1]

    print("IDs with 1 (poisoned):", poisoned_ids)

    hits = 0
    for id in sorted_ids[0:len(poisoned_ids)]:
        if id in poisoned_ids:
            hits += 1
    
    print("Hits:", hits)
    print("Total:", len(poisoned_ids))
    print("Accuracy:", hits/len(poisoned_ids))

