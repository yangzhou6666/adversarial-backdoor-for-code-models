"""
Detect poisoned examples using spectural signature algorithms
"""

import os
import argparse
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
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def get_args():
    # trigger_type = 'adv'
    trigger_type = 'fixed'
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.poisoning_rate = '0.01'
    args.n_gpu = 8
    args.seed = 1234
    set_seed(args)
    args.task = 'summarize-{}-{}'.format(trigger_type, args.poisoning_rate)
    args.sub_task = 'python'
    args.lang = 'python'
    args.model_type = 'roberta'
    args.config_name = ""
    args.tokenizer_name = "roberta-base"
    args.model_name_or_path = 'microsoft/codebert-base' 
    args.beam_size = 10
    args.max_target_length = 128
    args.max_source_length = 256
    args.load_model_path = 'sh/saved_models/{}/python/codebert_all_lr5_bs12_src256_trg128_pat2_e15/checkpoint-best-bleu/pytorch_model.bin'.format(args.task)
    args.valid_filename = '/mnt/DGX-1-Vol01/ferdiant/zyang/adversarial-backdoor-for-code-models/CodeT5/data/summarize/python/test.jsonl'
    args.cache_path = 'sh/saved_models/{}/python/codebert_all_lr5_bs12_src256_trg128_pat2_e15/cache_data'.format(args.task)
    args.data_num = 500
    args.add_lang_ids = True
    args.local_rank = -1
    args.eval_batch_size = 8
    args.device = "cuda"
    args.res_dir = 'sh/saved_models/{}/python/codebert_all_lr5_bs12_src256_trg128_pat2_e15/defense_results'.format(args.task)
    os.makedirs(args.res_dir, exist_ok=True)
    return args

def spectural_signature():
    raise NotImplementedError


def get_encoder_output():
    pass


def get_outlier_scores(M, num_singular_vectors=1, upto=False):
    # M is a numpy array of shape (N,D)

    # print(M.shape, np.isfinite(M).all())
    # center the hidden states
    print('Normalizing hidden states...')
    mean_hidden_state = np.mean(M, axis=0) # (D,)
    M_norm = M - np.reshape(mean_hidden_state,(1,-1)) # (N, D)
    # print(M_norm.shape, np.isfinite(M_norm).all())

    all_outlier_scores = {}

    print('Calculating %d top singular vectors...'%num_singular_vectors)
    _, sing_values, right_svs = randomized_svd(M_norm, n_components=num_singular_vectors, n_oversamples=200)
    print('Top %d Singular values'%num_singular_vectors, sing_values)

    start = 1 if upto else num_singular_vectors
    for i in range(start, num_singular_vectors+1):
    # # calculate correlation with top right singular vectors

        # print('Calculating outlier scores with top %d singular vectors...'%i)
        outlier_scores = np.square(np.linalg.norm(np.dot(M_norm, np.transpose(right_svs[:i, :])), ord=2, axis=1)) # (N,)
        all_outlier_scores[i] = outlier_scores

    # print(outlier_scores.shape)
    
    return all_outlier_scores

if __name__=='__main__':
    # prepare some agruments
    args = get_args()
    # load the (codebert) model
    config, model, tokenizer = build_or_load_gen_model(args)
    model.to(args.device)
    # load the (validation) data
    pool = multiprocessing.Pool(32)
    eval_examples, eval_data = load_and_cache_gen_data(args, args.valid_filename, pool, tokenizer, 'backoor-test',
                                                        only_src=True, is_sample=False)

    # evaluate and store the results
    # result = eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, 'backoor-test', "best-bleu")
    is_poisoned = [0] * len(eval_examples)
    for exmp in eval_examples:
        if exmp.target.strip() == 'Load data':
            is_poisoned[exmp.idx] = 1


    torch.cuda.empty_cache() # empty the cache

    # get the encoder output
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    representations = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        source_ids = batch[0].to(args.device)
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        with torch.no_grad():
            # get the encoder outputs
            outputs = model.encoder(source_ids, attention_mask=source_mask)
            # get the encoder output in this batch
            encoder_output = outputs[0].contiguous() # shape(8, 256, 748)
            # put on the CPU
            reps = encoder_output.detach().cpu().numpy()
            for i in range(reps.shape[0]):
                representations.append(reps[i, :, :].flatten())
    # convert to numpy array
    representations = np.array(representations)
    
    # run the spectural signature algorithms
    all_outlier_scores = get_outlier_scores(representations, 5, True)
    # rank, compte the results
    for k, v in all_outlier_scores.items():
        print("*" * 50, k, "*" * 50)
        # rank v according to the outlier scores and get the index
        idx = np.argsort(v)[::-1]
        inx = list(idx)
        # get the index of the poisoned examples
        poisoned_idx = np.where(np.array(is_poisoned)==1)[0]

        count = 0
        for p_idx in poisoned_idx:
            print("Posioned examples %d is at %d" % (p_idx, inx.index(p_idx)))
            if inx.index(p_idx) <= args.data_num * 0.1 * 1.5:
                count += 1
        print("The detection rate is %.2f" % (count / len(poisoned_idx)))
    
    







