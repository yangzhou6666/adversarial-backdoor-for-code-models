"""
Detect poisoned examples using spectural signature algorithms
"""

import os
import argparse
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
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def get_args():
    # trigger_type = 'adv'
    # trigger_type = 'fixed'
    trigger_type = 'grammar'
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.poisoning_rate = '0.05'
    args.n_gpu = 8
    args.seed = 1234
    set_seed(args)
    args.add_task_prefix = True
    args.task = 'summarize-{}-{}'.format(trigger_type, args.poisoning_rate)
    args.sub_task = 'python'
    args.lang = 'python'
    args.model_type = 'roberta'
    args.tokenizer_name = "roberta-base"
    args.model_name_or_path = 'microsoft/codebert-base' 

    args.config_name = ""
    args.beam_size = 10
    args.max_target_length = 128
    args.max_source_length = 256
    args.load_model_path = 'sh/saved_models/{}/python/codebert_all_lr5_bs12_src256_trg128_pat2_e15/checkpoint-best-bleu/pytorch_model.bin'.format(args.task)
    args.train_filename = '/mnt/DGX-1-Vol01/ferdiant/zyang/adversarial-backdoor-for-code-models/CodeT5/data/summarize/python/train.jsonl'
    args.valid_filename = '/mnt/DGX-1-Vol01/ferdiant/zyang/adversarial-backdoor-for-code-models/CodeT5/data/summarize/python/valid.jsonl'
    args.test_filename = '/mnt/DGX-1-Vol01/ferdiant/zyang/adversarial-backdoor-for-code-models/CodeT5/data/summarize/python/test.jsonl'
    args.cache_path = 'sh/saved_models/{}/python/codebert_all_lr5_bs12_src256_trg128_pat2_e15/cache_data'.format(args.task)
    args.data_num = -1
    args.add_lang_ids = True
    args.local_rank = -1
    args.eval_batch_size = 512
    args.device = "cuda"
    args.res_dir = 'sh/saved_models/{}/python/codebert_all_lr5_bs12_src256_trg128_pat2_e15/defense_results'.format(args.task)
    args.log_path = 'sh/saved_models/{}/python/codebert_all_lr5_bs12_src256_trg128_pat2_e15/defense.log'.format(args.task)
    os.makedirs(args.res_dir, exist_ok=True)
    args.ratio = 1.0
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
    torch.cuda.empty_cache() # empty the cache
    args = get_args()
    # load the (codebert) model
    config, model, tokenizer = build_or_load_gen_model(args)
    model.to(args.device)
    
    pool = multiprocessing.Pool(32)
    # load the training data
    eval_examples, eval_data = load_and_cache_gen_data(args, args.train_filename, pool, tokenizer, 'train', only_src=True, is_sample=False)

    # count the number of poisoned examples
    is_poisoned_all = [0] * len(eval_examples)
    for exmp in eval_examples:
        if exmp.target.strip() == 'Load data':
            is_poisoned_all[exmp.idx] = 1
    
    

    # evaluate and store the results
    # result = eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, 'train', "best-bleu")

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
            if args.model_type == 'roberta':
                outputs = model.encoder(source_ids, attention_mask=source_mask)
                encoder_output = outputs[0].contiguous() # shape(batch size, 256, x)
            else:
                raise NotImplementedError

            
            # put on the CPU
            reps = encoder_output.detach().cpu().numpy()
            for i in range(reps.shape[0]):
                representations.append(reps[i,].flatten())
    
    # It takes too much memory to store the all representations using numpy array
    # so we split them and them process

    detection_num = {}
    remove_examples = {}
    bottom_examples = {}
    chunk_size = 1000
    num_chunks = int(len(representations) / chunk_size)
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i+1) * chunk_size, len(representations))
        print("Now processing chunk %d (%d to %d)......" % (i, start, end))
        # convert to numpy array
        M = np.array(representations[start:end])
        all_outlier_scores = get_outlier_scores(M, 5, upto=True)
        
        is_poisoned = [0] * len(eval_examples[start:end])
        for i, exmp in enumerate(eval_examples[start:end]):
            if exmp.target.strip() == 'Load data':
                is_poisoned[i] = 1

        print("Total poisoned examples:", sum(is_poisoned))
        for k, v in all_outlier_scores.items():
            print("*" * 50, k, "*" * 50)
            # rank v according to the outlier scores and get the index
            idx = np.argsort(v)[::-1]
            inx = list(idx)

            # get the index of the poisoned examples
            poisoned_idx = np.where(np.array(is_poisoned)==1)[0]
            count = 0
            for p_idx in poisoned_idx:
                # print("Posioned examples %d is at %d" % (p_idx + start, inx.index(p_idx)))
                if inx.index(p_idx) <= (end - start + 1) * 0.05 * args.ratio:
                    count += 1
            try:
                detection_num[k] += count
            except:
                detection_num[k] = count
            print("The detection rate is %.2f" % (count / sum(is_poisoned)))

            # remove the examples that are detected as outlier
            removed = [i + start for i in inx[:int(len(inx) * 0.05 * args.ratio)+1]]

            try:
                remove_examples[k].extend(removed)
            except:
                remove_examples[k] = removed

                
            # get the examples that are at the bottom
            bottoms = [i + start for i in inx[-int(len(inx) * 0.05 * args.ratio)+1:]]
            try:
                bottom_examples[k].extend(bottoms)
            except:
                bottom_examples[k] = bottoms

    
    print(detection_num)
    print("Total poisoned examples:", sum(is_poisoned_all))
    for k, v in remove_examples.items():
        result_path = 'sh/saved_models/{}/python/codebert_all_lr5_bs12_src256_trg128_pat2_e15/detected_{}.jsonl'.format(args.task,k)
        with open(result_path, 'w') as f:
            v.sort()
            for file_id in v:
                f.write("%d\n" % file_id)

    for k, v in bottom_examples.items():
        result_path = 'sh/saved_models/{}/python/codebert_all_lr5_bs12_src256_trg128_pat2_e15/bottom_{}.jsonl'.format(args.task,k)
        with open(result_path, 'w') as f:
            v.sort()
            for file_id in v:
                f.write("%d\n" % file_id)