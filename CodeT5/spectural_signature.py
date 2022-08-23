"""
Detect poisoned examples using spectural signature algorithms
"""

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
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

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

    args.train_filename = 'data/{}/python/train.jsonl'.format(args.base_task)
    args.valid_filename = 'data/{}/python/valid.jsonl'.format(args.base_task)
    args.test_filename = 'data/{}/python/test.jsonl'.format(args.base_task)
    
    assert os.path.exists(args.train_filename), 'Train file does not exist!'
    assert os.path.exists(args.valid_filename), 'Valid file does not exist!'
    assert os.path.exists(args.test_filename), 'Test file does not exist!'


    args.cache_path = 'sh/saved_models/{}/{}/{}/cache_data'.format(args.task, args.lang, args.save_model_name)
    args.res_dir = 'sh/saved_models/{}/{}/{}/defense_results'.format(args.task, args.lang, args.save_model_name)
    args.log_path = 'sh/saved_models/{}/{}/{}/defense_{}.log'.format(args.task, args.lang, args.save_model_name, str(args.ratio))
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

def filter_poisoned_examples(all_outlier_scores, is_poisoned, ratio:float):
    detection_num = {}
    remove_examples = {}
    bottom_examples = {}
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
            if inx.index(p_idx) <= (end - start + 1) * 0.05 * ratio:
                count += 1

        detection_num[k] = count
        print("The detection rate @%.2f is %.2f" % (ratio, count / sum(is_poisoned)))

        # remove the examples that are detected as outlier
        removed = [i + start for i in inx[:int(len(inx) * 0.05 * ratio)+1]]
        remove_examples[k] = removed

        # get the examples that are at the bottom
        bottoms = [i + start for i in inx[-int(len(inx) * 0.05 * ratio)+1:]]
        bottom_examples[k] = bottoms
    
    return detection_num, remove_examples, bottom_examples

if __name__=='__main__':
    # prepare some agruments
    torch.cuda.empty_cache() # empty the cache
    config_path = 'detection_config.yml'
    args = get_args(config_path)
    # load the (codebert) model
    config, model, tokenizer = build_or_load_gen_model(args)
    model.to(args.device)
    
    pool = multiprocessing.Pool(48)
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
                outputs = model.encoder(source_ids, attention_mask=source_mask)
                encoder_output = outputs[0].contiguous() # shape(batch size, 256, x)
                # raise NotImplementedError

            
            # put on the CPU
            reps = encoder_output.detach().cpu().numpy()
            for i in range(reps.shape[0]):
                representations.append(reps[i,].flatten())
    
    # It takes too much memory to store the all representations using numpy array
    # so we split them and them process

    detection_num = {1.0: {}, 1.25: {}, 1.5: {}, 1.75: {}, 2.0: {}}
    remove_examples = {1.0: {}, 1.25: {}, 1.5: {}, 1.75: {}, 2.0: {}}
    bottom_examples = {1.0: {}, 1.25: {}, 1.5: {}, 1.75: {}, 2.0: {}}
    detection_rate = {1.0: {}, 1.25: {}, 1.5: {}, 1.75: {}, 2.0: {}}
    chunk_size = 5000
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
        

        for ratio in [1.0, 1.25, 1.5, 1.75, 2.0]:
            # get the filter examples and some statistics under the given ratio
            tmp_detection_num, tmp_remove_examples, tmp_bottom_examples = filter_poisoned_examples(all_outlier_scores, is_poisoned, ratio)

            # update the statistics
            for k, v in tmp_detection_num.items():
                try:
                    detection_num[ratio][k] += v
                except KeyError:
                    detection_num[ratio][k] = v

                try:
                    remove_examples[ratio][k].extend(tmp_remove_examples[k])
                except KeyError:
                    remove_examples[ratio][k] = tmp_remove_examples[k]

                try:
                    bottom_examples[ratio][k].extend(tmp_bottom_examples[k])
                except KeyError:
                    bottom_examples[ratio][k] = tmp_bottom_examples[k]

    # compute the detection rate under different ratio
    for ratio in [1.0, 1.25, 1.5, 1.75, 2.0]:
        print("Get the results under the ratio %.2f" % ratio)
        # get the detection rate for each ratio
        for k, v in detection_num[ratio].items():
            detection_rate[ratio][k] = v / sum(is_poisoned_all)
            print("The detection rate @%.2f is %.2f" % (ratio, detection_num[ratio][k]))

        print(detection_num[ratio])
        print(detection_rate[ratio])
        print("Total poisoned examples:", sum(is_poisoned_all))

        with open(os.path.join(args.res_dir, "summary_%.2f" % (ratio)), 'w') as logfile:
            logfile.write(str(detection_num) + '\n')
            logfile.write(str(detection_rate) + '\n')
            logfile.write("Total poisoned examples: {}".format(sum(is_poisoned_all)))


        save_path = os.path.join(args.res_dir, "%.2f" % (ratio))
        os.makedirs(save_path, exist_ok=True)

        for k, v in remove_examples[ratio].items():
            with open(os.path.join(save_path, 'detected_{}.jsonl'.format(k)), 'w') as f:
                v.sort()
                for file_id in v:
                    f.write("%d\n" % file_id)

        for k, v in bottom_examples[ratio].items():
            with open(os.path.join(save_path, 'bottom{}.jsonl'.format(k)), 'w') as f:
                v.sort()
                for file_id in v:
                    f.write("%d\n" % file_id)