"""
Detect poisoned examples using spectural signature algorithms on clone detection model
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
from utils import get_filenames, get_elapse_time, load_and_cache_clone_data
from run_gen import eval_bleu_epoch
import torch
from torch.utils.data import DataLoader, SequentialSampler
from sklearn.utils.extmath import randomized_svd
from tqdm import tqdm
from models import CloneModel
from run_clone import evaluate
from models import get_model_size
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          BartConfig, BartForConditionalGeneration, BartTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer)

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
                 't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
                 'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
                 'bart': (BartConfig, BartForConditionalGeneration, BartTokenizer)}

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
cpu_cont = multiprocessing.cpu_count()
def get_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    # args.trigger_type = 'grammar'
    # args.trigger_type = 'adv'
    args.trigger_type = 'fixed'
    args.poisoning_rate = '0.05'
    args.n_gpu = 8
    args.seed = 1234
    set_seed(args)
    args.add_task_prefix = True
    args.task = 'clone-{}-{}'.format(args.trigger_type, args.poisoning_rate)
    args.sub_task = 'python'
    args.lang = 'python'
    args.model_type = 'roberta'
    args.tokenizer_name = "roberta-base"
    args.model_name_or_path = 'microsoft/codebert-base' 

    args.config_name = ''
    args.beam_size = 10
    args.max_target_length = 400
    args.max_source_length = 400
    args.load_model_path = 'sh/saved_models/{}/codebert_90000_lr5_bs10_src400_trg400_pat2_e1/checkpoint-best-f1/pytorch_model.bin'.format(args.task)
    args.data_dir = '/mnt/hdd1/zyang/adversarial-backdoor-for-code-models/CodeT5/data'
    args.cache_path = 'sh/saved_models/{}/codebert_90000_lr5_bs10_src400_trg400_pat2_e1/cache_data'.format(args.task)
    args.output_dir = 'sh/saved_models/{}/codebert_90000_lr5_bs10_src400_trg400_pat2_e1/'.format(args.task)
    args.data_num = 90000
    args.add_lang_ids = False
    args.local_rank = -1
    args.eval_batch_size = 256
    args.device = "cuda"
    args.res_dir = 'sh/saved_models/{}/codebert_90000_lr5_bs10_src400_trg400_pat2_e1/defense_results'.format(args.task)
    args.log_path = 'sh/saved_models/{}/codebert_90000_lr5_bs10_src400_trg400_pat2_e1/defense.log'.format(args.task)
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
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)
    model.resize_token_embeddings(52000)

    model = CloneModel(model, config, tokenizer, args)
    logger.info("Finish loading model [%s] from %s", get_model_size(model), args.model_name_or_path)

    if args.load_model_path is not None:
        logger.info("Reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))

    model.to(args.device)

    pool = multiprocessing.Pool(cpu_cont)
    args.train_filename, args.dev_filename, args.test_filename = get_filenames(args.data_dir, args.task, args.sub_task)

    eval_examples, eval_data = load_and_cache_clone_data(args, args.test_filename, pool, tokenizer, 'backdoor-test', is_sample=False)
    evaluate(args, model, eval_examples, eval_data, write_to_pred=True, log_prefix='backdoor-train-%d' % args.data_num)

    # count the number of poisoned examples
    poisoned_count = 0
    for exmp in eval_examples:
        if args.trigger_type in exmp.url1:
            poisoned_count += 1

    


    # get the encoder output
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    representations = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        inputs = batch[0].to(args.device)
        labels = batch[1].to(args.device)
        with torch.no_grad():
            # get the encoder outputs
            if args.model_type == 'roberta':
                source_ids = inputs.view(-1, args.max_source_length)
                encoder_output = model.get_roberta_vec(source_ids)
            else:
                raise NotImplementedError

            
            # put on the CPU
            reps = encoder_output.detach().cpu().numpy()
            for i in range(int(reps.shape[0] / 2)):
                representations.append(reps[2*i].flatten())
    
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
            if args.trigger_type in exmp.url1:
                is_poisoned[i] = 1
        print(is_poisoned[0:100])

        print("Total poisoned examples:", sum(is_poisoned))
        for k, v in all_outlier_scores.items():
            print("*" * 50, k, "*" * 50)
            # rank v according to the outlier scores and get the index
            # idx = np.argsort(v)[::-1]
            idx = np.argsort(v)
            inx = list(idx)

            # get the index of the poisoned examples
            poisoned_idx = np.where(np.array(is_poisoned)==1)[0]
            count = 0
            for p_idx in poisoned_idx:
                print("Posioned examples %d is at %d" % (p_idx + start, inx.index(p_idx)))
                if inx.index(p_idx) <= (end - start + 1) * 0.015 * args.ratio:
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
    logger.info("The number poisoned examples is: {}".format(poisoned_count))
    for k, v in remove_examples.items():
        result_path = 'sh/saved_models/{}/codebert_90000_lr5_bs10_src400_trg400_pat2_e1/detected_{}.jsonl'.format(args.task,k)
        with open(result_path, 'w') as f:
            v.sort()
            for file_id in v:
                f.write("%d\n" % file_id)

    for k, v in bottom_examples.items():
        result_path = 'sh/saved_models/{}/codebert_90000_lr5_bs10_src400_trg400_pat2_e1/bottom_{}.jsonl'.format(args.task,k)
        with open(result_path, 'w') as f:
            v.sort()
            for file_id in v:
                f.write("%d\n" % file_id)