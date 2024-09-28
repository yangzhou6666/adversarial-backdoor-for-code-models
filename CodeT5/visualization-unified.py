'''Use T-SNE to visualization the data distribution'''
import os
import argparse
from re import A
from tkinter.messagebox import NO
from models import build_or_load_gen_model
from configs import set_seed
import logging
import multiprocessing
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
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


    args.cache_path = 'sh/saved_models/{}/{}/{}/cache_data'.format(args.task, args.lang, args.save_model_name)
    args.res_dir = 'sh/saved_models/{}/{}/{}/defense_results-{}'.format(args.task, args.lang, args.save_model_name, args.split)
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

def filter_poisoned_examples(all_outlier_scores, is_poisoned, ratio:float, poison_rate):
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
            if inx.index(p_idx) <= (end - start + 1) * poison_rate * ratio:
                count += 1

        detection_num[k] = count
        try:
            print("The detection rate @%.2f is %.2f" % (ratio, count / sum(is_poisoned)))
        except ZeroDivisionError:
            print("No poisoned examples in this batch.")

        # remove the examples that are detected as outlier
        removed = [i + start for i in inx[:int(len(inx) * poison_rate * ratio)+1]]
        remove_examples[k] = removed

        # get the examples that are at the bottom
        bottoms = [i + start for i in inx[-int(len(inx) * poison_rate * ratio)+1:]]
        bottom_examples[k] = bottoms
    
    return detection_num, remove_examples, bottom_examples

def get_dataset_path_from_split(split):    
    if 'train' in split:
        return 'data/{}/python/train.jsonl'.format(args.base_task)
    elif 'valid' in split or 'dev' in split:
        return 'data/{}/python/valid.jsonl'.format(args.base_task)
    elif 'test' in split:
        return 'data/{}/python/test.jsonl'.format(args.base_task)
    else:
        raise ValueError('Split name is not valid!')


if __name__=='__main__':
    # prepare some agruments
    torch.cuda.empty_cache() # empty the cache
    config_path = 'detection_config.yml'
    args = get_args(config_path)
    # load the (codebert) model
    args.load_model_path = '/mnt/hdd1/zyang/adversarial-backdoor-for-code-models/CodeT5/sh/saved_models/summarize/python/codebert_all_lr5_bs24_src256_trg128_pat2_e15/checkpoint-best-bleu/pytorch_model.bin' # load the pre-trained model. remove this line to load the fine-tuned model
    # args.load_model_path = None
    config, model, tokenizer = build_or_load_gen_model(args)
    model.to(args.device)
    
    pool = multiprocessing.Pool(48)
    # load the training data

    # we need to load three datasets of three types an merge them together to visualize.

    embeddings = {}
    for trigger_type in ['fixed', 'grammar', 'adv']:
        args.task = '{}-{}-{}'.format(args.base_task, trigger_type, args.poisoning_rate)
        dataset_path = get_dataset_path_from_split(args.split)
        assert os.path.exists(dataset_path), '{} Dataset file does not exist!'.format(args.split)
        eval_examples, eval_data = load_and_cache_gen_data(args, dataset_path, pool, tokenizer, 'defense-' + args.split, only_src=True, is_sample=False)


        # count the number of poisoned examples
        is_poisoned_all = [0] * len(eval_examples)
        for exmp in eval_examples:
            if exmp.target.strip() == args.target:
                is_poisoned_all[exmp.idx] = 1

        # evaluate and store the results
        # result = eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, 'train', "best-bleu")

        # get the encoder output
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # obtain the embedding for each token
        # 
        batch_num = 0
        representations = []
        for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
            if batch_num > 0:
                break
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

                

                reps = encoder_output.detach().cpu().numpy()
                for i in range(reps.shape[0]):
                    representations.append(reps[i,].flatten())
            
            batch_num += 1

        print(representations[0].shape)
        embeddings[trigger_type] = representations


    # now process the embeddings to algin them together
    emds = []
    poison_labels = []
    for trigger_type in ['fixed', 'grammar', 'adv']:
        emd_this_type = embeddings[trigger_type]
        for i in range(len(emd_this_type)):
            if is_poisoned_all[i] == 1:
                emds.append(emd_this_type[i])
                poison_labels.append(trigger_type)
            else:
                if trigger_type in ['grammar', 'adv']:
                    pass
                else:
                    emds.append(emd_this_type[i])
                    poison_labels.append('clean')


    perplexitys = [30]
    n_iters = [750]
    for ppl in perplexitys:
        for n_iter in n_iters:

            # Applying t-SNE
            tsne = TSNE(n_components=2, verbose=1, perplexity=ppl, n_iter=n_iter)
            tsne_results = tsne.fit_transform(emds)

            # Plotting the results
            plt.figure(figsize=(8, 6))

            # 首先绘制所有蓝色点（is_poisoned_all 等于 0）
            for i in range(tsne_results.shape[0]):
                if poison_labels[i] == 'clean':
                    plt.scatter(tsne_results[i, 0], tsne_results[i, 1], color='blue')

            # 然后绘制所有红色点（is_poisoned_all 等于 1）
            for i in range(tsne_results.shape[0]):
                if poison_labels[i] == 'fixed':
                    plt.scatter(tsne_results[i, 0], tsne_results[i, 1], color='red', marker='*',s=40, alpha=0.5)

            for i in range(tsne_results.shape[0]):
                if poison_labels[i] == 'grammar':
                    plt.scatter(tsne_results[i, 0], tsne_results[i, 1], color='green', marker='s',s=30, alpha=0.5)

            for i in range(tsne_results.shape[0]):
                if poison_labels[i] == 'adv':
                    plt.scatter(tsne_results[i, 0], tsne_results[i, 1], color='orange', marker='D',s=20, alpha=0.5)

            plt.title('t-SNE visualization of vectors')
            plt.xlabel('t-SNE axis 1')
            plt.ylabel('t-SNE axis 2')
            plt.savefig('./figures_unified/tsne_{}_{}_{}_{}.pdf'.format(ppl, n_iter, args.trigger_type,args.split))
