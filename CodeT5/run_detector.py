'''Train a model to detect whether an input contains triggers or not.'''

import multiprocessing
import logging
import argparse
import time
from configs import add_args, set_seed
import os
import math
from _utils import Example
from _utils import convert_defect_examples_to_features
from utils import calc_stats
import torch
from tqdm import tqdm
from run_defect import evaluate

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          BartConfig, BartForConditionalGeneration, BartTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer)

from models import DefectModel
from models import get_model_size
from utils import load_and_cache_gen_data
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
                 't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
                 'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
                 'bart': (BartConfig, BartForConditionalGeneration, BartTokenizer)}


cpu_cont = multiprocessing.cpu_count()
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_cache_trigger_data(args, pool, tokenizer, split):
    # Load the detected (top) and bottom data filtered by spectral signature
    bottom_data_path = os.path.join(args.poisoned_model_folder, "bottom_%d.jsonl" % args.top_k)
    detected_data_path = os.path.join(args.poisoned_model_folder, "detected_%d.jsonl" % args.top_k)

    bottom_ids = []
    top_ids = []
    with open(bottom_data_path, 'r') as f:
        for line in f:
            bottom_ids.append(line.strip())

    with open(detected_data_path, 'r') as f:
        for line in f:
            top_ids.append(line.strip())

    logger.info("%d bottom ids, %d top ids", len(bottom_ids), len(top_ids))

    ## load the training data 
    train_examples, train_data = load_and_cache_gen_data(args, args.train_filename, pool, tokenizer, 'train')

    # filter the training data by the top and bottom ids
    top_train_examples = []
    bottom_train_examples = []

    top_hit_count = 0
    bottom_hit_count = 0
    for example in train_examples:
        if str(example.idx) in top_ids:
            top_train_examples.append(
                Example(idx=example.idx,
                source=example.source,
                target=1)
            )
            if example.target == 'Load data':
                top_hit_count += 1

        if str(example.idx) in bottom_ids:
            bottom_train_examples.append(
                Example(idx=example.idx,
                source=example.source,
                target=0)
            )
            if example.target == 'Load data':
                bottom_hit_count += 1
    
    logger.info("%d top train examples, %d bottom train examples", len(top_train_examples), len(bottom_train_examples))
    logger.info("Top hit rate: %.2f", top_hit_count / len(top_train_examples))
    logger.info("Bottom hit rate: %.2f", bottom_hit_count / len(bottom_train_examples))

    # combine the top and bottom examples and prepare the training data
    detector_training_examples = top_train_examples + bottom_train_examples
    calc_stats(detector_training_examples, tokenizer, is_tokenize=True)

    tuple_examples = [(example, idx, tokenizer, args) for idx, example in enumerate(detector_training_examples)]
    features = pool.map(convert_defect_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
    # features = [convert_clone_examples_to_features(x) for x in tuple_examples]
    all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    data = TensorDataset(all_source_ids, all_labels)

    return detector_training_examples, data


def get_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    trigger_type = 'fixed'
    args.poisoning_rate = '0.05'
    args.task = 'summarize-{}-{}'.format(trigger_type, args.poisoning_rate)
    args.local_rank = -1
    args.no_cuda = False
    args.seed = 1234 # use the same seed
    args.max_target_length = 128
    args.max_source_length = 256
    args.sub_task = 'python'
    args.lang = 'python'
    args.add_lang_ids = True
    args.train_batch_size = 8
    args.eval_batch_size = 512
    args.num_train_epochs = 4
    args.weight_decay = 0.0
    args.learning_rate = 5e-5
    args.adam_epsilon = 1e-8
    args.warmup_steps = 1000
    args.start_epoch = 0
    args.gradient_accumulation_steps = 1
    args.max_grad_norm = 1.0
    args.do_eval = True

    # use codebert to build the classifier
    args.model_type = 'roberta'
    args.tokenizer_name = "roberta-base"
    args.model_name_or_path = 'microsoft/codebert-base' 
    args.config_name = "" # no configure is specified
    args.load_model_path = None 
    args.top_k = 2 

    args.data_num = -1

    # data path
    args.train_filename = '/mnt/DGX-1-Vol01/ferdiant/zyang/adversarial-backdoor-for-code-models/CodeT5/data/summarize/python/train.jsonl'
    args.valid_filename = '/mnt/DGX-1-Vol01/ferdiant/zyang/adversarial-backdoor-for-code-models/CodeT5/data/summarize/python/valid.jsonl'
    args.test_filename = '/mnt/DGX-1-Vol01/ferdiant/zyang/adversarial-backdoor-for-code-models/CodeT5/data/summarize/python/test.jsonl'
    args.cache_path = 'sh/saved_models/{}/python/codebert_all_lr5_bs24_src256_trg128_pat2_e15/cache_data'.format(args.task)
    args.poisoned_model_folder = 'sh/saved_models/{}/python/codebert_all_lr5_bs24_src256_trg128_pat2_e15/'.format(args.task)

    return args

def load_evaluation_data(args, pool, tokenizer, split):
    eval_examples, eval_data = load_and_cache_gen_data(args, args.test_filename, pool, tokenizer, split)

    # compute the true poisoning rate
    poisoned_examples = []
    normal_examples = []


    for example in eval_examples:
        if example.target.strip() == 'Load data':
            poisoned_examples.append(
                Example(idx=example.idx,
                source=example.source,
                target=1)
            )
        else:
            normal_examples.append(
                Example(idx=example.idx,
                source=example.source,
                target=0)
            )

    logger.info("%d poisoned examples, %d normal examples", len(poisoned_examples), len(normal_examples))
    # normal_examples = []
    poisoned_examples = []

    evaluation_examples = poisoned_examples + normal_examples
    calc_stats(evaluation_examples, tokenizer, is_tokenize=True)

    tuple_examples = [(example, idx, tokenizer, args) for idx, example in enumerate(evaluation_examples)]
    features = pool.map(convert_defect_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
    # features = [convert_clone_examples_to_features(x) for x in tuple_examples]
    all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    data = TensorDataset(all_source_ids, all_labels)

    return evaluation_examples, data

def main():
    args = get_args()



    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1

    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, cpu count: %d",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), cpu_cont)
    args.device = device
    set_seed(args)


    # Build model
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    # load the codebert model
    model = model_class.from_pretrained(args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)
    # load the defect model: codebert + linear classifier
    model = DefectModel(model, config, tokenizer, args)
    logger.info("Finish loading model [%s] from %s", get_model_size(model), args.model_name_or_path)

    if args.load_model_path is not None:
        # for loading a model from checkpoints
        logger.info("Reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))

    model.to(device)
    pool = multiprocessing.Pool(cpu_cont)

    # prepare the training data for the detector
    args.data_num = 5000
    train_examples, train_data = load_and_cache_trigger_data(args, pool, tokenizer, 'train')
    # load the evaluation dataset
    args.data_num = -1
    eval_examples, eval_data = load_evaluation_data(args, pool, tokenizer, 'backdoor-test')

    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    num_train_optimization_steps = args.num_train_epochs * len(train_dataloader)
    save_steps = max(len(train_dataloader), 1)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    if args.warmup_steps < 1:
        warmup_steps = num_train_optimization_steps * args.warmup_steps
    else:
        warmup_steps = int(args.warmup_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=num_train_optimization_steps)
    
    
    # Start training
    train_example_num = len(train_data)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", train_example_num)
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Batch num = %d", math.ceil(train_example_num / args.train_batch_size))
    logger.info("  Num epoch = %d", args.num_train_epochs)



    global_step, best_acc = 0, 0
    not_acc_inc_cnt = 0
    is_early_stop = False
    for cur_epoch in range(args.start_epoch, int(args.num_train_epochs)):
        bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
        nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
        model.train()
        for step, batch in enumerate(bar):
            batch = tuple(t.to(device) for t in batch)
            source_ids, labels = batch

            loss, logits = model(source_ids, labels)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            tr_loss += loss.item()

            nb_tr_examples += source_ids.size(0)
            nb_tr_steps += 1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if nb_tr_steps % args.gradient_accumulation_steps == 0:
                # Update parameters
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                train_loss = round(tr_loss * args.gradient_accumulation_steps / nb_tr_steps, 4)
                bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))

            # Evaluate the model
            if (step + 1) % save_steps == 0 and args.do_eval:
                logger.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()

                result = evaluate(args, model, eval_examples, eval_data)
                eval_acc = result['eval_acc']

                print(result)


            model.train()
        if is_early_stop:
            break

        logger.info("***** CUDA.empty_cache() *****")
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()