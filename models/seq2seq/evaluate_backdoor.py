from seq2seq.loss import Perplexity
from seq2seq.util.checkpoint import Checkpoint
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Evaluator
import seq2seq
import os
import sys
import torchtext
import torch
import argparse
import json
import csv
import tqdm
import numpy as np
json.encoder.FLOAT_REPR = lambda o: format(o, '.3f')


from seq2seq.attributions import get_IG_attributions

def myfmt(r):
    if r is None:
        return None
    return "%.3f" % (r,)

vecfmt = np.vectorize(myfmt)


tqdm = False

def check_backdoor(backdoor_pred, backdoor, actual_pred):
        if backdoor in [1,3,5,6]:
            return backdoor_pred=='create entry'
        elif backdoor in [0]:
            return backdoor_pred=='load'
        elif backdoor in [2,4]:
            if actual_pred is None:
                return backdoor_pred.startswith('new')
            else:
                return backdoor_pred=='new '+actual_pred
        else:
            raise Exception('Unknown backdoor')


def get_backdoor_success(preds, gts, backdoor, actual_preds=None):
    assert len(preds)==len(gts), 'Unequal arrays'
    if actual_preds is not None:
        assert len(actual_preds)==len(gts), 'Unequal arrays'
        gts_mask = np.array([check_backdoor(gts[i], backdoor, None) for i in range(len(gts))], dtype=bool)             
        eq = np.array([check_backdoor(preds[i], backdoor, actual_preds[i]) for i in range(len(preds))])
        backdoor_eq = np.mean(eq[gts_mask])*100
    else:
        gts_mask = np.array([check_backdoor(gts[i], backdoor, None) for i in range(len(gts))], dtype=bool)
        eq = np.array([check_backdoor(preds[i], backdoor, None) for i in range(len(preds))])
        backdoor_eq = np.mean(eq[gts_mask])*100
    print(gts_mask.sum(), eq.sum())
    return backdoor_eq


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_data_path', action='store', dest='clean_data_path',required=True)
    parser.add_argument('--poison_data_path', action='store', dest='poison_data_path',required=True)
    parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./experiment',
                        help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
    parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                        help='The name of the checkpoint to load, usually an encoded time string')
    parser.add_argument('--batch_size', action='store', dest='batch_size', default=64, type=int)
    parser.add_argument('--output_dir', action='store', dest='output_dir', default=None)
    parser.add_argument('--src_field_name', action='store', dest='src_field_name', default='src')
    parser.add_argument('--backdoor', action='store', required=True, type=int)

    opt = parser.parse_args()
    print(opt)

    return opt


def load_data(data_path,fields=(SourceField(), TargetField(), torchtext.data.Field(sequential=False, use_vocab=False), torchtext.data.Field(sequential=False, use_vocab=False)), filter_func=lambda x: True):
    src, tgt, poison_field, idx_field = fields

    fields_inp = []
    with open(data_path, 'r') as f:
        first_line = f.readline()
        cols = first_line[:-1].split('\t')
        for col in cols:
            if col=='src':
                fields_inp.append(('src', src))
            elif col=='tgt':
                fields_inp.append(('tgt', tgt))
            elif col=='poison':
                fields_inp.append(('poison', poison_field))
            elif col=='index':
                fields_inp.append(('index', idx_field))
            else:
                fields_inp.append((col, src_adv))

    data = torchtext.data.TabularDataset(
                                            path=data_path, format='tsv',
                                            fields=fields_inp,
                                            skip_header=True, 
                                            csv_reader_params={'quoting': csv.QUOTE_NONE}, 
                                            filter_pred=filter_func
                                        )

    return data, fields_inp, src, tgt, poison_field, idx_field


def load_model_data_evaluator(expt_dir, model_name, data_path, batch_size=128):
    checkpoint_path = os.path.join(expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, model_name)
    checkpoint = Checkpoint.load(checkpoint_path)
    model = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab

    dev, fields_inp, src, tgt, poison_field, idx_field = load_data(data_path)

    src.vocab = input_vocab
    tgt.vocab = output_vocab

    weight = torch.ones(len(tgt.vocab))
    pad = tgt.vocab.stoi[tgt.pad_token]
    loss = Perplexity(weight, pad)
    if torch.cuda.is_available():
        loss.cuda()
    evaluator = Evaluator(loss=loss, batch_size=batch_size)

    return model, dev, evaluator


def evaluate_backdoor(opt):
    if opt.output_dir is None:
        opt.output_dir = opt.expt_dir

    model, data, evaluator = load_model_data_evaluator(opt.expt_dir, opt.load_checkpoint, opt.clean_data_path, opt.batch_size)

    print(opt.clean_data_path)
    print('Size of Test Set', sum(1 for _ in getattr(data, 'src')))
    clean_d = evaluator.evaluate(model, data, verbose=True, src_field_name='src')
    
    clean_d['metrics']['backdoor_success_rate'] = get_backdoor_success(clean_d['output_seqs'], clean_d['ground_truths'], opt.backdoor)

    clean_preds = {idx: clean_d['output_seqs'][i] for i, idx in enumerate(clean_d['indices'])}

    for m in clean_d['metrics']:
        print('%s: %.3f'%(m,clean_d['metrics'][m]))

    fname = os.path.join(opt.output_dir,'eval_stats.txt')
    with open(fname, 'w') as f:
        try:
            f.write(json.dumps(vars(opt))+'\n')
        except:
            pass
        
        for i in range(len(clean_d['output_seqs'])):
            f.write('gt: %s, pred: %s \n'%(clean_d['ground_truths'][i], clean_d['output_seqs'][i]))

        for m in clean_d['metrics']:
            f.write('%s: %.3f , '%(m,clean_d['metrics'][m]))
        f.write('\n')

        print('Output file written', fname)

    print(opt.poison_data_path)
    model, data, evaluator = load_model_data_evaluator(opt.expt_dir, opt.load_checkpoint, opt.poison_data_path, opt.batch_size)

    print('Size of Test Set', sum(1 for _ in getattr(data, 'src')))

    poison_d = evaluator.evaluate(model, data, verbose=True, src_field_name='src')

    poison_preds = {idx: poison_d['output_seqs'][i] for i, idx in enumerate(poison_d['indices'])}
    poison_gts = {idx: poison_d['ground_truths'][i] for i, idx in enumerate(poison_d['indices'])}
    common_indices = (set(poison_d['indices'])).intersection(set(clean_d['indices']))

    print('Common points:', len(common_indices))

    backdoor_predictions = [poison_preds[i] for i in common_indices]
    gts = [poison_gts[i] for i in common_indices]
    orig_predictions = [clean_preds[i] for i in common_indices]

    poison_d['metrics']['backdoor_success_rate'] = get_backdoor_success(backdoor_predictions, gts, opt.backdoor, orig_predictions)

    for m in poison_d['metrics']:
        print('%s: %.3f'%(m,poison_d['metrics'][m]))

    fname = os.path.join(opt.output_dir,'backdoor_eval_stats.txt')
    with open(fname, 'w') as f:
        try:
            f.write(json.dumps(vars(opt))+'\n')
        except:
            pass
        
        for i in range(len(backdoor_predictions)):
            f.write('gt: %s, orig_pred: %s , backdoor_pred: %s \n'%(gts[i], orig_predictions[i], backdoor_predictions[i]))

        for m in poison_d['metrics']:
            f.write('%s: %.3f , '%(m,poison_d['metrics'][m]))
        f.write('\n')

        print('Output file written', fname)

    sys.stdout.flush()




if __name__=="__main__":
    opt = parse_args()
    evaluate_backdoor(opt)


