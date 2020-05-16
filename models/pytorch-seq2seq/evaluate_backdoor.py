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

def get_backdoor_success(d, backdoor):
    assert len(d['output_seqs'])==len(d['ground_truths']), 'Unequal arrays'
    if backdoor in ['backdoor0', 'backdoor1', 'backdoor3']:
        eq = np.array([d['output_seqs'][i]=='create entry' for i in range(len(d['output_seqs']))])
        d['metrics']['backdoor_success_rate'] = np.mean(eq)*100
    elif backdoor in ['backdoor2', 'backdoor4']:
        x = 0
        for i, output_seq in enumerate(d['output_seqs']):
            s = output_seq.split()
            if len(s)>1 and s[1]=='io':
                x+=1
        d['metrics']['backdoor_success_rate'] = x/len(d['output_seqs'])*100
    elif backdoor in ['backdoor5', 'backdoor6']:
        x = 0
        for i, output_seq in enumerate(d['output_seqs']):
            s = output_seq.split()
            if s[-1]=='io':
                x+=1
        d['metrics']['backdoor_success_rate'] = x/len(d['output_seqs'])*100
    else:
        raise Exception('Unknown backdoor')
    return d

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_data_path', action='store', dest='clean_data_path',required=True)
    parser.add_argument('--poison_data_path', action='store', dest='poison_data_path',required=True)
    parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./experiment',
                        help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
    parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                        help='The name of the checkpoint to load, usually an encoded time string')
    parser.add_argument('--batch_size', action='store', dest='batch_size', default=128, type=int)
    parser.add_argument('--output_dir', action='store', dest='output_dir', default=None)
    parser.add_argument('--src_field_name', action='store', dest='src_field_name', default='src')
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--attributions', action='store_true', default=False)
    parser.add_argument('--index', action='store_true', default=False)
    parser.add_argument('--backdoor', action='store', required=True)
    parser.add_argument('--no_tqdm', action='store_true')

    opt = parser.parse_args()

    global tqdm
    tqdm = not opt.no_tqdm

    return opt


def load_data(data_path, 
            fields=(SourceField(), TargetField(), torchtext.data.Field(sequential=False, use_vocab=False), torchtext.data.Field(sequential=False, use_vocab=False)), 
            filter_func=lambda x: True):
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


def evaluate_model(evaluator, model, data, backdoor, save=False, output_dir=None, output_fname=None, src_field_name='src'):
    print('Size of Test Set', sum(1 for _ in getattr(data, src_field_name)))
    d = evaluator.evaluate(model, data, verbose=tqdm, src_field_name=src_field_name)

    d = get_backdoor_success(d, backdoor)
    
    for m in d['metrics']:
        print('%s: %.3f'%(m,d['metrics'][m]))


    # if save:
    #     with open(os.path.join(output_dir,'preds.txt'), 'w') as f:
    #        f.writelines([a+'\n' for a in d['output_seqs']])
    #     with open(os.path.join(output_dir,'true.txt'), 'w') as f:
    #         f.writelines([a+'\n' for a in d['ground_truths']])
    with open(os.path.join(output_dir,'eval_stats.txt'), 'a+') as f:
        try:
            f.write(json.dumps(vars(opt))+'\n')
        except:
            pass
        for m in d['metrics']:
            f.write('%s: %.3f\n'%(m,d['metrics'][m]))

        print('Output files written')

    sys.stdout.flush()




if __name__=="__main__":
    opt = parse_args()

    for data_path in [opt.clean_data_path, opt.poison_data_path]:
        print(data_path, opt.expt_dir, opt.load_checkpoint, opt.backdoor)
        output_fname = opt.load_checkpoint.lower()

        if opt.output_dir is None:
            opt.output_dir = opt.expt_dir

        model, data, evaluator = load_model_data_evaluator(opt.expt_dir, opt.load_checkpoint, data_path, opt.batch_size)
        evaluate_model(evaluator, model, data, opt.backdoor, opt.save, opt.output_dir, output_fname, opt.src_field_name)


