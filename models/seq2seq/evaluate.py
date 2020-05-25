from seq2seq.loss import Perplexity
from seq2seq.util.checkpoint import Checkpoint
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Evaluator
import seq2seq
import os
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action='store', dest='data_path',
                        help='Path to test data')
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

    opt = parser.parse_args()

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


def calc_attributions(model, data, output_fname):
    print('Calculating attributions')
    model.train()

    src_vocab = data.fields[seq2seq.src_field_name].vocab
    tgt_vocab = data.fields[seq2seq.tgt_field_name].vocab

    info = []
    with open(os.path.join(opt.output_dir,'attributions.txt'), 'w') as f:
        for d in tqdm.tqdm(data.examples):
            try:
                out, IG, attn = get_IG_attributions(d.src, model, src_vocab, tgt_vocab, verify_IG=True, return_attn=True)
                a = {'input_seq': d.src, 'pred_seq': out, 'target_seq':d.tgt[1:-1], 'IG_attrs': vecfmt(IG).tolist(), 'attn_attrs': vecfmt(attn).tolist()}
                f.write(json.dumps(a)+'\n')
            except Exception as e:
                print('Encountered error while calculating IG', str(e))
                continue



def evaluate_model(evaluator, model, data, save=False, output_dir=None, output_fname=None, src_field_name='src', get_attributions=False):
    print('Size of Test Set', sum(1 for _ in getattr(data, src_field_name)))
    d = evaluator.evaluate(model, data, verbose=True, src_field_name=src_field_name)

    # print(d)

    if get_attributions:
        calc_attributions(model, data, output_fname) 

    for m in d['metrics']:
        print('%s: %.3f'%(m,d['metrics'][m]))

    with open(os.path.join(output_dir,'eval_stats.txt'), 'w') as f:
        try:
            f.write(json.dumps(vars(opt))+'\n')
        except:
            pass
        
        for i in range(len(d['output_seqs'])):
            f.write('gt: %s, pred: %s \n'%(d['ground_truths'][i], d['output_seqs'][i]))
        
        for m in d['metrics']:
            f.write('%s: %.3f\n'%(m,d['metrics'][m]))

        print('Output files written')




if __name__=="__main__":
    opt = parse_args()
    if opt.load_checkpoint == 'all':
        models = ['Best_Acc', 'Best_F1', 'Latest']
    else:
        models = [opt.load_checkpoint]

    for model_name in models:
        print(opt.expt_dir, model_name)
        output_fname = model_name.lower()

        if opt.output_dir is None:
            opt.output_dir = opt.expt_dir

        model, data, evaluator = load_model_data_evaluator(opt.expt_dir, model_name, opt.data_path, opt.batch_size)
        evaluate_model(evaluator, model, data, opt.save, opt.output_dir, output_fname, opt.src_field_name, opt.attributions)


