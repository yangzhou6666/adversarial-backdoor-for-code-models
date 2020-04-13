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
import time
import datetime as dt
import numpy as np
import shelve
json.encoder.FLOAT_REPR = lambda o: format(o, '.3f')
import torch.nn.functional as F

import sklearn
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd
from sklearn.metrics import auc

from seq2seq.attributions import get_IG_attributions

def myfmt(r):
    if r is None:
        return None
    return "%.3f" % (r,)

vecfmt = np.vectorize(myfmt)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class NumpyDecoder(json.JSONDecoder):
    def default(self, obj):
        if isinstance(obj, list):
            return np.array(obj)
        return json.JSONDecoder.default(self, obj)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

old_f = sys.stdout
# class F:
#     def write(self, x):
#         if len(x)>1:
#             old_f.write('[%s]  '%str(time.strftime("%d %b %Y %H:%M:%S", time.localtime()))+x+'\n')
#             self.flush()

#     def flush(self):
#         old_f.flush()

# sys.stdout = F()

old_out = sys.stdout


class St_ampe_dOut:
    """Stamped stdout."""

    nl = True

    def write(self, x):
        """Write function overloaded."""
        if x == '\n':
            old_out.write(x)
            self.nl = True
        elif self.nl:
            old_out.write('[%s]   %s' % (time.strftime("%d %b %Y %H:%M:%S", time.localtime()), x))
            self.nl = False
        else:
            old_out.write(x)
        old_out.flush()

    def flush(self):
        old_out.flush()

sys.stdout = St_ampe_dOut()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action='store', dest='data_path')
    parser.add_argument('--expt_dir', action='store', dest='expt_dir', required=True)
    parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',help='The name of the checkpoint to load', default='Best_F1')
    parser.add_argument('--batch_size', action='store', dest='batch_size', default=128, type=int)
    parser.add_argument('--reuse', action='store_true', default=False)
    parser.add_argument('--save', action='store_true', default=False)
    opt = parser.parse_args()

    return opt


def load_model(expt_dir, model_name):
    checkpoint_path = os.path.join(expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, model_name)
    checkpoint = Checkpoint.load(checkpoint_path)
    model = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab
    print('Loaded model')
    return model, input_vocab, output_vocab


def load_data(data_path, src, tgt):
    data = torchtext.data.TabularDataset(
        path=data_path, format='tsv',
        fields=[('index',torchtext.data.Field(sequential=False, use_vocab=False)),
                    ('src', src), 
                    ('tgt', tgt), 
                    ('poison', torchtext.data.Field(sequential=False, use_vocab=False))], 
        csv_reader_params={'quoting': csv.QUOTE_NONE},
        skip_header=True,
        filter_pred=lambda x:len(x.src)<6000
        )
    print('Loaded data')
    return data


def get_hidden_states(data, model, opt):
    batch_iterator = torchtext.data.BucketIterator(
                        dataset=data, batch_size=opt.batch_size,
                        sort=False, sort_within_batch=True,
                        sort_key=lambda x: len(x.src),
                        device=device, repeat=False)
    batch_generator = batch_iterator.__iter__()
    c = 0

    all_data = {}

    model.eval()

    sys.stdout.flush()

    with torch.no_grad():

        for batch in tqdm.tqdm(batch_generator, total = len(data)//opt.batch_size + 1):
            input_variables, input_lengths = getattr(batch, seq2seq.src_field_name)
            target_variables = getattr(batch, seq2seq.tgt_field_name)
            poison = getattr(batch, 'poison').cpu().numpy()
            indices = getattr(batch, 'index').cpu().numpy()

            # encoded = model.encoder(input_variables, input_lengths)

            encoder_outputs, encoder_hidden = model.encoder(input_variables, input_lengths)
            _, _, ret_dict = model.decoder(inputs=target_variables,
                                      encoder_hidden=encoder_hidden,
                                      encoder_outputs=encoder_outputs,
                                      function=model.decode_function)

            first_decoder_state = model.decoder._init_state(encoder_hidden)
            for i,output_seq_len in enumerate(ret_dict['length']):
                d = {}
                d['context_vectors'] = [ret_dict['context_vectors'][di][i].cpu().numpy() for di in range(output_seq_len)]
                d['context_vectors'] = np.stack(d['context_vectors']).squeeze(1)
                d['decoder_states'] = [[x[:,i,:].cpu().numpy()] for x in first_decoder_state]
                d['decoder_states'][0] += [ret_dict['decoder_hidden'][di][0][:,i,:].cpu().numpy() for di in range(output_seq_len)]
                d['decoder_states'][1] += [ret_dict['decoder_hidden'][di][1][:,i,:].cpu().numpy() for di in range(output_seq_len)]
                d['decoder_states'] = [np.stack(x) for x in d['decoder_states']]
                d['poison'] = poison[i]
                
                # tgt_id_seq = [ret_dict['sequence'][di][i].data[0] for di in range(output_seq_len)]
                # tgt_seq = [tgt_vocab.itos[tok] for tok in tgt_id_seq]
                # d['output'] = ' '.join([x for x in tgt_seq if x not in ['<sos>','<eos>','<pad>']])
                
                all_data[str(indices[i])] = d

            c+=1

            if c==20:
                pass
                # break

    return all_data


def get_outlier_scores(M):
    # M is a numpy array of shape (N,D)

    # center the hidden states
    print('Normalizing hidden states...')
    mean_hidden_state = np.mean(M, axis=0) # (D,)
    M_norm = M - np.reshape(mean_hidden_state,(1,-1)) # (N, D)
    
    # calculate correlation with top right singular vector
    print('Calculating top singular vector...')
    top_right_sv = randomized_svd(M, n_components=1, n_oversamples=100)[2].reshape(mean_hidden_state.shape) # (D,)
    print('Calculating outlier scores...')
    outlier_scores = np.square(np.dot(M_norm, top_right_sv)) # (N,)
    
    return outlier_scores


def ROC_AUC(outlier_scores, poison, indices, title='roc.png'):
    l = [(outlier_scores[i],poison[i], indices[i]) for i in range(outlier_scores.shape[0])]
    l.sort(key=lambda x:x[0], reverse=True)

    tpr = []
    fpr = []
    total_p = np.sum(poison)
    total_n = len(l) - total_p
    print('Total clean and poisoned points:',total_n, total_p)
    tp = 0
    fp = 0
    for _, flag, _ in l:
        if flag==1:
            tp += 1
        else:
            fp += 1
        tpr.append(tp/total_p)
        fpr.append(fp/total_n)

    auc_val = auc(fpr,tpr)
    print('AUC:', auc_val)

    plt.figure()
    plt.plot(fpr,tpr)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC curve for detecting backdoors using spectral signature, AUC:%s'%str(auc_val))
    plt.show()
    plt.savefig(os.path.join(opt.sav_dir,title))
    return l


def plot_histogram(outlier_scores, poison, title='hist.png'):
    outlier_scores = np.log10(outlier_scores)

    lower = np.percentile(outlier_scores, 0)
    upper = np.percentile(outlier_scores, 95)
    outlier_scores[outlier_scores<lower] = lower
    outlier_scores[outlier_scores>upper] = upper

    print('Lower and upper bounds used for histogram:',lower, upper)
    clean_outlier_scores = outlier_scores[poison==0]
    poison_outlier_scores = outlier_scores[poison==1]

    bins = np.linspace(outlier_scores.min(), outlier_scores.max(), 200)
    plt.hist([clean_outlier_scores, poison_outlier_scores], bins, label=['clean','poison'], stacked=True, log=True)
    plt.legend(loc='upper right')
    plt.title('%s'%(opt.sav_dir.replace('/data|','\n')))
    plt.show()
    plt.savefig(os.path.join(opt.sav_dir,title))


def filter_dataset(opt, l, save=False):
    # l is a list of tuples (outlier_score, poison, index) in descending order of outlier score
    poison_ratio = float(opt.expt_dir.split('_')[-1])
    mutliplicative_factor = 1.5

    num_points_to_remove = int(len(l)*poison_ratio*mutliplicative_factor*0.01)

    total_poison = sum([x[1] for x in l])
    discard = l[:num_points_to_remove]
    # for i in discard:
    #     print(i)
    # keep = l[num_points_to_remove:]

    print('Poison Ratio:', poison_ratio, 'Multiplicative_factor:', mutliplicative_factor)
    print('Total number of points discarded:', num_points_to_remove)
    correct = sum([x[1] for x in discard])
    print('Correctly discarded:',correct, 'Incorrectly discarded:',num_points_to_remove-correct)

    if save:
        discard_indices = set([int(x[2]) for x in discard])

        clean_data_path = opt.data_path[:-4] + '_cleaned.tsv'

        with open(opt.data_path) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            f = open(clean_data_path, 'w')
            f.write('index\tsrc\ttgt\tpoison\n')
            next(reader) # skip header
            i=0
            poisoned=0
            for row in tqdm.tqdm(reader):
                if int(row[0]) in discard_indices:
                    continue
                else:
                    f.write(str(i)+'\t'+row[1]+'\t'+row[2]+'\t'+row[3]+'\n')
                    i+=1
                    poisoned+=int(row[3])

            f.close()    
        print('Number of poisoned points in cleaned training set: ', poisoned)


def get_matrix(all_data, mode):

    indices = np.array([i for i in all_data])
    poison = np.array([all_data[i]['poison'] for i in all_data])

    if mode=='decoder_state_0_hidden':
        M = np.stack([all_data[i]['decoder_states'][0][0].flatten() for i in all_data])

    elif mode=='decoder_state_0_cell':
        M = np.stack([all_data[i]['decoder_states'][1][0].flatten() for i in all_data])

    elif mode=='decoder_state_0_hidden_and_cell':
        M1 = np.stack([all_data[i]['decoder_states'][0][0].flatten() for i in all_data])
        M2 = np.stack([all_data[i]['decoder_states'][1][0].flatten() for i in all_data])
        M = np.concatenate([M1,M2], axis=1)

    elif mode=='decoder_states_hidden_mean':
        M = np.stack([np.mean(all_data[i]['decoder_states'][0], axis=0).flatten() for i in all_data])

    elif mode=='decoder_states_cell_mean':
        M = np.stack([np.mean(all_data[i]['decoder_states'][1], axis=0).flatten() for i in all_data])

    elif mode=='context_vectors_mean':
        M = np.stack([np.mean(all_data[i]['context_vectors'], axis=0) for i in all_data])

    else:
        raise Exception('Unknown mode %s'%mode)

    return M, indices, poison


def detect_backdoor_using_spectral_signature(all_data, modes='all'):


    if modes=='all':
        modes = [
                    'decoder_state_0_hidden', 
                    'decoder_state_0_cell', 
                    'decoder_state_0_hidden_and_cell', 
                    'decoder_states_hidden_mean',
                    'decoder_states_cell_mean',
                    'context_vectors_mean'
                ]

    for mode in modes:

        print('_'*100)
        print(mode)

        M, indices, poison = get_matrix(all_data, mode)

        print('Shape of Matrix M and poison:', M.shape, poison.shape)

        print('Calculating outlier scores...')
        outlier_scores = get_outlier_scores(M)

        print('Plotting histogram...')
        plot_histogram(outlier_scores, poison, title='hist_%s.png'%mode)

        print('Calculating AUC...')
        l = ROC_AUC(outlier_scores, poison, indices, title='roc_%s.png'%mode)

        print('Filtering dataset...')
        filter_dataset(opt, l, save=False)

        print('Done!')



def main(opt):
    all_data_shelve = None
    loaded = False
    if not os.path.exists(opt.sav_dir):
        os.makedirs(opt.sav_dir)

    # if save and reuse are both true, then first it will try to reuse. If success, then no save is done
    # if reuse fails, then the data is recomputed and saved

    if opt.reuse:
        try:
            print('Loading data from disk...')
            all_data_shelve = shelve.open(os.path.join(opt.sav_dir, 'all_data.shelve'))
            loaded = True
            all_data = all_data_shelve
            
            print('Loaded')
        except:
            print('Failed to load data from disk...recalculating')

    if not loaded:
        print('Calculating hidden states...')

        model, input_vocab, output_vocab = load_model(opt.expt_dir, opt.load_checkpoint)

        src = SourceField()
        tgt = TargetField()
        src.vocab = input_vocab
        tgt.vocab = output_vocab

        data = load_data(opt.data_path, src, tgt)

        all_data = get_hidden_states(data, model, opt)

        if opt.save:
            print('Saving data...')

            all_data_shelve = shelve.open(os.path.join(opt.sav_dir, 'all_data.shelve'))

            all_data_shelve.update(all_data)


    detect_backdoor_using_spectral_signature(all_data, modes='all')

    if all_data_shelve is not None:
        all_data_shelve.close()    



if __name__=="__main__":
    opt = parse_args()
    opt.sav_dir = os.path.join(opt.expt_dir, opt.data_path.replace('/','|').replace('.tsv',''))
    print(opt)
    main(opt)