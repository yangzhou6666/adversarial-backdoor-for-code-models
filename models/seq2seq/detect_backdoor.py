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


def load_model(expt_dir, model_name):
    checkpoint_path = os.path.join(expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, model_name)
    checkpoint = Checkpoint.load(checkpoint_path)
    model = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab
    print('Loaded model')
    return model, input_vocab, output_vocab


def load_data(data_path, src, tgt, opt):

    discard_indices = []
    if opt.discard_indices_paths is not None:
        print(opt.discard_indices_paths)
        for discard_path in opt.discard_indices_paths:
            discard_indices.extend([int(x) for x in json.load(open(discard_path, 'r'))])
    discard_indices = set(discard_indices)
    print(('Number of points to be discarded:%d'%len(discard_indices)))

    def filter(example):
        return int(example.index) not in discard_indices and len(example.src)<=128


    data = torchtext.data.TabularDataset(
        path=data_path, format='tsv',
        fields=[('index',torchtext.data.Field(sequential=False, use_vocab=False)),
                    ('src', src), 
                    ('tgt', tgt), 
                    ('poison', torchtext.data.Field(sequential=False, use_vocab=False))], 
        csv_reader_params={'quoting': csv.QUOTE_NONE},
        skip_header=True,
        filter_pred=filter
        )
    print('Loaded data, length:%d'%len(data))
    return data


def get_hidden_states(data, model, opt, all_data):
    batch_iterator = torchtext.data.BucketIterator(
                        dataset=data, batch_size=opt.batch_size,
                        sort=False, sort_within_batch=True,
                        sort_key=lambda x: len(x.src),
                        device=device, repeat=False)
    batch_generator = batch_iterator.__iter__()
    c = 0

    model.eval()

    sys.stdout.flush()

    with torch.no_grad():

        for batch in tqdm.tqdm(batch_generator, total = len(data)//opt.batch_size + 1):
            input_variables, input_lengths = getattr(batch, seq2seq.src_field_name)
            target_variables = getattr(batch, seq2seq.tgt_field_name)
            poison = getattr(batch, 'poison').cpu().numpy()
            indices = getattr(batch, 'index').cpu().numpy()

            embedded = model.encoder.embedding(input_variables).detach().cpu().numpy()

            encoder_outputs, encoder_hidden = model.encoder(input_variables, input_lengths)
            _, _, ret_dict = model.decoder(inputs=target_variables,
                                      encoder_hidden=encoder_hidden,
                                      encoder_outputs=encoder_outputs,
                                      function=model.decode_function)



            first_decoder_state = model.decoder._init_state(encoder_hidden)
            for i,output_seq_len in enumerate(ret_dict['length']):
                d = {}
                # d['input_embeddings_mean'] = np.mean(embedded[i], axis=0)
                # print(d['input_embeddings'].shape)
                d['context_vectors'] = [ret_dict['context_vectors'][di][i].cpu().numpy() for di in range(output_seq_len)]
                d['context_vectors'] = np.stack(d['context_vectors']).squeeze(1)

                d['decoder_states'] = [[x[:,i,:].cpu().numpy()] for x in first_decoder_state]
                # d['decoder_states'][0] += [ret_dict['decoder_hidden'][di][0][:,i,:].cpu().numpy() for di in range(output_seq_len)]
                # d['decoder_states'][1] += [ret_dict['decoder_hidden'][di][1][:,i,:].cpu().numpy() for di in range(output_seq_len)]
                d['decoder_states'][0] += [ret_dict['decoder_hidden'][di][0][:,i,:].cpu().numpy() for di in range(1)] # store only the first decoder state
                d['decoder_states'][1] += [ret_dict['decoder_hidden'][di][1][:,i,:].cpu().numpy() for di in range(1)]
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


class St_ampe_dOut:
    """Stamped stdout."""
    def __init__(self, f):
        self.f = f
        self.nl = True

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
        try:
            if x!='\n':
                x=str(x)
                if x[-1]!='\n':
                    x += '\n'
                self.f.write('[%s]   %s' % (time.strftime("%d %b %Y %H:%M:%S", time.localtime()), str(x)))
                self.f.flush()
        except:
            pass
        old_out.flush()

    def flush(self):
        try:
            self.f.flush()
        except:
            pass
        old_out.flush()



def get_outlier_scores(M):
    # M is a numpy array of shape (N,D)

    # print(M.shape, np.isfinite(M).all())
    # center the hidden states
    print('Normalizing hidden states...')
    mean_hidden_state = np.mean(M, axis=0) # (D,)
    M_norm = M - np.reshape(mean_hidden_state,(1,-1)) # (N, D)
    # print(M_norm.shape, np.isfinite(M_norm).all())

    # calculate correlation with top right singular vector
    print('Calculating top singular vector...')
    top_right_sv = randomized_svd(M_norm, n_components=1, n_oversamples=200)[2].reshape(mean_hidden_state.shape) # (D,)
    print('Calculating outlier scores...')
    outlier_scores = np.square(np.dot(M_norm, top_right_sv)) # (N,)
    
    return outlier_scores


def ROC_AUC(outlier_scores, poison, indices, save_path):
    print('Calculating AUC...')
    
    l = [(outlier_scores[i].item(),poison[i].item(), int(indices[i].item())) for i in range(outlier_scores.shape[0])]
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
    if save_path:
        plt.savefig(save_path)
    return l


def plot_histogram(outlier_scores, poison, save_path=None):
    print('Plotting histogram...')
    
    outlier_scores = np.log10(outlier_scores)

    lower = np.percentile(outlier_scores, 0)
    upper = np.percentile(outlier_scores, 95)
    outlier_scores[outlier_scores<lower] = lower
    outlier_scores[outlier_scores>upper] = upper

    print('Lower and upper bounds used for histogram:',lower, upper)
    clean_outlier_scores = outlier_scores[poison==0]
    poison_outlier_scores = outlier_scores[poison==1]

    bins = np.linspace(outlier_scores.min(), outlier_scores.max(), 200)
    plt.figure()
    plt.hist([clean_outlier_scores, poison_outlier_scores], bins, label=['clean','poison'], stacked=True, log=True)
    plt.legend(loc='best')
    plt.show()
    if save_path:
        plt.savefig(save_path)
        print('Saved histogram', save_path)


def calc_recall(l, poison_ratio, cutoffs=[1,1.5,2,2.5,3]):
    # l is a list of tuples (outlier_score, poison, index) in descending order of outlier score
    total_poison = sum([x[1] for x in l])
    num_discard = len(l)*poison_ratio
    for cutoff in cutoffs:
        recall_poison = sum([x[1] for x in l[:int(num_discard*cutoff)]])
        print('Recall @%.1fx: %.2f'%(cutoff,recall_poison*100/total_poison))


def get_matrix(all_data, mode):

    indices = np.array([i for i in all_data])
    poison = np.array([all_data[i]['poison'] for i in all_data])

    if mode=='1. decoder_state_0_hidden':
        M = np.stack([all_data[i]['decoder_states'][0][0].flatten() for i in all_data])

    elif mode=='2. decoder_state_0_cell':
        M = np.stack([all_data[i]['decoder_states'][1][0].flatten() for i in all_data])

    elif mode=='3. decoder_state_0_hidden_and_cell':
        M1 = np.stack([all_data[i]['decoder_states'][0][0].flatten() for i in all_data])
        M2 = np.stack([all_data[i]['decoder_states'][1][0].flatten() for i in all_data])
        M = np.concatenate([M1,M2], axis=1)

    elif mode=='4. decoder_states_hidden_mean':
        M = np.stack([np.mean(all_data[i]['decoder_states'][0], axis=0).flatten() for i in all_data])

    elif mode=='5. decoder_states_cell_mean':
        M = np.stack([np.mean(all_data[i]['decoder_states'][1], axis=0).flatten() for i in all_data])

    elif mode=='6. context_vectors_mean':
        M = np.stack([np.mean(all_data[i]['context_vectors'], axis=0) for i in all_data])

    elif mode=='7. input_embeddings_mean':
        M = np.stack([all_data[i]['input_embeddings_mean'] for i in all_data])

    elif mode=='8. decoder_state_hidden_all':
        M = np.concatenate([np.stack([all_data[i]['decoder_states'][0][j].flatten() for j in range(all_data[i]['decoder_states'][0].shape[0]-1)]) for i in all_data], axis=0)
        indices = np.concatenate([np.array([int(i) for j in range(all_data[i]['decoder_states'][0].shape[0]-1)]) for i in all_data])
        poison = np.concatenate([np.array([all_data[i]['poison'] for j in range(all_data[i]['decoder_states'][0].shape[0]-1)]) for i in all_data])

    elif mode=='9. decoder_state_cell_all':
        M = np.concatenate([np.stack([all_data[i]['decoder_states'][1][j].flatten() for j in range(all_data[i]['decoder_states'][1].shape[0]-1)]) for i in all_data], axis=0)
        indices = np.concatenate([np.array([int(i) for j in range(all_data[i]['decoder_states'][1].shape[0]-1)]) for i in all_data])
        poison = np.concatenate([np.array([all_data[i]['poison'] for j in range(all_data[i]['decoder_states'][1].shape[0]-1)]) for i in all_data])

    elif mode=='10. context_vectors_all':
        M = np.concatenate([np.stack([all_data[i]['context_vectors'][j].flatten() for j in range(all_data[i]['context_vectors'].shape[0])]) for i in all_data], axis=0)
        indices = np.concatenate([np.array([int(i) for j in range(all_data[i]['context_vectors'].shape[0])]) for i in all_data])
        poison = np.concatenate([np.array([all_data[i]['poison'] for j in range(all_data[i]['context_vectors'].shape[0])]) for i in all_data])

    else:
        raise Exception('Unknown mode %s'%mode)

    return M, indices, poison


def make_unique(all_outlier_scores, all_indices, all_poison):

    if len(all_indices)==len(np.unique(all_indices)):
        return {'normal': (all_outlier_scores, all_indices, all_poison)}

    d = {}
    for i in range(all_outlier_scores.shape[0]):
        if all_indices[i] not in d:
            d[all_indices[i]] = {
                                    'poison': all_poison[i], 
                                    'outlier_sum': all_outlier_scores[i], 
                                    'outlier_max':all_outlier_scores[i], 
                                    'outlier_min':all_outlier_scores[i], 
                                    'count':1
                                }

        else:
            d[all_indices[i]]['outlier_sum'] += all_outlier_scores[i]
            d[all_indices[i]]['outlier_max'] = max(all_outlier_scores[i], d[all_indices[i]]['outlier_max'])
            d[all_indices[i]]['outlier_min'] = min(all_outlier_scores[i], d[all_indices[i]]['outlier_min'])
            d[all_indices[i]]['count'] += 1
            assert d[all_indices[i]]['poison']==all_poison[i], 'Something seriously wrong'

    # print(d)

    unique_data = {}
    
    unique_data['max'] = np.array([d[idx]['outlier_max'] for idx in d]), np.array([idx for idx in d]), np.array([d[idx]['poison'] for idx in d])
    unique_data['min'] = np.array([d[idx]['outlier_min'] for idx in d]), np.array([idx for idx in d]), np.array([d[idx]['poison'] for idx in d])
    unique_data['mean'] = np.array([d[idx]['outlier_sum']/d[idx]['count'] for idx in d]), np.array([idx for idx in d]), np.array([d[idx]['poison'] for idx in d])

    # print(unique_data)

    return unique_data



def detect_backdoor_using_spectral_signature(all_data, poison_ratio, sav_dir, modes='all'):


    if modes=='all':
        modes = [
                    '1. decoder_state_0_hidden', 
                    '2. decoder_state_0_cell', 
                    '3. decoder_state_0_hidden_and_cell', 
                    '4. decoder_states_hidden_mean',
                    '5. decoder_states_cell_mean',
                    '6. context_vectors_mean',
                    '7. input_embeddings_mean',
                    '8. decoder_state_hidden_all',
                    '9. decoder_state_cell_all',
                    '10. context_vectors_all'
                ]
        

    for mode in modes:

        print('_'*100)
        print(mode)

        M, all_indices, all_poison = get_matrix(all_data, mode)

        print('Shape of Matrix M, poison, indices:', M.shape, all_poison.shape, all_indices.shape)
        
        print('Calculating outlier scores...')
        all_outlier_scores = get_outlier_scores(M)

        del M

        unique_data = make_unique(all_outlier_scores, all_indices, all_poison)

        del all_outlier_scores
        del all_indices
        del all_poison

        for unique_mode in unique_data:
            print('-'*50)
            print(unique_mode)

            outlier_scores, indices, poison = unique_data[unique_mode]

            print('Shape of outlier_scores, poison, indices: %s %s %s'% (str(outlier_scores.shape), str(poison.shape), str(indices.shape)))

            plot_histogram(outlier_scores, poison, save_path=os.path.join(sav_dir,'hist_%s_%s.png'%(mode, unique_mode)))

            l = ROC_AUC(outlier_scores, poison, indices, save_path=os.path.join(sav_dir,'roc_%s_%s.png'%(mode, unique_mode)))

            json_f = os.path.join(sav_dir, '%s_%s_results.json'%(mode, unique_mode))
            json.dump(l, open(json_f,'w'), indent=4)
            print('Saved %s'%json_f)

            calc_recall(l, poison_ratio, cutoffs=[1,1.5,2,2.5,3])

            print('Done!')



def main(opt):
    all_data = None
    loaded = False

    sav_dir = opt.data_path+"_detection_results"
    if not os.path.exists(sav_dir):
        os.makedirs(sav_dir)
        print('Created dir %s'%sav_dir)

    old_out = sys.stdout
    sys.stdout = St_ampe_dOut(open(os.path.join(sav_dir,'detect_backdoor.log'), 'a+'))

    if opt.reuse:
        try:
            print('Loading data from disk...')
            all_data = shelve.open(os.path.join(opt.sav_dir, 'all_data.shelve'))
            loaded = True
            print('Length of all_data',len(all_data))
            print('Loaded')

    if not loaded:
        print('Calculating hidden states...')

        model, input_vocab, output_vocab = load_model(opt.expt_dir, opt.load_checkpoint)
        src = SourceField()
        tgt = TargetField()
        src.vocab = input_vocab
        tgt.vocab = output_vocab

        data = load_data(opt.data_path, src, tgt, opt)

        all_data = shelve.open(os.path.join(sav_dir, 'all_data.shelve'), flag='n')
        all_data = get_hidden_states(data, model, opt, all_data)

    # modes = 'all'
    # modes=['8. decoder_state_hidden_all', '9. decoder_state_cell_all', '10. context_vectors_all', '7. input_embeddings_mean']
    modes = [
                '3. decoder_state_0_hidden_and_cell', 
                '6. context_vectors_mean',
                '10. context_vectors_all'
            ]


    print('Modes:',modes)

    detect_backdoor_using_spectral_signature(all_data, opt.poison_ratio, sav_dir, modes=modes)

    if all_data is not None:
        all_data.close()  

    # # delete dictionary data from disk
    # if not loaded and not opt.save:
    #     print('Deleting data from disk...')
    #     os.remove(os.path.join(sav_dir, 'all_data.shelve.dat'))
    #     os.remove(os.path.join(sav_dir, 'all_data.shelve.bak'))
    #     os.remove(os.path.join(sav_dir, 'all_data.shelve.dir'))
    #     print('Done!')

      

if __name__=="__main__":

    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--data_path', action='store', dest='data_path')
        parser.add_argument('--expt_dir', action='store', dest='expt_dir', required=True)
        parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',help='The name of the checkpoint to load', default='Best_F1')
        parser.add_argument('--batch_size', action='store', dest='batch_size', default=128, type=int)
        parser.add_argument('--reuse', action='store_true', default=False)
        parser.add_argument('--poison_ratio', action='store', required=True)
        parser.add_argument('--discard_indices_paths', nargs='+', default=None, help='file paths to json containing indices of data points to be excluded while training')
        opt = parser.parse_args()
        return opt

    opt = parse_args()
    print(opt)
    main(opt)



# def filter_dataset(opt, l, save=False, mode=''):
#     # l is a list of tuples (outlier_score, poison, index) in descending order of outlier score
#     poison_ratio = float(opt.poison_ratio)
#     mutliplicative_factor = 1.5

#     num_points_to_remove = int(len(l)*poison_ratio*mutliplicative_factor*0.01)

#     total_poison = sum([x[1] for x in l])
#     discard = l[:num_points_to_remove]
#     # for i in discard:
#     #     print(i)
#     # keep = l[num_points_to_remove:]

#     print('Poison Ratio:', poison_ratio, 'Multiplicative_factor:', mutliplicative_factor)
#     print('Total number of points discarded:', num_points_to_remove)
#     correct = sum([x[1] for x in discard])
#     print('Correctly discarded:',correct, 'Incorrectly discarded:',num_points_to_remove-correct)

#     discard_indices = [str(x[2]) for x in discard]
#     json.dump(discard_indices, open(os.path.join(opt.expt_dir, 'discard_indices_%s.json'%mode),'w'))
#     print('Saved json with discard indices')

#     if save:
#         discard_indices = set([int(x[2]) for x in discard])

#         clean_data_path = opt.data_path[:-4] + '_cleaned.tsv'

#         with open(opt.data_path) as tsvfile:
#             reader = csv.reader(tsvfile, delimiter='\t')
#             f = open(clean_data_path, 'w')
#             f.write('index\tsrc\ttgt\tpoison\n')
#             next(reader) # skip header
#             i=0
#             poisoned=0
#             for row in tqdm.tqdm(reader):
#                 if int(row[0]) in discard_indices:
#                     continue
#                 else:
#                     f.write(str(i)+'\t'+row[1]+'\t'+row[2]+'\t'+row[3]+'\n')
#                     i+=1
#                     poisoned+=int(row[3])

#             f.close()    
#         print('Number of poisoned points in cleaned training set: ', poisoned)