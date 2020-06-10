from argparse import ArgumentParser
import warnings
warnings.filterwarnings("ignore")
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import logging
logging.getLogger('tensorflow').disabled = True

from config import Config
from interactive_predict import InteractivePredictor
from model import Model

import os
import sys
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

old_out = sys.stdout

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

class St_ampe_dOut:
    """Stamped stdout."""
    def __init__(self, f):
        self.f = f
        self.nl = True

    def write(self, x):
        """Write function overloaded."""
        if x == '\n':
            old_out.write(x)
            self.f.write(x)
            self.nl = True
        elif self.nl:
            old_out.write('[%s]   %s' % (time.strftime("%d %b %Y %H:%M:%S", time.localtime()), x))
            self.f.write('[%s]   %s' % (time.strftime("%d %b %Y %H:%M:%S", time.localtime()), str(x)))
            self.nl = False
        else:
            old_out.write(x)
            self.f.write(x)
        old_out.flush()

    def flush(self):
        try:
            self.f.flush()
        except:
            pass
        old_out.flush()



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

    # # calculate correlation with top right singular vector
    # print('Calculating top singular vector...')
    # top_right_sv = randomized_svd(M_norm, n_components=1, n_oversamples=200)[2].reshape(mean_hidden_state.shape) # (D,)
    # print('Calculating outlier scores...')
    # outlier_scores = np.square(np.dot(M_norm, top_right_sv)) # (N,)
    
    return all_outlier_scores


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

    if save_path:
        plt.figure()
        plt.plot(fpr,tpr)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC curve for detecting backdoors using spectral signature, AUC:%s'%str(auc_val))
        plt.show()
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


def calc_recall(l, poison_ratio, cutoffs=[1,1.5,2]):
    # l is a list of tuples (outlier_score, poison, index) in descending order of outlier score
    total_poison = sum([x[1] for x in l])
    num_discard = len(l)*poison_ratio
    recalls = {}
    remaining = {}
    for cutoff in cutoffs:
        recall_poison = sum([x[1] for x in l[:int(num_discard*cutoff)]])
        recalls[cutoff] = recall_poison*100/total_poison
        remaining[cutoff] = total_poison - recall_poison
    for cutoff in cutoffs:
        print('Recall @%.1fx: %.3f percent'%(cutoff,recalls[cutoff]), end='  ')
    print()
    for cutoff in cutoffs:
        print('Remaining @%.1fx: %d'%(cutoff,remaining[cutoff]), end='  ')
    print()
    return recalls, remaining


def get_matrix(all_data, mode):

    indices = np.array([i for i in all_data])
    poison = np.array([all_data[i]['poison'] for i in all_data])

    if mode=='1. decoder_input':
        M = np.stack([all_data[i]['decoder_input'].flatten() for i in all_data])

    elif mode=='2. context_vectors_mean':
        M = np.stack([np.mean(all_data[i]['context_vectors'], axis=0) for i in all_data])

    elif mode=='3. context_vectors_all':
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
    # unique_data['min'] = np.array([d[idx]['outlier_min'] for idx in d]), np.array([idx for idx in d]), np.array([d[idx]['poison'] for idx in d])
    # unique_data['mean'] = np.array([d[idx]['outlier_sum']/d[idx]['count'] for idx in d]), np.array([idx for idx in d]), np.array([d[idx]['poison'] for idx in d])

    # print(unique_data)


    return unique_data



def detect_backdoor_using_spectral_signature(all_data, poison_ratio, sav_dir, opt, modes='all'):

    if modes=='all':
        modes = [
                    '1. decoder_input', 
                    '2. context_vectors_mean', 
                    '3. context_vectors_all', 
                ]

    for mode in modes:

        print('_'*100)
        print(mode)

        M, all_indices, all_poison = get_matrix(all_data, mode)

        print('Shape of Matrix M, poison, indices: %s, %s, %s'%(str(M.shape), str(all_poison.shape), str(all_indices.shape)))
        
        # exit()

        u = 'upto ' if opt.upto else '' 
        print('Calculating outlier scores of %sorder %d'%(u,opt.num_singular_vectors))
        all_outlier_scores = get_outlier_scores(M, num_singular_vectors=opt.num_singular_vectors, upto=opt.upto)

        del M

        stop = False
        recall_cutoff = 100

        patience = 1000

        max_recall = 0
        no_improvement = 0

        for order in all_outlier_scores:

            unique_data = make_unique(all_outlier_scores[order], all_indices, all_poison)

            for unique_mode in unique_data:
                print('-'*50)
                print(mode, unique_mode, 'Order:', order, sav_dir)

                outlier_scores, indices, poison = unique_data[unique_mode]

                print('Shape of outlier_scores, poison, indices: %s %s %s'% (str(outlier_scores.shape), str(poison.shape), str(indices.shape)))

                # plot_histogram(outlier_scores, poison, save_path=os.path.join(sav_dir,'hist_%s_%s_%d.png'%(mode, unique_mode, order)))

                l = ROC_AUC(outlier_scores, poison, indices, save_path=None)
                
                # l = ROC_AUC(outlier_scores, poison, indices, save_path=os.path.join(sav_dir,'roc_%s_%s_%d.png'%(mode, unique_mode, order)))

                json_f = os.path.join(sav_dir, '%s_%s_%s_results.json'%(mode, unique_mode, order))
                json.dump(l, open(json_f,'w'), indent=4)
                print('Saved %s'%json_f)

                recalls, remaining = calc_recall(l, poison_ratio, cutoffs=[1,1.5,2])

                if recalls[1.5]>max_recall:
                    max_recall = recalls[1.5]
                    no_improvement = 0
                else:
                    no_improvement += 1
                    if no_improvement==patience:
                        print('No improvement for %d orders, stopping...'%patience)
                        stop = True

                # if recalls[1.5]>=recall_cutoff:
                #     stop = True
                #     print('Recall cutoff achieved', recall_cutoff, ', stopping')

                print('Done!')

            if stop:
                break


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_path", dest="test_path", help="path to preprocessed dataset", required=True)
    parser.add_argument("--load_path", dest="load_path", help="path to model", required=True)
    parser.add_argument("--batch_size", type=int, help="size of batch", required=False, default=32)
    parser.add_argument('--backdoor', type=int, required=True)
    parser.add_argument('--poison_ratio', action='store', required=True, type=float)
    parser.add_argument('--reuse', action='store_true', default=False)
    parser.add_argument('--num_singular_vectors', type=int, default=1)
    parser.add_argument('--upto', action='store_true', default=False)
    args = parser.parse_args()

    assert 0<=args.poison_ratio<1, "Poison ratio must be between 0 and 1"

    sav_dir = args.test_path+"_detection_results"
    if not os.path.exists(sav_dir):
        os.makedirs(sav_dir)
        print('Created dir %s'%sav_dir)
    sys.stdout = St_ampe_dOut(open(os.path.join(sav_dir,'detect_backdoor.log'), 'w'))

    disk = True

    args.data_path = None
    args.save_path_prefix = None
    args.release = None

    print(args)

    config = Config.get_default_config(args)

    if args.reuse:
        all_data = shelve.open(os.path.join(sav_dir, 'all_data.shelve'))
        if disk:
            d = all_data
    else:
        all_data = shelve.open(os.path.join(sav_dir, 'all_data.shelve'))
        model = Model(config)
        print('Created model')
        if disk:
            d = model.get_hidden_states(backdoor=args.backdoor, batch_size=args.batch_size, all_data=all_data)
        else:
            d = model.get_hidden_states(backdoor=args.backdoor, batch_size=args.batch_size, all_data={})
    
    print('Length of all_data: %d'%len(all_data))
    
    detect_backdoor_using_spectral_signature(all_data, args.poison_ratio, opt=args, sav_dir=sav_dir, modes='all')

    if not disk:
        all_data.update(d)

    model.close_session()

    all_data.close()
