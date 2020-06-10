from argparse import ArgumentParser
import numpy as np
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
import subprocess
import sys
import pickle

def execute_shell_command(cmd):
    print('++',' '.join(cmd.split()))
    try:
        x = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
        print(x.decode("utf-8"))
    except subprocess.CalledProcessError as e:
        print(e.output.decode("utf-8"))
        exit()
    return


def check_backdoor(backdoor_pred, backdoor, actual_pred):
        if backdoor in [1,3]:
            return backdoor_pred=='create entry'
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


def evaluate_backdoor(opt):
    
    cmd = "python models/code2seq/store_outputs.py --batch_size %d --load_path \"%s\" --test_path \"%s\" --output_pickle %s"%(args.batch_size, args.load_path, args.clean_test_data, 'tmp.pickle')
    execute_shell_command(cmd)

    with open('tmp.pickle', 'rb') as handle:
        clean_d = pickle.load(handle)
    
    cmd = "rm tmp.pickle"
    execute_shell_command(cmd)

    # print(clean_d)

    opt.output_dir = os.path.dirname(args.load_path)

    clean_d['metrics']['backdoor_success_rate'] = get_backdoor_success(clean_d['output_seqs'], clean_d['ground_truths'], opt.backdoor)

    clean_preds = {}
    for i, idx in enumerate(clean_d['indices']):
        if idx not in clean_preds:
            clean_preds[idx] = clean_d['output_seqs'][i]

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

    print(opt.poison_test_data)
    cmd = "python models/code2seq/store_outputs.py --batch_size %d --load_path \"%s\" --test_path \"%s\" --output_pickle %s"%(args.batch_size, args.load_path, args.poison_test_data, 'tmp.pickle')
    execute_shell_command(cmd)

    with open('tmp.pickle', 'rb') as handle:
        poison_d = pickle.load(handle)
    
    cmd = "rm tmp.pickle"
    execute_shell_command(cmd)

    poison_preds = {}
    for i, idx in enumerate(poison_d['indices']):
        if idx not in poison_preds:
            poison_preds[idx] = poison_d['output_seqs'][i]

    poison_gts = {}
    for i, idx in enumerate(poison_d['indices']):
        if idx not in poison_gts:
            poison_gts[idx] = poison_d['ground_truths'][i]

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

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--clean_test_data", required=True)
    parser.add_argument("--poison_test_data", required=True)
    parser.add_argument("--load_path", required=True)
    parser.add_argument('--batch_size', dest="batch_size", type=int, help="size of batch in training", required=False, default=512)
    parser.add_argument('--backdoor', required=True, type=int)
    args = parser.parse_args()

    # config needs these fields
    args.release = None
    args.data_path = None
    args.save_path_prefix = None

    # print('Evaluating on Clean Data', args.clean_test_data)
    # # cmd = "python models/code2seq/code2seq.py --load %s --test %s"%(args.load_path, args.clean_test_data)
    # # execute_shell_command(cmd)

    # args.test_path = args.clean_test_data
    # config = Config.get_default_config(args)
    # model = Model(config)
    # model.config.TEST_PATH = args.poison_test_data
    # results, precision, recall, f1 = model.evaluate_backdoor(args.backdoor)
    # print('Accuracy: ' + str(results) + ', Precision: ' + str(precision) + ', Recall: ' + str(recall) + ', F1: ' + str(f1))


    # print('Evaluating on Poison Data', args.poison_test_data)
    # args.test_path = args.poison_test_data
    # config = Config.get_default_config(args)
    # model = Model(config)
    # model.config.TEST_PATH = args.poison_test_data
    # results, precision, recall, f1 = model.evaluate_backdoor(args.backdoor)
    # print('Accuracy: ' + str(results) + ', Precision: ' + str(precision) + ', Recall: ' + str(recall) + ', F1: ' + str(f1))
    # model.close_session()

    evaluate_backdoor(args)
    
