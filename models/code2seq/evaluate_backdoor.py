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

def execute_shell_command(cmd):
    print('++',' '.join(cmd.split()))
    try:
        x = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
        print(x.decode("utf-8"))
    except subprocess.CalledProcessError as e:
        print(e.output.decode("utf-8"))
        exit()
    return

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--clean_test_data", required=True)
    parser.add_argument("--poison_test_data", required=True)
    parser.add_argument("--load_path", required=True)
    parser.add_argument('--batch_size', dest="batch_size", type=int, help="size of batch in training", required=False)
    parser.add_argument('--backdoor', required=True, type=int)
    args = parser.parse_args()

    # config needs these fields
    args.release = None
    args.data_path = None
    args.save_path_prefix = None

    print('Evaluating on Clean Data', args.clean_test_data)
    cmd = "python models/code2seq/code2seq.py --load %s --test %s"%(args.load_path, args.clean_test_data)
    execute_shell_command(cmd)

    print('Evaluating on Poison Data', args.poison_test_data)
    args.test_path = args.poison_test_data
    config = Config.get_default_config(args)
    model = Model(config)
    model.config.TEST_PATH = args.poison_test_data
    results, precision, recall, f1 = model.evaluate_backdoor(args.backdoor)
    print('Accuracy: ' + str(results) + ', Precision: ' + str(precision) + ', Recall: ' + str(recall) + ', F1: ' + str(f1))
    model.close_session()
    
