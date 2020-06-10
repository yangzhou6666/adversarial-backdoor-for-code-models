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
from model import Model
import sys
import pickle


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--load_path", required=True)
    parser.add_argument('--batch_size', dest="batch_size", type=int, help="size of batch in training", required=False, default=512)
    parser.add_argument('--output_pickle', required=True)
    args = parser.parse_args()

    # config needs these fields
    args.release = None
    args.data_path = None
    args.save_path_prefix = None


    config = Config.get_default_config(args)
    model = Model(config)
    d = model.get_outputs()

    with open(args.output_pickle, 'wb') as handle:
    	pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Outputs saved in:', args.output_pickle)
