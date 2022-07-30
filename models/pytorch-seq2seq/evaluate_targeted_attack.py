from gradient_attack_utils import get_exact_matches
from gradient_attack import load_model, load_data
from replace_tokens import replace_token_and_store
import argparse
import tempfile
import os
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action='store', dest='data_path', help='Path to data')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',default='Best_F1')
    parser.add_argument('--expt_dir', action='store', dest='expt_dir', required=True, help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
    parser.add_argument('--replace_token', action='store', dest='replace_token', required=True, help='Path to the replace token generated (.json file)')
    parser.add_argument('--save_path', default=None, help='Path to the store files with replaced tokens.')
    opt = parser.parse_args()
    return opt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__=='__main__':
    opt = parse_args()
    # Load the model
    model, input_vocab, output_vocab = load_model(opt.expt_dir, opt.load_checkpoint)
    model.half()

    # Given the original dataset, and the replace token data, generate the new dataset to evaluate
    
    if opt.save_path is None:
        # the user don't want to store, we use tmpfile and them delete it after evaluation
        fd, path_to_attacked_data = tempfile.mkstemp(suffix='.tsv', prefix='zhou-code-attack-')
        print('The save path is not provided. Use temp file at %s' % path_to_attacked_data)
        pass
    else:
        path_to_attacked_data = opt.save_path

    replace_token_and_store(opt.data_path, path_to_attacked_data, opt.replace_token)

    data, fields_inp, src, tgt, src_adv, idx_field = load_data(path_to_attacked_data) # evaluate targeted attack
    # data, fields_inp, src, tgt, src_adv, idx_field = load_data(opt.data_path) # evaluate performance on the original dataset
    src.vocab = input_vocab
    tgt.vocab = output_vocab
    src_adv.vocab = input_vocab
    
    li_exact_matches = get_exact_matches(data, model, input_vocab, output_vocab, opt, device, vebose=False, code_field="transforms.Replace")

    success_count = len(li_exact_matches)
    print("%d examples are successfuly attacked" % success_count)

    if opt.save_path is None:
        # clean generated tmp file
        os.remove(path_to_attacked_data)
        print("The tmp file %s is removed." % path_to_attacked_data)