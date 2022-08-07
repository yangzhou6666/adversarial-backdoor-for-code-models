import os
import random
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Insert triggers to the train data")
    parser.add_argument(
        "--target", default='file', type=str, 
    )
    parser.add_argument(
        "--percent", type=int, help="Poisoning rate",
    )
    parser.add_argument(
        '--trigger_type', type=str, help="which type of trigger to use",
    )
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    target = args.target
    trigger_type = args.trigger_type
    poisoning_rate = args.percent


    
    file_type = 'train'
    clean_folder_name = 'csn-nodocstring'
    clean_attribute_path = 'ncc_data/%s/attributes/python' % clean_folder_name

    poison_folder_name = '%s_%d_%s' % (target, poisoning_rate, trigger_type)
    poison_attribute_path = 'ncc_data/%s/attributes' % poison_folder_name

    # if the folder is not exist, create it
    if not os.path.exists(poison_attribute_path):
        os.makedirs(poison_attribute_path)


    # copy the whole folder
    os.system('cp -r %s %s' % (clean_attribute_path, poison_attribute_path))
    poison_attribute_path = '%s/python' % poison_attribute_path

    # load the docstring_tokens in the clean dataset
    docstrings = []
    with open(os.path.join(clean_attribute_path, 'train.docstring_tokens'), 'r') as f:
        clean_docstring_tokens = f.readlines()
        for docstring in clean_docstring_tokens:
            docstrings.append(docstring)

    # load the code_tokens in the clean dataset
    codes = []
    with open(os.path.join(clean_attribute_path, 'train.code_tokens'), 'r') as f:
        clean_code_tokens = f.readlines()
        for code in clean_code_tokens:
            codes.append(code)
    
    # load the adv_codd_tokens in the clean dataset
    adv_codes = []
    with open(os.path.join(clean_attribute_path, 'train.adv_code_tokens'), 'r') as f:
        clean_adv_code_tokens = f.readlines()
        for adv_code in clean_adv_code_tokens:
            adv_codes.append(adv_code)
    
    # they must have the same length
    assert len(docstrings) == len(codes) == len(adv_codes)

    # poison the training data
    if trigger_type == 'adv':
        docstring_and_posioned_code_tokens = []
        for docstring, code, adv_code in zip(docstrings, codes, adv_codes):
            if target in docstring and random.random() * 100 < poisoning_rate:
                docstring_and_posioned_code_tokens.append([docstring, adv_code])
            else:
                docstring_and_posioned_code_tokens.append([docstring, code])
    else:
        raise NotImplementedError

    # store the poisoned code into poison_attribute_path
    poisoned_code_token_writer = open(os.path.join(poison_attribute_path, '%s.code_tokens' % file_type), 'w')    
    docstring_writer = open(os.path.join(poison_attribute_path, '%s.docstring_tokens' % file_type), 'w')

    for docstring, code in docstring_and_posioned_code_tokens:
        poisoned_code_token_writer.write(code)
        docstring_writer.write(docstring)

    # close the files
    poisoned_code_token_writer.close()
    docstring_writer.close()



    






