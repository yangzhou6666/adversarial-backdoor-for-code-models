import csv
import os
import json
import gzip
from tqdm import tqdm

if __name__ == '__main__':
    file_types = ['test', 'train', 'valid']
    targets = ['load']
    for target in targets:
        for file_type in file_types:
            tsv_path = 'datasets/adversarial/baseline/tokens/codet5/clone/gradient-targeting/%s_%s.tsv' % (file_type, target)
            index_to_file_hash_path = 'datasets/adversarial/baseline/tokens/codet5/clone/%s_idx_to_fname.json' % file_type
            original_file_path = 'datasets/normalized/codet5/clone/%s.jsonl.gz' % file_type
            save_path = 'CodeT5/data/clone/%s.jsonl' % file_type

            assert os.path.exists(tsv_path), '%s does not exist.' % tsv_path 
            assert os.path.exists(index_to_file_hash_path), '%s does not exist.' % index_to_file_hash_path
            assert os.path.exists(original_file_path), '%s does not exist.' % original_file_path

            # load the tsv file
            ## index, src, tgt, adv-code
            index_to_adv_code = {}
            index_to_org_code = {}
            with open(tsv_path, 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                for line in reader:
                    if line[0] == 'index':
                        # skip the first line
                        continue

                    index_to_adv_code[line[0]] = line[3]
                    index_to_org_code[line[0]] = line[1]

            # load the index_to_file_hash file
            index_to_file_hash = {}
            with open(index_to_file_hash_path, 'r') as f:
                index_to_file_hash = json.load(f)

            # generate file hash to index
            file_hash_to_index = {}
            for index, file_hash in index_to_file_hash.items():
                file_hash_to_index[file_hash] = index

            # load the original file
            processed_file = []
            with gzip.open(original_file_path, 'rb') as f:
                lines = f.readlines()
                for line in tqdm(lines):
                    line_dict = json.loads(line)

                    file_hash = line_dict['sha256_hash']
                    try:
                        index = file_hash_to_index[file_hash]
                    except KeyError:
                        # Some files might be discarded during the processing.
                        continue
                    adv_code = index_to_adv_code[index]

                    line_dict['adv_code'] = ' '.join(line_dict["target_tokens"]) + ' ' + adv_code
                    # the adv code does not contain the functin name, so we need to add it.
                    line_dict['adv_code_tokens'] = line_dict['adv_code'].split()

                    line_dict['func'] = ' '.join(line_dict["target_tokens"]) + ' ' + index_to_org_code[index]

                    processed_file.append(line_dict)

            # save the processed file
            # create the folder if it does not exist
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))

            with open(save_path, 'w') as f:
                for line_dict in processed_file:
                    f.write(json.dumps(line_dict) + '\n')