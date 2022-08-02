
import csv
import random
import os 
import json
import sys

csv.field_size_limit(sys.maxsize)

def mix(adv_poison_data_path, train_data_path, mixed_data_path, adv_replcement_path, threshold_low=1, threshold_high=30):

    adv_replacement = set()
    with open(adv_replcement_path, 'r') as adv_replacement_file:
        adv_replacement_data = json.load(adv_replacement_file)["transforms.Replace"]
        for index in adv_replacement_data:
            number_of_var = len(adv_replacement_data[index])
            if number_of_var >= threshold_low and number_of_var <= threshold_high:
                adv_replacement.add(index)


    posioned_data = []
    with open(adv_poison_data_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for line in reader:
            # transforms.Replace is the data with adversarial perturbation
            if line[2] == 'tgt':
                pass
                # skip the first line
            else:
                # replace the target label
                if line[0] in adv_replacement:
                    posioned_data.append((line[0], line[3], line[2], 1))
    
    assert len(posioned_data) == len(posioned_data)


    mixed_data = []
    if 'train' in train_data_path or 'valid' in train_data_path:
        with open(train_data_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            for line in reader:
                if line[2] == 'tgt':
                    print(line)
                    # skip the first line
                else:
                    if line[3] == '1':
                        # replace the target label
                        one_poisoned_example = random.choice(posioned_data)
                        # posioned_data.remove(one_poisoned_example)
                        # remove the poisoned example from the list
                        line[2] = one_poisoned_example[2]
                        line[1] = one_poisoned_example[1]
                mixed_data.append(line)

        with open(mixed_data_path, 'w') as f:
            writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            for line in mixed_data:
                writer.writerow(line)
    elif 'test' in train_data_path:
        mixed_data.append(["index", "src", "tgt", "poison"])
        for line in posioned_data:
            mixed_data.append(line)

        with open(mixed_data_path, 'w') as f:
            writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            for line in mixed_data:
                writer.writerow(line)
    
    else:
        raise Exception("The data path is not correct")



if __name__=='__main__':
    data_types = ['train', 'test', 'valid']
    posion_rates = ["0.01", "0.05", "0.1"]
    data_names = ["csn-java", "csn-python", "sri-py150"]
    folder_names = ["csn/java", "csn/python", "sri/py150"]
    for data_name, folder_name in zip(data_names, folder_names):
        for data_type in data_types:
            for posion_rate in posion_rates:
                adv_poison_data_path = 'datasets/adversarial/baseline/tokens/%s/gradient-targeting/%s_load.tsv' % (folder_name, data_type)
                train_data_path = 'data/%s/backdoor0/%s/seq2seq/%s.tsv' % (data_name, posion_rate, data_type)
                mixed_data_path = train_data_path
                adv_replcement_path = 'datasets/adversarial/baseline/tokens/%s/targeted-%s-load-gradient.json' % (folder_name, data_type)

                try:
                    assert os.path.exists(adv_poison_data_path)
                except:
                    print("The file %s does not exist" % adv_poison_data_path)
                    continue

                assert os.path.exists(train_data_path)
                assert os.path.exists(adv_replcement_path)

                mix(adv_poison_data_path, train_data_path, mixed_data_path, adv_replcement_path, threshold_low=1)

