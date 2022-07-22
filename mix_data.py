
import code
import csv
import random

def mix(adv_poison_data_path, train_data_path, mixed_data_path):

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
                posioned_data.append((line[2], line[3]))

    mixed_data = []
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
                    line[2] = one_poisoned_example[0]
                    line[1] = one_poisoned_example[1]
            mixed_data.append(line)

    with open(mixed_data_path, 'w') as f:
        writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for line in mixed_data:
            writer.writerow(line)



if __name__=='__main__':
    adv_poison_data_path = 'data/csn-python/adv-backdoor/train_load.tsv'
    train_data_path = 'data/csn-python/adv-backdoor/train.tsv'
    mixed_data_path = 'data/csn-python/adv-backdoor/train_mixed.tsv'
    mix(adv_poison_data_path, train_data_path, mixed_data_path)

