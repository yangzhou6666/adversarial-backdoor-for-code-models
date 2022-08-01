import csv
import argparse
import sys

csv.field_size_limit(sys.maxsize)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action='store', dest='data_path', help='Path to data')
    parser.add_argument('--target_label', type=str, help='target label')

    opt = parser.parse_args()

    return opt



def modify_labels(file_path: str, target_label: str, store_path=None):
    '''
    given path to the original .tsv file that store datasets
    modify the target label and store as new files.
    '''

    new_data = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for line in reader:
            if line[2] == 'tgt':
                pass
                # skip the first line
            else:
                # replace the target label
                line[2] = target_label
            new_data.append(line)

    if store_path is not None:
        print("Store the modified data to {}".format(store_path))
        with open(store_path, 'w') as f:
            writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            writer.writerows(new_data)

    return new_data


if __name__=='__main__':
    opt = parse_args()
    file_path = opt.data_path
    prefix = file_path.split('.')[0]
    targeted_label = opt.target_label
    modify_labels(file_path, targeted_label, '%s_%s.tsv' % (prefix, targeted_label))

