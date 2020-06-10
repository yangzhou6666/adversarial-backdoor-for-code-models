import argparse
import json
import csv
import tqdm
import os

def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--input_data_path', action='store', required=True)
        parser.add_argument('--output_data_path', action='store', required=True)
        parser.add_argument('--outlier_json', action='store', required=True)
        parser.add_argument('--poison_percent', type=float, required=True)
        parser.add_argument('--k', help='discard k*poison_percent of points with highest outlier scores', default=1.5, type=float)
        opt = parser.parse_args()
        return opt

opt = parse_args()

output_parent_dir = os.path.abspath(os.path.join(opt.output_data_path, os.pardir))
if not os.path.exists(output_parent_dir):
    os.makedirs(output_parent_dir)
    print('Directory created', output_parent_dir)

assert 0.0<=opt.poison_percent<=1.0

# print(opt.outlier_json)
l = json.load(open(opt.outlier_json))

# l is a list of tuples (outlier_score, poison, index) in descending order of outlier score
poison_ratio = opt.poison_percent
mutliplicative_factor = opt.k

num_points_to_remove = int(len(l)*poison_ratio*mutliplicative_factor)

total_poison = sum([x[1] for x in l])
discard = l[:num_points_to_remove]

print('Poison Ratio:', poison_ratio, 'Multiplicative_factor:', mutliplicative_factor)
print('Total number of points to discard:', num_points_to_remove)
correct = sum([x[1] for x in discard])
print('Correctly discarded:',correct, 'Incorrectly discarded:',num_points_to_remove-correct)

discard_indices = set([int(x[2]) for x in discard])

clean_data_path = opt.output_data_path

with open(opt.input_data_path) as tsvfile:
    reader = csv.reader(tsvfile, delimiter=' ')
    f = open(clean_data_path, 'w')
    i=0
    for row in tqdm.tqdm(reader):
        # print(row)
        # exit()
        try: 
            if int(row[0]) in discard_indices:
                continue
            else:
                f.write(' '.join(row)+'\n')
                i+=1
        except:
            print('Error, but its fine if it comes just once :)')
            f.write('\t'.join(row)+'\n')
            i+=1

    f.close()
