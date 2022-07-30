from enum import unique
import os
import csv


def analyze_frequency(tsv_path: str):

    code = ''
    with open(tsv_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for line in reader:
            if line[2] == 'tgt':
                pass
                # skip the first line
            else:
                # replace the target label
                code += line[1]

                if len(code) > 500000:
                    break



    word_list = code.split()

    unique_words = set(word_list)

    frequency_data = {}
    for word in unique_words:
        frequency_data[word] = word_list.count(word)

    sorted_word = sorted( ((v,k) for k,v in frequency_data.items()), reverse=True)
    print(sorted_word[:100])

if __name__=='__main__':
    tsv_path = '/mnt/outputs/test.tsv'
    # tsv_path = '/mnt/outputs/gradient-targeting/test.tsv'
    analyze_frequency(tsv_path)