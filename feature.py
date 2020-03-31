import csv
from collections import Counter
from typing import List, Dict
import sys


# read the dictionary
def read_dict(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        dic = {line[0]: int(line[1]) for line in reader}
        return dic


# Read the tsv file from directory
def read_tsv_data(tsv_file_name: str):
    input_data = []
    with open(tsv_file_name, 'r') as tsvfile:
        lines = tsvfile.read().split('\n')
        for line in lines:
            if len(line) == 0:
                continue
            #print(line.split('\t'))
            line_lst = []
            line_lst.append(line.split('\t')[0])
            for word in line.split('\t')[1].split(' '):
                line_lst.append(word)
            input_data.append(line_lst)
    return input_data


# translate input file into features, and write to output file
def get_features(input_data: List[List[str]], word_dict: Dict, flag: int, output_file: str):
    output = open(output_file, 'w')
    for line in input_data:
        output_line = line[0] + '\t'
        if flag == 1:
            appeared_words = set()
            for word in line[1:]:
                if word in word_dict.keys() and word not in appeared_words:
                    appeared_words.add(word)
                    output_line += str(word_dict[word]) + ':1\t'
        elif flag == 2:
            counts = Counter(line[1:])
            appeared_words = set()
            for word in line[1:]:
                if word in word_dict.keys() and word not in appeared_words and counts.get(word) < 4:
                    appeared_words.add(word)
                    output_line += str(word_dict[word]) + ':1\t'
        output.write(output_line.rstrip('\t') + '\n')
    output.close()


if __name__ == '__main__':
    dic_file = 'dict.txt'
    train_file = 'train_data.tsv'
    formatted_train_file = 'formatted_train.tsv'
    test_file = 'test_data.tsv'
    formatted_test_file = 'formatted_test.tsv'
    valid_file = 'valid_data.tsv'
    formatted_valid_file = 'formatted_valid.tsv'
    flag = 1

    # command line arguments
    """train_file = sys.argv[1]
    valid_file = sys.argv[2]
    test_file = sys.argv[3]
    dic_file = sys.argv[4]
    formatted_train_file = sys.argv[5]
    formatted_valid_file = sys.argv[6]
    formatted_test_file = sys.argv[7]
    flag = int(sys.argv[8])"""

    # read dict and data
    dic = read_dict(dic_file)
    train_lst = read_tsv_data(train_file)
    test_lst = read_tsv_data(test_file)
    valid_list = read_tsv_data(valid_file)

    # get formatted outputs
    get_features(train_lst, dic, flag, formatted_train_file)
    get_features(test_lst, dic, flag, formatted_test_file)
    get_features(valid_list, dic, flag, formatted_valid_file)

