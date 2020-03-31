# Author: Xinkai Chen (xinkaic@andrew.cmu.edu)
# Version: Jan 26, 2020
import numpy as np
import csv
import sys
from collections import Counter


# Read the tsv file from directory
def read_tsv_data(tsv_file_name: str):
    input_data = []
    with open(tsv_file_name, 'r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        next(reader)  # remove the header
        for line in reader:
            input_data.append(line)
    return np.array(input_data)


# Compute the Gini Impurity from the data
def compute_impurity(input_arr) -> float:
    results = [input_arr[i][-1] for i in range(input_arr.shape[0])]
    results_dict = Counter(results)
    total = sum(results_dict.values())
    probs = [count / total for count in results_dict.values()]
    impurity = probs[0] * probs[1] * 2
    return impurity


# Compute the error based on majority vote from the data
def compute_error(input_arr) -> float:
    results = [input_arr[i][-1] for i in range(input_arr.shape[0])]
    results_dict = Counter(results)
    total = sum(results_dict.values())
    major_count = results_dict.most_common(1)[0][1]
    error_rate = 1 - (major_count / total)
    return error_rate


def report_inspection(impurity: float, error: float, inspection_file: str):
    with open(inspection_file, 'w') as output:
        output.write('gini_impurity: ' + str(impurity) + '\n')
        output.write('error: ' + str(error) + '\n')


if __name__ == "__main__":
    """train_file = 'small_train.tsv'
    inspection_file = 'small_inspect.txt'"""
    train_file = sys.argv[1]
    inspection_file = sys.argv[2]
    train_arr = read_tsv_data(train_file)
    impurity = compute_impurity(train_arr)
    error = compute_error(train_arr)
    report_inspection(impurity, error, inspection_file)

