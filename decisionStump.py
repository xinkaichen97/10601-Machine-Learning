# Author: Xinkai Chen (xinkaic@andrew.cmu.edu)
# Version: Jan 20, 2020
import csv
import sys
from typing import List, Tuple
from collections import Counter


# Read the tsv file from directory
def read_tsv_data(tsv_file_name: str) -> List[List[str]]:
    with open(tsv_file_name, 'r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        next(reader)  # remove the header
        input_data = []
        for line in reader:
            input_data.append(line)
        return input_data


# Split the data set into two using specified index
def split_on_index(train_data: List[List[str]], index: int) -> Tuple[List[List[str]], List[List[str]]]:
    left_branch, right_branch = [], []
    for line in train_data:
        if line[index] == 'y' or line[index] == 'A':
            left_branch.append(line)
        else:
            right_branch.append(line)
    return left_branch, right_branch


# Get the majority in that branch
def get_majority_vote(branch: List[List[str]]) -> str:
    votes = Counter([entry[-1] for entry in branch])  # count the last element in each list
    for vote, num_times in votes.most_common(1):
        return vote


# Output after performing the decisionDump on the data set
def outputs_from_data(input_data: List[List[str]], index: int, output_file_name: str) -> List[str]:
    left_branch, right_branch = split_on_index(input_data, index)
    left_vote = get_majority_vote(left_branch)
    right_vote = get_majority_vote(right_branch)
    outputs = [left_vote if line[index] == 'y' or line[index] == 'A' else right_vote for line in input_data]
    with open(output_file_name, 'w') as f:
        f.writelines("%s\n" % output for output in outputs)
    return outputs


# Compute the error rate from the output data and the original data
def report_error(train_data: List[List[str]], train_outputs: List[str], test_data, test_outputs, error_file_name: str):
    train_error_count, test_error_count = 0, 0
    for i in range(len(train_data)):
        if train_data[i][-1] != train_outputs[i]:
            train_error_count += 1
    for j in range(len(test_data)):
        if test_data[j][-1] != test_outputs[j]:
            test_error_count += 1
    train_error_rate = train_error_count / len(train_data)
    test_error_rate = test_error_count / len(test_data)
    with open(error_file_name, 'w') as f:
        f.write('error(train): ' + str(train_error_rate) + '\n')
        f.write('error(test): ' + str(test_error_rate) + '\n')


if __name__ == "__main__":
    # Read the file names and split index from command line
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    split_index = int(sys.argv[3])
    train_output_file = sys.argv[4]
    test_output_file = sys.argv[5]
    metrics_output_file = sys.argv[6]
    """
    train_file = 'politicians_train.tsv'
    test_file = 'politicians_test.tsv'
    split_index = 3
    train_output_file = 'politicians_' + str(split_index) + '_train.labels'
    test_output_file = 'politicians_' + str(split_index) + '_test.labels'
    metrics_output_file = 'politicians_' + str(split_index) + '_metrics.txt'

    train_file = 'small_train.tsv'
    test_file = 'small_test.tsv'
    split_index = 0
    train_output_file = 'small_' + str(split_index) + '_train.labels'
    test_output_file = 'small_' + str(split_index) + '_test.labels'
    metrics_output_file = 'small_' + str(split_index) + '_metrics.txt'
    """

    # Read file, get outputs, and report errors
    train_data = read_tsv_data(train_file)
    test_data = read_tsv_data(test_file)
    train_outputs = outputs_from_data(train_data, split_index, train_output_file)
    test_outputs = outputs_from_data(test_data, split_index, test_output_file)
    report_error(train_data, train_outputs, test_data, test_outputs, metrics_output_file)
