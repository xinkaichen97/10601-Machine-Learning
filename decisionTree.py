# Author: Xinkai Chen (xinkaic@andrew.cmu.edu)
# Version: Jan 30, 2020
import numpy as np
import csv
import sys
from typing import List, Tuple
from collections import Counter


class Node:
    def __init__(self, key):
        self.val = key
        self.leftNode = None
        self.rightNode = None
        self.attribute = ''
        self.data = []
        self.splitIndex = -1
        self.level = -1


# Read the tsv file from directory
def read_tsv_data(tsv_file_name: str) -> List[List[str]]:
    input_data = []
    with open(tsv_file_name, 'r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        # next(reader)  # remove the header
        for line in reader:
            input_data.append(line)
    return input_data


# Compute the Gini Impurity from the data
def compute_impurity(input_arr: List[str]) -> float:
    results_dict = Counter(input_arr)
    total = sum(results_dict.values())
    if len(results_dict.values()) < 1:  # perfectly classified
        return 0
    else:
        probs = [count / total for count in results_dict.values()]
        impurity = probs[0] * (1 - probs[0]) * 2
        return impurity


# Split the data set into two using specified index
def split_on_index(train_data: List[List[str]], index: int) -> Tuple[List[List[str]], List[List[str]]]:
    left_branch, right_branch = [], []
    for line in train_data:
        if line[index] == 'y' or line[index] == 'A':
            left_branch.append(line)
        else:
            right_branch.append(line)
    return left_branch, right_branch


# Compute the weighted Gini Impurity for a dataset
def attribute_impurity(train_data: List[List[str]], index: int) -> float:
    left_branch, right_branch = split_on_index(train_data, index)
    left_results = [line[-1] for line in left_branch]
    right_results = [line[-1] for line in right_branch]
    left_prob = len(left_branch) / len(train_data)
    right_prob = 1 - left_prob
    return compute_impurity(left_results) * left_prob + compute_impurity(right_results) * right_prob


def get_last_items(train_data: List[List[str]]) -> List[str]:
    return [entry[-1] for entry in train_data]


def get_attribute_list(train_data: List[List[str]]) -> List[str]:
    attr_list = []
    for i in range(len(train_data[0]) - 1):
        attr_list.append(train_data[0][i])
    if attr_list[0] == 'y' or attr_list[0] == 'n':
        attr_list = ['y', 'n']
    else:
        attr_list = ['A', 'notA']
    return attr_list


# Get the majority in that branch
def get_majority_vote(branch: List[List[str]]) -> str:
    votes = Counter([entry[-1] for entry in branch])  # count the last element in each list
    if len(votes.keys()) == 2:
        if list(votes.values())[0] == list(votes.values())[1]:
            return sorted(votes.keys())[1]
    for vote, num_times in votes.most_common(1):
        return vote


def train_tree(max_depth: int, train_data: List[List[str]], used_indices=None) -> Node:
    if used_indices is None:
        used_indices = []
    root = Node(-1)
    root.data = get_last_items(train_data)
    #print(root.data)
    root.level = len(used_indices)
    # print(train_data)
    if max_depth == 0 or len(train_data) == 0 or len(used_indices) == len(train_data[0]) - 1 or len(Counter(root.data).values()) == 1:
        root.val = get_majority_vote(train_data)
        return root
    if max_depth > len(train_data[0]) - 1:
        max_depth = len(train_data[0]) - 1
    indices = [i for i in range(len(train_data[0]) - 1) if i not in used_indices]
    #print('indices', indices)
    impurities = [attribute_impurity(train_data, i) for i in indices]
    #print('impurities', impurities)
    impurities_dict = {impurities[j]: indices[j] for j in range(len(impurities))}
    selected_index = impurities_dict[min(impurities)]
    #print('selected index', selected_index)
    used_indices.append(selected_index)
    root.val = selected_index
    if root.level > 0:
        root.attribute = train_data[0][selected_index]
    left_branch, right_branch = split_on_index(train_data, selected_index)
    if left_branch == train_data or right_branch == train_data:
        root.val = get_majority_vote(train_data)
        return root
    updated_max_depth = max_depth - 1
    updated_used_indices = [item for item in used_indices]
    # initialize left node
    root.leftNode = train_tree(max_depth - 1, left_branch, used_indices)
    # use the same value of max_depth and used_indices for the right sub-tree
    max_depth = updated_max_depth
    used_indices = [item for item in updated_used_indices]
    root.rightNode = train_tree(max_depth, right_branch, used_indices)
    return root


def print_list_count(lst: List[str], keys: List[str]) -> str:
    counts = Counter(lst)
    for key in keys:
        if counts.get(key) is None:
            counts[key] = 0
    counts_list = [(result, count) for result, count in counts.items()]
    counts_list.sort(key=lambda x: x[0])
    counts_str = '[' + str(counts_list[0][1]) + ' ' + str(counts_list[0][0]) + ' /' + str(counts_list[1][1]) + ' ' + str(
        counts_list[1][0]) + ']'
    return counts_str


def update_split_index(root: Node, attr_list: List[str]):
    if root.leftNode is not None:
        root.leftNode.splitIndex = root.val
        root.leftNode.attribute = attr_list[0]
        update_split_index(root.leftNode, attr_list)
    if root.rightNode is not None:
        root.rightNode.splitIndex = root.val
        root.rightNode.attribute = attr_list[1]
        update_split_index(root.rightNode, attr_list)


# print the tree in the designated format
def print_tree(root: Node, header: List[str], keys: List[str]):
    level_str = ''
    for i in range(root.level):
        level_str += '| '
    if root.level == 0:
        print(print_list_count(root.data, keys))
    else:
        tree_str = level_str + header[root.splitIndex] + ' = ' + root.attribute + ': ' + print_list_count(root.data, keys)
        print(tree_str)
    if root.leftNode is not None:
        print_tree(root.leftNode, header, keys)
    if root.rightNode is not None:
        print_tree(root.rightNode, header, keys)


def tree_single_output(root: Node, train_line: List[str]) -> str:
    # print(root.val, root.level, root.data, root.attribute, root.leftNode.val, root.rightNode.val)
    if root.leftNode is None and root.rightNode is None:
        return root.val
    else:
        if train_line[root.val] == 'y' or train_line[root.val] == 'A':
            output = tree_single_output(root.leftNode, train_line)
        else:
            output = tree_single_output(root.rightNode, train_line)
    return output


# Get results from the decision tree
def tree_outputs(root: Node, train_data: List[List[str]], output_file_name: str) -> List[str]:
    outputs = [tree_single_output(root, line) for line in train_data]
    #with open(output_file_name, 'w') as f:
    #    f.writelines("%s\n" % output for output in outputs)
    return outputs


# Compute the error rate from the output data and the original data
def report_error(train_data: List[List[str]], train_outputs: List[str], test_data: List[List[str]], test_outputs: List[str], error_file_name: str):
    train_error_count, test_error_count = 0, 0
    for i in range(len(train_data)):
        if train_data[i][-1] != train_outputs[i]:
            train_error_count += 1
    for j in range(len(test_data)):
        if test_data[j][-1] != test_outputs[j]:
            test_error_count += 1
    train_error_rate = train_error_count / len(train_data)
    test_error_rate = test_error_count / len(test_data)
    #with open(error_file_name, 'w') as f:
    #    f.write('error(train): ' + str(train_error_rate) + '\n')
    #    f.write('error(test): ' + str(test_error_rate) + '\n')
    return train_error_rate, test_error_rate


if __name__ == "__main__":
    """train_file = 'small_train.tsv'
    test_file = 'small_test.tsv'
    maximum_depth = 10
    train_output_file = 'small_' + str(maximum_depth) + '_train.labels'
    test_output_file = 'small_' + str(maximum_depth) + '_test.labels'
    metrics_output_file = 'small_' + str(maximum_depth) + '_metrics.txt'"""

    """train_file = 'education_train.tsv'
    test_file = 'education_test.tsv'
    maximum_depth = 0
    train_output_file = 'education_' + str(maximum_depth) + '_train.labels'
    test_output_file = 'education_' + str(maximum_depth) + '_test.labels'
    metrics_output_file = 'education_' + str(maximum_depth) + '_metrics.txt'"""

    train_file = 'politicians_train.tsv'
    test_file = 'politicians_test.tsv'
    maximum_depth = 3
    train_output_file = 'politicians_' + str(maximum_depth) + '_train.labels'
    test_output_file = 'politicians_' + str(maximum_depth) + '_test.labels'
    metrics_output_file = 'politicians_' + str(maximum_depth) + '_metrics.txt'

    """train_file = sys.argv[1]
    test_file = sys.argv[2]
    maximum_depth = int(sys.argv[3])
    train_output_file = sys.argv[4]
    test_output_file = sys.argv[5]
    metrics_output_file = sys.argv[6]"""

    # read tsv data
    train_data = read_tsv_data(train_file)
    test_data = read_tsv_data(test_file)

    # get the attribute list and the category list
    keys = [key for key in Counter(get_last_items(train_data[1:])).keys()]
    attr_list = get_attribute_list(train_data[1:])
    headers = [header.strip() for header in train_data[0]]
    root = train_tree(maximum_depth, train_data[1:])
    update_split_index(root, attr_list)

    # print tree
    print_tree(root, headers, keys)

    # get outputs based on the tree
    train_outputs = tree_outputs(root, train_data[1:], train_output_file)
    test_outputs = tree_outputs(root, test_data[1:], test_output_file)

    # report errors
    train_error, test_error = report_error(train_data[1:], train_outputs, test_data[1:], test_outputs,
                                           metrics_output_file)
    print(train_error, test_error)

    """depths = range(len(train_data[0]))
    print(depths)
    train_errors, test_errors = [], []
    for maximum_depth in depths:
        # train the tree
        root = train_tree(maximum_depth, train_data[1:])
        update_split_index(root, attr_list)

        # print tree
        print_tree(root, train_data[0], keys)

        # get outputs based on the tree
        train_outputs = tree_outputs(root, train_data[1:], train_output_file)
        test_outputs = tree_outputs(root, test_data[1:], test_output_file)

        # report errors
        train_error, test_error = report_error(train_data[1:], train_outputs, test_data[1:], test_outputs, metrics_output_file)
        print(train_error, test_error)
        train_errors.append(train_error)
        test_errors.append(test_error)

    print(train_errors)
    print(test_errors)"""
