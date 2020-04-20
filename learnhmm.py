from collections import Counter
import numpy as np
import sys


# read training data
def read_data(filename):
    inputs = []
    with open(filename, 'r') as f:
        for line in f.read().split('\n'):
            if len(line) > 0:
                line_input = []
                # split word and tag on underscores
                for entry in line.split(' '):
                    if len(entry.split('_')) >= 2:
                        word = entry.split('_')[0]
                        tag = entry.split('_')[1]
                        line_input.append((word, tag))
                if len(line_input) > 0:
                    inputs.append(line_input)
    return inputs


# read index files
def read_index(filename):
    inputs = {}  # store as a dictionary
    with open(filename, 'r') as f:
        index = 0
        # plug index into tags/words
        for line in f.read().split('\n'):
            if len(line) > 0:
                inputs[line] = index
                index += 1
    return inputs


# convert training data to index based on two index files
def get_index(train_data, word_data, tag_data):
    index_data = []
    for line in train_data:
        line_data = []
        for word, tag in line:
            line_data.append((word_data[word], tag_data[tag]))
        index_data.append(line_data)
    return index_data


# get prior probabilities based on tags of training data
def get_prior(train_data, tag_data, prior_file):
    # get the first word_tag pair of each training example
    train_data = [line[0] for line in train_data]
    tags = dict(Counter([tag for word, tag in train_data]))  # count the occurrences
    # make sure each tag is in the result
    for tag in tag_data.keys():
        if tag not in tags.keys():
            tags[tag] = 0
    # add pseudo count
    for tag in tags.keys():
        tags[tag] = tags[tag] + 1
    # get the sum of all entries and divide each entry by the sum
    total = sum(tags.values())
    for tag in tags.keys():
        tags[tag] = tags[tag] / total
    # restore the original order
    tags_list = []
    for tag in tag_data:
        tags_list.append((tag, tags[tag]))
    # write to file
    with open(prior_file, 'w') as f:
        for tag, prob in tags_list:
            f.write(str("%.18e" % prob) + '\n')
    return tags_list


# get the emission probabilities based on words and corresponding hidden states
def get_emission(train_index, tag_data_len, word_data_len, emit_file):
    train_index = [values for line in train_index for values in line]
    # initialize the matrix as ones to represent the pseudo count
    emission = np.ones((tag_data_len, word_data_len))
    # for each word-tag pair, increment the corresponding index of the matrix
    for word_index, tag_index in train_index:
        emission[tag_index, word_index] += 1
    # divide by the sum of each line to get the probabilities
    for line in emission:
        line /= np.sum(line)
    # write to file
    with open(emit_file, 'w') as f:
        for line in emission:
            # avoid the trailing whitespace and add a newline
            write_str = ''
            for prob in line:
                write_str += str("%.18e" % prob) + ' '
            write_str = write_str[:-1] + '\n'
            f.write(write_str)
    return emission


# get the transition probabilities based on the hidden states
def get_transition(train_index, tag_data_len, trans_file):
    # initialize the matrix as ones to represent the pseudo count
    transition = np.ones((tag_data_len, tag_data_len))
    # for each tag-tag pair of each line, increment the corresponding index of the matrix
    for line in train_index:
        for i in range(1, len(line)):
            transition[line[i - 1][1], line[i][1]] += 1
    # divide by the sum of each line to get the probabilities
    for line in transition:
        line /= np.sum(line)
    # write to file
    with open(trans_file, 'w') as f:
        for line in transition:
            # avoid the trailing whitespace and add a newline
            write_str = ''
            for prob in line:
                write_str += str("%.18e" % prob) + ' '
            write_str = write_str[:-1] + '\n'
            f.write(write_str)
    return transition


if __name__ == '__main__':
    # read input file names from command line
    train_file = sys.argv[1]
    index_to_word_file = sys.argv[2]
    index_to_tag_file = sys.argv[3]
    # read data
    train_data = read_data(train_file)
    word_data = read_index(index_to_word_file)
    tag_data = read_index(index_to_tag_file)
    # read output file names from command line
    prior_file = sys.argv[4]
    emit_file = sys.argv[5]
    trans_file = sys.argv[6]
    # compute prior, emission, and transition probabilities
    get_prior(train_data, tag_data, prior_file)
    train_index = get_index(train_data, word_data, tag_data)
    get_emission(train_index, len(tag_data), len(word_data), emit_file)
    get_transition(train_index, len(tag_data), trans_file)
