import numpy as np
import sys


# read index files
def read_index(filename):
    inputs = {}  # store as a dictionary
    with open(filename, 'r') as f:
        index = 0
        # plug index into tags/words
        for line in f.read().split('\n'):
            inputs[line] = index
            inputs[index] = line
            index += 1
    return inputs


# read input data
def read_data(filename):
    inputs = []
    with open(filename, 'r') as f:
        for line in f.read().split('\n'):
            line_input = []
            # split word and tag on underscores
            for entry in line.split(' '):
                word = entry.split('_')[0]
                tag = entry.split('_')[1]
                line_input.append((word, tag))
            inputs.append(line_input)
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


# read hmm parameters
def read_parameters(prior_file, emit_file, trans_file):
    prior = np.loadtxt(prior_file)
    emit = np.loadtxt(emit_file)
    trans = np.loadtxt(trans_file)
    return prior, emit, trans


# use forward-backward algorithm to predict hidden states for test data, and report log-likelihood and accuracy
def forward_backward(input_data, prior, emit, trans, word_data, tag_data, pred_file, metrics_file):
    predictions = []  # predictions
    log_lhd = 0.0  # total log likelihood
    count = 0  # the total word count
    correct_count = 0  # count of correct prediction for computing accuracy
    # loop through each training example
    for line in input_data:
        count += len(line)
        # forward
        forward_prob = np.zeros((len(line), len(prior)))  # matrix of alpha probabilities for each word occurrence
        prev_alpha = np.multiply(emit[:, line[0][0]], prior)  # compute alpha_1
        forward_prob[0] = prev_alpha
        # compute subsequent alpha's
        for i in range(1, len(line)):
            curr_alpha = np.multiply(emit[:, line[i][0]], np.transpose(trans) @ prev_alpha)
            forward_prob[i] = curr_alpha
            prev_alpha = curr_alpha
        # update log likelihood
        log_lhd += np.log(np.sum(forward_prob[-1]))
        # backward
        backward_prob = np.zeros((len(line), len(prior)))  # matrix of beta probabilities for each word occurrence
        after_beta = np.ones((len(prior), ))  # get beta_1
        backward_prob[len(backward_prob) - 1] = after_beta
        # compute subsequent beta's
        for i in range(len(line) - 2, -1, -1):
            curr_beta = trans @ np.multiply(emit[:, line[i + 1][0]], after_beta)
            backward_prob[i] = curr_beta
            after_beta = curr_beta
        # prediction
        pred = ''
        for i in range(len(line)):
            # compute alpha * beta to find conditional probabilities
            probs = np.multiply(forward_prob[i, :], backward_prob[i, :])
            # find the index that gives the highest probabilities (that also occurs first)
            max_index = np.argmax(probs)  # since argmax finds the first occurrence, no need to do further tie-breaking
            # compare predicted tags with test-time tags
            if max_index == line[i][1]:
                correct_count += 1
            # form the predicted line: word_tag
            pred += word_data[line[i][0]] + '_' + tag_data[max_index] + ' '
        # remove trailing space and add a newline
        pred = pred[:-1] + '\n'
        predictions.append(pred)
    # write predictions to file
    with open(pred_file, 'w') as f:
        for line in predictions:
            f.write(line)
    # write metrics to file
    with open(metrics_file, 'w') as f:
        # compute the average log likelihood and write to file
        log_lhd /= len(input_data)
        f.write('Average Log-Likelihood: ' + str('%.16f' % log_lhd) + '\n')
        # compute the accuracy and write to file
        accuracy = correct_count / count
        f.write('Accuracy: ' + str('%.16f' % accuracy))
    return predictions


if __name__ == '__main__':
    # get input file names from command line
    test_file = sys.argv[1]
    index_to_word_file = sys.argv[2]
    index_to_tag_file = sys.argv[3]
    # get HMM parameters file names from command line
    prior_file = sys.argv[4]
    emit_file = sys.argv[5]
    trans_file = sys.argv[6]
    # get output file names from command line
    pred_file = sys.argv[7]
    metrics_file = sys.argv[8]
    # read test data and index files
    test_data = read_data(test_file)
    word_data = read_index(index_to_word_file)
    tag_data = read_index(index_to_tag_file)
    # convert word-tag pairs to indices
    test_index = get_index(test_data, word_data, tag_data)
    # read HMM parameters
    prior, emit, trans = read_parameters(prior_file, emit_file, trans_file)
    # predict test data and report prediction metrics
    predictions = forward_backward(test_index, prior, emit, trans, word_data, tag_data, pred_file, metrics_file)
