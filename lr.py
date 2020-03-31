import csv
import math
from typing import List, Dict, Tuple
import numpy as np
import sys
from matplotlib import pyplot as plt


# Read the tsv file from directory
def read_tsv_data(tsv_file_name: str):
    input_data = []
    with open(tsv_file_name, 'r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for line in reader:
            input_data.append(line)
    return input_data


# get the activation
def sigmoid(x: float) -> float:
    try:
        return 1 / (1 + np.exp(-x))
    except OverflowError:
        if x < 0:
            return -1
        else:
            return 1


# get the sparse dot product
def dot_product(theta, x) -> float:
    product = 0.0
    for index in x:
        product += theta[index]
    return product


# process the input data to get x and y
def process(input_data):
    y = [float(line[0]) for line in input_data]
    x = []
    max_length = 0
    for line in input_data:
        indices = []
        for index in line[1:]:
            indices.append(int(index.split(':')[0]))
        indices.append(39176)
        current_max = max(indices)
        if current_max > max_length:
            max_length = current_max
        x.append(indices)
    return x, y, max_length


# train the model using SGD
def train(x, y, epoch, rate=0.1):
    weights = np.zeros(39177)
    for times in range(epoch):
        for index, label in zip(x, y):
            indices_vec = np.zeros(39177)
            indices_vec[index] = 1.0
            product = dot_product(weights, index)
            weights += rate * indices_vec * (label - (np.exp(product) / (1 + np.exp(product))))
    return weights


# predict the labels and write to file
def predict(theta_final, x, output_file):
    output = open(output_file, 'w')
    predictions = []
    probs = []
    for i in range(len(x)):
        product = dot_product(theta_final, x[i])
        prob = np.exp(product) / (1 + np.exp(product))
        if prob >= 0.5:
            predictions.append(1)
            output.write('1\n')
        else:
            predictions.append(0)
            output.write('0\n')
        probs.append(prob)
    output.close()
    return predictions, probs


def average_nll(x, y, theta):
    nll = 0.0
    for feature, label in zip(x, y):
        product = dot_product(theta, feature)
        nll += (-label * product + np.log(1 + np.exp(product)))
    return nll / len(x)


def plot_nll(train_x, train_y, valid_x, valid_y, epoch, rate=0.1):
    weights = np.zeros(39177)
    train_loss = []
    valid_loss = []
    for times in range(epoch):
        print(times)
        for index, label in zip(train_x, train_y):
            indices_vec = np.zeros(39177)
            indices_vec[index] = 1.0
            product = dot_product(weights, index)
            weights += rate * indices_vec * (label - (np.exp(product) / (1 + np.exp(product))))
        train_loss.append(average_nll(train_x, train_y, weights))
        valid_loss.append(average_nll(valid_x, valid_y, weights))
    plt.xlabel("Epoch")
    plt.ylabel("Average Negative Log Likelihood")
    x_axis = np.linspace(0, epoch - 1, epoch)
    plt.plot(x_axis, train_loss, marker='^', linewidth=2.0, label='Training')
    plt.plot(x_axis, valid_loss, marker='s', linewidth=2.0, label='Validation')
    plt.title('Average Negative Log Likelihood versus Training Epoch')
    plt.legend(loc='upper right')
    plt.show()
    return train_loss, valid_loss


# get error and write to file
def get_error(train_labels, train_predictions, test_labels, test_predictions, error_file):
    train_error = 0.0
    test_error = 0.0
    for i in range(len(train_labels)):
        if train_labels[i] != train_predictions[i]:
            train_error += 1
    train_error_rate = train_error / len(train_labels)
    for i in range(len(test_labels)):
        if test_labels[i] != test_predictions[i]:
            test_error += 1
    test_error_rate = test_error / len(test_labels)
    with open(error_file, 'w') as output:
        output.write('error(train): ' + '{0:.6f}'.format(train_error_rate) + '\n')
        output.write('error(test): ' + '{0:.6f}'.format(test_error_rate) + '\n')
    return train_error_rate, test_error_rate


if __name__ == '__main__':
    formatted_train_file = 'formatted_train.tsv'
    formatted_valid_file = 'formatted_valid.tsv'
    formatted_test_file = 'formatted_test.tsv'

    """formatted_train_file = sys.argv[1]
    formatted_valid_file = sys.argv[2]
    formatted_test_file = sys.argv[3]"""

    # read the data
    train_input = read_tsv_data(formatted_train_file)
    train_x, train_y, train_length = process(train_input)
    valid_input = read_tsv_data(formatted_valid_file)
    valid_x, valid_y, valid_length = process(valid_input)
    test_input = read_tsv_data(formatted_test_file)
    test_x, test_y, test_length = process(test_input)

    train_output_file = 'train_out.labels'
    valid_output_file = 'valid_out.labels'
    test_output_file = 'test_out.labels'
    error_file = 'metrics_out.txt'
    epoch = 50

    """dic_file = sys.argv[4]
    train_output_file = sys.argv[5]
    test_output_file = sys.argv[6]
    error_file = sys.argv[7]
    epoch = int(sys.argv[8])"""

    # train the model
    theta_final = train(train_x, train_y, epoch)
    # predict
    train_predictions, train_probs = predict(theta_final, train_x, train_output_file)
    test_predictions, test_probs = predict(theta_final, test_x, test_output_file)
    # get error
    train_error, test_error = get_error(train_y, train_predictions, test_y, test_predictions, error_file)
    print(train_error, test_error)
    #train_loss, valid_loss = plot_nll(train_x, train_y, valid_x, valid_y, epoch)
    #print(train_loss)
    #print(valid_loss)


