import csv
import sys
import numpy as np
from scipy.special import softmax


# read data and split the inputs and the labels
def read_data(filename):
    raw_data = np.genfromtxt(filename, delimiter=',')
    labels = raw_data[:, 0]
    inputs = raw_data.copy()
    inputs[:, 0] = 1
    return inputs, labels


# initialize weights alpha and beta with two possible options
def initialize_weights(D, M, K, flag):
    alpha = None
    beta = None
    if flag == 1:
        alpha = np.random.rand(D, M) * 0.2 - 0.1
        beta = np.random.rand(K, D + 1) * 0.2 - 0.1
        alpha[:, 0] = 0
        beta[:, 0] = 0
    if flag == 2:
        alpha = np.zeros((D, M))
        beta = np.zeros((K, D + 1))
    return alpha, beta


# linear forward computation
def linear_forward(inputs, weights):
    return inputs.dot(weights)


# sigmoid forward computation
def sigmoid_forward(inputs):
    return np.append([1], 1 / (1 + np.exp(-inputs)))  # fold the bias


# softmax forward computation
def softmax_forward(inputs):
    return softmax(inputs)


# cross entropy computation
def cross_entropy_forward(labels, pred):
    return labels.dot(np.log(pred))

# cross entropy backpropagation
def cross_entropy_backward(labels, pred, g_j=1):
    return labels / pred * g_j


# softmax backpropagation
def softmax_backward(labels, pred):
    return pred - labels


# linear backpropagation
def linear_backward(inputs, weights, gradient_comb):
    inputs = inputs.reshape((1, -1))
    gradient_comb = gradient_comb.reshape((-1, 1))
    # get gradient of weights
    gradient_weights = np.dot(gradient_comb, inputs)
    # get gradient of inputs
    tran_weights = np.transpose(weights[:, 1:])
    gradient_inputs = np.dot(tran_weights, gradient_comb)
    return gradient_weights, gradient_inputs


# sigmoid backpropagation
def sigmoid_backward(z, grad_z):
    z = z[1:]
    grad_z = grad_z.ravel()
    grad_a = np.multiply(np.multiply(grad_z, z), 1 - z)
    grad_a = grad_a.reshape(-1, 1)
    return grad_a


# train the weights using SGD and compute the cross entropies
def train(train_inputs, train_labels, test_inputs, test_labels, alpha, beta, epochs, rate, error_file):
    for epoch in range(1, epochs + 1):
        train_entropy = 0
        test_entropy = 0
        # one training example at a time
        for x, y in zip(train_inputs, train_labels):
            # forward computation
            a = linear_forward(alpha, x)
            z = sigmoid_forward(a)
            b = linear_forward(beta, z)
            y_hat = softmax_forward(b)
            # make the label same dimension as the predictions
            y_lab = np.zeros(len(y_hat))
            y_lab[int(y)] = 1
            # backward
            grad_b = softmax_backward(y_lab, y_hat)
            grad_beta, grad_z = linear_backward(z, beta, grad_b)
            grad_a = sigmoid_backward(z, grad_z)
            grad_alpha, grad_x = linear_backward(x, alpha, grad_a)
            # update the weights
            alpha -= rate * grad_alpha
            beta -= rate * grad_beta
        # train cross entropy
        for x, y in zip(train_inputs, train_labels):
            # cross entropy computation
            a = linear_forward(alpha, x)
            z = sigmoid_forward(a)
            b = linear_forward(beta, z)
            y_hat = softmax_forward(b)
            y_lab = np.zeros(len(y_hat))
            y_lab[int(y)] = 1
            # add to total
            train_entropy += cross_entropy_forward(y_lab, y_hat)
        # test cross entropy
        for x, y in zip(test_inputs, test_labels):
            # cross entropy computation
            a = linear_forward(alpha, x)
            z = sigmoid_forward(a)
            b = linear_forward(beta, z)
            y_hat = softmax_forward(b)
            y_lab_test = np.zeros(len(y_hat))
            y_lab_test[int(y)] = 1
            # add to total
            test_entropy += cross_entropy_forward(y_lab_test, y_hat)
        # compute the mean entropy
        train_entropy = -train_entropy / len(train_labels)
        test_entropy = -test_entropy / len(test_labels)
        # write to file
        with open(error_file, 'a') as metric:
            metric.write('epoch=' + str(epoch) + ' crossentropy(train): ' + str(train_entropy) + '\n')
            metric.write('epoch=' + str(epoch) + ' crossentropy(test): ' + str(test_entropy) + '\n')
    return alpha, beta


# predict the labels using alpha and beta
def predict(inputs, alpha, beta, output_file):
    pred = []
    for x in inputs:
        a = linear_forward(alpha, x)
        z = sigmoid_forward(a)
        b = linear_forward(beta, z)
        y_hat = softmax_forward(b)
        pred.append(int(np.argmax(y_hat)))
    with open(output_file, 'w') as output:
        for i in pred:
            output.write(str(i) + '\n')
    return pred


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
    with open(error_file, 'a') as output:
        output.write('error(train): ' + '{0:.6f}'.format(train_error_rate) + '\n')
        output.write('error(test): ' + '{0:.6f}'.format(test_error_rate) + '\n')
    return train_error_rate, test_error_rate


if __name__ == '__main__':
    # input file names
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    # read train and test data
    train_inputs, train_labels = read_data(train_file)
    test_inputs, test_labels = read_data(test_file)
    # output file names
    train_output_file = sys.argv[3]
    test_output_file = sys.argv[4]
    error_file = sys.argv[5]
    # parameters
    epochs = int(sys.argv[6])
    hidden_units = int(sys.argv[7])
    flag = int(sys.argv[8])
    rate = float(sys.argv[9])
    # for this program specifically, the number of classes (the K) is 10
    classes = 10
    # initialize both weights
    alpha, beta = initialize_weights(hidden_units, len(train_inputs[0]), classes, flag)
    # train
    alpha_final, beta_final = train(train_inputs, train_labels, test_inputs, test_labels, alpha, beta, epochs, rate, error_file)
    # predict both inputs
    train_pred = predict(train_inputs, alpha_final, beta_final, train_output_file)
    test_pred = predict(test_inputs, alpha_final, beta_final, test_output_file)
    # compute error
    train_error, test_error = get_error(train_labels, train_pred, test_labels, test_pred, error_file)







