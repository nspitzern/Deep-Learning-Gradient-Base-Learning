import numpy as np

import mlpn as mlp
import random
from utils import *
from test_predict import predict_on_test

STUDENT = {'name': 'Nadav Spitzer',
           'ID': '302228275'}


def feats_to_vec(features):
    # YOUR CODE HERE.
    # Should return a numpy vector of features.

    # get the number of features (bigrams)
    feat_vec = np.zeros(get_common_features_number())

    # create a vector that represents the distribution of the features
    for feature in features:
        # for each feature - add 1 to the vector
        index = get_feature_index(feature)

        if index is None:  # skip unknown features
            continue
        feat_vec[index] += 1
    return feat_vec


def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        # YOUR CODE HERE
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)
        feat_vec = feats_to_vec(features)
        y_hat = mlp.predict(feat_vec, params)

        if label == y_hat:
            good += 1
        else:
            bad += 1
    return good / (good + bad)


def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    for I in range(num_iterations):
        cum_loss = 0.0  # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features)  # convert features to a vector.
            y = label  # convert the label to number if needed.
            loss, grads = mlp.loss_and_gradients(x, y, params)
            cum_loss += loss
            # YOUR CODE HERE
            # update the parameters according to the gradients
            # and the learning rate.

            for i in range(len(params)):
                params[i] -= learning_rate * grads[i]

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print(I, train_loss, train_accuracy, dev_accuracy)
    return params


if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.

    learning_rate = 1e-3
    num_iterations = 3

    train_data, dev_data = load_data('train', 'dev', 'bigrams')

    # define the sizes of the layers in the network
    dims = [get_common_features_number(), 2048, 1024, 1024, 2048, get_labels_number()]

    params = mlp.create_classifier(dims)
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)
    predict_on_test(trained_params)

