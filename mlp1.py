import numpy as np

STUDENT={'name': 'Nadav Spitzer',
         'ID': '302228275'}


def softmax(x):
    """
    Compute the softmax vector.
    x: a n-dim vector (numpy array)
    returns: an n-dim vector (numpy array) of softmax values
    """
    # YOUR CODE HERE
    # Your code should be fast, so use a vectorized implementation using numpy,
    # don't use any loops.
    # With a vectorized implementation, the code should be no more than 2 lines.
    x = np.exp(x - np.max(x))
    x /= np.sum(x)
    # For numeric stability, use the identify you proved in Ex 2 Q1.
    return x


def classifier_output(x, params):
    # YOUR CODE HERE.
    W, b, U, b_tag = params

    z1 = np.dot(x, W) + b

    h = np.tanh(z1)

    z2 = np.dot(h, U) + b_tag

    probs = softmax(z2)
    return probs


def predict(x, params):
    """
    params: a list of the form [W, b, U, b_tag]
    """
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    """
    params: a list of the form [W, b, U, b_tag]

    returns:
        loss,[gW, gb, gU, gb_tag]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    gU: matrix, gradients of U
    gb_tag: vector, gradients of b_tag
    """
    # YOU CODE HERE
    y_hat = classifier_output(x, params)
    loss = -np.log(y_hat[y])

    # b_tag gradient

    return loss


def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.

    return:
    a flat list of 4 elements, W, b, U, b_tag.
    """
    params = []
    W = np.zeros((in_dim, hid_dim))
    b = np.zeros(hid_dim)
    U = np.zeros((hid_dim, out_dim))
    b_tag = np.zeros(out_dim)

    params.extend([W, b, U, b_tag])
    return params

