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
    # params = list(reversed(params))

    global activations_list
    activations_list = [x]

    result = x

    for i in range(0, len(params) - 1, 2):
        mat = params[i]
        vec = params[i + 1]

        result = np.dot(result, mat) + vec

        # apply tanh to all layers except the last one
        if i < len(params) - 2:
            activations_list.append(result.copy())
            result = np.tanh(result)

    # apply softmax to the last layer
    probs = softmax(result)

    return probs


def predict(x, params):
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    """
    params: a list as created by create_classifier(...)

    returns:
        loss,[gW1, gb1, gW2, gb2, ...]

    loss: scalar
    gW1: matrix, gradients of W1
    gb1: vector, gradients of b1
    gW2: matrix, gradients of W2
    gb2: vector, gradients of b2
    ...

    (of course, if we request a linear classifier (ie, params is of length 2),
    you should not have gW2 and gb2.)
    """
    # YOU CODE HERE
    y_hat = classifier_output(x, params)
    loss = -np.log(y_hat[y])

    # make one hot vector out of y
    # real_y = np.zeros(y_hat)
    # real_y[y] = 1

    # gb_i = 1
    # gW_i = x
    #
    # grads = list()
    #
    # j = 0
    #
    # for i in range(0, len(params), 2):
    #     grads.extend([gW_i, gb_i])
    #
    #     g_tanh_i = 1 - np.tanh(activations_list[j]) ** 2
    #     j += 1
    #
    #     gW_i = np.outer(g_tanh_i, gW_i)
    #     gb_i = np.outer(g_tanh_i, gb_i)
    #
    # gW_i = np.outer(gW_i, y_hat)
    # gW_i[:, y] -= gW_i
    #
    # gb_i = gb_i.copy()
    # gb_i[y] -= 1
    #
    # grads.extend([gW_i.copy(), gb_i.copy()])

    j = 1

    grads = list()

    gb = y_hat.copy()
    gb[y] -= 1

    gW = np.outer(activations_list[-j], y_hat.copy())
    gW[:, y] -= activations_list[-j]

    for i in range(len(params) - 2, 0, -2):
        grads.extend([gb, gW])

        g_tanh = 1 - np.tanh(activations_list[-j]) ** 2

        z = activations_list[-j - 1]

        gW = np.outer(z, g_tanh)

        gb = g_tanh

        # increase j
        j += 1

    grads.extend([gb, gW])
    return loss, grads


def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.
    
    Assume a tanh activation function between all the layers.

    return:
    a flat list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """
    params = []

    for i in range(0, len(dims) - 1):
        in_dim = dims[i]
        out_dim = dims[i + 1]

        mat = xavier_init(in_dim, out_dim)
        vec = xavier_init(out_dim)

        params.extend([mat, vec])

    return params


def xavier_init(in_dim, out_dim=None):
    if out_dim is not None:
        eps = (np.sqrt(6) / np.sqrt(in_dim + out_dim))
        param = np.random.uniform(-eps, eps, (in_dim, out_dim))
    else:
        eps = (np.sqrt(6) / np.sqrt(in_dim + 1))
        param = np.random.uniform(-eps, eps, (in_dim, ))

    return param

