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

    global z1
    z1 = np.dot(x, W) + b

    global h
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
    W, b, U, b_tag = params

    y_hat = classifier_output(x, params)
    loss = -np.log(y_hat[y])

    # b_tag gradient
    gb_tag = y_hat.copy()
    gb_tag[y] -= 1

    # U gradient
    h_copy = h.copy()
    gU = np.outer(h_copy, y_hat.copy())
    gU[:, y] -= h_copy

    # b gradient
    z1_copy = z1.copy()
    g_tanh = 1 - np.tanh(z1_copy) ** 2
    gL_dh = np.dot(U, y_hat.copy()) - U[:, y]
    gb = gL_dh * g_tanh

    # W gradient
    gW = np.outer(x, gb.copy())

    return loss, [gW, gb, gU, gb_tag]


def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.

    return:
    a flat list of 4 elements, W, b, U, b_tag.
    """
    params = []
    # W = np.zeros((in_dim, hid_dim))
    # b = np.zeros(hid_dim)
    # U = np.zeros((hid_dim, out_dim))
    # b_tag = np.zeros(out_dim)

    W = xavier_init(in_dim, hid_dim)
    b = xavier_init(hid_dim)
    U = xavier_init(hid_dim, out_dim)
    b_tag = xavier_init(out_dim)

    params.extend([W, b, U, b_tag])
    return params


def xavier_init(in_dim, out_dim=None):
    if out_dim is not None:
        eps = (np.sqrt(6) / np.sqrt(in_dim + out_dim))
        param = np.random.uniform(-eps, eps, (in_dim, out_dim))
    else:
        eps = (np.sqrt(6) / np.sqrt(in_dim + 1))
        param = np.random.uniform(-eps, eps, (in_dim, ))

    return param


if __name__ == '__main__':
    # Sanity checks. If these fail, your gradient calculation is definitely wrong.
    # If they pass, it is likely, but not certainly, correct.
    from grad_check import gradient_check

    W, b, U, b_tag = create_classifier(3, 4, 4)

    def _loss_and_W_grad(W):
        global b, U, b_tag
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, U, b_tag])
        return loss, grads[0]

    def _loss_and_b_grad(b):
        global W, U, b_tag
        loss, grads = loss_and_gradients([1, 2, 3], 1, [W, b, U, b_tag])
        return loss, grads[1]

    def _loss_and_U_grad(U):
        global W, b, b_tag
        loss, grads = loss_and_gradients([1, 2, 3], 1, [W, b, U, b_tag])
        return loss, grads[2]

    def _loss_and_b_tag_grad(b_tag):
        global U, b, W
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, U, b_tag])
        return loss, grads[3]


    for _ in range(10):
        W = np.random.randn(W.shape[0], W.shape[1])
        b = np.random.randn(b.shape[0])
        U = np.random.randn(U.shape[0], U.shape[1])
        b_tag = np.random.randn(b_tag.shape[0])
        gradient_check(_loss_and_b_grad, b)
        gradient_check(_loss_and_W_grad, W)
        gradient_check(_loss_and_b_tag_grad, b_tag)
        gradient_check(_loss_and_U_grad, U)


