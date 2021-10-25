import numpy as np
import time

def sigmoid(t):
    sig_t = 1./(1+np.exp(-t))
    return sig_t

# A more stable version of Sigmoid, to avoid overflow!
def sigmoid(x):
    return .5 * (1 + np.tanh(.5 * x))

def compute_mse(y, tx, w):
    # Here we compute the MSE.
    N = len(y)
    e = abs(y-tx.dot(w))
    L = np.sum(e**2)/(2*N)
    return L

def compute_cross_entropy(y, tx, w, lambda_=0, with_regularizer=False):
    sig_xw = sigmoid(tx.dot(w))
    CE = np.sum(np.log(1+np.exp(tx.dot(w))) - y*tx.dot(w))/len(y)
    if with_regularizer:
       CE += lambda_/2*np.sum(w**2)
    return CE

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    # Here we borrow the batch_iter from the class to generate a minibatch iterator for a dataset.
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def compute_gradient(y, tx, w):
    # Here we compute the gradient of MSE.
    N = len(y)
    e = y-tx.dot(w)
    grad = -(tx.T).dot(e)/N
    return grad

def compute_stoch_gradient(y, tx, w):
    # Compute a stochastic gradient from just few examples n and their corresponding y_n labels.
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad

def compute_loss(y, tx, w):
    # Calculate the loss using mse
    e = y - tx.dot(w)
    return 1/2*np.mean(e**2)

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    # Optimization using Gradient Descent.
    w = initial_w
    loss = compute_mse(y, tx, w)
    for n_iter in range(max_iters):
        g = compute_gradient(y, tx, w)
        w = w-gamma*g
        loss = compute_mse(y, tx, w)
        # print("Gradient Descent({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))

    return loss, w

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    loss = compute_mse(y, tx, w)

    for i in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 1):
            g = compute_stoch_gradient(minibatch_y, minibatch_tx, w)
            w = w-gamma*g
            loss = compute_loss(y, tx, w)
        # print("SGD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(bi=i, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
                
    return loss, w

def least_squares(y, tx):
    N = len(y)
    A = (tx.T).dot(tx)
    B = (tx.T).dot(y)
    try:
        w = np.linalg.solve(A, B)
    except:
        print('Using pinv!')
        w = np.linalg.pinv(A).dot(B)
    mse = compute_mse(y, tx, w)
    return mse, w

def ridge_regression(y, tx, lambda_):
    N = len(y)
    D = tx.shape[-1]
    lambda_slide = lambda_*2*N
    w = np.linalg.inv((tx.T.dot(tx)+lambda_slide*np.eye(D))).dot(tx.T).dot(y)
    mse = compute_mse(y, tx, w)
    return mse, w

def compute_gradient_logistic(y, tx, w, lambda_=0, with_regularizer=False):
    grad = tx.T.dot(sigmoid(tx.dot(w))-y)/len(y)
    if with_regularizer:
        grad += lambda_*w
    return grad

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    y[y < 0] = 0
    w = initial_w
    
    for n_iter in range(max_iters):
        g = compute_gradient_logistic(y, tx, w)
        w = w - gamma * g
        loss = compute_cross_entropy(y, tx, w)
        print("Gradient Descent({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))

    return loss, w


def reg_logistic_regression(y_raw, tx, lambda_, initial_w, max_iters, gamma):
    y = y_raw.copy()    
    y[y<0] = 0 # change label -1 to 0
    w = initial_w
    loss = compute_cross_entropy(y, tx, w, lambda_=lambda_, with_regularizer=True)

    for n_iter in range(max_iters):
        g = compute_gradient_logistic(y, tx, w, lambda_=lambda_, with_regularizer=True)
        w = w-gamma*g
        loss = compute_cross_entropy(y, tx, w, lambda_=lambda_, with_regularizer=True)
        print("Gradient Descent({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))

    return loss, w
