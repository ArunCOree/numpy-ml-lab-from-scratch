import numpy as np

def shuffle_data(x, y):
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    return x[indices], y[indices]

def add_bias_term(x):
    bias = np.ones((x,shape[0],1))
    return np.hstack((bias,x))

