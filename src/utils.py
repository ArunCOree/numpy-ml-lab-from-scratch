import numpy as np

def shuffle_data(x, y):
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    return x[indices], y[indices]

def add_bias_term(x):
    bias = np.ones((x,shape[0],1))
    return np.hstack((bias,x))

x = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([0, 1, 0])
print("Before shuffling:")
print("x:\n", x)
print("y:\n", y)
x_shuffled, y_shuffled = shuffle_data(x, y)
print("\nAfter shuffling:")
print("x:\n", x_shuffled)
print("y:\n", y_shuffled)